#include "pipeline/TrackingPipeline.hpp"

#include <algorithm>
#include <tuple>
#include <utility>

#include "tracking/Quality.hpp"

namespace pipeline {

TrackingPipeline::TrackingPipeline(config::PipelineConfig config)
    : detector_(nullptr),
      config_(std::move(config)),
      tracker_(std::make_unique<tracking::CsrtTracker>()),
      kalman_(std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman)) {}

TrackingPipeline::TrackingPipeline(detection::Detector& detector, config::PipelineConfig config)
    : TrackingPipeline(std::move(config)) {
    detector_ = &detector;
    rescue_ = std::make_unique<rescue::RescueStrategy>(
        *detector_,
        config_.rescue.roiScale,
        config_.rescue.intervalFrames,
        config_.rescue.minIoU);
}

void TrackingPipeline::initializeWithBox(
    const cv::Mat& frame,
    const BoundingBox& box,
    const std::string& label,
    float score) {
    target_ = TrackingTarget{nextTrackId_++, label, box, score};
    if (!tracker_) {
        tracker_ = std::make_unique<tracking::CsrtTracker>();
    } else {
        tracker_->reset();
    }
    tracker_->initialize(frame, box);

    if (!kalman_) {
        kalman_ = std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman);
    }
    kalman_->reset(box);

    lastSuccessFrame_ = frameIndex_;
    lastFullFrameAttempt_ = frameIndex_;
}

void TrackingPipeline::clear() {
    target_.reset();
    if (tracker_) {
        tracker_->reset();
    }
    kalman_.reset();
    lastSuccessFrame_ = frameIndex_;
    lastFullFrameAttempt_ = frameIndex_;
}

PipelineOutput TrackingPipeline::step(const cv::Mat& frame) {
    return step(frame, std::nullopt);
}

PipelineOutput TrackingPipeline::step(
    const cv::Mat& frame,
    const std::optional<BoundingBox>& groundTruth) {
    ++frameIndex_;
    const std::tuple<int, int> frameSize{frame.cols, frame.rows};

    if (!target_) {
        const auto detection = initializeTarget(frame);
        PipelineOutput output{};
        output.frameIndex = frameIndex_;
        output.target = target_;
        if (target_) {
            output.smoothedBox = target_->box;
            output.rawBox = target_->box;
        }
        output.detection = detection;
        output.lost = !target_.has_value();
        return output;
    }

    const auto [rawBox, qualityValid] = track(frame, frameSize);

    const bool detectorAvailable = detector_ != nullptr;
    bool forceRescue = false;
    if (detectorAvailable && rawBox && groundTruth && config_.rescue.forceRescueIoU > 0.0f) {
        const float groundTruthIoU = rawBox->iou(*groundTruth);
        forceRescue = groundTruthIoU < config_.rescue.forceRescueIoU;
    }

    const bool measurementValid = qualityValid && !forceRescue;
    auto smoothedBox = smooth(rawBox, measurementValid);

    bool finalValid = measurementValid;
    auto mutableRaw = rawBox;
    if (!measurementValid) {
        const auto recovered = attemptRescue(frame, frameSize, rawBox);
        if (recovered) {
            mutableRaw = recovered;
            smoothedBox = smooth(recovered, true);
            finalValid = true;
        }
    }

    if (finalValid && mutableRaw) {
        target_->update(*mutableRaw);
        lastSuccessFrame_ = frameIndex_;
    }

    PipelineOutput output{};
    output.frameIndex = frameIndex_;
    output.target = target_;
    output.rawBox = mutableRaw;
    output.smoothedBox = smoothedBox;
    output.lost = !finalValid;
    return output;
}


std::optional<detection::Detection> TrackingPipeline::initializeTarget(const cv::Mat& frame) {
    if (!detector_) {
        return std::nullopt;
    }

    auto detections = detector_->detect(frame);
    if (detections.empty()) {
        return std::nullopt;
    }
    const auto bestIt = std::max_element(
        detections.begin(),
        detections.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.score < rhs.score; });
    target_ = TrackingTarget{nextTrackId_++, bestIt->label, bestIt->box, bestIt->score};
    if (!tracker_) {
        tracker_ = std::make_unique<tracking::CsrtTracker>();
    }
    tracker_->initialize(frame, bestIt->box);
    if (!kalman_) {
        kalman_ = std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman);
    }
    kalman_->reset(bestIt->box);
    lastSuccessFrame_ = frameIndex_;
    lastFullFrameAttempt_ = frameIndex_;
    return *bestIt;
}

std::pair<std::optional<BoundingBox>, bool> TrackingPipeline::track(
    const cv::Mat& frame,
    std::tuple<int, int> frameSize) {
    const auto tracked = tracker_->update(frame);
    if (!tracked) {
        return {std::nullopt, false};
    }
    const bool valid = tracking::isQualityAcceptable(
        *tracked,
        target_ ? std::optional<BoundingBox>(target_->box) : std::nullopt,
        frameSize,
        config_.quality.minArea,
        config_.quality.minAspectRatio,
        config_.quality.maxAspectRatio,
        config_.quality.minIoU);
    return {tracked, valid};
}

std::optional<BoundingBox> TrackingPipeline::smooth(
    const std::optional<BoundingBox>& box,
    bool validMeasurement) {
    if (!kalman_) {
        kalman_ = std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman);
    }
    return kalman_->smooth(box, validMeasurement);
}

std::optional<BoundingBox> TrackingPipeline::attemptRescue(
    const cv::Mat& frame,
    std::tuple<int, int> frameSize,
    const std::optional<BoundingBox>& rawBox) {
    if (!target_) {
        return std::nullopt;
    }

    const int framesSinceSuccess = (lastSuccessFrame_ >= 0) ? (frameIndex_ - lastSuccessFrame_) : frameIndex_;

    bool roiAttemptFailed = false;
    if (rescue_ && rescue_->shouldTrigger(frameIndex_, lastSuccessFrame_)) {
        const auto recovered = rescue_->recover(
            frame,
            *target_,
            frameIndex_,
            frameSize,
            rawBox ? std::optional<BoundingBox>(*rawBox) : std::optional<BoundingBox>(target_->box));
        if (recovered) {
            reinitializeTrackers(frame, *recovered);
            lastSuccessFrame_ = frameIndex_;
            lastFullFrameAttempt_ = frameIndex_;
            return recovered;
        }
        roiAttemptFailed = true;
    }

    const bool shouldRunFullFrameRecovery =
        detector_ &&
        (lastFullFrameAttempt_ != frameIndex_) &&
        ((config_.rescue.fullFrameInterval > 0 && framesSinceSuccess >= config_.rescue.fullFrameInterval) ||
         roiAttemptFailed);

    if (shouldRunFullFrameRecovery) {
        lastFullFrameAttempt_ = frameIndex_;
        if (auto detection = runFullFrameRecovery(frame)) {
            target_->update(detection->box, detection->score);
            reinitializeTrackers(frame, detection->box);
            lastSuccessFrame_ = frameIndex_;
            return detection->box;
        }
    }

    if (config_.rescue.maxLostFrames > 0 && framesSinceSuccess >= config_.rescue.maxLostFrames) {
        target_.reset();
        tracker_.reset();
        kalman_.reset();
        lastSuccessFrame_ = frameIndex_;
        lastFullFrameAttempt_ = frameIndex_;
    }

    return std::nullopt;
}

void TrackingPipeline::reinitializeTrackers(const cv::Mat& frame, const BoundingBox& box) {
    if (!tracker_) {
        tracker_ = std::make_unique<tracking::CsrtTracker>();
    } else {
        tracker_->reset();
    }
    tracker_->initialize(frame, box);

    if (!kalman_) {
        kalman_ = std::make_unique<smoothing::BoxKalmanFilter>(config_.kalman);
    }
    kalman_->reset(box);
}

std::optional<detection::Detection> TrackingPipeline::runFullFrameRecovery(const cv::Mat& frame) {
    if (!detector_) {
        return std::nullopt;
    }

    auto detections = detector_->detect(frame);
    const detection::Detection* bestMatch = nullptr;
    for (const auto& detection : detections) {
        if (detection.label != target_->label) {
            continue;
        }
        if (!bestMatch || detection.score > bestMatch->score) {
            bestMatch = &detection;
        }
    }
    if (!bestMatch) {
        return std::nullopt;
    }
    return *bestMatch;
}

}  // namespace pipeline


