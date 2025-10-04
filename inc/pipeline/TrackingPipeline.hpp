#pragma once

#include <memory>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "config/Config.hpp"
#include "detection/Detection.hpp"
#include "pipeline/PipelineOutput.hpp"
#include "rescue/RescueStrategy.hpp"
#include "smoothing/BoxKalmanFilter.hpp"
#include "tracking/CsrtTracker.hpp"

namespace pipeline {

class TrackingPipeline {
public:
    explicit TrackingPipeline(config::PipelineConfig config = {});
    TrackingPipeline(detection::Detector& detector, config::PipelineConfig config = {});

    void initializeWithBox(
        const cv::Mat& frame,
        const BoundingBox& box,
        const std::string& label = "target",
        float score = 1.0f);

    bool hasTarget() const { return target_.has_value(); }
    void clear();

    PipelineOutput step(const cv::Mat& frame);
    PipelineOutput step(const cv::Mat& frame, const std::optional<BoundingBox>& groundTruth);

private:
    std::optional<detection::Detection> initializeTarget(const cv::Mat& frame);
    std::pair<std::optional<BoundingBox>, bool> track(const cv::Mat& frame, std::tuple<int, int> frameSize);
    std::optional<BoundingBox> smooth(const std::optional<BoundingBox>& box, bool validMeasurement);
    std::optional<BoundingBox> attemptRescue(
        const cv::Mat& frame,
        std::tuple<int, int> frameSize,
        const std::optional<BoundingBox>& rawBox);
    void reinitializeTrackers(const cv::Mat& frame, const BoundingBox& box);
    std::optional<detection::Detection> runFullFrameRecovery(const cv::Mat& frame);

    detection::Detector* detector_{nullptr};
    config::PipelineConfig config_;
    std::unique_ptr<tracking::CsrtTracker> tracker_;
    std::unique_ptr<smoothing::BoxKalmanFilter> kalman_;
    std::unique_ptr<rescue::RescueStrategy> rescue_;
    std::optional<TrackingTarget> target_;
    int frameIndex_{0};
    int lastSuccessFrame_{-1};
    int nextTrackId_{1};
    int lastFullFrameAttempt_{-1};
};

}  // namespace pipeline

