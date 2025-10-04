#include "rescue/RescueStrategy.hpp"

#include <algorithm>
#include <tuple>

namespace rescue {

RescueStrategy::RescueStrategy(
    detection::Detector& detector,
    float roiScale,
    int triggerInterval,
    float minIoU)
    : detector_(detector),
      roiScale_(roiScale),
      triggerInterval_(triggerInterval),
      minIoU_(minIoU) {}

bool RescueStrategy::shouldTrigger(int frameIndex, int lastSuccessFrame) {
    if (frameIndex - lastSuccessFrame < triggerInterval_) {
        return false;
    }
    if (lastTriggerFrame_ == frameIndex) {
        return false;
    }
    lastTriggerFrame_ = frameIndex;
    return true;
}

std::optional<pipeline::BoundingBox> RescueStrategy::recover(
    const cv::Mat& frame,
    pipeline::TrackingTarget& target,
    int frameIndex,
    std::tuple<int, int> frameSize,
    const std::optional<pipeline::BoundingBox>& lastKnownBox) {
    static_cast<void>(frameIndex);
    const pipeline::BoundingBox reference = lastKnownBox.value_or(target.box);
    const auto [frameWidth, frameHeight] = frameSize;
    pipeline::BoundingBox roi = reference.scale(roiScale_)
                                       .clamp(static_cast<float>(frameWidth), static_cast<float>(frameHeight));

    auto detections = detector_.detect(frame, roi);
    auto candidates = filterCandidates(detections, target.label, reference, roi);
    if (candidates.empty()) {
        return std::nullopt;
    }

    const auto best = std::max_element(
        candidates.begin(),
        candidates.end(),
        [](const auto& lhs, const auto& rhs) {
            if (lhs.second != rhs.second) {
                return lhs.second < rhs.second;
            }
            return lhs.first.score < rhs.first.score;
        });

    const pipeline::BoundingBox translated = translateBox(best->first.box, roi);
    target.update(translated, best->first.score);
    return translated;
}

std::vector<std::pair<detection::Detection, float>> RescueStrategy::filterCandidates(
    const std::vector<detection::Detection>& detections,
    const std::string& label,
    const pipeline::BoundingBox& reference,
    const pipeline::BoundingBox& roi) const {
    std::vector<std::pair<detection::Detection, float>> filtered;
    filtered.reserve(detections.size());
    for (const auto& detection : detections) {
        if (detection.label != label) {
            continue;
        }
        const pipeline::BoundingBox globalBox = translateBox(detection.box, roi);
        const float overlap = globalBox.iou(reference);
        if (overlap < minIoU_) {
            continue;
        }
        detection::Detection updatedDetection = detection;
        updatedDetection.box = detection.box;  // keep relative box for translation step
        filtered.emplace_back(updatedDetection, overlap);
    }
    return filtered;
}

pipeline::BoundingBox RescueStrategy::translateBox(
    const pipeline::BoundingBox& box,
    const pipeline::BoundingBox& roi) {
    return {box.x + roi.x, box.y + roi.y, box.width, box.height};
}

}  // namespace rescue
