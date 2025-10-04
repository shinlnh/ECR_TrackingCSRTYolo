#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "detection/Detection.hpp"
#include "pipeline/Types.hpp"

namespace rescue {

class RescueStrategy {
public:
    RescueStrategy(
        detection::Detector& detector,
        float roiScale,
        int triggerInterval,
        float minIoU);

    bool shouldTrigger(int frameIndex, int lastSuccessFrame);

    std::optional<pipeline::BoundingBox> recover(
        const cv::Mat& frame,
        pipeline::TrackingTarget& target,
        int frameIndex,
        std::tuple<int, int> frameSize,
        const std::optional<pipeline::BoundingBox>& lastKnownBox);

private:
    std::vector<std::pair<detection::Detection, float>> filterCandidates(
        const std::vector<detection::Detection>& detections,
        const std::string& label,
        const pipeline::BoundingBox& reference,
        const pipeline::BoundingBox& roi) const;

    static pipeline::BoundingBox translateBox(
        const pipeline::BoundingBox& box,
        const pipeline::BoundingBox& roi);

    detection::Detector& detector_;
    float roiScale_{1.6f};
    int triggerInterval_{12};
    float minIoU_{0.1f};
    int lastTriggerFrame_{-1};
};

}  // namespace rescue
