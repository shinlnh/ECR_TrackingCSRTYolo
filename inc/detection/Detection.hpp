#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "pipeline/Types.hpp"

namespace detection {

struct Detection {
    std::string label;
    float score{0.0f};
    pipeline::BoundingBox box;
};

class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) = 0;
};

}  // namespace detection
