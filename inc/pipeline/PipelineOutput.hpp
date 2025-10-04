#pragma once

#include <optional>

#include "detection/Detection.hpp"
#include "pipeline/Types.hpp"

namespace pipeline {

struct PipelineOutput {
    int frameIndex{0};
    std::optional<TrackingTarget> target;
    std::optional<BoundingBox> smoothedBox;
    std::optional<BoundingBox> rawBox;
    std::optional<detection::Detection> detection;
    bool lost{true};
};

}  // namespace pipeline
