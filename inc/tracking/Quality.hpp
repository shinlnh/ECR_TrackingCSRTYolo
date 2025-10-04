#pragma once

#include <optional>
#include <tuple>

#include "pipeline/Types.hpp"

namespace tracking {

bool isBoxInsideFrame(const pipeline::BoundingBox& box, float frameWidth, float frameHeight);

bool isBoxSizeValid(const pipeline::BoundingBox& box, float minArea);

bool isAspectRatioValid(const pipeline::BoundingBox& box, float minRatio, float maxRatio);

bool isQualityAcceptable(
    const pipeline::BoundingBox& box,
    const std::optional<pipeline::BoundingBox>& previous,
    std::tuple<int, int> frameSize,
    float minArea,
    float minRatio,
    float maxRatio,
    float minIoU);

}  // namespace tracking
