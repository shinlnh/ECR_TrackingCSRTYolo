#include "tracking/Quality.hpp"

namespace tracking {

bool isBoxInsideFrame(const pipeline::BoundingBox& box, float frameWidth, float frameHeight) {
    const float x = box.x;
    const float y = box.y;
    const float w = box.width;
    const float h = box.height;
    return x >= 0.0f && y >= 0.0f && (x + w) <= frameWidth && (y + h) <= frameHeight;
}

bool isBoxSizeValid(const pipeline::BoundingBox& box, float minArea) {
    return box.area() >= minArea;
}

bool isAspectRatioValid(const pipeline::BoundingBox& box, float minRatio, float maxRatio) {
    if (box.height == 0.0f) {
        return false;
    }
    const float ratio = box.width / box.height;
    return ratio >= minRatio && ratio <= maxRatio;
}

bool isQualityAcceptable(
    const pipeline::BoundingBox& box,
    const std::optional<pipeline::BoundingBox>& previous,
    std::tuple<int, int> frameSize,
    float minArea,
    float minRatio,
    float maxRatio,
    float minIoU) {
    const auto [width, height] = frameSize;
    if (!isBoxInsideFrame(box, static_cast<float>(width), static_cast<float>(height))) {
        return false;
    }
    if (!isBoxSizeValid(box, minArea)) {
        return false;
    }
    if (!isAspectRatioValid(box, minRatio, maxRatio)) {
        return false;
    }
    if (previous && box.iou(*previous) < minIoU) {
        return false;
    }
    return true;
}

}  // namespace tracking
