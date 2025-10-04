#include "detection/StaticBoxDetector.hpp"

#include <vector>

namespace detection {

StaticBoxDetector::StaticBoxDetector(std::string label, std::optional<pipeline::BoundingBox> predefined)
    : label_(std::move(label)), box_(predefined) {}

std::vector<Detection> StaticBoxDetector::detect(
    const cv::Mat& frame,
    const std::optional<pipeline::BoundingBox>& roi) {
    const int width = frame.cols;
    const int height = frame.rows;

    pipeline::BoundingBox box = resolveBox(width, height);
    if (roi) {
        pipeline::BoundingBox clipped = box.intersect(*roi);
        if (clipped.area() <= 0.0f) {
            return {};
        }
        pipeline::BoundingBox relative(
            clipped.x - roi->x,
            clipped.y - roi->y,
            clipped.width,
            clipped.height);
        return {Detection{label_, 0.85f, relative}};
    }
    return {Detection{label_, 0.9f, box}};
}

pipeline::BoundingBox StaticBoxDetector::resolveBox(int width, int height) const {
    if (box_) {
        return *box_;
    }
    const float w = static_cast<float>(width) * 0.2f;
    const float h = static_cast<float>(height) * 0.3f;
    const float x = (static_cast<float>(width) - w) * 0.5f;
    const float y = (static_cast<float>(height) - h) * 0.5f;
    return {x, y, w, h};
}

}  // namespace detection
