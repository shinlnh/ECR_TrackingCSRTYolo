#include "detection/NCNNSsdDetector.hpp"

#include <vector>

namespace detection {

NCNNSsdDetector::NCNNSsdDetector(std::string modelPath, cv::Size inputSize)
    : modelPath_(std::move(modelPath)), inputSize_(inputSize) {
    loadModel();
}

void NCNNSsdDetector::loadModel() {
    // TODO: Integrate actual NCNN/Vulkan model loading.
}

std::vector<Detection> NCNNSsdDetector::detect(
    const cv::Mat& frame,
    const std::optional<pipeline::BoundingBox>& roi) {
    cv::Mat region = frame;
    if (roi) {
        const auto rect = roi->toRect();
        const cv::Rect roiRect(
            static_cast<int>(rect.x),
            static_cast<int>(rect.y),
            static_cast<int>(rect.width),
            static_cast<int>(rect.height));
        region = frame(roiRect & cv::Rect(0, 0, frame.cols, frame.rows));
    }

    static_cast<void>(region);
    return {};
}

}  // namespace detection
