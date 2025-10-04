#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "detection/Detection.hpp"

namespace detection {

class YoloV5Detector final : public Detector {
public:
    YoloV5Detector(const std::string& modelPath,
                   const std::string& labelPath,
                   float confidenceThreshold,
                   float nmsThreshold,
                   std::optional<std::string> focusLabel,
                   std::string reportedLabel,
                   int inputSize = 640);

    std::vector<Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) override;

private:
    void loadLabels(const std::string& labelPath);
    std::optional<int> resolveFocusLabel(const std::optional<std::string>& focusLabel) const;
    static cv::Mat letterbox(const cv::Mat& image, int inputSize, float& scale, int& padX, int& padY);

    cv::dnn::Net net_;
    std::vector<std::string> labels_;
    float confidenceThreshold_{0.25f};
    float nmsThreshold_{0.45f};
    std::optional<int> focusClassId_;
    std::string reportedLabel_;
    int inputSize_{640};
};

}  // namespace detection
