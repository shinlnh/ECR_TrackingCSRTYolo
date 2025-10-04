#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "detection/Detection.hpp"

namespace detection {

class OnnxSsdDetector final : public detection::Detector {
public:
    OnnxSsdDetector(const std::string& modelPath,
                    const std::string& labelPath,
                    float confidenceThreshold,
                    std::optional<std::string> focusLabel,
                    std::string reportedLabel,
                    cv::Size inputSize = {300, 300});

    std::vector<Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) override;

private:
    void loadLabels(const std::string& labelPath);

    cv::dnn::Net net_;
    std::vector<std::string> labels_;
    float confidenceThreshold_{0.4f};
    std::optional<int> focusClassId_;
    std::string reportedLabel_;
    cv::Size inputSize_;
    double scale_{1.0 / 127.5};
    cv::Scalar mean_{127.5, 127.5, 127.5};
    bool swapRB_{true};
};

}  // namespace detection

