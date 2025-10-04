#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "detection/Detection.hpp"

namespace detection {

class NCNNSsdDetector final : public Detector {
public:
    explicit NCNNSsdDetector(std::string modelPath, cv::Size inputSize = {300, 300});

    std::vector<Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) override;

private:
    void loadModel();

    std::string modelPath_;
    cv::Size inputSize_;
};

}  // namespace detection
