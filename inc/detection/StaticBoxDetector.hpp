#pragma once

#include <optional>
#include <string>

#include "detection/Detection.hpp"

namespace detection {

class StaticBoxDetector final : public Detector {
public:
    explicit StaticBoxDetector(
        std::string label = "person",
        std::optional<pipeline::BoundingBox> predefined = std::nullopt);

    std::vector<Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) override;

private:
    pipeline::BoundingBox resolveBox(int width, int height) const;

    std::string label_;
    std::optional<pipeline::BoundingBox> box_;
};

}  // namespace detection
