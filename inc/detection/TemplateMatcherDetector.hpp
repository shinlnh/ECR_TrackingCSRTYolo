#pragma once

#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "detection/Detection.hpp"

namespace detection {

class TemplateMatcherDetector final : public detection::Detector {
public:
    TemplateMatcherDetector(const cv::Mat& initialFrame,
                            const pipeline::BoundingBox& initialBox,
                            std::string label,
                            double matchThreshold = 0.65,
                            int searchPadding = 48,
                            double templateUpdateRate = 0.18);

    std::vector<Detection> detect(
        const cv::Mat& frame,
        const std::optional<pipeline::BoundingBox>& roi = std::nullopt) override;

    void updateTrackedBox(const cv::Mat& frame, const pipeline::BoundingBox& box);

    void setPadding(int padding) { padding_ = padding; }

private:
    pipeline::BoundingBox clampToImage(const pipeline::BoundingBox& box, const cv::Size& size) const;
    cv::Rect clampRect(const cv::Rect2f& rect, const cv::Size& size) const;
    cv::Mat buildTemplate(const cv::Mat& frame, const pipeline::BoundingBox& box) const;
    void refreshTemplate(const cv::Mat& frame, const pipeline::BoundingBox& box);

    cv::Mat templateGray_;  // CV_32F, normalised to [0,1]
    cv::Size templateSize_;
    pipeline::BoundingBox lastBox_;
    std::string label_;
    double threshold_;
    int padding_;
    double updateRate_;
};

}  // namespace detection

