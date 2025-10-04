#include "detection/TemplateMatcherDetector.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {

cv::Rect ensureMinimumSize(const cv::Rect& rect, const cv::Size& minSize, const cv::Size& bounds) {
    cv::Rect expanded = rect;
    if (expanded.width < minSize.width) {
        const int deficit = minSize.width - expanded.width;
        expanded.x = std::max(0, expanded.x - deficit / 2);
        expanded.width = std::min(bounds.width - expanded.x, expanded.width + deficit);
    }
    if (expanded.height < minSize.height) {
        const int deficit = minSize.height - expanded.height;
        expanded.y = std::max(0, expanded.y - deficit / 2);
        expanded.height = std::min(bounds.height - expanded.y, expanded.height + deficit);
    }
    return expanded;
}

cv::Rect enlargeWithPadding(const cv::Rect& rect, int padding, const cv::Size& bounds) {
    cv::Rect padded = rect;
    padded.x = std::max(0, padded.x - padding);
    padded.y = std::max(0, padded.y - padding);
    padded.width = std::min(bounds.width - padded.x, padded.width + padding * 2);
    padded.height = std::min(bounds.height - padded.y, padded.height + padding * 2);
    return padded;
}

}  // namespace

namespace detection {

TemplateMatcherDetector::TemplateMatcherDetector(const cv::Mat& initialFrame,
                                                 const pipeline::BoundingBox& initialBox,
                                                 std::string label,
                                                 double matchThreshold,
                                                 int searchPadding,
                                                 double templateUpdateRate)
    : templateSize_(
          std::max(8, static_cast<int>(std::round(initialBox.width))),
          std::max(8, static_cast<int>(std::round(initialBox.height)))),
      lastBox_(initialBox),
      label_(std::move(label)),
      threshold_(matchThreshold),
      padding_(searchPadding),
      updateRate_(templateUpdateRate) {
    templateGray_ = buildTemplate(initialFrame, initialBox);
    if (templateGray_.empty()) {
        templateSize_ = {std::max(8, static_cast<int>(initialFrame.cols * 0.05)),
                         std::max(8, static_cast<int>(initialFrame.rows * 0.05))};
        pipeline::BoundingBox fallback{0.0f, 0.0f,
                                       static_cast<float>(templateSize_.width),
                                       static_cast<float>(templateSize_.height)};
        templateGray_ = buildTemplate(initialFrame, fallback);
        lastBox_ = clampToImage(initialBox, initialFrame.size());
    }
}

std::vector<Detection> TemplateMatcherDetector::detect(
    const cv::Mat& frame,
    const std::optional<pipeline::BoundingBox>& roi) {
    if (templateGray_.empty() || frame.empty()) {
        return {};
    }

    const cv::Size frameSize = frame.size();
    cv::Rect searchRect;
    if (roi) {
        searchRect = clampRect(roi->toRect(), frameSize);
    } else {
        cv::Rect base = clampRect(lastBox_.toRect(), frameSize);
        searchRect = enlargeWithPadding(base, padding_, frameSize);
    }

    searchRect = ensureMinimumSize(searchRect, templateSize_, frameSize);
    if (searchRect.width < templateSize_.width || searchRect.height < templateSize_.height) {
        return {};
    }

    cv::Mat region = frame(searchRect).clone();
    cv::Mat regionGray;
    cv::cvtColor(region, regionGray, cv::COLOR_BGR2GRAY);
    regionGray.convertTo(regionGray, CV_32F, 1.0 / 255.0);

    cv::Mat result;
    const cv::Size resultSize{
        regionGray.cols - templateGray_.cols + 1,
        regionGray.rows - templateGray_.rows + 1};

    if (resultSize.width <= 0 || resultSize.height <= 0) {
        return {};
    }

    cv::matchTemplate(regionGray, templateGray_, result, cv::TM_CCOEFF_NORMED);

    double maxVal = 0.0;
    double minVal = 0.0;
    cv::Point maxLoc;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    if (!std::isfinite(maxVal) || maxVal < threshold_) {
        return {};
    }

    const float width = static_cast<float>(templateGray_.cols);
    const float height = static_cast<float>(templateGray_.rows);
    const float globalX = static_cast<float>(searchRect.x + maxLoc.x);
    const float globalY = static_cast<float>(searchRect.y + maxLoc.y);

    pipeline::BoundingBox globalBox{globalX, globalY, width, height};
    lastBox_ = clampToImage(globalBox, frameSize);

    refreshTemplate(frame, lastBox_);

    pipeline::BoundingBox reportedBox = roi
        ? pipeline::BoundingBox{
              static_cast<float>(maxLoc.x),
              static_cast<float>(maxLoc.y),
              width,
              height}
        : lastBox_;

    Detection detection;
    detection.label = label_;
    detection.score = static_cast<float>(maxVal);
    detection.box = reportedBox;
    return {detection};
}

void TemplateMatcherDetector::updateTrackedBox(const cv::Mat& frame, const pipeline::BoundingBox& box) {
    lastBox_ = clampToImage(box, frame.size());
    refreshTemplate(frame, lastBox_);
}

pipeline::BoundingBox TemplateMatcherDetector::clampToImage(const pipeline::BoundingBox& box, const cv::Size& size) const {
    return box.clamp(static_cast<float>(size.width), static_cast<float>(size.height));
}

cv::Rect TemplateMatcherDetector::clampRect(const cv::Rect2f& rect, const cv::Size& size) const {
    const float x = std::clamp(rect.x, 0.0f, static_cast<float>(size.width));
    const float y = std::clamp(rect.y, 0.0f, static_cast<float>(size.height));
    const float maxW = std::max(0.0f, static_cast<float>(size.width) - x);
    const float maxH = std::max(0.0f, static_cast<float>(size.height) - y);
    const float width = std::clamp(rect.width, 0.0f, maxW);
    const float height = std::clamp(rect.height, 0.0f, maxH);
    return {
        static_cast<int>(std::floor(x)),
        static_cast<int>(std::floor(y)),
        std::max(1, static_cast<int>(std::round(width))),
        std::max(1, static_cast<int>(std::round(height)))};
}

cv::Mat TemplateMatcherDetector::buildTemplate(const cv::Mat& frame, const pipeline::BoundingBox& box) const {
    if (frame.empty() || box.width <= 1.0f || box.height <= 1.0f) {
        return {};
    }
    const cv::Rect region = clampRect(box.toRect(), frame.size());
    if (region.width <= 1 || region.height <= 1) {
        return {};
    }
    cv::Mat patch = frame(region).clone();
    cv::Mat gray;
    cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    if (templateSize_.width > 0 && templateSize_.height > 0 &&
        (gray.cols != templateSize_.width || gray.rows != templateSize_.height)) {
        cv::resize(gray, gray, templateSize_);
    }
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    cv::Mat normalized;
    cv::normalize(gray, normalized, 0.0, 1.0, cv::NORM_MINMAX);
    return normalized;
}

void TemplateMatcherDetector::refreshTemplate(const cv::Mat& frame, const pipeline::BoundingBox& box) {
    if (templateGray_.empty()) {
        templateGray_ = buildTemplate(frame, box);
        return;
    }

    cv::Mat fresh = buildTemplate(frame, box);
    if (fresh.empty() || fresh.size() != templateGray_.size()) {
        return;
    }

    templateGray_ = templateGray_ * (1.0 - updateRate_) + fresh * updateRate_;
}

}  // namespace detection
