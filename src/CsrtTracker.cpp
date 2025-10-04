#include "tracking/CsrtTracker.hpp"

#include <cmath>

namespace tracking {

CsrtTracker::CsrtTracker() : tracker_(cv::TrackerCSRT::create()) {}

bool CsrtTracker::initialize(const cv::Mat& frame, const pipeline::BoundingBox& box) {
    tracker_ = cv::TrackerCSRT::create();
    const cv::Rect rect(
        static_cast<int>(std::round(box.x)),
        static_cast<int>(std::round(box.y)),
        static_cast<int>(std::round(box.width)),
        static_cast<int>(std::round(box.height)));
    tracker_->init(frame, rect);
    initialized_ = true;
    return initialized_;
}

std::optional<pipeline::BoundingBox> CsrtTracker::update(const cv::Mat& frame) {
    if (!initialized_) {
        return std::nullopt;
    }
    cv::Rect rect;
    if (!tracker_->update(frame, rect)) {
        initialized_ = false;
        return std::nullopt;
    }
    return pipeline::BoundingBox(
        static_cast<float>(rect.x),
        static_cast<float>(rect.y),
        static_cast<float>(rect.width),
        static_cast<float>(rect.height));
}

void CsrtTracker::reset() {
    tracker_ = cv::TrackerCSRT::create();
    initialized_ = false;
}

}  // namespace tracking

