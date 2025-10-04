#pragma once

#include <optional>

#include <opencv2/tracking.hpp>

#include "pipeline/Types.hpp"

namespace tracking {

class CsrtTracker {
public:
    CsrtTracker();

    bool initialize(const cv::Mat& frame, const pipeline::BoundingBox& box);
    std::optional<pipeline::BoundingBox> update(const cv::Mat& frame);
    void reset();

private:
    cv::Ptr<cv::TrackerCSRT> tracker_;
    bool initialized_{false};
};

}  // namespace tracking
