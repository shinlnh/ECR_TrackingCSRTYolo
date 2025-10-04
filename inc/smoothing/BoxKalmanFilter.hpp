#pragma once

#include <optional>

#include <opencv2/video/tracking.hpp>

#include "pipeline/Types.hpp"

namespace smoothing {

struct KalmanParams {
    float processNoise{1e-2f};
    float measurementNoise{1e-1f};
    float clampPosition{15.0f};
    float clampScale{20.0f};
};

class BoxKalmanFilter {
public:
    explicit BoxKalmanFilter(const KalmanParams& params = {});

    void reset(const pipeline::BoundingBox& box);

    std::optional<pipeline::BoundingBox> smooth(
        const std::optional<pipeline::BoundingBox>& box,
        bool validMeasurement);

private:
    pipeline::BoundingBox predict();
    pipeline::BoundingBox correct(const pipeline::BoundingBox& measurement);
    pipeline::BoundingBox clampBox(const pipeline::BoundingBox& box) const;

    KalmanParams params_{};
    cv::KalmanFilter filter_;
    bool initialized_{false};
    std::optional<pipeline::BoundingBox> lastBox_;
};

}  // namespace smoothing
