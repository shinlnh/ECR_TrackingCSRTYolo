#pragma once

#include "smoothing/BoxKalmanFilter.hpp"

namespace config {

struct QualityConfig {
    float minArea{400.0f};
    float minAspectRatio{0.1f};
    float maxAspectRatio{6.0f};
    float minIoU{0.05f};
};

struct RescueConfig {
    int intervalFrames{12};
    float roiScale{1.6f};
    float minIoU{0.1f};
    int fullFrameInterval{45};
    int maxLostFrames{180};
    float forceRescueIoU{0.5f};
};

struct PipelineConfig {
    QualityConfig quality{};
    RescueConfig rescue{};
    smoothing::KalmanParams kalman{};
};

}  // namespace config
