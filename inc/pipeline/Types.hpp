#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <tuple>

#include <opencv2/core.hpp>

namespace pipeline {

struct BoundingBox {
    float x{0.0f};
    float y{0.0f};
    float width{0.0f};
    float height{0.0f};

    constexpr BoundingBox() = default;

    constexpr BoundingBox(float x_, float y_, float w_, float h_)
        : x(x_), y(y_), width(w_), height(h_) {}

    [[nodiscard]] float area() const {
        return std::max(width, 0.0f) * std::max(height, 0.0f);
    }

    [[nodiscard]] std::array<float, 2> center() const {
        return {x + width * 0.5f, y + height * 0.5f};
    }

    [[nodiscard]] cv::Rect2f toRect() const {
        return {x, y, width, height};
    }

    [[nodiscard]] BoundingBox intersect(const BoundingBox& other) const {
        const float x1 = std::max(x, other.x);
        const float y1 = std::max(y, other.y);
        const float x2 = std::min(x + width, other.x + other.width);
        const float y2 = std::min(y + height, other.y + other.height);
        return {x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1)};
    }

    [[nodiscard]] float iou(const BoundingBox& other) const {
        const float inter = intersect(other).area();
        const float uni = area() + other.area() - inter;
        return (uni > 0.0f) ? (inter / uni) : 0.0f;
    }

    [[nodiscard]] BoundingBox clamp(float frameWidth, float frameHeight) const {
        const float clampedX = std::clamp(x, 0.0f, frameWidth);
        const float clampedY = std::clamp(y, 0.0f, frameHeight);
        const float maxW = std::max(0.0f, frameWidth - clampedX);
        const float maxH = std::max(0.0f, frameHeight - clampedY);
        const float clampedW = std::clamp(width, 0.0f, maxW);
        const float clampedH = std::clamp(height, 0.0f, maxH);
        return {clampedX, clampedY, clampedW, clampedH};
    }

    [[nodiscard]] BoundingBox scale(float factor) const {
        const auto [cx, cy] = center();
        const float newW = width * factor;
        const float newH = height * factor;
        return {cx - newW * 0.5f, cy - newH * 0.5f, newW, newH};
    }
};

struct TrackingTarget
{
    int id{0};
    std::string label;
    BoundingBox box;
    float score{0.0f};

    void update(const BoundingBox& newBox, std::optional<float> newScore = std::nullopt)
    {
        box = newBox;
        if (newScore)
        {
            score = *newScore;
        }
    }
};

}  // namespace pipeline



