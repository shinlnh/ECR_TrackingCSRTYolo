#include "smoothing/BoxKalmanFilter.hpp"

#include <algorithm>

namespace {

pipeline::BoundingBox stateToBox(const cv::Mat& state) {
    const float cx = state.at<float>(0);
    const float cy = state.at<float>(1);
    const float w = state.at<float>(2);
    const float h = state.at<float>(3);
    return {cx - w * 0.5f, cy - h * 0.5f, w, h};
}

cv::Mat boxToMeasurement(const pipeline::BoundingBox& box) {
    const auto [cx, cy] = box.center();
    cv::Mat measurement(4, 1, CV_32F);
    measurement.at<float>(0) = cx;
    measurement.at<float>(1) = cy;
    measurement.at<float>(2) = box.width;
    measurement.at<float>(3) = box.height;
    return measurement;
}

}  // namespace

namespace smoothing {

BoxKalmanFilter::BoxKalmanFilter(const KalmanParams& params)
    : params_(params), filter_(8, 4, 0) {
    filter_.transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
    for (int i = 0; i < 4; ++i) {
        filter_.transitionMatrix.at<float>(i, i + 4) = 1.0f;
    }
    filter_.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    for (int i = 0; i < 4; ++i) {
        filter_.measurementMatrix.at<float>(i, i) = 1.0f;
    }
    filter_.processNoiseCov = cv::Mat::eye(8, 8, CV_32F) * params_.processNoise;
    filter_.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * params_.measurementNoise;
    filter_.errorCovPost = cv::Mat::eye(8, 8, CV_32F);
}

void BoxKalmanFilter::reset(const pipeline::BoundingBox& box) {
    const auto [cx, cy] = box.center();
    filter_.statePost = cv::Mat::zeros(8, 1, CV_32F);
    filter_.statePost.at<float>(0) = cx;
    filter_.statePost.at<float>(1) = cy;
    filter_.statePost.at<float>(2) = box.width;
    filter_.statePost.at<float>(3) = box.height;
    filter_.statePost.at<float>(4) = 0.0f;
    filter_.statePost.at<float>(5) = 0.0f;
    filter_.statePost.at<float>(6) = 0.0f;
    filter_.statePost.at<float>(7) = 0.0f;
    filter_.statePre = filter_.statePost.clone();
    filter_.errorCovPost = cv::Mat::eye(8, 8, CV_32F);
    initialized_ = true;
    lastBox_ = box;
}

std::optional<pipeline::BoundingBox> BoxKalmanFilter::smooth(
    const std::optional<pipeline::BoundingBox>& box,
    bool validMeasurement) {
    if (!initialized_) {
        if (!box) {
            return std::nullopt;
        }
        reset(*box);
        return box;
    }

    pipeline::BoundingBox estimate = predict();
    if (validMeasurement && box) {
        estimate = correct(*box);
    }
    const pipeline::BoundingBox smoothed = clampBox(estimate);
    lastBox_ = smoothed;
    return smoothed;
}

pipeline::BoundingBox BoxKalmanFilter::predict() {
    const cv::Mat prediction = filter_.predict();
    return stateToBox(prediction);
}

pipeline::BoundingBox BoxKalmanFilter::correct(const pipeline::BoundingBox& measurement) {
    const cv::Mat measurementVec = boxToMeasurement(measurement);
    const cv::Mat corrected = filter_.correct(measurementVec);
    return stateToBox(corrected);
}

pipeline::BoundingBox BoxKalmanFilter::clampBox(const pipeline::BoundingBox& box) const {
    if (!lastBox_) {
        return box;
    }
    const auto [lastCx, lastCy] = lastBox_->center();
    const auto [cx, cy] = box.center();
    const float clampedCx = std::clamp(cx, lastCx - params_.clampPosition, lastCx + params_.clampPosition);
    const float clampedCy = std::clamp(cy, lastCy - params_.clampPosition, lastCy + params_.clampPosition);
    const float clampedW = std::clamp(
        box.width,
        lastBox_->width - params_.clampScale,
        lastBox_->width + params_.clampScale);
    const float clampedH = std::clamp(
        box.height,
        lastBox_->height - params_.clampScale,
        lastBox_->height + params_.clampScale);
    return {
        clampedCx - clampedW * 0.5f,
        clampedCy - clampedH * 0.5f,
        clampedW,
        clampedH};
}

}  // namespace smoothing
