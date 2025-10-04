#include "detection/OnnxSsdDetector.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace detection {

namespace {

pipeline::BoundingBox makeBoundingBox(float cx1,
                                      float cy1,
                                      float cx2,
                                      float cy2,
                                      float offsetX,
                                      float offsetY,
                                      float frameWidth,
                                      float frameHeight) {
    const float x1 = std::clamp(cx1 * frameWidth + offsetX, 0.0f, frameWidth + offsetX);
    const float y1 = std::clamp(cy1 * frameHeight + offsetY, 0.0f, frameHeight + offsetY);
    const float x2 = std::clamp(cx2 * frameWidth + offsetX, 0.0f, frameWidth + offsetX);
    const float y2 = std::clamp(cy2 * frameHeight + offsetY, 0.0f, frameHeight + offsetY);

    const float left = std::min(x1, x2);
    const float top = std::min(y1, y2);
    const float right = std::max(x1, x2);
    const float bottom = std::max(y1, y2);

    return {left, top, std::max(0.0f, right - left), std::max(0.0f, bottom - top)};
}

}  // namespace

OnnxSsdDetector::OnnxSsdDetector(const std::string& modelPath,
                                 const std::string& labelPath,
                                 float confidenceThreshold,
                                 std::optional<std::string> focusLabel,
                                 std::string reportedLabel,
                                 cv::Size inputSize)
    : confidenceThreshold_(confidenceThreshold),
      reportedLabel_(std::move(reportedLabel)),
      inputSize_(inputSize) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load ONNX model from " + modelPath);
    }

    loadLabels(labelPath);

    if (focusLabel) {
        const auto it = std::find_if(
            labels_.begin(), labels_.end(),
            [&](const std::string& candidate) {
                return candidate == *focusLabel;
            });
        if (it != labels_.end()) {
            focusClassId_ = static_cast<int>(std::distance(labels_.begin(), it));
        }
    }

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void OnnxSsdDetector::loadLabels(const std::string& labelPath) {
    std::ifstream file(labelPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + labelPath);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }

    if (labels_.empty()) {
        throw std::runtime_error("Label list is empty in " + labelPath);
    }
}

std::vector<Detection> OnnxSsdDetector::detect(
    const cv::Mat& frame,
    const std::optional<pipeline::BoundingBox>& roi) {
    if (frame.empty()) {
        return {};
    }

    cv::Rect roiRect(0, 0, frame.cols, frame.rows);
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    if (roi) {
        const cv::Rect2f rect = roi->toRect();
        const float x = std::clamp(rect.x, 0.0f, static_cast<float>(frame.cols));
        const float y = std::clamp(rect.y, 0.0f, static_cast<float>(frame.rows));
        const float w = std::clamp(rect.width, 0.0f, static_cast<float>(frame.cols) - x);
        const float h = std::clamp(rect.height, 0.0f, static_cast<float>(frame.rows) - y);
        int roiX = static_cast<int>(std::round(x));
        int roiY = static_cast<int>(std::round(y));
        int roiW = static_cast<int>(std::round(w));
        int roiH = static_cast<int>(std::round(h));
        if (roiX >= frame.cols || roiY >= frame.rows) {
            return {};
        }
        const int maxW = frame.cols - roiX;
        const int maxH = frame.rows - roiY;
        if (maxW <= 0 || maxH <= 0) {
            return {};
        }
        roiW = std::max(1, std::min(roiW, maxW));
        roiH = std::max(1, std::min(roiH, maxH));
        cv::Rect candidate(roiX, roiY, roiW, roiH);
        roiRect = candidate & cv::Rect(0, 0, frame.cols, frame.rows);
        if (roiRect.width <= 0 || roiRect.height <= 0) {
            return {};
        }
        offsetX = static_cast<float>(roiRect.x);
        offsetY = static_cast<float>(roiRect.y);
    }

    const cv::Mat region = frame(roiRect);
    cv::Mat blob = cv::dnn::blobFromImage(region, scale_, inputSize_, mean_, swapRB_, false);

    net_.setInput(blob);
    cv::Mat output = net_.forward();

    const int detections = output.size[2];
    const int step = output.size[3];
    std::vector<Detection> results;
    results.reserve(static_cast<std::size_t>(detections));

    const float roiWidth = static_cast<float>(roiRect.width);
    const float roiHeight = static_cast<float>(roiRect.height);

    for (int i = 0; i < detections; ++i) {
        const float* data = output.ptr<float>(0, 0, i);
        if (step < 7) {
            continue;
        }
        const int classId = static_cast<int>(data[1]);
        const float confidence = data[2];
        if (confidence < confidenceThreshold_) {
            continue;
        }
        if (focusClassId_ && classId != *focusClassId_) {
            continue;
        }

        const float left = data[3];
        const float top = data[4];
        const float right = data[5];
        const float bottom = data[6];

        pipeline::BoundingBox box = makeBoundingBox(
            left, top, right, bottom, offsetX, offsetY, roiWidth, roiHeight);
        if (box.area() <= 1.0f) {
            continue;
        }

        Detection detection;
        const std::string className =
            (classId >= 0 && classId < static_cast<int>(labels_.size()))
                ? labels_[classId]
                : reportedLabel_;
        detection.label = reportedLabel_;
        detection.score = confidence;
        detection.box = box;
        results.emplace_back(std::move(detection));
    }

    return results;
}

}  // namespace detection
