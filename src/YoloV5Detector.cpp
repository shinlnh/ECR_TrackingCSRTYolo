#include "detection/YoloV5Detector.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace detection {
namespace {
constexpr float kScoreEpsilon = 1e-6f;
}  // namespace

YoloV5Detector::YoloV5Detector(const std::string& modelPath,
                               const std::string& labelPath,
                               float confidenceThreshold,
                               float nmsThreshold,
                               std::optional<std::string> focusLabel,
                               std::string reportedLabel,
                               int inputSize)
    : confidenceThreshold_(confidenceThreshold),
      nmsThreshold_(nmsThreshold),
      reportedLabel_(std::move(reportedLabel)),
      inputSize_(inputSize) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load YOLOv5 model from " + modelPath);
    }

    loadLabels(labelPath);
    focusClassId_ = resolveFocusLabel(focusLabel);

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void YoloV5Detector::loadLabels(const std::string& labelPath) {
    std::ifstream stream(labelPath);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open label file: " + labelPath);
    }

    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }

    if (labels_.empty()) {
        throw std::runtime_error("Label file is empty: " + labelPath);
    }
}

std::optional<int> YoloV5Detector::resolveFocusLabel(const std::optional<std::string>& focusLabel) const {
    if (!focusLabel) {
        return std::nullopt;
    }
    const auto it = std::find(labels_.begin(), labels_.end(), *focusLabel);
    if (it == labels_.end()) {
        return std::nullopt;
    }
    return static_cast<int>(std::distance(labels_.begin(), it));
}

cv::Mat YoloV5Detector::letterbox(const cv::Mat& image, int inputSize, float& scale, int& padX, int& padY) {
    const int width = image.cols;
    const int height = image.rows;
    const float r = std::min(static_cast<float>(inputSize) / static_cast<float>(width),
                             static_cast<float>(inputSize) / static_cast<float>(height));
    const int newWidth = static_cast<int>(std::round(width * r));
    const int newHeight = static_cast<int>(std::round(height * r));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newWidth, newHeight));

    const int dw = inputSize - newWidth;
    const int dh = inputSize - newHeight;
    padX = dw / 2;
    padY = dh / 2;

    cv::Mat padded(inputSize, inputSize, image.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(padX, padY, newWidth, newHeight)));

    scale = r;
    return padded;
}

std::vector<Detection> YoloV5Detector::detect(
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

    cv::Mat region = frame(roiRect).clone();

    float scale = 1.0f;
    int padX = 0;
    int padY = 0;
    cv::Mat input = letterbox(region, inputSize_, scale, padX, padY);

    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(inputSize_, inputSize_), cv::Scalar(), true, false);
    net_.setInput(blob);
    cv::Mat output = net_.forward();

    if (output.dims != 3) {
        return {};
    }

    const int rows = output.size[1];
    const int dimensions = output.size[2];
    cv::Mat outMat(rows, dimensions, CV_32F, output.ptr<float>());

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int i = 0; i < rows; ++i) {
        const float* row = outMat.ptr<float>(i);
        const float objectness = row[4];
        if (objectness < confidenceThreshold_) {
            continue;
        }

        const float* classScores = row + 5;
        const int numClasses = dimensions - 5;
        const cv::Mat scores(1, numClasses, CV_32F, const_cast<float*>(classScores));

        cv::Point classIdPoint;
        double maxClassScore = 0.0;
        cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classIdPoint);
        const int classId = classIdPoint.x;
        const float confidence = objectness * static_cast<float>(maxClassScore);
        if (confidence < confidenceThreshold_) {
            continue;
        }
        if (focusClassId_ && classId != *focusClassId_) {
            continue;
        }

        const float cx = row[0];
        const float cy = row[1];
        const float width = row[2];
        const float height = row[3];

        const float boxX = (cx - width * 0.5f - static_cast<float>(padX)) / scale;
        const float boxY = (cy - height * 0.5f - static_cast<float>(padY)) / scale;
        const float boxW = width / scale;
        const float boxH = height / scale;

        const float clampedX = std::clamp(boxX + offsetX, 0.0f, static_cast<float>(frame.cols));
        const float clampedY = std::clamp(boxY + offsetY, 0.0f, static_cast<float>(frame.rows));
        const float maxW = std::max(0.0f, static_cast<float>(frame.cols) - clampedX);
        const float maxH = std::max(0.0f, static_cast<float>(frame.rows) - clampedY);
        const float clampedW = std::clamp(boxW, 0.0f, maxW);
        const float clampedH = std::clamp(boxH, 0.0f, maxH);

        if (clampedW <= kScoreEpsilon || clampedH <= kScoreEpsilon) {
            continue;
        }

        boxes.emplace_back(static_cast<int>(std::round(clampedX)),
                           static_cast<int>(std::round(clampedY)),
                           static_cast<int>(std::round(clampedW)),
                           static_cast<int>(std::round(clampedH)));
        confidences.push_back(confidence);
        classIds.push_back(classId);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold_, nmsThreshold_, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices) {
        if (idx < 0 || idx >= static_cast<int>(boxes.size())) {
            continue;
        }
        const cv::Rect& rect = boxes[idx];
        Detection detection;
        detection.label = reportedLabel_;
        detection.score = confidences[idx];
        detection.box = {
            static_cast<float>(rect.x),
            static_cast<float>(rect.y),
            static_cast<float>(rect.width),
            static_cast<float>(rect.height)};
        detections.emplace_back(std::move(detection));
    }

    return detections;
}

}  // namespace detection
