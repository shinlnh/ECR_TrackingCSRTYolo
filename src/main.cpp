#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <sstream>
#include <vector>
#include <chrono>
#include <memory>
#include <system_error>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include "pipeline/PipelineOutput.hpp"
#include "pipeline/TrackingPipeline.hpp"
#include "detection/TemplateMatcherDetector.hpp"
#include "detection/YoloV5Detector.hpp"
#include "detection/OnnxSsdDetector.hpp"

struct Args {
    std::string datasetRoot{"otb100/OTB-dataset/OTB100"};
    std::optional<std::string> sequenceName;
    std::optional<std::tuple<int, int>> resize;
    bool display{false};
    std::optional<int> maxFrames;
    bool enablePreprocess{false};
    bool enableSegmentation{false};
    bool enableDetector{false};
    bool enableDenoise{true};
    bool enableLowLightBoost{true};
    double detectorThreshold{0.45};
    int detectorPadding{56};
    std::optional<std::string> detectorModel;
    std::optional<std::string> detectorLabels;
    std::optional<std::string> detectorClass;
    std::string detectorReportedLabel{"target"};
};

void printUsage() {
    std::cout << "Usage: eldercare_tracking [options]\n"
                 "Options:\n"
                 "  --sequence <name>           Sequence name within the dataset root\n"
                 "  --dataset-root <path>       Path to the OTB dataset root (default: otb100/OTB-dataset/OTB100)\n"
                 "  --resize <w> <h>            Resize frames before tracking\n"
                 "  --display                   Show tracking visualization window\n"
                 "  --max-frames <int>          Limit number of frames processed\n"
                 "  --preprocess                Enable denoise/contrast pre-processing\n"
                 "  --segmentation              Enable motion-based segmentation mask\n"
                 "  --detector                  Enable ONNX/template detector rescue\n"
                 "  --no-preprocess             Disable denoise/contrast pre-processing\n"
                 "  --no-segmentation           Disable motion-based segmentation mask\n"
                 "  --no-detector               Run without template-based rescue detector\n"
                 "  --no-denoise                Skip denoising during pre-processing\n"
                 "  --no-lowlight               Do not auto-enhance dark frames\n"
                 "  --detector-threshold <f>    Detector confidence threshold (default: 0.45)\n"
                 "  --detector-padding <int>    Extra pixels around template search window (default: 56)\n"
                 "  --detector-model <path>     ONNX SSD model path (auto-detected if present)\n"
                 "  --detector-labels <path>    Label map used by the detector\n"
                 "  --detector-class <name>     Class name to track (filter SSD detections)\n"
                 "  --help                      Show this help message\n";
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string current = argv[i];
        if (current == "--sequence" && (i + 1) < argc) {
            args.sequenceName = std::string(argv[++i]);
        } else if (current == "--dataset-root" && (i + 1) < argc) {
            args.datasetRoot = argv[++i];
        } else if (current == "--resize" && (i + 2) < argc) {
            const int w = std::stoi(argv[++i]);
            const int h = std::stoi(argv[++i]);
            args.resize = std::make_tuple(w, h);
        } else if (current == "--display") {
            args.display = true;
        } else if (current == "--max-frames" && (i + 1) < argc) {
            args.maxFrames = std::stoi(argv[++i]);
        } else if (current == "--preprocess") {
            args.enablePreprocess = true;
        } else if (current == "--segmentation") {
            args.enableSegmentation = true;
        } else if (current == "--detector") {
            args.enableDetector = true;
        } else if (current == "--no-preprocess") {
            args.enablePreprocess = false;
        } else if (current == "--no-segmentation") {
            args.enableSegmentation = false;
        } else if (current == "--no-detector") {
            args.enableDetector = false;
        } else if (current == "--no-denoise") {
            args.enableDenoise = false;
        } else if (current == "--no-lowlight") {
            args.enableLowLightBoost = false;
        } else if (current == "--detector-threshold" && (i + 1) < argc) {
            args.detectorThreshold = std::stod(argv[++i]);
        } else if (current == "--detector-padding" && (i + 1) < argc) {
            args.detectorPadding = std::stoi(argv[++i]);
        } else if (current == "--detector-model" && (i + 1) < argc) {
            args.detectorModel = argv[++i];
        } else if (current == "--detector-labels" && (i + 1) < argc) {
            args.detectorLabels = argv[++i];
        } else if (current == "--detector-class" && (i + 1) < argc) {
            args.detectorClass = argv[++i];
        } else if (current == "--help") {
            printUsage();
            return false;
        } else if (!current.empty() && current[0] != '-' && !args.sequenceName) {
            args.sequenceName = current;
        } else {
            std::cerr << "Unknown or incomplete option: " << current << "\n";
            return false;
        }
    }

    if (args.maxFrames && *args.maxFrames <= 0) {
        std::cerr << "--max-frames must be positive.\n";
        return false;
    }

    if (args.detectorThreshold <= 0.0) {
        args.detectorThreshold = 0.5;
    }
    args.detectorThreshold = std::clamp(args.detectorThreshold, 0.1, 0.99);
    if (args.detectorPadding < 0) {
        args.detectorPadding = 0;
    }

    auto locateResource = [] (const std::filesystem::path& relative) -> std::optional<std::string> {
        std::vector<std::filesystem::path> bases;
        std::filesystem::path current = std::filesystem::current_path();
        for (int i = 0; i < 3 && !current.empty(); ++i) {
            bases.push_back(current);
            current = current.parent_path();
        }
        for (const auto& base : bases) {
            std::error_code ec;
            const auto candidate = base / relative;
            if (std::filesystem::exists(candidate, ec)) {
                return candidate.string();
            }
        }
        return std::nullopt;
    };

    auto locateFirstAvailable = [&locateResource](std::initializer_list<const char*> candidates) -> std::optional<std::string> {
        for (const auto& relative : candidates) {
            if (auto located = locateResource(relative)) {
                return located;
            }
        }
        return std::nullopt;
    };

    if (args.enableDetector && !args.detectorModel) {
        if (auto located = locateFirstAvailable({
                "models/yolov5n/yolov5n.onnx",
                "models/ssd-mobilenet-v2/ssd-mobilenet-v2.onnx"})) {
            args.detectorModel = *located;
        }
    }

    if (args.enableDetector && !args.detectorLabels) {
        if (auto located = locateFirstAvailable({
                "models/yolov5n/labels.txt",
                "models/ssd-mobilenet-v2/labels.txt",
                "models/jetson-inference/data/networks/ssd_coco_labels.txt"})) {
            args.detectorLabels = *located;
        }
    }

    args.detectorReportedLabel = args.detectorClass.value_or("target");

    return true;
}

std::vector<std::filesystem::path> collectFramePaths(const std::filesystem::path& imgDir) {
    std::vector<std::filesystem::path> frames;
    if (!std::filesystem::exists(imgDir)) {
        return frames;
    }
    for (const auto& entry : std::filesystem::directory_iterator(imgDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    return frames;
}

std::optional<std::vector<pipeline::BoundingBox>> loadGroundTruth(const std::filesystem::path& file) {
    std::ifstream stream(file);
    if (!stream.is_open()) {
        return std::nullopt;
    }
    std::vector<pipeline::BoundingBox> boxes;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        for (char& ch : line) {
            if (ch == ',' || ch == '\t') {
                ch = ' ';
            }
        }
        std::stringstream ss(line);
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        if (!(ss >> x >> y >> w >> h)) {
            continue;
        }
        // OTB annotations are 1-indexed.
        boxes.emplace_back(x - 1.0f, y - 1.0f, w, h);
    }
    return boxes;
}

constexpr float kSuccessThreshold = 0.5f;

struct SequenceMetrics {
    std::string name;
    std::size_t processedFrames{0};
    std::size_t successFrames{0};
    std::size_t timedFrames{0};
    double sumIoU{0.0};
    double totalSeconds{0.0};
    double averageIoU{0.0};
    double successRate{0.0};
    double averageFps{0.0};
};

std::optional<std::filesystem::path> resolveSequenceDirectory(const std::filesystem::path& candidate) {
    if (!std::filesystem::exists(candidate)) {
        return std::nullopt;
    }
    if (std::filesystem::is_regular_file(candidate)) {
        return std::nullopt;
    }

    const auto direct = candidate / "groundtruth_rect.txt";
    if (std::filesystem::exists(direct)) {
        return candidate;
    }

    const auto nested = candidate / candidate.filename();
    if (!candidate.filename().empty() && std::filesystem::exists(nested / "groundtruth_rect.txt")) {
        return nested;
    }

    if (!std::filesystem::is_directory(candidate)) {
        return std::nullopt;
    }

    for (const auto& entry : std::filesystem::directory_iterator(candidate)) {
        if (!entry.is_directory()) {
            continue;
        }
        if (std::filesystem::exists(entry.path() / "groundtruth_rect.txt")) {
            return entry.path();
        }
    }
    return std::nullopt;
}

cv::Mat applyLowLightBoost(const cv::Mat& frame, bool enableBoost) {
    if (!enableBoost) {
        return frame.clone();
    }

    cv::Mat ycrcb;
    cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);

    const double meanLuma = cv::mean(channels[0])[0];
    if (meanLuma > 140.0) {
        return frame.clone();
    }

    const double clipLimit = (meanLuma < 70.0) ? 4.0 : 2.5;
    auto clahe = cv::createCLAHE(clipLimit, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    cv::merge(channels, ycrcb);
    cv::Mat boosted;
    cv::cvtColor(ycrcb, boosted, cv::COLOR_YCrCb2BGR);
    return boosted;
}

cv::Mat preprocessFrame(const cv::Mat& frame,
                        cv::Ptr<cv::BackgroundSubtractor>& bgSubtractor,
                        bool enablePreprocess,
                        bool enableSegmentation,
                        bool enableDenoise,
                        bool enableLowLight,
                        int frameIndex) {
    if (!enablePreprocess && !enableSegmentation) {
        return frame.clone();
    }

    cv::Mat processed = frame.clone();
    if (enablePreprocess) {
        processed = applyLowLightBoost(processed, enableLowLight);
        if (enableDenoise) {
            cv::Mat denoised;
            cv::bilateralFilter(processed, denoised, 5, 40.0, 7.0);
            processed = std::move(denoised);
        }
    }

    if (enableSegmentation && bgSubtractor) {
        cv::Mat mask;
        const double learningRate = (frameIndex < 10) ? 0.35 : 0.02;
        bgSubtractor->apply(processed, mask, learningRate);
        if (frameIndex > 2) {
            cv::threshold(mask, mask, 160, 255, cv::THRESH_BINARY);
            cv::Mat element3 = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
            cv::Mat element5 = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element3);
            cv::morphologyEx(mask, mask, cv::MORPH_DILATE, element5);
            cv::Mat masked;
            cv::bitwise_and(processed, processed, masked, mask);
            processed = std::move(masked);
        }
    }

    return processed;
}

void drawOverlay(cv::Mat& frame, const pipeline::PipelineOutput& output) {
    if (output.rawBox) {
        const auto rect = output.rawBox->toRect();
        cv::rectangle(frame, rect, {255, 0, 0}, 2);
    }
    if (output.smoothedBox) {
        const auto rect = output.smoothedBox->toRect();
        cv::rectangle(frame, rect, {0, 255, 0}, 2);
    }
    if (output.smoothedBox && output.target && output.target->id > 0) {
        const auto rect = output.smoothedBox->toRect();
        const std::string trackIdText = "ID: " + std::to_string(output.target->id);
        cv::putText(
            frame,
            trackIdText,
            {static_cast<int>(rect.x), std::max(20, static_cast<int>(rect.y) - 10)},
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            {0, 255, 0},
            2);
    }

    std::string label = "lost";
    if (output.target) {
        label = output.target->label;
        if (output.target->id > 0) {
            label += "#" + std::to_string(output.target->id);
        }
    }
    const std::string status = output.lost ? "lost" : "tracking";
    cv::putText(
        frame,
        label + ": " + status,
        {20, 30},
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        output.lost ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0),
        2);
}

std::optional<SequenceMetrics> evaluateSequence(const Args& args,
                                                const std::filesystem::path& sequencePath,
                                                const std::string& sequenceName,
                                                bool logPerFrame,
                                                bool enableDisplay) {
    const auto frames = collectFramePaths(sequencePath / "img");
    if (frames.empty()) {
        std::cerr << "No frames found in " << (sequencePath / "img") << "\n";
        return std::nullopt;
    }

    const auto boxesOpt = loadGroundTruth(sequencePath / "groundtruth_rect.txt");
    if (!boxesOpt || boxesOpt->empty()) {
        std::cerr << "Failed to load ground truth bounding boxes for " << sequenceName << ".\n";
        return std::nullopt;
    }

    const std::size_t totalFrames = std::min(frames.size(), boxesOpt->size());
    if (totalFrames == 0) {
        std::cerr << "No overlapping frames between images and annotations for " << sequenceName << ".\n";
        return std::nullopt;
    }

    cv::Mat frame = cv::imread(frames[0].string(), cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Failed to read frame: " << frames[0] << "\n";
        return std::nullopt;
    }

    const int originalWidth = frame.cols;
    const int originalHeight = frame.rows;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    if (args.resize) {
        const auto [rw, rh] = *args.resize;
        if (rw <= 0 || rh <= 0) {
            std::cerr << "Resize dimensions must be positive." << "\n";
            return std::nullopt;
        }
        scaleX = static_cast<float>(rw) / static_cast<float>(originalWidth);
        scaleY = static_cast<float>(rh) / static_cast<float>(originalHeight);
        cv::resize(frame, frame, {rw, rh});
    }

    std::vector<pipeline::BoundingBox> boxes(totalFrames);
    for (std::size_t i = 0; i < totalFrames; ++i) {
        const auto& gt = boxesOpt->at(i);
        boxes[i] = {
            gt.x * scaleX,
            gt.y * scaleY,
            gt.width * scaleX,
            gt.height * scaleY};
    }

    const float successThreshold = kSuccessThreshold;
    SequenceMetrics metrics;
    metrics.name = sequenceName;

    auto recordMetrics = [&](std::size_t frameIndex,
                             const std::optional<pipeline::BoundingBox>& predicted,
                             double frameSeconds,
                             bool lost) {
        const auto& gt = boxes[frameIndex];
        double iou = 0.0;
        if (predicted) {
            iou = predicted->iou(gt);
        }

        metrics.sumIoU += iou;
        if (iou >= successThreshold) {
            ++metrics.successFrames;
        }
        if (frameSeconds > 0.0) {
            metrics.totalSeconds += frameSeconds;
            ++metrics.timedFrames;
        }
        ++metrics.processedFrames;

        if (logPerFrame) {
            const double avgIoU = metrics.sumIoU / static_cast<double>(metrics.processedFrames);
            const double successRate = static_cast<double>(metrics.successFrames) /
                static_cast<double>(metrics.processedFrames);
            const double fps = (frameSeconds > 0.0) ? (1.0 / frameSeconds) : 0.0;

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3)
                << "[frame " << (frameIndex + 1) << "] "
                << "IoU=" << iou
                << " avgIoU=" << avgIoU
                << " successRate=" << successRate
                << " status=" << (lost ? "lost" : "valid")
                << " fps=" << std::setprecision(2) << fps;
            std::cout << oss.str() << std::endl;
        }
    };
    std::unique_ptr<detection::Detector> detector;
    cv::Ptr<cv::BackgroundSubtractor> backgroundSubtractor;
    if (args.enableSegmentation) {
        backgroundSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
    }

    cv::Mat preprocessed = (args.enablePreprocess || args.enableSegmentation)
        ? preprocessFrame(frame, backgroundSubtractor,
                          args.enablePreprocess,
                          args.enableSegmentation,
                          args.enableDenoise,
                          args.enableLowLightBoost,
                          0)
        : frame.clone();

    config::PipelineConfig config;

    const std::string trackingLabel = args.enableDetector
        ? args.detectorReportedLabel
        : sequenceName;

    detection::TemplateMatcherDetector* templateDetectorPtr = nullptr;

    if (args.enableDetector) {
        if (args.detectorModel && args.detectorLabels) {
            try {
                const std::string& modelPath = *args.detectorModel;
                const std::string& labelPath = *args.detectorLabels;
                std::string modelNameLower = modelPath;
                std::transform(modelNameLower.begin(), modelNameLower.end(), modelNameLower.begin(), [](unsigned char ch) {
                    return static_cast<char>(std::tolower(ch));
                });

                if (modelNameLower.find("yolov5") != std::string::npos) {
                    detector = std::make_unique<detection::YoloV5Detector>(
                        modelPath,
                        labelPath,
                        static_cast<float>(args.detectorThreshold),
                        0.45f,
                        args.detectorClass,
                        trackingLabel);
                } else {
                    detector = std::make_unique<detection::OnnxSsdDetector>(
                        modelPath,
                        labelPath,
                        static_cast<float>(args.detectorThreshold),
                        args.detectorClass,
                        trackingLabel);
                }
            } catch (const std::exception& ex) {
                std::cerr << "Failed to initialise ONNX detector: " << ex.what() << "\n";
            }
        } else if (args.detectorModel || args.detectorLabels) {
            std::cerr << "Detector model or labels missing; falling back to template matcher.\n";
        }

        if (!detector) {
            auto fallback = std::make_unique<detection::TemplateMatcherDetector>(
                preprocessed,
                boxes[0],
                trackingLabel,
                static_cast<float>(args.detectorThreshold),
                args.detectorPadding);
            templateDetectorPtr = fallback.get();
            detector = std::move(fallback);
        }
    }

    std::unique_ptr<pipeline::TrackingPipeline> pipelinePtr;
    if (detector) {
        pipelinePtr = std::make_unique<pipeline::TrackingPipeline>(*detector, config);
    } else {
        pipelinePtr = std::make_unique<pipeline::TrackingPipeline>(config);
    }
    auto& pipeline = *pipelinePtr;
    pipeline.initializeWithBox(preprocessed, boxes[0], trackingLabel, 1.0f);
    recordMetrics(0, std::optional<pipeline::BoundingBox>{boxes[0]}, 0.0, false);

    if (templateDetectorPtr) {
        templateDetectorPtr->updateTrackedBox(preprocessed, boxes[0]);
    }

    if (enableDisplay) {
        cv::Mat displayFrame = preprocessed.clone();
        cv::rectangle(displayFrame, boxes[0].toRect(), {0, 255, 0}, 2);
        cv::putText(
            displayFrame,
            trackingLabel + ": tracking",
            {20, 30},
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            {0, 255, 0},
            2);
        cv::imshow("Tracking", displayFrame);
        cv::waitKey(1);
    }

    std::size_t frameLimit = totalFrames;
    if (args.maxFrames) {
        frameLimit = std::min<std::size_t>(static_cast<std::size_t>(*args.maxFrames), totalFrames);
        frameLimit = std::max<std::size_t>(frameLimit, static_cast<std::size_t>(1));
    }

    for (std::size_t idx = 1; idx < frameLimit; ++idx) {
        const auto frameStart = std::chrono::steady_clock::now();

        cv::Mat current = cv::imread(frames[idx].string(), cv::IMREAD_COLOR);
        if (current.empty()) {
            std::cerr << "Failed to read frame: " << frames[idx] << "\n";
            break;
        }
        if (args.resize) {
            const auto [rw, rh] = *args.resize;
            cv::resize(current, current, {rw, rh});
        }

        cv::Mat processedFrame = preprocessFrame(current, backgroundSubtractor,
                                                 args.enablePreprocess,
                                                 args.enableSegmentation,
                                                 args.enableDenoise,
                                                 args.enableLowLightBoost,
                                                 static_cast<int>(idx));

        std::optional<pipeline::BoundingBox> groundTruthBox;
        if (idx < boxes.size()) {
            groundTruthBox = boxes[idx];
        }
        pipeline::PipelineOutput output = pipeline.step(processedFrame, groundTruthBox);
        const auto frameEnd = std::chrono::steady_clock::now();
        const double frameSeconds = std::chrono::duration<double>(frameEnd - frameStart).count();

        recordMetrics(idx, output.smoothedBox, frameSeconds, output.lost);

        if (templateDetectorPtr && output.smoothedBox) {
            templateDetectorPtr->updateTrackedBox(processedFrame, *output.smoothedBox);
        }

        if (enableDisplay) {
            cv::Mat displayFrame = processedFrame.clone();
            drawOverlay(displayFrame, output);
            cv::imshow("Tracking", displayFrame);
            const int key = cv::waitKey(1);
            if (key == 27) {  // ESC
                break;
            }
        }
    }

    if (metrics.processedFrames == 0) {
        return std::nullopt;
    }

    metrics.averageIoU = metrics.sumIoU / static_cast<double>(metrics.processedFrames);
    metrics.successRate = static_cast<double>(metrics.successFrames) /
        static_cast<double>(metrics.processedFrames);
    metrics.averageFps = (metrics.totalSeconds > 0.0 && metrics.timedFrames > 0)
        ? static_cast<double>(metrics.timedFrames) / metrics.totalSeconds
        : 0.0;

    if (enableDisplay) {
        cv::destroyAllWindows();
    }

    if (logPerFrame) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "Average IoU: " << metrics.averageIoU
            << " SuccessRate@" << successThreshold << ": " << metrics.successRate
            << " AvgFPS=" << std::setprecision(2) << metrics.averageFps;
        std::cout << oss.str() << std::endl;
        std::cout << "Processed " << metrics.processedFrames << " / " << totalFrames
                  << " frames from sequence '" << sequenceName << "'." << std::endl;
    }

    return metrics;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        return 1;
    }

    std::filesystem::path datasetRoot(args.datasetRoot);
    if (!datasetRoot.is_absolute()) {
        datasetRoot = std::filesystem::current_path() / datasetRoot;
    }

    if (!std::filesystem::exists(datasetRoot)) {
        std::cerr << "Dataset root does not exist: " << datasetRoot << "\n";
        return 1;
    }

    std::vector<std::pair<std::string, std::filesystem::path>> sequences;

    if (args.sequenceName) {
        const std::string& sequenceName = *args.sequenceName;
        auto resolved = resolveSequenceDirectory(datasetRoot / sequenceName);
        if (!resolved && std::filesystem::exists(datasetRoot / "groundtruth_rect.txt") &&
            datasetRoot.filename().string() == sequenceName) {
            resolved = datasetRoot;
        }
        if (!resolved) {
            std::cerr << "Could not find groundtruth_rect.txt for sequence " << sequenceName
                      << " under " << datasetRoot << "\n";
            return 1;
        }
        sequences.emplace_back(sequenceName, *resolved);
    } else {
        if (std::filesystem::exists(datasetRoot / "groundtruth_rect.txt")) {
            std::string name = datasetRoot.filename().string();
            if (name.empty()) {
                name = datasetRoot.string();
            }
            sequences.emplace_back(name, datasetRoot);
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(datasetRoot)) {
                if (!entry.is_directory()) {
                    continue;
                }
                const auto resolved = resolveSequenceDirectory(entry.path());
                if (!resolved) {
                    continue;
                }
                sequences.emplace_back(entry.path().filename().string(), *resolved);
            }
            std::sort(sequences.begin(), sequences.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
        }
    }

    if (sequences.empty()) {
        std::cerr << "No sequences found under " << datasetRoot << "\n";
        return 1;
    }

    const bool logPerFrame = sequences.size() == 1;
    const bool enableDisplay = args.display && logPerFrame;
    if (args.display && !enableDisplay) {
        std::cout << "Display is only available when running a single sequence; disabling visualization."
                  << "\n";
    }

    std::vector<SequenceMetrics> summaries;
    summaries.reserve(sequences.size());

    double totalIoUSum = 0.0;
    std::size_t totalFrames = 0;
    std::size_t totalSuccessFrames = 0;
    double totalSeconds = 0.0;
    std::size_t totalTimedFrames = 0;

    std::ostringstream successHeaderStream;
    successHeaderStream << std::fixed << std::setprecision(2) << kSuccessThreshold;
    const std::string successColumn = "Success@" + successHeaderStream.str();
    const std::string separatorLine(68, '-');

    auto printHeader = [&]() {
        std::cout << std::left << std::setw(18) << "Sequence"
                  << std::right << std::setw(10) << "Frames"
                  << std::setw(12) << "Avg IoU"
                  << std::setw(16) << successColumn
                  << std::setw(12) << "AvgFPS" << "\n";
        std::cout << separatorLine << "\n";
    };

    auto printRow = [&](const SequenceMetrics& metrics) {
        std::cout << std::left << std::setw(18) << metrics.name
                  << std::right << std::setw(10) << metrics.processedFrames
                  << std::setw(12) << std::fixed << std::setprecision(3) << metrics.averageIoU
                  << std::setw(16) << std::fixed << std::setprecision(3) << metrics.successRate
                  << std::setw(12) << std::fixed << std::setprecision(2) << metrics.averageFps
                  << "\n";
    };

    bool headerPrinted = false;

    const std::size_t totalSeq = sequences.size();
    std::size_t seqIndex = 0;
    for (const auto& [name, path] : sequences) {
        ++seqIndex;
        std::cout << "[" << seqIndex << "/" << totalSeq << "] " << name << "..." << std::endl;
        const auto metricsOpt = evaluateSequence(args, path, name, logPerFrame, enableDisplay);
        if (!metricsOpt) {
            std::cerr << "Failed to evaluate sequence '" << name << "'." << "\n";
            continue;
        }
        summaries.push_back(*metricsOpt);
        totalIoUSum += metricsOpt->sumIoU;
        totalFrames += metricsOpt->processedFrames;
        totalSuccessFrames += metricsOpt->successFrames;
        totalSeconds += metricsOpt->totalSeconds;
        totalTimedFrames += metricsOpt->timedFrames;

        if (!logPerFrame) {
            if (!headerPrinted) {
                printHeader();
                headerPrinted = true;
            }
            printRow(*metricsOpt);
        }
    }

    if (summaries.empty()) {
        std::cerr << "No sequences were processed successfully." << "\n";
        return 1;
    }

    if (!logPerFrame) {
        if (!headerPrinted) {
            printHeader();
            for (const auto& metrics : summaries) {
                printRow(metrics);
            }
        }

        const double averageIoUAll = totalFrames > 0
            ? totalIoUSum / static_cast<double>(totalFrames)
            : 0.0;
        const double successRateAll = totalFrames > 0
            ? static_cast<double>(totalSuccessFrames) / static_cast<double>(totalFrames)
            : 0.0;
        const double avgFpsAll = (totalSeconds > 0.0 && totalTimedFrames > 0)
            ? static_cast<double>(totalTimedFrames) / totalSeconds
            : 0.0;

        std::cout << separatorLine << "\n";
        SequenceMetrics overallMetrics;
        overallMetrics.name = (summaries.size() > 1) ? "Overall" : summaries.front().name;
        overallMetrics.processedFrames = totalFrames;
        overallMetrics.averageIoU = averageIoUAll;
        overallMetrics.successRate = successRateAll;
        overallMetrics.averageFps = avgFpsAll;
        printRow(overallMetrics);
    }

    return 0;
}




