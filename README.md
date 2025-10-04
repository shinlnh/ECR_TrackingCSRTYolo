# Pipeline Tracking Elder Care Robot (C++)

This project implements the requested real-time tracking pipeline in modern C++ (C++17) with OpenCV. The code mirrors the original Python design while targeting low-latency deployments on embedded platforms such as Jetson Nano.

## Module Map

1. **Types (`include/pipeline/Types.hpp`)** - bounding-box math and `TrackingTarget` bookkeeping.
2. **Detection (`include/detection`)** - detector interface plus the SSD/NCNN stub (`NCNNSsdDetector`) and a `StaticBoxDetector` for dry runs.
3. **Tracking (`include/tracking`)** - CSRT wrapper (`CsrtTracker.hpp`) and bounding-box quality filters (`Quality.hpp`).
4. **Rescue (`include/rescue/RescueStrategy.hpp`)** - ROI expansion, class filtering, IoU-based re-ranking, and re-init hooks.
5. **Smoothing (`include/smoothing/BoxKalmanFilter.hpp`)** - Kalman-based smoothing with clamp limits on position and scale.
6. **Pipeline (`include/pipeline/TrackingPipeline.hpp`)** - orchestrates detection, tracking, rescue, and smoothing to deliver per-frame outputs.
7. **App (`src/main.cpp`)** - command-line entry point using a configurable GStreamer pipeline and optional live visualization.

All implementations live in `src/`, with the same filenames as their headers.

## Building

1. Install a C++17 toolchain and OpenCV (built with `opencv_contrib` to access CSRT).
2. Configure and build with CMake:
   ```powershell
   cmake -S . -B build
   cmake --build build
   ```
3. Run the demo binary, supplying your GStreamer pipeline string:
   ```powershell
   .\build\eldercare_tracking --camera-pipeline "<gst string>" --use-mock-detector --display
   ```

When integrating the real detector, provide `--model <path>` instead of `--use-mock-detector`.

## Next Steps

- Replace the stubbed `NCNNSsdDetector::loadModel`/`detect` with the actual NCNN/Vulkan flow you plan to deploy.
- Export the SSD graph to a Jetson-optimized engine once the NCNN/Vulkan implementation is validated.
- Tune the thresholds in `config::PipelineConfig` to match your hardware and environment.

