cmake_minimum_required(VERSION 3.16)
project(eldercare_tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio tracking video dnn)

# Print OpenCV information for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

set(PIPELINE_SOURCES
    src/TrackingPipeline.cpp
    src/RescueStrategy.cpp
    src/CsrtTracker.cpp
    src/Quality.cpp
    src/BoxKalmanFilter.cpp
    src/NCNNSsdDetector.cpp
    src/StaticBoxDetector.cpp
    src/TemplateMatcherDetector.cpp
    src/OnnxSsdDetector.cpp
    src/YoloV5Detector.cpp)

add_library(eldercare_pipeline STATIC ${PIPELINE_SOURCES})

target_include_directories(eldercare_pipeline PUBLIC 
    inc
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(eldercare_pipeline PUBLIC ${OpenCV_LIBS})

add_executable(ECR_Tracking
    src/main.cpp)

target_include_directories(ECR_Tracking PRIVATE 
    inc
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ECR_Tracking PRIVATE eldercare_pipeline)

add_executable(ECR_Tracking_otb
    src/otb_evaluator.cpp)

target_include_directories(ECR_Tracking_otb PRIVATE 
    inc
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ECR_Tracking_otb PRIVATE eldercare_pipeline)


