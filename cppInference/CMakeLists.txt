cmake_minimum_required(VERSION 3.19)
project(ReportOnNeuralNetworkInference)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.2/bin/nvcc")

string(APPEND CMAKE_CXX_FLAGS " -std=c++17 -pthread -fPIC -fopenmp")
string(APPEND CMAKE_C_FLAGS " -std=c11 -fPIC -fopenmp")
string(APPEND CMAKE_CUDA_FLAGS "-cudart=shared -lcudadevrt -gencode arch=compute_75,code=sm_75 -std=c++17")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE(STATUS "debug")
else ()
    MESSAGE(STATUS "release")
    string(APPEND CMAKE_CXX_FLAGS " -O3 -march=native -mtune=native -funroll-all-loops")
    string(APPEND CMAKE_C_FLAGS " -O3 -fPIC -funroll-all-loops")
    string(APPEND CMAKE_CUDA_FLAGS " -use_fast_math -O3 -m64 --no-exceptions")
endif ()

find_package(OpenCV 4.5.3 EXACT)
find_package(CUDA 11.2 REQUIRED EXACT)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenMP)

enable_language(CUDA)

file(GLOB APP_SOURCES
        "app/*.cpp"
        "videoio/*.cpp"
        "videoio/codec/*.cpp"
        "detection/*.cpp"
        "common/*.cpp"
        "config/*.cpp"
        "config/json/*.cpp"
        "main_video_app.cpp"
        )

cuda_add_executable(app ${APP_SOURCES})
target_include_directories(
        app
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(
        app
        opencv_video
        opencv_videoio
        opencv_core
        opencv_highgui
        nvinfer
        nvinfer_plugin
        nlohmann_json::nlohmann_json
)

cuda_add_executable(mnist_digits main_mnist_digits.cpp common/logger.cpp common/util.cpp)
target_include_directories(
        mnist_digits
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(
        mnist_digits
        opencv_core
        opencv_imgcodecs
        nvinfer
        nvinfer_plugin
)

cuda_add_executable(mnist_digits_batch3 main_mnist_digits_batch3.cpp common/logger.cpp common/util.cpp)
target_include_directories(
        mnist_digits_batch3
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(
        mnist_digits_batch3
        opencv_core
        opencv_imgcodecs
        nvinfer
        nvinfer_plugin
)