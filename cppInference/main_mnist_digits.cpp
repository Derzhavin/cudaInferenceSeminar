//
// Created by denis on 25.03.2022.
//

#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <common/util.h>
#include <common/logger.h>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage: ./mnist_digits <engine> <image>" << std::endl;
        return 0;
    }

    const std::string engine_filename(argv[1]);

    std::ifstream engine_file_stream(engine_filename, std::ios::binary);

    if (engine_file_stream.fail())
        return 0;

    engine_file_stream.seekg(0, std::ifstream::end);
    auto file_size = engine_file_stream.tellg();
    engine_file_stream.seekg(0, std::ifstream::beg);

    std::vector<char> engine_data(file_size);
    engine_file_stream.read(engine_data.data(), file_size);

    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    initLibNvInferPlugins(nullptr, "");

    util::UniquePtr<nvinfer1::ICudaEngine> engine;
    engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr));

    util::UniquePtr<nvinfer1::IExecutionContext> context;

    if (!(context = util::UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext())))
        return 0;

    int32_t input_layer_id = engine->getBindingIndex("input_1");

    context->setBindingDimensions(input_layer_id, engine->getBindingDimensions(input_layer_id));

    auto output_layer_id = engine->getBindingIndex("output_1");

    void* gpu_input_layer_mem;
    void* gpu_output_layer_mem;

    int32_t input_layer_size = util::getMemorySize(engine->getBindingDimensions(input_layer_id), sizeof(float));
    if (cudaMalloc(&gpu_input_layer_mem, input_layer_size) != cudaSuccess)
        return 0;

    int32_t output_layer_size = util::getMemorySize(engine->getBindingDimensions(output_layer_id), sizeof(float));
    if (cudaMalloc(&gpu_output_layer_mem, output_layer_size) != cudaSuccess)
        return 0;

    void* bindings[] = {gpu_input_layer_mem, gpu_output_layer_mem};

    std::string image_filename(argv[2]);
    cv::Mat img = cv::imread(image_filename, cv::IMREAD_GRAYSCALE), normalized_img;
    cv::resize(img, img, cv::Size(28, 28), cv::INTER_CUBIC);
    img.convertTo(normalized_img, CV_32F);

    if (cudaMemcpy(gpu_input_layer_mem, (void*)normalized_img.data, input_layer_size, cudaMemcpyHostToDevice) != cudaSuccess)
        return 0;

    context->executeV2(bindings);

    float probabilities[10];

    if (cudaMemcpy((void*)&probabilities[0], gpu_output_layer_mem, output_layer_size, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 0;

    int max_cls = 0;
    for (int i = 0; i < 10; ++i)
    {
        if (probabilities[max_cls] < probabilities[i])
            max_cls = i;

        std::cout << i << ": " << probabilities[i] << std::endl;
    }
    std::cout << "class: " << max_cls << std::endl;
    return 0;
}