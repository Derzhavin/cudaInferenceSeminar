//
// Created by denis on 25.03.2022.
//

#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <NvInferRuntime.h>
#include <common/util.h>
#include <common/logger.h>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[])
{
    if (argc != 5)
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

    util::UniquePtr<nvinfer1::ICudaEngine> engine;
    engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr));

    util::UniquePtr<nvinfer1::IExecutionContext> context;

    if (!(context = util::UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext())))
        return 0;

    int32_t input_layer_id = engine->getBindingIndex("flatten_input:0");

    context->setBindingDimensions(input_layer_id, engine->getBindingDimensions(input_layer_id));

    auto output_layer_id = engine->getBindingIndex("Identity:0");

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
    std::string image_filename1(argv[3]);
    std::string image_filename2(argv[4]);

    auto p_cnn_input_buffer = std::unique_ptr<float[]>(new float [3 * 28 * 28 * 1]);

    std::vector<cv::Mat> cnn_batch_input(3);
    for (int i = 0; i < 3; ++i)
        cnn_batch_input[i] = cv::Mat(28, 28, CV_32FC1, p_cnn_input_buffer.get() + i * 28 * 28, 0);

    std::vector<cv::Mat> imgs(3);
    imgs[0] = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);
    imgs[1] = cv::imread(image_filename1, cv::IMREAD_GRAYSCALE);
    imgs[2] = cv::imread(image_filename2, cv::IMREAD_GRAYSCALE);

    for (int i = 0; i < 3; ++i)
    {
        cv::Mat resized_img;
        cv::resize(imgs[i], resized_img, cv::Size(28, 28), cv::INTER_CUBIC);
        resized_img.convertTo(cnn_batch_input[i], CV_32FC1);
    }

    if (cudaMemcpy(gpu_input_layer_mem, (void*)p_cnn_input_buffer.get(), input_layer_size, cudaMemcpyHostToDevice) != cudaSuccess)
        return 0;

    context->executeV2(bindings);

    float probabilities[3 * 10];

    if (cudaMemcpy((void*)&probabilities[0], gpu_output_layer_mem, output_layer_size, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 0;

    for (int i = 0; i < 3; ++i) {
        float *elem = (float*)probabilities + i * 10;

        int max_cls = 0;
        for (int j = 0; j < 10; ++j)
        {
            if (elem[max_cls] < elem[j])
                max_cls = j;

            std::cout << j << ": " << elem[j] << std::endl;
        }
        std::cout << "class: " << max_cls << std::endl;
    }
    return 0;
}