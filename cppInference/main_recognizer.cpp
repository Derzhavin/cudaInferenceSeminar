//
// Created by denis on 25.03.2022.
//

#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <NvInferRuntime.h>
#include <common/util.h>
#include <common/logger.h>
#include <opencv2/imgproc.hpp>
#include "config/config.h"
#include "config/json/JsonConfigParser.h"
#include "omp.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: ./main_recognizer <config>" << std::endl;
        return 0;
    }

    Config config;
    const std::string config_path(argv[1]);

    std::vector<IJsonConfigParser*> parsers
    {
            new ImagesReaderJsonParser(),
            new ObjectsRecognizerJsonParser()
    };
    JsonConfigParser json_config_parser(parsers);

    if (!json_config_parser.parseFile(config_path, config))
    {
        std::cout << "Failed to open config file" << std::endl;
        return 0;
    }

    std::ifstream engine_file_stream(config.engine_file_path, std::ios::binary);

    if (engine_file_stream.fail())
    {
        std::cout << "failed to read engine" << std::endl;
        return 0;
    }

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
    {
        std::cout << "failed to create context" << std::endl;
        return 0;
    }
    int32_t input_layer_id = engine->getBindingIndex(config.input_tensor.c_str());

    context->setBindingDimensions(input_layer_id, engine->getBindingDimensions(input_layer_id));

    auto output_layer_id = engine->getBindingIndex(config.output_tensor.c_str());

    void* gpu_input_layer_mem;
    void* gpu_output_layer_mem;

    int32_t input_layer_size = util::getMemorySize(engine->getBindingDimensions(input_layer_id), sizeof(float));
    if (cudaMalloc(&gpu_input_layer_mem, input_layer_size) != cudaSuccess) {
        std::cout << "failed to allocate input layer" << std::endl;
        return 0;
    }

    int32_t output_layer_size = util::getMemorySize(engine->getBindingDimensions(output_layer_id), sizeof(float));
    if (cudaMalloc(&gpu_output_layer_mem, output_layer_size) != cudaSuccess)
    {
        std::cout << "failed to allocate output layer" << std::endl;
        return 0;
    }

    void* bindings[] = {gpu_input_layer_mem, gpu_output_layer_mem};

    auto dims = engine->getBindingDimensions(input_layer_id);
    const auto channels_num = dims.d[3];
    const auto cnn_w = dims.d[2];
    const auto cnn_h = dims.d[1];

    auto p_cnn_input_buffer = std::unique_ptr<float[]>(new float [config.batch_size * cnn_w * cnn_h * channels_num]);
    auto p_probs = std::unique_ptr<float[]>(new float [config.batch_size * config.classes.size()]{0});

    std::vector<cv::Mat> cnn_batch_input(config.batch_size);
    for (int i = 0; i < config.batch_size; ++i)
        cnn_batch_input[i] = cv::Mat(cnn_w, cnn_h, CV_32FC(channels_num), p_cnn_input_buffer.get() + i * config.batch_size * cnn_w * cnn_h * channels_num, 0);

    const auto num_of_executions = config.images_path.size() / config.batch_size;
    for (int i = 0; i < num_of_executions; ++i)
    {
//#pragma omp parallel for shared(config, cnn_batch_input, cnn_h, cnn_w, channels_num, i) default(none) num_threads(config.batch_size)
        for (int j = 0; j < config.batch_size; ++j)
        {
            cv::Mat input_img = cv::imread( config.images_path[i * config.batch_size + j]);
            cv::resize(input_img, input_img, cv::Size(cnn_w, cnn_h));
            input_img.convertTo(cnn_batch_input[j], CV_32FC(channels_num));
        }

        if (cudaMemcpy(gpu_input_layer_mem, (void*)p_cnn_input_buffer.get(), input_layer_size, cudaMemcpyHostToDevice) != cudaSuccess)
            return 0;

        context->execute(config.batch_size, bindings);

        if (cudaMemcpyAsync((void *)&p_probs.get()[0], gpu_output_layer_mem, output_layer_size, cudaMemcpyDeviceToHost) != cudaSuccess)
            return 0;

        for (int j = 0; j < config.batch_size; ++j)
        {
            float *object_probs = (float*)p_probs.get() + j * config.classes.size();

            int max_cls = 0;
            for (int k = 0; k < config.classes.size(); ++k)
            {
                if (object_probs[max_cls] < object_probs[k])
                    max_cls = k;

                std::cout << "\t" << k << ": " << object_probs[k] << std::endl;
            }
            std::cout << "car brend[" << (i * config.batch_size + j) << "] = " << config.classes[max_cls] << "(class_no =" << max_cls << ", prob = " << object_probs[max_cls] << ")" << std::endl;
        }
    }

    cudaFree(gpu_input_layer_mem);
    cudaFree(gpu_output_layer_mem);
    return 0;
}