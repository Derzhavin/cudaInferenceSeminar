//
// Created by denis on 19.02.2022.
//

#include "TfObjDetApiDetector.h"

#include "NvInferPlugin.h"
#include <fstream>
#include "common/logger.h"

const char* TfObjDetApiDetector::input_layer_name = "input_tensor:0";
const char* TfObjDetApiDetector::output_layer_num_detections_name = "num_detections";
const char* TfObjDetApiDetector::output_layer_detection_boxes = "detection_boxes";
const char* TfObjDetApiDetector::output_layer_detection_scores = "detection_scores";
const char* TfObjDetApiDetector::output_layer_detection_classes = "detection_classes";
const short TfObjDetApiDetector::num_detections = 100;

bool TfObjDetApiDetector::isReadyImpl() const
{
    return _ready;
}
unsigned int TfObjDetApiDetector::detectionsNumImpl()
{
    return num_detections;
}

unsigned int TfObjDetApiDetector::inputImgWidthImpl() const
{
    unsigned int input_layer_id = _engine->getBindingIndex(input_layer_name);
    auto dims = _engine->getBindingDimensions(input_layer_id);
    return dims.d[1];
}

unsigned int TfObjDetApiDetector::inputImgHeightImpl() const
{
    unsigned int input_layer_id = _engine->getBindingIndex(input_layer_name);
    auto dims = _engine->getBindingDimensions(input_layer_id);
    return dims.d[2];
}

TfObjDetApiDetector::TfObjDetApiDetector(const std::string &filename, bool use_cuda_graph):
        _ready(false),
        _gpu_input_layer_mem(nullptr),
        _gpu_detection_boxes_mem(nullptr),
        _gpu_detection_classes_mem(nullptr),
        _gpu_num_detections_mem(nullptr),
        _gpu_detection_scores_mem(nullptr),
        _cudaStream(nullptr),
        _engine(nullptr),
        _use_cuda_graph(use_cuda_graph),
        _cuda_graph_created(false)
{
    std::vector<char> engine_data;

    if (!(_ready = readEngineFile(filename, engine_data)))
        return;

    if (!(_ready = buildEngine(engine_data)))
        return;

    if (!(_ready = createContext()))
        return;

    if (!(_ready = createBindings()))
        return;

    _ready = cudaStreamCreate(&_cudaStream) == cudaSuccess;
}

TfObjDetApiDetector::~TfObjDetApiDetector()
{
    cudaFree(_gpu_detection_scores_mem);
    cudaFree(_gpu_detection_classes_mem);
    cudaFree(_gpu_detection_boxes_mem);
    cudaFree(_gpu_num_detections_mem);
    cudaFree(_gpu_input_layer_mem);
}

bool TfObjDetApiDetector::readEngineFile(const std::string &filename, std::vector<char> &engine_data)
{
    std::ifstream engine_file_stream(filename, std::ios::binary);

    if (engine_file_stream.fail())
        return false;

    engine_file_stream.seekg(0, std::ifstream::end);
    auto file_size = engine_file_stream.tellg();
    engine_file_stream.seekg(0, std::ifstream::beg);

    engine_data.resize(file_size);
    engine_file_stream.read(engine_data.data(), file_size);

    return true;
}

bool TfObjDetApiDetector::buildEngine(std::vector<char> &engine_data)
{
    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    initLibNvInferPlugins(nullptr, "");
    _engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr));
    return _engine.get() != nullptr;
}

bool TfObjDetApiDetector::createContext()
{
    _context = util::UniquePtr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());
    return _context != nullptr;
}

bool TfObjDetApiDetector::createBindings()
{
    int32_t input_layer_id = _engine->getBindingIndex(input_layer_name);

    if (input_layer_id == -1)
        return false;

    if (_engine->getBindingDataType(input_layer_id) != nvinfer1::DataType::kFLOAT)
        return false;

    _context->setBindingDimensions(input_layer_id, _engine->getBindingDimensions(input_layer_id));

    auto num_detections_id = _engine->getBindingIndex(output_layer_num_detections_name);
    auto detection_boxes_id = _engine->getBindingIndex(output_layer_detection_boxes);
    auto detection_scores_id = _engine->getBindingIndex(output_layer_detection_scores);
    auto detection_classes_id = _engine->getBindingIndex(output_layer_detection_classes);

    if (num_detections_id == -1 || detection_boxes_id == -1 || detection_scores_id == -1 || detection_classes_id == -1)
        return false;

    _input_layer_size = util::getMemorySize(_engine->getBindingDimensions(input_layer_id), sizeof(float));
    if (cudaMalloc(&_gpu_input_layer_mem, _input_layer_size) != cudaSuccess)
        return false;

    _num_detections_size = util::getMemorySize(_engine->getBindingDimensions(num_detections_id), sizeof(float));
    if (cudaMalloc(&_gpu_num_detections_mem, _num_detections_size) != cudaSuccess)
        return false;

    _detection_boxes_size = util::getMemorySize(_engine->getBindingDimensions(detection_boxes_id), sizeof(float));
    if (cudaMalloc(&_gpu_detection_boxes_mem, _detection_boxes_size) != cudaSuccess)
        return false;

    _detection_scores_size = util::getMemorySize(_engine->getBindingDimensions(detection_scores_id), sizeof(float));
    if (cudaMalloc(&_gpu_detection_scores_mem, _detection_scores_size) != cudaSuccess)
        return false;

    _detection_classes_size = util::getMemorySize(_engine->getBindingDimensions(detection_classes_id), sizeof(int32_t));
    if (cudaMalloc(&_gpu_detection_classes_mem, _detection_classes_size) != cudaSuccess)
        return false;

    _bindings[0] = _gpu_input_layer_mem;
    _bindings[1] = _gpu_num_detections_mem;
    _bindings[2] = _gpu_detection_boxes_mem;
    _bindings[3] = _gpu_detection_scores_mem;
    _bindings[4] = _gpu_detection_classes_mem;

    return true;
}

DetectedObjectsInfo TfObjDetApiDetector::detectImpl(void* buffer, bool &detection_status, unsigned int images_num) {
    detection_status = false;
    if (!_ready || _engine->getMaxBatchSize() < images_num)
        return std::move(DetectedObjectsInfo());

    if (cudaMemcpyAsync(_gpu_input_layer_mem, buffer, _input_layer_size, cudaMemcpyHostToDevice, _cudaStream) != cudaSuccess)
        return std::move(DetectedObjectsInfo());

    if (_use_cuda_graph)
    {
        if (!_cuda_graph_created)
        {
            cudaStreamBeginCapture(_cudaStream, cudaStreamCaptureModeGlobal);
            _context->enqueueV2(_bindings, _cudaStream, nullptr);
            if (cudaStreamEndCapture(_cudaStream, &_graph) != cudaSuccess)
                return std::move(DetectedObjectsInfo());
            cudaGraphInstantiate(&_instance, _graph, nullptr, nullptr, 0);
            _cuda_graph_created= true;
        }

        cudaGraphLaunch(_instance, _cudaStream);
    }
    else
    {
        _context->enqueue(images_num, _bindings, _cudaStream, nullptr);
    }
    cudaStreamSynchronize(_cudaStream);

    auto *boxes   = new float[_detection_boxes_size / sizeof (float )];
    auto *scores  = new float[_detection_scores_size / sizeof (float )];
    auto *classes = new int32_t[_detection_classes_size / sizeof (int32_t )];

    bool gpu_copy_failed =  cudaMemcpyAsync(boxes,   _gpu_detection_boxes_mem,   _detection_boxes_size,   cudaMemcpyDeviceToHost, _cudaStream) != cudaSuccess ||
                            cudaMemcpyAsync(scores,  _gpu_detection_scores_mem,  _detection_scores_size,  cudaMemcpyDeviceToHost, _cudaStream) != cudaSuccess ||
                            cudaMemcpyAsync(classes, _gpu_detection_classes_mem, _detection_classes_size, cudaMemcpyDeviceToHost, _cudaStream) != cudaSuccess;

    if (gpu_copy_failed)
    {
        delete [] boxes;
        delete [] scores;
        delete [] classes;
        return std::move(DetectedObjectsInfo());
    }

    detection_status = true;
    DetectedObjectsInfo detectedObjectsInfo(num_detections, images_num, boxes, scores, classes);
    return std::move(detectedObjectsInfo);
}