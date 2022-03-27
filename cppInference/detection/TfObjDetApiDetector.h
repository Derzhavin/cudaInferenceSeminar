//
// Created by denis on 19.02.2022.
//

#ifndef VISION_TFOBJDETAPIDETECTOR_H
#define VISION_TFOBJDETAPIDETECTOR_H

#include "IObjectDetector.h"

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "common/util.h"
#include <string>
#include <vector>

class TfObjDetApiDetector: public IObjectDetector<TfObjDetApiDetector>
{
    friend class IObjectDetector<TfObjDetApiDetector>;

    bool _ready;

    util::UniquePtr<nvinfer1::ICudaEngine> _engine;
    util::UniquePtr<nvinfer1::IExecutionContext> _context;

    cudaStream_t _cudaStream;

    cudaGraph_t _graph;
    cudaGraphExec_t _instance;
    bool _use_cuda_graph;
    bool _cuda_graph_created;

    void* _gpu_input_layer_mem;
    void* _gpu_num_detections_mem;
    void* _gpu_detection_boxes_mem;
    void* _gpu_detection_scores_mem;
    void* _gpu_detection_classes_mem;

    void* _bindings[5];

    int32_t _input_layer_size;
    int32_t _num_detections_size;
    int32_t _detection_boxes_size;
    int32_t _detection_scores_size;
    int32_t _detection_classes_size;

    std::string _input_layer_name;

    bool isReadyImpl() const;
    unsigned int inputImgWidthImpl() const;
    unsigned int inputImgHeightImpl() const;
    DetectedObjectsInfo detectImpl(void* buffer, bool& detection_status, unsigned int images_num = 1);
    unsigned int detectionsNumImpl();

    static bool readEngineFile(const std::string& filename, std::vector<char> &engine_data);
    bool buildEngine(std::vector<char> &engine_data);
    bool createContext();
    bool createBindings();

public:
    static const char* output_layer_num_detections_name;
    static const char* output_layer_detection_boxes;
    static const char* output_layer_detection_scores;
    static const char* output_layer_detection_classes;
    static const short num_detections;
    explicit TfObjDetApiDetector(const std::string& filename, const std::string& input_tensor, bool use_cuda_graph = false);
    ~TfObjDetApiDetector();
};


#endif //VISION_TFOBJDETAPIDETECTOR_H