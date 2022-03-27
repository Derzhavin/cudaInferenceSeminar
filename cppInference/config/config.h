//
// Created by denis on 19.02.2022.
//

#ifndef VISION_CONFIG_H
#define VISION_CONFIG_H


#include <string>
#include <vector>

struct Config
{
    float scores_threshold;
    unsigned short batch_size;
    unsigned short frames_buffer_size;
    bool use_cuda_graph;
    std::string engine_file_path;
    std::string file_capture_path;
    std::string input_tensor;
    std::vector<std::string> classes;
};

#endif //VISION_CONFIG_H
