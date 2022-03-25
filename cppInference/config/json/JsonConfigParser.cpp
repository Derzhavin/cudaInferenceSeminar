//
// Created by denis on 19.02.2022.
//

#include "JsonConfigParser.h"

#include <fstream>


bool JsonConfigParser::parseJson(nlohmann::json& json_obj, Config& config)
{
    for (auto & parser : _parsers)
    {
        if (!parser->parseJson(json_obj, config))
            return false;
    }
    return true;
}

bool JsonConfigParser::parseFile(const std::string &filename, Config& config)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        return false;

    nlohmann::json jf(nlohmann::json::parse(ifs));
    auto res = parseJson(jf, config);

    return res;
}

JsonConfigParser::JsonConfigParser(std::vector<IJsonConfigParser*> &parsers): _parsers(parsers) {}

bool FileCaptureJsonParser::parseJson(nlohmann::json &json_obj, Config &config)
{
    auto file_capture_json_obj = json_obj["FileCapture"];

    config.file_capture_path = file_capture_json_obj["Path"];
    config.frames_buffer_size = file_capture_json_obj["FramesBufferSize"];

    return true;
}

bool ObjectDetectorJsonParser::parseJson(nlohmann::json &json_obj, Config &config)
{
    auto object_detection_json = json_obj["ObjectDetection"];

    config.batch_size        = object_detection_json["BatchSize"];
    config.engine_file_path  = object_detection_json["EnginePath"];
    config.use_cuda_graph  = object_detection_json["UseCudaGraph"];

    auto classes_json = object_detection_json["Classes"];

    config.classes.clear();
    for (auto &cls: classes_json)
    {
        config.classes.push_back(cls);
    }
    return true;
}

bool CullingJsonParser::parseJson(nlohmann::json &json_obj, Config &config) {
    auto culling_json = json_obj["CullingDetections"];

    config.scores_threshold = culling_json["ScoresThreshold"];
    return true;
}
