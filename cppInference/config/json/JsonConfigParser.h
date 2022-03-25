//
// Created by denis on 19.02.2022.
//

#ifndef VISION_JSONCONFIGPARSER_H
#define VISION_JSONCONFIGPARSER_H

#include "config/config.h"

#include <nlohmann/json.hpp>
#include <variant>

class IJsonConfigParser
{
public:
    IJsonConfigParser() = default;
    virtual bool parseJson(nlohmann::json& json_obj, Config& config) = 0;
};
class ObjectDetectorJsonParser: public IJsonConfigParser
{
public:
    ObjectDetectorJsonParser() = default;
    bool parseJson(nlohmann::json& json_obj, Config& config) override;
};

class FileCaptureJsonParser: public IJsonConfigParser
{
public:
    FileCaptureJsonParser() = default;
    bool parseJson(nlohmann::json& json_obj, Config& config) override;
};

class CullingJsonParser: public IJsonConfigParser
{
public:
    CullingJsonParser() = default;
    bool parseJson(nlohmann::json& json_obj, Config& config) override;
};

class JsonConfigParser
{
    std::vector<IJsonConfigParser*> _parsers;
public:
    JsonConfigParser(std::vector<IJsonConfigParser*>& parsers);
    bool parseJson(nlohmann::json& json_obj, Config& config);
    bool parseFile(const std::string &filename, Config& config);
};


#endif //VISION_JSONCONFIGPARSER_H
