#include "config/config.h"
#include "config/json/JsonConfigParser.h"

#include <iostream>
#include <videoio/FileCapture.h>
#include <detection/TfObjDetApiDetector.h>
#include <app/App.h>

int main(int argc, char *argv[]) {
    if (argc != 2)
    {
        std::cout << "Usage: ./app <config> path" << std::endl;
        return 0;
    }

    Config config;
    const std::string config_path(argv[1]);

    std::vector<IJsonConfigParser*> parsers
    {
        new ObjectDetectorJsonParser(),
        new FileCaptureJsonParser(),
        new CullingDetectionsJsonParser()
    };
    JsonConfigParser json_config_parser(parsers);

    if (!json_config_parser.parseFile(config_path, config))
    {
        std::cout << "Failed to open config file" << std::endl;
        return 0;
    }

    FileCapture video_capture(config.file_capture_path, config.frames_buffer_size);

    TfObjDetApiDetector objects_detector(config.engine_file_path, config.use_cuda_graph);

    App< decltype(video_capture), decltype(objects_detector)> app(config, video_capture, objects_detector);
    app.run();

    return 0;
}
