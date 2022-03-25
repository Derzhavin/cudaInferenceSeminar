//
// Created by denis on 18.02.2022.
//

#ifndef VISION_FILECAPTURE_H
#define VISION_FILECAPTURE_H

#include "IVideoCapture.h"

#include <opencv2/videoio.hpp>

class FileCapture: public IVideoCapture<FileCapture>{

    friend class IVideoCapture<FileCapture>;

    cv::VideoCapture _video_capture;

    bool readFrameImpl(cv::Mat& mat);
    bool isOpenedImpl();
    void closeImpl();
    std::pair<uint , uint > resolutionImpl();

public:
    explicit FileCapture(const std::string& path, unsigned short frames_buffer_size);
    ~FileCapture();
};


#endif //VISION_FILECAPTURE_H
