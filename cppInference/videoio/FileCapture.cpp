//
// Created by denis on 18.02.2022.
//

#include "FileCapture.h"

FileCapture::FileCapture(const std::string &path, unsigned short frames_buffer_size): _video_capture(path) {
    _video_capture.set(cv::CAP_PROP_BUFFERSIZE, frames_buffer_size);
}
FileCapture::~FileCapture()
{
    close();
}


bool FileCapture::readFrameImpl(cv::Mat &mat) {
    return _video_capture.read(mat);
}

bool FileCapture::isOpenedImpl() {
    return _video_capture.isOpened();
}

void FileCapture::closeImpl() {
    _video_capture.release();
}

std::pair<uint , uint> FileCapture::resolutionImpl()
{
    uint w = _video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
    uint h = _video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    return std::make_pair(w, h);
}
