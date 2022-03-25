//
// Created by denis on 23.02.2022.
//

#ifndef VISION_APP_H
#define VISION_APP_H

#include "config/config.h"
#include "videoio/IVideoCapture.h"
#include "detection/IObjectDetector.h"

#include <ctime>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

template <  class VideoCaptureImpl,
            class ObjectDetectorImpl>
class App
{
    Config _config;
    IVideoCapture<VideoCaptureImpl>& _video_capture;
    IObjectDetector<ObjectDetectorImpl>& _objects_detector;

    inline bool checkGeometry(const float& y_min, const float& x_min, const float& y_max, const float& x_max);

public:
    explicit App(
        Config& config,
        VideoCaptureImpl& videoCapture,
        ObjectDetectorImpl& objects_detector
    );
    void run();
};

template <class VideoCaptureImpl,
        class ObjectDetectorImpl>
App<    VideoCaptureImpl,
        ObjectDetectorImpl>::App(
            Config& config,
            VideoCaptureImpl& videoCapture,
            ObjectDetectorImpl& objects_detector
):  _config(config),
    _video_capture(videoCapture),
    _objects_detector(objects_detector)
{

}

template <  class VideoCaptureImpl,
        class ObjectDetectorImpl>
void App<    VideoCaptureImpl,
        ObjectDetectorImpl>::run()
{
    if (!_video_capture.isOpened())
    {
        std::cout << "Failed to open video capture" << std::endl;
        return;
    }

    if (!_objects_detector.isReady())
    {
        std::cout << "Failed to create objects detector" << std::endl;
        return;
    }

    const short cnn_input_w = _objects_detector.inputImgWidth();
    const short cnn_input_h = _objects_detector.inputImgHeight();
    auto p_cnn_input_buffer = std::unique_ptr<float[]>(new float [3 * cnn_input_w * cnn_input_h * _config.batch_size]);

    std::vector<cv::Mat> cnn_batch_input(_config.batch_size);
    for (int i = 0; i < _config.batch_size; ++i)
        cnn_batch_input[i] = cv::Mat(cnn_input_h, cnn_input_w, CV_32FC3, p_cnn_input_buffer.get() + 3 * cnn_input_w * cnn_input_h * i, 0);

    cv::Mat resized_frame;
    cv::Mat cnn_input;

    const auto frame_width = _video_capture.resolution().first;
    const auto frame_height = _video_capture.resolution().second;
    const auto objects_num = _config.batch_size * _objects_detector.detectionsNum();

    const auto text_colour = cv::Scalar(0, 255, 0);
    const auto rect_colour = cv::Scalar(0, 255, 0);

    while (true)
    {
        std::vector<cv::Mat> frames(_config.batch_size);

        for (int i = 0; i < _config.batch_size; ++i)
        {
            while (frames[i].empty())
            {
                _video_capture.readFrame(frames[i]);
            }
            cv::resize(frames[i], resized_frame, cv::Size(cnn_input_w, cnn_input_h));
            resized_frame.convertTo(cnn_batch_input[i], CV_32FC3);
        }

        bool detection_status = false;

        auto start_detect = std::chrono::system_clock::now();

        DetectedObjectsInfo detected_objects_info(_objects_detector.detect(p_cnn_input_buffer.get(), detection_status, _config.batch_size));

        auto dur_detect = (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_detect).count() / 1000;
        std::cout << "detection time in ms: " << dur_detect << std::endl;

        if(!detection_status)
        {
            std::cout << "Failed to detect" << std::endl;
            return;
        }

        for (int obj_i = 0; obj_i < objects_num; ++obj_i)
        {
            if (detected_objects_info.classes()[obj_i] > -1 && detected_objects_info.scores()[obj_i] >= _config.scores_threshold)
            {
                const float y_min = (float)frame_height * detected_objects_info.boxes()[obj_i * 4    ];
                const float x_min = (float)frame_width * detected_objects_info.boxes() [obj_i * 4 + 1];
                const float y_max = (float)frame_height * detected_objects_info.boxes()[obj_i * 4 + 2];
                const float x_max = (float)frame_width * detected_objects_info.boxes() [obj_i * 4 + 3];

                if (checkGeometry(y_min, x_min, y_max, x_max))
                {
                    const auto frame_no = obj_i / _objects_detector.detectionsNum();

                    cv::Rect frame_rect(x_min, y_min, x_max - x_min, y_max - y_min);

                    const auto cls = detected_objects_info.classes()[obj_i];

                    auto p1 = cv::Point(x_min, y_min);
                    auto p2 = cv::Point(x_max, y_max);
                    cv::rectangle(frames[frame_no], p1, p2, rect_colour, 2);
                    cv::putText(frames[frame_no], _config.classes[(size_t)cls], p1, cv::FONT_HERSHEY_DUPLEX, 2, text_colour, false);
                    cv::imshow("frame", frames[frame_no]);

                    if(cv::waitKey(1) == 'q')
                        return;
                }
            }
        }
    }
}

template<class VideoCaptureImpl, class ObjectDetectorImpl>
bool
App<VideoCaptureImpl, ObjectDetectorImpl>::checkGeometry(const float &y_min, const float &x_min, const float &y_max,
                                                         const float &x_max)
{
    const auto frame_width = _video_capture.resolution().first;
    const auto frame_height = _video_capture.resolution().second;

    return !(std::isnan(x_min) || std::isnan(y_min) || std::isnan(x_max) || std::isnan(y_max) ||
             std::isinf(x_min) || std::isinf(y_min) || std::isinf(x_max) || std::isinf(y_max) ||
             x_min >= frame_width || y_min >= frame_height || x_max >= frame_width || y_max >= frame_height ||
             x_min <= 0 || y_min <= 0 || x_max <= 0 || y_max <= 0 ||
             std::abs(x_max - x_min) < 1 || std::abs(y_max - y_min) < 1);
}


#endif //VISION_APP_H
