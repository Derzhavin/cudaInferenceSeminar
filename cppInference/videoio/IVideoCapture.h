//
// Created by denis on 18.02.2022.
//

#ifndef VISION_IVIDEOCAPTURE_H
#define VISION_IVIDEOCAPTURE_H

#include <opencv2/core.hpp>


template<typename VideoCaptureImpl>
class IVideoCapture
{
    friend VideoCaptureImpl;

    IVideoCapture() = default;

    VideoCaptureImpl& impl()
    {
        return static_cast<VideoCaptureImpl&>(*this);
    }

public:
    inline bool isOpened()
    {
        return impl().isOpenedImpl();
    }
    inline bool readFrame(cv::Mat& frame)
    {
        return impl().readFrameImpl(frame);
    }
    inline void close()
    {
        impl().closeImpl();
    }
    inline std::pair<uint , uint > resolution()
    {
        return impl().resolutionImpl();
    }
};


#endif //VISION_IVIDEOCAPTURE_H
