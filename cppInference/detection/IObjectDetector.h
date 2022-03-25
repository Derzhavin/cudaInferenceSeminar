//
// Created by denis on 19.02.2022.
//

#ifndef VISION_IOBJECTDETECTOR_H
#define VISION_IOBJECTDETECTOR_H


#include <cstddef>
#include <memory>


class DetectedObjectsInfo
{
    std::unique_ptr<float> _boxes;
    std::unique_ptr<float> _scores;
    std::unique_ptr<int32_t> _classes;
    unsigned short _num_objects_per_batch;
    unsigned short _batch_size;

public:
    explicit DetectedObjectsInfo();
    explicit DetectedObjectsInfo(unsigned short num_objects_per_batch, unsigned short batch_size, float *boxes,
                                 float *scores,
                                 int32_t *classes);

    const float *const boxes() const;
    const float *const scores() const;
    const int32_t *const classes() const;
    unsigned short numObjectsPerBatch() const;
    unsigned short batchSize() const;
};

template<typename ObjectDetectorImpl>
class IObjectDetector
{
    friend ObjectDetectorImpl;

    IObjectDetector() = default;

    ObjectDetectorImpl& impl()
    {
        return static_cast<ObjectDetectorImpl&>(*this);
    }

public:
    inline bool isReady()
    {
        return impl().isReadyImpl();
    }
    inline DetectedObjectsInfo detect(void* buffer, bool& detection_status, unsigned int images_num = 1)
    {
        return impl().detectImpl(buffer, detection_status, images_num);
    }
    inline unsigned int inputImgWidth()
    {
        return impl().inputImgWidthImpl();
    }
    inline unsigned int inputImgHeight()
    {
        return impl().inputImgHeightImpl();
    }
    inline unsigned int detectionsNum()
    {
        return impl().detectionsNumImpl();
    }
    inline DetectedObjectsInfo detectedObjectsInfo()
    {
        return impl().detectedObjectsInfoImpl();
    }
};


#endif //VISION_IOBJECTDETECTOR_H