//
// Created by denis on 22.02.2022.
//
#include "IObjectDetector.h"

DetectedObjectsInfo::DetectedObjectsInfo():
        _num_objects_per_batch(0),
        _boxes(std::unique_ptr<float>(nullptr)),
        _scores(std::unique_ptr<float>(nullptr)),
        _classes(std::unique_ptr<int32_t>(nullptr))
{

}

DetectedObjectsInfo::DetectedObjectsInfo(unsigned short num_objects_per_batch, unsigned short batch_size, float *boxes,
                                         float *scores,
                                         int32_t *classes) :
        _num_objects_per_batch(num_objects_per_batch),
        _batch_size(batch_size),
        _boxes(std::unique_ptr<float>(boxes)),
        _scores(std::unique_ptr<float>(scores)),
        _classes(std::unique_ptr<int32_t>(classes))
{

}

const float *const DetectedObjectsInfo::boxes() const
{
    return _boxes.get();
}

const float *const DetectedObjectsInfo::scores() const
{
    return _scores.get();
}

const int32_t *const DetectedObjectsInfo::classes() const
{
    return _classes.get();
}

unsigned short DetectedObjectsInfo::numObjectsPerBatch() const
{
    return _num_objects_per_batch;
}

unsigned short DetectedObjectsInfo::batchSize() const
{
    return _batch_size;
}
