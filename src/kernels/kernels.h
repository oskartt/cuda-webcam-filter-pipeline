#pragma once

#include <opencv2/opencv.hpp>

namespace cuda_filter
{

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

    void applyHDRGPU(const cv::Mat &input, cv::Mat &output,
                     float exposure = 1.0f,
                     float gamma = 2.2f);

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

    namespace cuda
    {
#ifdef __CUDACC__
        __host__ __device__ inline int divUp(int a, int b)
        {
            return (a + b - 1) / b;
        }
#endif
    }

} // namespace cuda_filter