#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>
#include <math.h>

namespace cuda_filter
{

#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            output[(y * width + x) * channels + c] =
                static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    __global__ void hdrToneMappingKernel(const unsigned char *input,
                                         unsigned char *output,
                                         int width,
                                         int height,
                                         int channels,
                                         float exposure,
                                         float gamma)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int idx = (y * width + x) * channels;

        for (int c = 0; c < channels; c++)
        {
            float color = input[idx + c] / 255.0f;

            // Simple global HDR tone mapping
            float mapped = 1.0f - expf(-color * exposure);

            // Gamma correction
            mapped = powf(mapped, 1.0f / gamma);

            mapped = min(max(mapped, 0.0f), 1.0f);

            output[idx + c] = static_cast<unsigned char>(mapped * 255.0f);
        }
    }

    void applyHDRGPU(const cv::Mat &input, cv::Mat &output, float exposure, float gamma)
    {
        if (input.empty())
        {
            PLOG_ERROR << "Input image is empty";
            return;
        }

        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();

        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        hdrToneMappingKernel<<<gridDim, blockDim>>>(
            d_input,
            d_output,
            width,
            height,
            channels,
            exposure,
            gamma);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
    }

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        float *h_kernel = new float[kernelSize * kernelSize];

        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);

        delete[] h_kernel;
    }

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;
        int radius = kernelSize / 2;

        float *h_kernel = new float[kernelSize * kernelSize];

        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0.0f;

                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ix = std::min(std::max(x + kx, 0), width - 1);
                            int iy = std::min(std::max(y + ky, 0), height - 1);

                            float kernelValue = h_kernel[(ky + radius) * kernelSize + (kx + radius)];
                            float pixelValue = input.at<cv::Vec3b>(iy, ix)[c];

                            sum += pixelValue * kernelValue;
                        }
                    }

                    output.at<cv::Vec3b>(y, x)[c] =
                        static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }

        delete[] h_kernel;
    }

} // namespace cuda_filter