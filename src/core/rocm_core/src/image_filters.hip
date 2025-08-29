#include "rocm_core/image_filters.h"
#include <hip/hip_runtime.h>

// --- ROCm Kernels ---

__global__ void rgb_to_grayscale_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_index = y * width + x;
        int rgb_index = pixel_index * 3;

        unsigned char r = input[rgb_index];
        unsigned char g = input[rgb_index + 1];
        unsigned char b = input[rgb_index + 2];

        output[pixel_index] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

__global__ void box_blur_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float r = 0, g = 0, b = 0;
        int count = 0;

        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int nx = x + i;
                int ny = y + j;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int index = (ny * width + nx) * 3;
                    r += input[index];
                    g += input[index + 1];
                    b += input[index + 2];
                    count++;
                }
            }
        }

        int out_index = (y * width + x) * 3;
        output[out_index] = (unsigned char)(r / count);
        output[out_index + 1] = (unsigned char)(g / count);
        output[out_index + 2] = (unsigned char)(b / count);
    }
}

// --- Host-side Launcher Functions ---

int launch_rgb_to_grayscale_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    hipLaunchKernelGGL(rgb_to_grayscale_kernel, numBlocks, threadsPerBlock, 0, 0, d_input, d_output, width, height);

    return hipGetLastError();
}

int launch_box_blur_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    hipLaunchKernelGGL(box_blur_kernel, numBlocks, threadsPerBlock, 0, 0, d_input, d_output, width, height);

    return hipGetLastError();
}
