#include "cuda_core/image_filters.h"
#include <cuda_runtime.h>
#include <stdio.h>

// This file contains implementations of more advanced image processing filters
// that run as CUDA kernels on the GPU.

// --- Gaussian Blur Kernel ---

/**
 * @brief A 3x3 Gaussian kernel defined in constant memory.
 *
 * Constant memory is a special type of read-only memory on the GPU that is cached.
 * It is very fast for cases where all threads in a warp access the same memory location,
 * which is true when reading from a convolution kernel like this one.
 */
__constant__ int G_KERNEL[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};
const int G_KERNEL_DIVISOR = 16; // Sum of the kernel weights

/**
 * @brief Kernel to apply a 3x3 convolution, specifically for Gaussian blur.
 *
 * Each thread calculates the value for one output pixel. It reads a 3x3 neighborhood
 * of pixels from the input image, multiplies them by the corresponding kernel weights,
 * sums them up, and writes the final normalized value to the output image.
 */
__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sum = 0;

        // Iterate over the 3x3 kernel neighborhood.
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int neighbor_x = x + j;
                int neighbor_y = y + i;

                // Simple "clamp-to-edge" border handling.
                // For pixels near the edge, we clamp the coordinates to be within the image bounds.
                if (neighbor_x < 0) neighbor_x = 0;
                if (neighbor_x >= width) neighbor_x = width - 1;
                if (neighbor_y < 0) neighbor_y = 0;
                if (neighbor_y >= height) neighbor_y = height - 1;

                // Read the neighbor pixel value and apply the kernel weight.
                sum += input[neighbor_y * width + neighbor_x] * G_KERNEL[i + 1][j + 1];
            }
        }
        output[y * width + x] = (unsigned char)(sum / G_KERNEL_DIVISOR);
    }
}

int launch_gaussian_blur_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + 15) / 16, (height + 15) / 16);
    gaussian_blur_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    // Return the last error from the asynchronous kernel launch.
    return cudaGetLastError();
}


// --- Sobel Filter Kernel ---

/**
 * @brief Kernel to apply a Sobel edge detection filter.
 *
 * This kernel calculates the image gradient at each pixel by convolving with two
 * Sobel operators (Gx and Gy). The final output pixel value is the magnitude
 * of the gradient, which highlights edges.
 */
__global__ void sobel_filter_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Sobel operators for detecting horizontal and vertical edges.
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // We only process interior pixels, as the Sobel operator requires a 3x3 neighborhood.
    // A more robust implementation would handle borders.
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int sumX = 0;
        int sumY = 0;

        // Apply the Gx and Gy kernels.
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        // Calculate the magnitude of the gradient.
        // The true magnitude is sqrt(Gx^2 + Gy^2), but for performance,
        // a common approximation is |Gx| + |Gy|.
        int magnitude = abs(sumX) + abs(sumY);

        // Clamp the magnitude to the valid 0-255 range for an 8-bit grayscale image.
        output[y * width + x] = (unsigned char)(magnitude > 255 ? 255 : magnitude);
    } else {
        // For border pixels, just write black.
        if (x < width && y < height) {
            output[y * width + x] = 0;
        }
    }
}

int launch_sobel_filter_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + 15) / 16, (height + 15) / 16);
    sobel_filter_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    return cudaGetLastError();
}
