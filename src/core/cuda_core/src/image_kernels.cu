#include "cuda_core/image_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>

// --- CUDA Kernel ---

/**
 * @brief A __global__ function (kernel) that is executed on the GPU.
 * Each thread in the grid is responsible for converting one pixel from RGB to grayscale.
 */
__global__ void rgb_to_grayscale_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    // Calculate the unique global thread ID for the x and y dimensions.
    // This provides a 2D mapping of threads to the 2D image pixels.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check that the thread's coordinates are within the image bounds.
    // This is necessary because the number of threads in the grid might be
    // larger than the number of pixels (if dimensions are not a multiple of block size).
    if (x < width && y < height) {
        int pixel_index = y * width + x;
        int rgb_index = pixel_index * 3;

        // Load the RGB values for the pixel.
        unsigned char r = input[rgb_index];
        unsigned char g = input[rgb_index + 1];
        unsigned char b = input[rgb_index + 2];

        // Apply the standard luminosity formula for grayscale conversion.
        // Using float coefficients for precision before casting back to unsigned char.
        // Y = 0.299*R + 0.587*G + 0.114*B
        output[pixel_index] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}


// --- Host-side Launcher Function ---

int launch_rgb_to_grayscale_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    if (!d_input || !d_output || width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }

    // Define the dimensions of the thread blocks. 16x16 (256 threads) is a common choice.
    const dim3 threadsPerBlock(16, 16);

    // Calculate the number of blocks needed in each dimension to cover the entire image.
    // The ceiling division `(a + b - 1) / b` ensures we have enough blocks.
    const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel on the device.
    // The <<<...>>> syntax is specific to CUDA C++ and is used to specify
    // the execution configuration (grid dimensions, block dimensions).
    rgb_to_grayscale_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    // cudaGetLastError() is used to check for errors from asynchronous kernel launches.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return err;
    }

    // It's often a good practice to synchronize after a kernel launch if the CPU
    // needs the results immediately, but this function will remain asynchronous.
    // cudaDeviceSynchronize();

    return cudaSuccess;
}
