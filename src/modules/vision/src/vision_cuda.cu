#include "vision/include/vision_cuda.hpp"
#include "shared/types/vision_types.h"
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <iostream>

namespace trackie::vision {

// CUDA Kernel to draw bounding boxes on an image
__global__ void draw_boxes_kernel(uchar3* image_data, int width, int height, DetectionResult* detections, int num_detections) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int pixel_index = y * width + x;
    uchar3 color = {0, 255, 0}; // Green color for the box

    for (int i = 0; i < num_detections; ++i) {
        DetectionResult box = detections[i];

        // Check if the pixel is on the border of the bounding box
        bool on_top_border = (y == box.y_min || y == box.y_max) && (x >= box.x_min && x <= box.x_max);
        bool on_side_border = (x == box.x_min || x == box.x_max) && (y >= box.y_min && y <= box.y_max);

        if (on_top_border || on_side_border) {
            image_data[pixel_index] = color;
        }
    }
}

// C++ Wrapper function to launch the CUDA kernel
void draw_detections_cuda(cv::Mat& image, const std::vector<DetectionResult>& detections) {
    if (image.empty() || detections.empty()) {
        return;
    }

    if (!image.isContinuous()) {
        std::cerr << "CUDA drawing requires a continuous cv::Mat." << std::endl;
        // Or we could clone it: image = image.clone();
        return;
    }

    if (image.type() != CV_8UC3) {
        std::cerr << "CUDA drawing currently only supports CV_8UC3 images." << std::endl;
        return;
    }

    int width = image.cols;
    int height = image.rows;
    size_t image_size = width * height * sizeof(uchar3);
    size_t detections_size = detections.size() * sizeof(DetectionResult);

    uchar3* d_image_data = nullptr;
    DetectionResult* d_detections = nullptr;

    // Allocate memory on the GPU
    cudaError_t err = cudaMalloc(&d_image_data, image_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for image: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_detections, detections_size);
    if (err != cudaSuccess) {
        cudaFree(d_image_data);
        std::cerr << "Failed to allocate device memory for detections: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Copy data from CPU to GPU
    cudaMemcpy(d_image_data, image.ptr<uchar3>(), image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_detections, detections.data(), detections_size, cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                    (height + threads_per_block.y - 1) / threads_per_block.y);

    // Launch the kernel
    draw_boxes_kernel<<<num_blocks, threads_per_block>>>(d_image_data, width, height, d_detections, detections.size());

    // Copy the modified image back from GPU to CPU
    cudaMemcpy(image.ptr<uchar3>(), d_image_data, image_size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_image_data);
    cudaFree(d_detections);
}

} // namespace trackie::vision
