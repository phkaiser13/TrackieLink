#pragma once

#include "vision_types.h" // For DetectionResult
#include <vector>

// Forward declaration for OpenCV's Mat
namespace cv {
    class Mat;
}

namespace trackie::vision {

/**
 * @brief Draws detection bounding boxes on an image using CUDA.
 *
 * This function takes an image and a list of detections, and uses a CUDA
 * kernel to draw the bounding boxes directly on the GPU for acceleration.
 *
 * @param image The input/output image (cv::Mat). The boxes will be drawn on this image.
 *              The Mat must be continuous.
 * @param detections A vector of DetectionResult structs to draw.
 */
void draw_detections_cuda(cv::Mat& image, const std::vector<DetectionResult>& detections);

} // namespace trackie::vision
