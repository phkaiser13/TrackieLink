#ifndef CUDA_CORE_IMAGE_FILTERS_H
#define CUDA_CORE_IMAGE_FILTERS_H

// This header defines the C-style interface for launching more advanced
// image filtering CUDA kernels.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Applies a 3x3 Gaussian blur to a grayscale image on the GPU.
 *
 * @param d_input A device pointer to the input grayscale image data.
 * @param d_output A device pointer to the output blurred image data.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @return An error code (0 for success) corresponding to cudaError_t.
 */
int launch_gaussian_blur_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height);

/**
 * @brief Applies a Sobel edge detection filter to a grayscale image on the GPU.
 *
 * This kernel calculates the gradient magnitude at each pixel to highlight edges.
 *
 * @param d_input A device pointer to the input grayscale image data.
 * @param d_output A device pointer to the output edge-detected image data.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @return An error code (0 for success).
 */
int launch_sobel_filter_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height);


#ifdef __cplusplus
}
#endif

#endif // CUDA_CORE_IMAGE_FILTERS_H
