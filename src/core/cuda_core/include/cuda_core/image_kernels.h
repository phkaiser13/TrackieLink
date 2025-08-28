#ifndef CUDA_CORE_IMAGE_KERNELS_H
#define CUDA_CORE_IMAGE_KERNELS_H

// This header defines the C-style interface for launching CUDA kernels
// related to image processing. Keeping these separate from memory management
// improves modularity.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converts a 3-channel (RGB) image to a 1-channel grayscale image on the GPU.
 *
 * This function is a host-side wrapper that configures and launches the CUDA kernel.
 *
 * @param d_input A device pointer to the input image data (unsigned char, interleaved RGB).
 * @param d_output A device pointer to the output grayscale image data (unsigned char).
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @return An error code (0 for success) corresponding to cudaError_t.
 */
int launch_rgb_to_grayscale_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CORE_IMAGE_KERNELS_H
