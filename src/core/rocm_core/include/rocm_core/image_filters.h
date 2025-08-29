#ifndef IMAGE_FILTERS_H
#define IMAGE_FILTERS_H

/**
 * @brief Lauches the RGB to grayscale conversion kernel on the ROCm device.
 */
int launch_rgb_to_grayscale_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height);

/**
 * @brief Lauches the box blur kernel on the ROCm device.
 */
int launch_box_blur_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height);

#endif // IMAGE_FILTERS_H
