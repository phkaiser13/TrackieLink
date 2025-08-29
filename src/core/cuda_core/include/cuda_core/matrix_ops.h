#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

/**
 * @brief Lauches the matrix multiplication kernel on the CUDA device.
 */
int launch_matrix_mul_kernel(const float* a_d, const float* b_d, float* c_d, int width);

#endif // MATRIX_OPS_H
