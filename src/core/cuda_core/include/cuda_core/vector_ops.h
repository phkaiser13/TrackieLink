#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

/**
 * @brief Lauches the vector addition kernel on the CUDA device.
 */
int launch_vector_add_kernel(const float* a_d, const float* b_d, float* c_d, int size);

/**
 * @brief Lauches the vector subtraction kernel on the CUDA device.
 */
int launch_vector_sub_kernel(const float* a_d, const float* b_d, float* c_d, int size);

/**
 * @brief Lauches the vector multiplication kernel on the CUDA device.
 */
int launch_vector_mul_kernel(const float* a_d, const float* b_d, float* c_d, int size);


#endif // VECTOR_OPS_H
