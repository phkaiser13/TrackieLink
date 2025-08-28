#ifndef CUDA_LINALG_MATRIX_OPS_H
#define CUDA_LINALG_MATRIX_OPS_H

// This header defines the C-style interface for launching CUDA kernels
// related to linear algebra operations.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs matrix-vector multiplication on the GPU (y = A * x).
 *
 * This function launches a CUDA kernel where each thread is responsible for
 * calculating one element of the output vector y.
 * The matrix A is assumed to be stored in row-major order.
 *
 * @param d_matrix A device pointer to the matrix A (float*).
 * @param d_vector_in A device pointer to the input vector x (float*).
 * @param d_vector_out A device pointer to the output vector y (float*).
 * @param num_rows The number of rows in matrix A.
 * @param num_cols The number of columns in matrix A (must equal the size of vector x).
 * @return An error code (0 for success) corresponding to cudaError_t.
 */
int launch_matrix_vector_mult_kernel(const float* d_matrix, const float* d_vector_in, float* d_vector_out, int num_rows, int num_cols);


#ifdef __cplusplus
}
#endif

#endif // CUDA_LINALG_MATRIX_OPS_H
