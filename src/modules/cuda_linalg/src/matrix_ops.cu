#include "cuda_linalg/matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * @brief CUDA kernel for matrix-vector multiplication (y = A * x).
 *
 * This kernel is designed so that each thread in the grid is responsible for
 * computing a single element of the output vector `y`. This is a common and
 * intuitive way to parallelize this operation.
 *
 * @param A The input matrix (row-major layout).
 * @param x The input vector.
 * @param y The output vector.
 * @param num_rows The number of rows in matrix A.
 * @param num_cols The number of columns in matrix A.
 */
__global__ void matrix_vector_mult_kernel(const float* A, const float* x, float* y, int num_rows, int num_cols) {
    // Calculate the global thread ID. In this 1D grid, this directly
    // corresponds to the row of the matrix and the element of the output vector.
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to ensure we don't read or write out of allocated memory.
    if (row < num_rows) {
        float sum = 0.0f;

        // Each thread computes the dot product of its assigned row with the input vector x.
        for (int col = 0; col < num_cols; ++col) {
            sum += A[row * num_cols + col] * x[col];
        }

        // Write the final result to the output vector.
        y[row] = sum;
    }
}


int launch_matrix_vector_mult_kernel(const float* d_matrix, const float* d_vector_in, float* d_vector_out, int num_rows, int num_cols) {
    // Basic validation of input pointers.
    if (!d_matrix || !d_vector_in || !d_vector_out) {
        return cudaErrorInvalidValue;
    }

    // We use a 1D grid of threads. Each thread will process one row.
    // 256 is a common block size, balancing parallelism and resource usage.
    const int threadsPerBlock = 256;

    // Calculate the number of blocks needed to cover all rows.
    const int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel.
    matrix_vector_mult_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_vector_in, d_vector_out, num_rows, num_cols);

    // Check for and return any errors from the asynchronous kernel launch.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error in %s: %s\n", __func__, cudaGetErrorString(err));
    }
    return err;
}
