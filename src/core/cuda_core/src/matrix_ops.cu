#include "cuda_core/matrix_ops.h"
#include <cuda_runtime.h>

// --- CUDA Kernel ---

__global__ void matrix_mul_kernel(const float* a, const float* b, float* c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; ++i) {
            sum += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = sum;
    }
}

// --- Host-side Launcher Function ---

int launch_matrix_mul_kernel(const float* a_d, const float* b_d, float* c_d, int width) {
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_mul_kernel<<<numBlocks, threadsPerBlock>>>(a_d, b_d, c_d, width);

    return cudaGetLastError();
}
