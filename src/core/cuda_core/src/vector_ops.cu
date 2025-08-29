#include "cuda_core/vector_ops.h"
#include <cuda_runtime.h>

// --- CUDA Kernels ---

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_sub_kernel(const float* a, const float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] - b[i];
    }
}

__global__ void vector_mul_kernel(const float* a, const float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] * b[i];
    }
}


// --- Host-side Launcher Functions ---

int launch_vector_add_kernel(const float* a_d, const float* b_d, float* c_d, int size) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, size);
    return cudaGetLastError();
}

int launch_vector_sub_kernel(const float* a_d, const float* b_d, float* c_d, int size) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vector_sub_kernel<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, size);
    return cudaGetLastError();
}

int launch_vector_mul_kernel(const float* a_d, const float* b_d, float* c_d, int size) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vector_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, size);
    return cudaGetLastError();
}
