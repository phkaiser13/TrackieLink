#include "rocm_core/vector_ops.h"
#include <hip/hip_runtime.h>

// --- ROCm Kernels ---

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
    hipLaunchKernelGGL(vector_add_kernel, blocksPerGrid, threadsPerBlock, 0, 0, a_d, b_d, c_d, size);
    return hipGetLastError();
}

int launch_vector_sub_kernel(const float* a_d, const float* b_d, float* c_d, int size) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vector_sub_kernel, blocksPerGrid, threadsPerBlock, 0, 0, a_d, b_d, c_d, size);
    return hipGetLastError();
}

int launch_vector_mul_kernel(const float* a_d, const float* b_d, float* c_d, int size) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vector_mul_kernel, blocksPerGrid, threadsPerBlock, 0, 0, a_d, b_d, c_d, size);
    return hipGetLastError();
}
