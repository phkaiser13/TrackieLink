#include "cuda_core/memory.h"
#include <cuda_runtime.h>
#include <stdio.h>

// This file provides simple C-style wrappers around common CUDA memory operations.
// This helps to decouple the main application logic from direct CUDA API calls
// and provides a single place for error checking.

/**
 * @brief A helper macro to check for CUDA errors and print them.
 * This is a common pattern in CUDA programming. It makes error handling
 * much more concise.
 */
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; \
    } \
} while(0)

int cuda_malloc_device(void** devPtr, size_t size) {
    CUDA_CHECK(cudaMalloc(devPtr, size));
    return cudaSuccess;
}

int cuda_free_device(void* devPtr) {
    // cudaFree can accept a NULL pointer, so we don't need to check.
    CUDA_CHECK(cudaFree(devPtr));
    return cudaSuccess;
}

int cuda_memcpy_host_to_device(void* dst, const void* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    return cudaSuccess;
}

int cuda_memcpy_device_to_host(void* dst, const void* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    return cudaSuccess;
}
