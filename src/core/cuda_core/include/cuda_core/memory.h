#ifndef CUDA_CORE_MEMORY_H
#define CUDA_CORE_MEMORY_H

#include <stddef.h>

// The __cplusplus macro is defined by C++ compilers.
// Using extern "C" ensures that the function names are not mangled
// by the C++ compiler, so they can be easily called from C or C++ code.
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A simple wrapper to allocate memory on the GPU.
 * @param devPtr A pointer to a device pointer that will hold the address of the allocated memory.
 * @param size The size of the memory to allocate in bytes.
 * @return An error code (0 for success) corresponding to cudaError_t.
 */
int cuda_malloc_device(void** devPtr, size_t size);

/**
 * @brief A simple wrapper to free memory on the GPU.
 * @param devPtr The device pointer to free.
 * @return An error code (0 for success).
 */
int cuda_free_device(void* devPtr);

/**
 * @brief A simple wrapper to copy memory from the host (CPU) to the device (GPU).
 * @param dst Device destination memory address.
 * @param src Host source memory address.
 * @param count Size in bytes to copy.
 * @return An error code (0 for success).
 */
int cuda_memcpy_host_to_device(void* dst, const void* src, size_t count);

/**
 * @brief A simple wrapper to copy memory from the device (GPU) to the host (CPU).
 * @param dst Host destination memory address.
 * @param src Device source memory address.
 * @param count Size in bytes to copy.
 * @return An error code (0 for success).
 */
int cuda_memcpy_device_to_host(void* dst, const void* src, size_t count);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CORE_MEMORY_H
