# Defines the cuda_linalg library, which provides CUDA-accelerated
# linear algebra routines, fundamental for ML and HPC tasks.

add_library(cuda_linalg STATIC
    src/matrix_ops.cu
)

# Expose the public include directory so other modules can use these routines.
target_include_directories(cuda_linalg PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# This library depends on the CUDA runtime. By linking against the
# CUDAToolkit::cudart target, we ensure it has the necessary headers and libraries.
target_link_libraries(cuda_linalg PUBLIC
    CUDAToolkit::cudart
)
