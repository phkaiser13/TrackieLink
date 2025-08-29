# Defines the cuda_core library, which contains all low-level
# CUDA kernels and GPU management utilities.

add_library(cuda_core STATIC
    src/memory.cu
    src/image_kernels.cu
    src/image_filters.cu
    src/vector_ops.cu
    src/matrix_ops.cu
)

# Expose the public include directory. Any module that uses cuda_core
# will get this include path.
target_include_directories(cuda_core PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# This library needs to link against the CUDA libraries.
# The `CUDAToolkit` package found in the root CMakeLists provides the
# CUDAToolkit::cudart target, which is the CUDA Runtime library.
target_link_libraries(cuda_core PUBLIC
    CUDAToolkit::cudart
)
