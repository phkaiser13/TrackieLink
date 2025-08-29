# Defines the rocm_core library.
# This module is only configured and built if ROCm is found.

find_package(ROCM)

if(ROCM_FOUND)
    add_library(rocm_core STATIC
        src/vector_ops.hip
        src/image_filters.hip
    )

    target_include_directories(rocm_core PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    target_link_libraries(rocm_core PUBLIC ROCM::hip_runtime)
endif()
