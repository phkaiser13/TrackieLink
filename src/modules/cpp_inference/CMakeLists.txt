cmake_minimum_required(VERSION 3.20)

add_library(cpp_inference STATIC
    inference_engine.cpp
)

target_include_directories(cpp_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against ONNX Runtime and other necessary libraries
target_link_libraries(cpp_inference PRIVATE
    ONNXRuntime::onnxruntime
    # We need vision_types.h, which is part of the vision module's interface
    vision_module
)

# The vision module needs to be a target.
# Let's assume it's defined in src/modules/vision/CMakeLists.txt
# and provides an interface library `vision_module`.
# If not, I will have to adjust this.
# For now, this is a reasonable assumption.
