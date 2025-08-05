#
# Author: Pedro h. Garcia <phkaiser13>.
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

add_library(vision_processing STATIC
        src/yolo_processor.cpp
        src/face_processor.cpp
        src/midas_processor.cpp
)

target_include_directories(vision_processing PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# A biblioteca de visão depende do OpenCV (renomeado para opencv_manual) e da nossa API de inferência em C.
target_link_libraries(vision_processing PUBLIC
        opencv_manual
        c_inference
)
