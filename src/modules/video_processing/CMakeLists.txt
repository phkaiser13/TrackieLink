#
# Author: Pedro h. Garcia <phkaiser13>.
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

add_library(video_processing STATIC
        handler/video_handler.cpp
)

target_include_directories(video_processing PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Vincula aos componentes específicos do OpenCV necessários para processamento de vídeo.
# Estes alvos são fornecidos pelo find_package(OpenCV) no CMakeLists.txt raiz.
target_link_libraries(video_processing PUBLIC
        OpenCV::core
        OpenCV::imgproc
        OpenCV::highgui
        OpenCV::videoio
)
