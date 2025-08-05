#
# Author: Pedro h. Garcia <phkaiser13>.
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

add_library(trackie_utils STATIC
        base64.cpp
)

target_include_directories(trackie_utils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)