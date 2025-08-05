#
# Author: Pedro H. Garcia <phkaiser13>
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

# Cria a biblioteca estática 'function_handler_lib'
add_library(function_handler_lib STATIC
        src/function_handler.cpp
)

# Diretórios de inclusão para a biblioteca
target_include_directories(function_handler_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Linka com a biblioteca de visão (vision_processing)
# Supondo que ela já tenha sido adicionada via add_subdirectory em um nível acima
target_link_libraries(function_handler_lib PUBLIC
        vision_processing
)
