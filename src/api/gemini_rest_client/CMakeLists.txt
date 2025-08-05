#
# Author: Pedro h. Garcia <phkaiser13>.
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

# --- Definição da Biblioteca do Cliente REST ---
add_library(gemini_rest_client STATIC
        apic/gemini_client.cpp
)

# --- Expondo os Cabeçalhos Públicos ---
target_include_directories(gemini_rest_client PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${NLOHMANN_JSON_INCLUDE_DIR}
)

# --- Vinculando Dependências ---
# Substituímos CURL::libcurl por libcurl_manual, conforme definido no CMakeLists.txt raiz
target_link_libraries(gemini_rest_client PUBLIC
        libcurl_manual
        trackie_utils
)
