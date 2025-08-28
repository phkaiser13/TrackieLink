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
)

# --- Vinculando Dependências ---
# Vinculamos ao alvo moderno `CURL::CURL` fornecido pelo `find_package`.
target_link_libraries(gemini_rest_client PUBLIC
        CURL::CURL
        nlohmann_json
        trackie_utils
)
