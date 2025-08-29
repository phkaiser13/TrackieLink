#
# Author: Pedro h. Garcia <phkaiser13>.
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

add_library(gemini_service_lib STATIC
        src/gemini_service.cpp
)

target_include_directories(gemini_service_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# O serviço Gemini é um dos componentes de mais alto nível.
# Ele depende da interface AI e de uma implementação (provedor),
# além de outros handlers.
target_link_libraries(gemini_service_lib PUBLIC
        ai_core                   # A interface abstrata para todos os provedores
        gemini_provider           # A implementação específica do Gemini
        function_handler_lib
        audio_processing
)
