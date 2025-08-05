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
# Ele depende do cliente REST, do handler de funções e do processador de áudio.
target_link_libraries(gemini_service_lib PUBLIC
        gemini_rest_client        # Certifique-se que este target existe e está correto
        function_handler_lib      # Certifique-se que este target existe e está correto
        audio_processing          # Certifique-se que este target existe e está correto
)
