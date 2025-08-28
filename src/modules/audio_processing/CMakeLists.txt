#
# Author: Pedro h. Garcia <phkaiser13>.
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

# --- Biblioteca de Processamento de Áudio ---
# Cria uma biblioteca estática que encapsula toda a interação
# com a biblioteca PortAudio (captura e reprodução de áudio)
add_library(audio_processing STATIC
        handler/audio_handler.cpp
)

# --- Inclusão de Cabeçalhos Públicos ---
# Incluímos o diretório local e também o cabeçalho da camada cpp_core,
# pois o AudioHandler pode ser exposto para uso no núcleo da aplicação.
target_include_directories(audio_processing PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/cpp_core/include
)

# --- Vinculação com PortAudio ---
# PortAudio::portaudio é o alvo moderno importado que deve ser
# fornecido por um `find_package(PortAudio)` bem-sucedido.
target_link_libraries(audio_processing PRIVATE
        PortAudio::portaudio
)
