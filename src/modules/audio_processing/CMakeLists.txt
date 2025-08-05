#
# Author: Pedro h. Garcia <phkaiser13>.
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

# --- Definição da Biblioteca de Processamento de Áudio ---
#
# Criamos uma biblioteca estática para encapsular toda a interação
# com a biblioteca PortAudio, tanto para captura quanto para reprodução.
add_library(audio_processing STATIC
        handler/audio_handler.cpp
)

# --- Expondo os Cabeçalhos Públicos ---
#
# O cabeçalho 'audio_handler.hpp' definirá a interface da nossa classe
# AudioHandler, que será usada pelo 'cpp_core'.
target_include_directories(audio_processing PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/cpp_core/include
)

# --- Vinculando Dependências ---
#
# Nossa biblioteca é um wrapper em torno da PortAudio.
# O alvo 'portaudio_manual' foi definido manualmente no CMakeLists.txt raiz.
target_link_libraries(audio_processing PUBLIC
        portaudio_manual
)
