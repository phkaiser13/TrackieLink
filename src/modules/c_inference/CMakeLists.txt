#
# Author: Pedro H. Garcia <phkaiser13>
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

# --- Definição da Biblioteca de Inferência em C ---
#
# Criamos uma biblioteca estática, escrita em C puro, para encapsular
# toda a interação com o ONNX Runtime.
add_library(c_inference STATIC
        Inf/inference.c
)

# --- Expondo os Cabeçalhos Públicos ---
#
# O diretório 'include' contém 'inference.h', que define a API pública
# da nossa biblioteca. Qualquer código que queira usar nosso motor de
# inferência (como o 'cpp_core') precisará deste cabeçalho.
target_include_directories(c_inference PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Vinculando Dependência: ONNX Runtime ---
#
# O alvo 'onnxruntime_manual' foi definido no CMakeLists.txt principal.
# Usamos 'PUBLIC' porque os headers expõem tipos do ONNX Runtime.
target_link_libraries(c_inference PUBLIC
        onnxruntime_manual
)
