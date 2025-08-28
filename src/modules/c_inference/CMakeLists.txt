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
        ${CMAKE_SOURCE_DIR}/src/shared/types
)

# --- Vinculando Dependência: ONNX Runtime ---
#
# Vinculamos ao alvo 'ONNXRuntime::onnxruntime' fornecido pelo find_package.
# Usamos 'PUBLIC' porque os headers expostos (inference.h) incluem tipos
# do ONNX Runtime (como OrtApi), então qualquer consumidor desta biblioteca
# também precisa conhecer os cabeçalhos do ONNX.
target_link_libraries(c_inference PUBLIC
        ONNXRuntime::onnxruntime
)
