#
# Author: Pedro H. Garcia <phkaiser13>.
# Licensed under the vyAI Social Commons License 1.0
# See the LICENSE file in the project root.
#
# You are free to use, modify, and share this file under the terms of the license,
# provided proper attribution and open distribution are maintained.
#

# --- Definição da Biblioteca do Loader MessagePack ---
#
# Criamos uma biblioteca estática para encapsular a lógica de carregamento
# e parsing dos nossos arquivos de configuração .msgpack.
add_library(msgpack_loader STATIC
        pkcore/packloader.cpp
)

# --- Expondo os Cabeçalhos Públicos ---
#
# Assim como no módulo gemini_rest_client, usamos 'PUBLIC' para que
# qualquer alvo que se vincule a 'msgpack_loader' (como o 'cpp_core')
# tenha acesso automático ao diretório 'include'.
target_include_directories(msgpack_loader PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${NLOHMANN_JSON_INCLUDE_DIR}
)

# --- Vinculando Dependências ---
#
# Esta biblioteca tem duas dependências principais:
# 1. msgpack-c: A biblioteca C++ para manipulação de dados MessagePack.
# 2. nlohmann_json: Para conversão dos dados para JSON estruturado.
#
# Nota: Esses alvos devem ter sido definidos no CMakeLists.txt raiz com os nomes:
# msgpack_manual e nlohmann_manual.
target_link_libraries(msgpack_loader PUBLIC
        msgpack_manual
        nlohmann_manual
)
