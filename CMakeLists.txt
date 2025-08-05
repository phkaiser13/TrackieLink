cmake_minimum_required(VERSION 3.16)
project(TrackieLink VERSION 1.0 LANGUAGES C CXX)

# --- Padrões da Linguagem ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# --- Caminho para Dependências ---
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party)

# --- Definição de Dependências de Terceiros como Alvos INTERFACE ---
# Esta é a maneira moderna de gerenciar dependências locais.
# Criamos um alvo para cada biblioteca que encapsula suas propriedades (includes, libs).

# --- ONNX Runtime ---
add_library(onnxruntime INTERFACE) # <<-- CORREÇÃO: Nome do alvo simplificado
target_include_directories(onnxruntime SYSTEM INTERFACE ${THIRD_PARTY_DIR}/OnnxRuntimeWin/include)
target_link_libraries(onnxruntime INTERFACE ${THIRD_PARTY_DIR}/OnnxRuntimeWin/lib/onnxruntime.lib)
message(STATUS " - Configurado: ONNX Runtime (manual)")

# --- libcurl ---
add_library(CURL::libcurl INTERFACE IMPORTED) # <<-- CORREÇÃO: Usando nome de alvo padrão para compatibilidade
set_target_properties(CURL::libcurl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${THIRD_PARTY_DIR}/CurlWinLibrary/include"
        INTERFACE_LINK_LIBRARIES "${THIRD_PARTY_DIR}/CurlWinLibrary/lib/libcurl.a"
)
message(STATUS " - Configurado: libcurl (manual)")

# --- msgpack-c ---
add_library(msgpack-c::msgpack-c INTERFACE IMPORTED) # <<-- CORREÇÃO: Usando nome de alvo padrão
set_target_properties(msgpack-c::msgpack-c PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${THIRD_PARTY_DIR}/mspacklib/include"
)
message(STATUS " - Configurado: msgpack-c (manual)")

# --- PortAudio ---
add_library(PortAudio::portaudio INTERFACE IMPORTED) # <<-- CORREÇÃO: Usando nome de alvo padrão
set_target_properties(PortAudio::portaudio PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${THIRD_PARTY_DIR}/PortaudioWinLib/include"
        INTERFACE_LINK_LIBRARIES "${THIRD_PARTY_DIR}/PortaudioWinLib/lib/x64/ReleaseMinDependency/portaudio_x64.lib"
)
message(STATUS " - Configurado: PortAudio (manual)")

# --- OpenCV ---
add_library(OpenCV::OpenCV INTERFACE IMPORTED) # <<-- CORREÇÃO: Usando nome de alvo padrão
set_target_properties(OpenCV::OpenCV PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${THIRD_PARTY_DIR}/opencv-Win/include"
        INTERFACE_LINK_LIBRARIES "${THIRD_PARTY_DIR}/opencv-Win/x64/vc16/lib/opencv_world4120.lib"
)
message(STATUS " - Configurado: OpenCV (manual)")

# --- nlohmann-json ---
add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED) # <<-- CORREÇÃO: Usando nome de alvo padrão
set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${THIRD_PARTY_DIR}/jsonlib/include"
)
message(STATUS " - Configurado: nlohmann-json (manual)")

# --- Módulo Rust ---
message(STATUS "Configurando a compilação do módulo Rust (rust_core)...")
add_custom_target(
        build_rust_core ALL
        COMMAND cargo build --release
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/core/rust_core
        COMMENT "Compilando a biblioteca 'rust_core' com Cargo..."
        VERBATIM
)

add_library(rust_core_lib STATIC IMPORTED) # <<-- CORREÇÃO: GLOBAL não é necessário aqui
set_target_properties(rust_core_lib PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/core/rust_core/target/release/rust_core.lib
)
# A dependência é do alvo final no Rust, não da biblioteca importada
# add_dependencies(rust_core_lib build_rust_core) # <<-- CORREÇÃO: Esta linha é redundante e pode causar problemas

# --- Organização dos Submódulos ---
message(STATUS "Configurando os módulos C/C++...")

# A ordem é crucial para que as dependências sejam conhecidas antes de serem usadas.
# Nível 0: Utilitários e APIs de baixo nível
add_subdirectory(src/shared/utils)
add_subdirectory(src/api/gemini_rest_client)
add_subdirectory(src/modules/c_inference)
add_subdirectory(src/modules/msgpack_loader)

# Nível 1: Processadores de I/O e Visão
add_subdirectory(src/modules/audio_processing)
add_subdirectory(src/modules/video_processing)
add_subdirectory(src/modules/vision)

# Nível 2: Lógica de Negócios
add_subdirectory(src/modules/function_calling)

# Nível 3: Serviços de Longa Duração
add_subdirectory(src/services/gemini_service)

# Nível 4: O Orquestrador Principal
add_subdirectory(src/core/cpp_core)