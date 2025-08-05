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
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# --- ONNX Runtime ---
add_library(onnxruntime_manual INTERFACE)
target_include_directories(onnxruntime_manual SYSTEM INTERFACE ${THIRD_PARTY_DIR}/onnxruntime-win-x64-1.18.0/include)
target_link_libraries(onnxruntime_manual INTERFACE ${THIRD_PARTY_DIR}/onnxruntime-win-x64-1.18.0/lib/onnxruntime.lib)
message(STATUS " - Configurado: ONNX Runtime (manual)")

# --- libcurl ---
add_library(libcurl_manual INTERFACE)
target_include_directories(libcurl_manual SYSTEM INTERFACE ${THIRD_PARTY_DIR}/curl-8.9.0_1-win64-mingw/include)
target_link_libraries(libcurl_manual INTERFACE ${THIRD_PARTY_DIR}/curl-8.15.0_4-win64-mingw/lib/libcurl.a)
message(STATUS " - Configurado: libcurl (manual)")

# --- msgpack-c ---
add_library(msgpack_manual INTERFACE)
target_include_directories(msgpack_manual SYSTEM INTERFACE ${THIRD_PARTY_DIR}/msgpack-c-cpp-6.1.0/include)
message(STATUS " - Configurado: msgpack-c (manual)")

# --- PortAudio ---
add_library(portaudio_manual INTERFACE)
target_include_directories(portaudio_manual SYSTEM INTERFACE ${THIRD_PARTY_DIR}/portaudio/include)
target_link_libraries(portaudio_manual INTERFACE ${THIRD_PARTY_DIR}/portaudio/lib/portaudio_x64.lib)
message(STATUS " - Configurado: PortAudio (manual)")

# --- OpenCV ---
add_library(opencv_manual INTERFACE)
target_include_directories(opencv_manual SYSTEM INTERFACE ${THIRD_PARTY_DIR}/opencv-4.10.0-windows/include)
target_link_libraries(opencv_manual INTERFACE
        ${THIRD_PARTY_DIR}/opencv-4.10.0-windows/x64/vc16/lib/opencv_world4100.lib
)
message(STATUS " - Configurado: OpenCV (manual)")

# --- nlohmann-json ---
add_library(nlohmann_manual INTERFACE)
target_include_directories(nlohmann_manual SYSTEM INTERFACE ${THIRD_PARTY_DIR}/nlohmann-json-3.11.3/include)
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

add_library(rust_core_lib STATIC IMPORTED GLOBAL)
set_target_properties(rust_core_lib PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/core/rust_core/target/release/rust_core.lib
)
add_dependencies(rust_core_lib build_rust_core)

# --- Organização dos Submódulos ---
message(STATUS "Configurando os módulos C/C++...")

# Nível 0
add_subdirectory(src/api/gemini_rest_client)
add_subdirectory(src/modules/c_inference)
add_subdirectory(src/modules/msgpack_loader)

# Nível 1
add_subdirectory(src/modules/audio_processing)
add_subdirectory(src/modules/video_processing)
add_subdirectory(src/modules/vision)

# Nível 2
add_subdirectory(src/modules/function_calling)

# Nível 3
add_subdirectory(src/services/gemini_service)

# Nível 4
add_subdirectory(src/core/cpp_core)
