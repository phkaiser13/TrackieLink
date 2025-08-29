cmake_minimum_required(VERSION 3.20)

# 1. Definição do Projeto e Linguagens
# Habilitamos C, CXX, e opcionalmente CUDA e OBJCXX (Objective-C++) para máxima flexibilidade.
project(TrackieLink VERSION 1.0 LANGUAGES C CXX)

# Em plataformas Apple, habilitamos o Objective-C++ para integração nativa.
if(APPLE)
    message(STATUS "Plataforma Apple detectada. Habilitando Objective-C++.")
    enable_language(OBJCXX)
endif()

# 2. Padrões da Linguagem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Para Apple, definimos o padrão do Objective-C++ se a linguagem estiver habilitada
if(CMAKE_OBJCXX_COMPILER)
    message(STATUS "Compilador Objective-C++ detectado. Configurando padrão C++17 para OBJCXX.")
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

# 3. Gestão de Dependências com Conan
# O toolchain do Conan é injetado via linha de comando (CMAKE_TOOLCHAIN_FILE)
# ou incluído aqui para conveniência do desenvolvedor.
if(EXISTS "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
    include("${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
    message(STATUS "Toolchain do Conan encontrado e carregado.")
endif()

message(STATUS "Procurando dependências via Conan (CMakeDeps)...")
# O Conan gera os arquivos 'Find<Lib>.cmake' necessários para o CMake.
find_package(CURL REQUIRED)
find_package(ONNXRuntime REQUIRED)
find_package(OpenCV REQUIRED) # Conan lida com os componentes internamente
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED) # Threads ainda é melhor gerenciado pelo CMake

message(STATUS "Dependências externas encontradas com sucesso.")

# 3.5. Gestão de Dependências via FetchContent (para PortAudio)
include(FetchContent)
message(STATUS "Configurando PortAudio via FetchContent...")

FetchContent_Declare(
    PortAudio
    GIT_REPOSITORY https://git.assembla.com/portaudio.git
    GIT_TAG v19.7.0 # Use a stable tag
)

FetchContent_MakeAvailable(PortAudio)

# O alvo `portaudio_static` é criado pelo build do PortAudio.
# Crie um alias para facilitar o uso.
add_library(portaudio ALIAS portaudio_static)

message(STATUS "PortAudio configurado com sucesso.")

# Para as bibliotecas "header-only" incluídas no repositório,
# criamos alvos INTERFACE para fácil utilização nos submódulos.
message(STATUS "Configurando dependências internas (header-only)...")

# nlohmann-json
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/jsonlib/include
)
# msgpack-c
add_library(msgpackc INTERFACE)
target_include_directories(msgpackc INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/mspacklib/include
)

message(STATUS " - nlohmann_json: Configurado como alvo INTERFACE")
message(STATUS " - msgpack-c: Configurado como alvo INTERFACE")

# 4. Suporte Opcional para CUDA
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit encontrado. Habilitando a linguagem CUDA.")
    enable_language(CUDA)
    add_definitions(-DWITH_CUDA)
else()
    message(STATUS "CUDA Toolkit não encontrado. Funcionalidades de GPU serão desabilitadas.")
endif()

# Suporte Opcional para ROCm
find_package(ROCM QUIET)
if(ROCM_FOUND)
    message(STATUS "ROCm found. Enabling HIP language.")
    enable_language(HIP)
    add_definitions(-DWITH_ROCM)
else()
    message(STATUS "ROCm not found. ROCm-based GPU features will be disabled.")
endif()

# 5. Compilação do Módulo Rust (rust_core)
message(STATUS "Configurando a compilação do módulo Rust (rust_core)...")

# Define o caminho de saída para a biblioteca Rust para ser consistente.
set(RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/rust_target)
set(RUST_CARGO_ARGS --release --target-dir ${RUST_TARGET_DIR})

add_custom_target(
    build_rust_core ALL
    COMMAND cargo build ${RUST_CARGO_ARGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/core/rust_core
    COMMENT "Compilando a biblioteca 'rust_core' com Cargo..."
    VERBATIM
)

# A biblioteca Rust é importada como um alvo para que outros possam depender dela.
# O nome do arquivo .a ou .lib pode variar entre plataformas.
if(WIN32)
    set(RUST_LIB_PATH ${RUST_TARGET_DIR}/release/rust_core.lib)
else()
    set(RUST_LIB_PATH ${RUST_TARGET_DIR}/release/librust_core.a)
endif()

add_library(rust_core_lib STATIC IMPORTED)
set_target_properties(rust_core_lib PROPERTIES
    IMPORTED_LOCATION "${RUST_LIB_PATH}"
)
add_dependencies(rust_core_lib build_rust_core)

# 6. Organização dos Submódulos e Alvos do Projeto
message(STATUS "Configurando os módulos do projeto...")

# Cores (Nível 0 - Fundações)
add_subdirectory(src/core/c_core)
if(CUDAToolkit_FOUND)
    add_subdirectory(src/core/cuda_core)
endif()
if(ROCM_FOUND)
    add_subdirectory(src/core/rocm_core)
endif()
if(APPLE)
    # These modules will only be configured and built on Apple platforms.
    add_subdirectory(src/core/objc_core)
    add_subdirectory(src/modules/metal_renderer)
endif()
# O Rust já é compilado acima.

# --- Núcleo de IA Abstrato (Nível 0.5) ---
# O ai_core é uma biblioteca de interface (apenas cabeçalhos) para que
# todos os outros componentes possam usar a interface AIProvider.
add_library(ai_core INTERFACE)
target_include_directories(ai_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Utilitários e Provedores (Nível 1)
add_subdirectory(src/shared/utils)
add_subdirectory(src/providers/gemini_provider)
add_subdirectory(src/providers/openai_provider)
add_subdirectory(src/providers/anthropic_provider)
add_subdirectory(src/providers/meta_provider)
if(CUDAToolkit_FOUND)
    add_subdirectory(src/modules/cuda_linalg)
endif()
# add_subdirectory(src/modules/c_inference) # Deprecated in favor of cpp_inference
add_subdirectory(src/modules/cpp_inference) # New C++ inference engine
add_subdirectory(src/modules/msgpack_loader)

# Módulos de Processamento (Nível 2)
add_subdirectory(src/modules/audio_processing)
add_subdirectory(src/modules/video_processing)
add_subdirectory(src/modules/vision)

# Lógica de Negócios e Serviços (Nível 3)
add_subdirectory(src/modules/function_calling)
add_subdirectory(src/Services/gemini_service)

# Orquestrador Principal (Nível 4)
add_subdirectory(src/core/cpp_core)

# 7. Habilitação de Testes com CTest
# O GTest é encontrado aqui, mas só será usado nos submódulos que definem testes.
find_package(GTest REQUIRED)
enable_testing()
include(CTest)

message(STATUS "Configuração do CMake concluída.")