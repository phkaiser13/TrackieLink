# TrackieLink

**TrackieLink** é a reescrita de alto desempenho e multiplataforma da aplicação de assistente de IA multimodal Trackie. Este projeto utiliza uma arquitetura poliglota para maximizar o desempenho, a segurança e a manutenibilidade, combinando o poder do C++, C e Rust.

## Visão Geral da Arquitetura

O projeto é dividido em módulos claros com responsabilidades bem definidas:

-   **Core (`cpp_core`)**: O coração da aplicação em C++, responsável por orquestrar todos os outros módulos, gerenciar threads e o ciclo de vida da aplicação.
-   **API Client (`gemini_rest_client`)**: Um cliente C++ para interagir com a API REST do Google Gemini.
-   **Inferência (`c_inference`)**: Um motor de inferência de alto desempenho escrito em C puro, utilizando o ONNX Runtime para executar modelos de IA (YOLO, Reconhecimento Facial, etc.).
-   **Segurança e Lógica (`rust_core`)**: Componentes críticos em termos de segurança de memória e concorrência, escritos em Rust.
-   **Processamento de I/O**: Módulos dedicados em C++ para lidar com áudio (`audio_processing` com PortAudio) e vídeo (`video_processing` com OpenCV).
-   **Configuração (`msgpack_loader`)**: Um loader eficiente para configurações e prompts do sistema, usando o formato binário MessagePack.

## Compilando o Projeto

### Pré-requisitos

Você precisará ter as seguintes ferramentas e bibliotecas de desenvolvimento instaladas:

-   CMake (versão 3.16+)
-   Um compilador C++ moderno (GCC, Clang, MSVC)
-   O toolchain do Rust (incluindo `cargo`)
-   **Bibliotecas de Desenvolvimento:**
    -   libcurl
    -   ONNX Runtime
    -   PortAudio
    -   OpenCV 4+
    -   msgpack-c
    -   nlohmann-json

### Passos para Compilar

1.  Clone o repositório:
    ```bash
    git clone [URL_DO_REPOSITORIO]
    cd TrackieLink
    ```

2.  Crie um diretório de build:
    ```bash
    mkdir build
    cd build
    ```

3.  Execute o CMake para configurar o projeto. Você pode precisar ajudar o CMake a encontrar as bibliotecas se elas não estiverem em um caminho padrão:
    ```bash
    cmake .. -DCMAKE_PREFIX_PATH="/caminho/para/onnxruntime;/caminho/para/outra/lib"
    ```

4.  Compile o projeto:
    ```bash
    cmake --build . --config Release
    ```

5.  O executável `TrackieLink` estará localizado no diretório `build/src/core/cpp_core/`.

## Uso

```bash
./TrackieLink [--mode <camera|screen>] [--show_preview]