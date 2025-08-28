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

O projeto utiliza o **Conan** para gerenciar as dependências C++ e o **CMake** como sistema de build. O processo foi simplificado com scripts para facilitar a compilação.

### Pré-requisitos

Você precisará ter as seguintes ferramentas instaladas e configuradas no seu `PATH`:

-   **CMake** (versão 3.20+)
-   Um **Compilador C++** moderno (GCC, Clang, ou MSVC)
-   O **Toolchain do Rust** (incluindo `cargo`)
-   **Conan** (versão 2.x): `pip install conan`

### Passos para Compilar

O processo de compilação é automatizado através de scripts. Eles irão instalar as dependências via Conan, baixar os modelos de IA necessários e compilar o projeto.

1.  **Clone o repositório:**
    ```bash
    git clone [URL_DO_REPOSITORIO]
    cd TrackieLink
    ```

2.  **Execute o script de build para a sua plataforma:**

    -   **Para Linux ou macOS:**
        ```bash
        ./Scripts/Build.sh
        ```

    -   **Para Windows:**
        ```cmd
        .\Scripts\Build.bat
        ```

3.  **Concluído!** O script cuidará de tudo. Ao final, o executável `TrackieLink` estará localizado no diretório de build:
    -   `build/src/core/cpp_core/TrackieLink` (Linux/macOS)
    -   `build\src\core\cpp_core\Release\TrackieLink.exe` (Windows)

## Uso

```bash
./TrackieLink [--mode <camera|screen>] [--show_preview]