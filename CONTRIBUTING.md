# Contribuição para o TrackieLink

Primeiramente, obrigado por considerar contribuir para o TrackieLink! Estamos construindo um projeto ambicioso e toda ajuda é bem-vinda. Este documento fornece um conjunto de diretrizes para contribuir com o projeto.

## Código de Conduta

Este projeto e todos que participam dele são regidos pelo nosso [Código de Conduta](CODE_OF_CONDUCT.md). Ao participar, você concorda em seguir seus termos.

## Como Posso Contribuir?

### Reportando Bugs

Se você encontrar um bug, por favor, certifique-se de que ele ainda não foi reportado criando uma [Issue](https://github.com/seu-usuario/TrackieLink/issues). Ao abrir uma nova issue, por favor, use o template de "Bug Report" e inclua o máximo de detalhes possível.

### Sugerindo Melhorias

Se você tem uma ideia para uma nova funcionalidade ou uma melhoria para uma existente, crie uma [Issue](https://github.com/seu-usuario/TrackieLink/issues) usando o template de "Feature Request".

### Sua Primeira Contribuição de Código

Não sabe por onde começar? Você pode procurar por issues com a tag `good-first-issue`.

## Guia de Desenvolvimento

### Configurando o Ambiente

Para compilar e desenvolver o TrackieLink, você precisará de um ambiente de desenvolvimento com as seguintes ferramentas e bibliotecas:

-   **Ferramentas de Build:**
    -   CMake (versão 3.20+)
    -   Um compilador C++ moderno (GCC, Clang, MSVC)
    -   Toolchain do Rust (Cargo)
    -   (Opcional) NVIDIA CUDA Toolkit (para desenvolvimento de GPU)
    -   (Opcional) Xcode ou ferramentas de linha de comando (em macOS, para desenvolvimento Objective-C/Metal)

-   **Dependências de Bibliotecas:**
    -   libcurl
    -   ONNX Runtime
    -   PortAudio
    -   OpenCV (4.x)
    -   pthread (em sistemas POSIX)

Recomendamos instalar essas dependências usando o gerenciador de pacotes do seu sistema (e.g., `apt` no Ubuntu, `brew` no macOS, `vcpkg` no Windows).

### Processo de Build

1.  Clone o repositório.
2.  Crie um diretório de build: `mkdir build && cd build`
3.  Configure o projeto com o CMake: `cmake ..`
4.  Compile o projeto: `cmake --build .`

### Estilo de Código

-   **C/C++:** Siga o estilo [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) como referência geral.
-   **Rust:** Use `cargo fmt` para formatar o código automaticamente.
-   **CMake:** Mantenha o `CMakeLists.txt` limpo e bem comentado.

### Processo de Pull Request

1.  Faça um fork do repositório e crie seu branch a partir do `main`.
2.  Se você adicionou código que deve ser testado, adicione testes.
3.  Garanta que a suíte de testes (quando implementada) passe.
4.  Certifique-se de que seu código segue as diretrizes de estilo.
5.  Abra um Pull Request usando o template fornecido.

Obrigado por sua contribuição!
