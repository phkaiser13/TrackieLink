# Arquitetura do TrackieLink

Este documento descreve a arquitetura de alto nível do TrackieLink. O objetivo é fornecer uma visão geral de como os componentes do projeto estão organizados e como eles interagem.

## Filosofia de Design

O TrackieLink é construído sobre uma arquitetura poliglota, projetada para alavancar as forças de diferentes linguagens de programação para tarefas específicas, garantindo alto desempenho, segurança e manutenibilidade.

A filosofia central é a **separação de responsabilidades**:
-   **Desempenho Crítico e Baixo Nível:** C e CUDA são usados para tarefas que exigem controle manual de memória, otimizações de baixo nível e computação massivamente paralela na GPU.
-   **Segurança e Concorrência:** Rust é a escolha para componentes onde a segurança de memória e a prevenção de corridas de dados são primordiais.
-   **Abstração de Alto Nível e Orquestração:** C++ serve como a "cola" do projeto, orquestrando os diferentes módulos e implementando a lógica de negócios complexa.
-   **Integração com Plataforma Nativa:** Objective-C/Metal são usados para integração profunda com o ecossistema da Apple, garantindo a melhor experiência e desempenho em dispositivos macOS.

## Componentes Principais (`src/core`)

O coração do TrackieLink é composto por várias bibliotecas "core", cada uma com uma responsabilidade clara:

-   `c_core`: (A ser implementado) Uma base em C puro, fornecendo estruturas de dados fundamentais, utilitários e uma API C estável (ABI) para interoperabilidade entre todas as outras linguagens.

-   `cpp_core`: O orquestrador principal da aplicação. Escrito em C++, ele é responsável por inicializar o sistema, gerenciar o ciclo de vida dos módulos, e encadear as operações (e.g., capturar vídeo, passar para inferência, enviar para a API, etc.).

-   `rust_core`: Contém lógica crítica de segurança. Ele lida com tarefas como parsing de dados de fontes não confiáveis, gerenciamento de estado concorrente e outras operações onde a segurança de memória é vital. Ele se comunica com o C++ através de uma camada FFI (Foreign Function Interface).

-   `cuda_core`: (A ser implementado) Aceleração por GPU. Este módulo contém kernels CUDA para operações de computação intensiva, como pré-processamento de imagens, transformações de tensores e algoritmos de álgebra linear.

-   `objc_core`: (A ser implementado) Integração com o ecossistema Apple. Este módulo contém código Objective-C++ (`.mm`) que interage com APIs nativas do macOS, como AVFoundation para captura de vídeo eficiente ou Metal para renderização e computação acelerada por GPU em Apple Silicon.

## Módulos (`src/modules`)

Os módulos são bibliotecas de funcionalidades específicas que são utilizadas pelo `cpp_core`.

-   `c_inference`: Um wrapper em C puro para o ONNX Runtime, fornecendo uma API simples para carregar modelos e executar inferências.
-   `video_processing` / `audio_processing`: Módulos que lidam com a captura e o pré-processamento de I/O de mídia, usando OpenCV e PortAudio. Partes desses módulos serão descarregadas para o `cuda_core` para aceleração.
-   `vision_processing`: Implementa a lógica para processar os resultados da inferência (e.g., desenhar caixas delimitadoras, extrair rostos).
-   ... e outros.

## Comunicação Inter-Módulos

-   **C++ <-> Rust:** Através de FFI, usando `cxx` ou uma ponte C manual.
-   **C++ <-> C:** Invocação direta.
-   **C++ <-> CUDA:** O `cpp_core` invoca wrappers que lançam os kernels do `cuda_core`.
-   **C++ <-> Objective-C++:** O `cpp_core` pode chamar diretamente funções em arquivos `.mm` compilados, permitindo uma integração transparente em plataformas Apple.

Esta arquitetura modular nos permite desenvolver, testar e otimizar cada parte do sistema de forma independente, enquanto mantemos uma base de código coesa e robusta.
