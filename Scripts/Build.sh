#!/bin/bash
set -e

# Cria os diretórios caso não existam
mkdir -p src/WorkTools/Models/Contextual
mkdir -p build

echo "Instalando dependências com Conan..."
# Instala as dependências do conanfile.py, gerando os arquivos do CMake no diretório 'build'
# --build=missing: Compila pacotes da fonte se um binário pré-compilado não for encontrado.
# -c tools.system.package_manager:mode=install: Autoriza o Conan a instalar dependências de sistema (ex: apt, yum).
# -c tools.system.package_manager:sudo=True: Usa sudo para o gerenciador de pacotes do sistema.
conan install . --output-folder=build --build=missing -c tools.system.package_manager:mode=install -c tools.system.package_manager:sudo=True

echo "Dependências instaladas."

echo "Baixando modelos de IA..."

MODEL_URL_BASE="https://github.com/phkaiser13/TrackieAssets/releases/download/Base.LINK.1.0/"

declare -a models=(
  "midas_v21_small_256.onnx"
  "w600k_mbf.onnx"
  "yolov8-face.onnx"
  "yolov8n.onnx"
)

for model in "${models[@]}"; do
  echo "Baixando modelo $model..."
  curl -L -o "src/WorkTools/Models/Contextual/$model" "$MODEL_URL_BASE$model"
done

echo "Modelos baixados."

echo "Configurando o projeto com CMake..."
# Configura o projeto. O toolchain do Conan será encontrado automaticamente em 'build/conan_toolchain.cmake'
cmake -S . -B build

echo "Compilando o projeto..."

cmake --build build

echo "Processo finalizado."
echo "O executável está em: build/src/core/cpp_core/TrackieLink"
