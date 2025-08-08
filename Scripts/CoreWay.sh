
#!/bin/bash
set -e

# Cria os diretórios caso não existam
mkdir -p src/third_party
mkdir -p src/WorkTools/Models/Contextual

echo "Baixando bibliotecas..."

LIB_URL_BASE="https://github.com/phkaiser13/TrackieAssets/releases/download/Lib.LINK.1.0/"

declare -a libs=(
  "CurlWayLibrary.zip"
  "OnnxRuntimeWay.zip"
  "opencv-Way.zip"
  "PortaudioWay.zip"
)

for lib in "${libs[@]}"; do
  echo "Baixando $lib..."
  curl -L -o "src/third_party/$lib" "$LIB_URL_BASE$lib"

  echo "Extraindo $lib..."
  unzip -o "src/third_party/$lib" -d src/third_party/

  echo "Removendo arquivo zip $lib..."
  rm "src/third_party/$lib"
done

echo "Bibliotecas baixadas e extraídas."

echo "Baixando modelos..."

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

echo "Rodando cmake para buildar projeto..."

cmake -S . -B build

echo "Buildando o projeto..."

cmake --build build

echo "Processo finalizado."
