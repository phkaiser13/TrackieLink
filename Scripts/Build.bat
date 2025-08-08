@echo off
setlocal enabledelayedexpansion

REM Criar diretórios se não existirem
if not exist src\third_party mkdir src\third_party
if not exist src\WorkTools\Models\Contextual mkdir src\WorkTools\Models\Contextual

REM Ferramenta para descompactar (usar powershell Expand-Archive)
set POWERSHELL=PowerShell -Command "Expand-Archive -Force"

echo Baixando bibliotecas...

REM Baixar e extrair bibliotecas
set LIB_URL_BASE=https://github.com/phkaiser13/TrackieAssets/releases/download/Lib.LINK.1.0/

for %%f in (
    CurlWinLibrary.zip
    OnnxRuntimeWin.zip
    opencv-Win.zip
    PortaudioWinLib.zip
) do (
    echo Baixando %%f...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%LIB_URL_BASE%%%f', 'src\third_party\%%f')"
    echo Extraindo %%f...
    %POWERSHELL% "src\third_party\%%f" "src\third_party"
    echo Deletando zip %%f...
    del src\third_party\%%f
)

echo Bibliotecas baixadas e extraídas.

echo Baixando modelos...

set MODEL_URL_BASE=https://github.com/phkaiser13/TrackieAssets/releases/download/Base.LINK.1.0/

for %%m in (
    midas_v21_small_256.onnx
    w600k_mbf.onnx
    yolov8-face.onnx
    yolov8n.onnx
) do (
    echo Baixando modelo %%m...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%MODEL_URL_BASE%%%m', 'src\WorkTools\Models\Contextual\%%m')"
)

echo Modelos baixados.

REM Build com CMake
echo Rodando cmake para buildar projeto...
cmake -S . -B build
cmake --build build
REM Opcional: buildar com cmake --build
echo Buildando o projeto...
cmake --build build

echo Processo finalizado.
pause
