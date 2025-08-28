@echo off
setlocal

REM Criar diretórios se não existirem
if not exist "src\WorkTools\Models\Contextual" mkdir "src\WorkTools\Models\Contextual"
if not exist "build" mkdir "build"

echo Instalando dependencias com Conan...
REM Adicionamos as flags para o Conan instalar dependencias de sistema, se necessário (principalmente para Linux/WSL).
conan install . --output-folder=build --build=missing -c tools.system.package_manager:mode=install -c tools.system.package_manager:sudo=True
if %errorlevel% neq 0 (
    echo Erro durante a instalacao com Conan.
    exit /b %errorlevel%
)

echo Dependências instaladas.

echo Baixando modelos de IA...

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
echo Configurando o projeto com CMake...
cmake -S . -B build
if %errorlevel% neq 0 (
    echo Erro durante a configuracao com CMake.
    exit /b %errorlevel%
)

echo Compilando o projeto...
cmake --build build --config Release
if %errorlevel% neq 0 (
    echo Erro durante a compilacao.
    exit /b %errorlevel%
)

echo Processo finalizado.
echo O executavel esta em: build\src\core\cpp_core\Release\TrackieLink.exe
