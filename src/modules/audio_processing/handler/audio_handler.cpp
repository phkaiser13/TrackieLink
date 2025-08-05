/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "audio_handler.hpp"
#include <portaudio.h>
#include <stdexcept>
#include <iostream>

namespace trackie::audio {

// --- Macro Auxiliar para Tratamento de Erros da PortAudio ---
#define CHECK_PA_ERROR(err) \
    if (err != paNoError) { \
        throw std::runtime_error(std::string("PortAudio error: ") + Pa_GetErrorText(err)); \
    }

// --- Construtor e Destrutor ---

AudioHandler::AudioHandler() {
    _initialize();
}

AudioHandler::~AudioHandler() {
    _terminate();
}

// --- Métodos de Ciclo de Vida Interno ---

void AudioHandler::_initialize() {
    if (m_is_initialized.load()) return;
    std::cout << "[Audio] Inicializando PortAudio..." << std::endl;
    CHECK_PA_ERROR(Pa_Initialize());
    m_is_initialized.store(true);
}

void AudioHandler::_terminate() {
    if (!m_is_initialized.load()) return;
    std::cout << "[Audio] Finalizando PortAudio..." << std::endl;
    stopInputStream();
    stopOutputStream();
    CHECK_PA_ERROR(Pa_Terminate());
    m_is_initialized.store(false);
}

// --- Implementação do Stream de Entrada (Captura) ---

void AudioHandler::startInputStream(const AudioInputCallback& callback) {
    if (m_input_stream) {
        std::cout << "[Audio] Stream de entrada já está ativo." << std::endl;
        return;
    }

    m_input_callback = callback;

    PaStreamParameters inputParameters;
    inputParameters.device = Pa_GetDefaultInputDevice();
    if (inputParameters.device == paNoDevice) {
        throw std::runtime_error("Erro: Nenhum dispositivo de entrada de áudio padrão encontrado.");
    }
    inputParameters.channelCount = AUDIO_CHANNELS;
    inputParameters.sampleFormat = paFloat32; // Usamos float32 para processamento mais fácil
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    std::cout << "[Audio] Abrindo stream de entrada a " << AUDIO_SEND_SAMPLE_RATE << " Hz..." << std::endl;

    CHECK_PA_ERROR(Pa_OpenStream(
        &m_input_stream,
        &inputParameters,
        NULL, // Sem saída
        AUDIO_SEND_SAMPLE_RATE,
        AUDIO_CHUNK_SIZE,
        paClipOff, // Sem clipping
        &AudioHandler::paInputStreamCallback,
        this // Passa o ponteiro 'this' como userData
    ));

    CHECK_PA_ERROR(Pa_StartStream(m_input_stream));
    std::cout << "[Audio] Escutando microfone..." << std::endl;
}

void AudioHandler::stopInputStream() {
    if (m_input_stream) {
        std::cout << "[Audio] Parando stream de entrada..." << std::endl;
        CHECK_PA_ERROR(Pa_StopStream(m_input_stream));
        CHECK_PA_ERROR(Pa_CloseStream(m_input_stream));
        m_input_stream = nullptr;
        m_input_callback = nullptr; // Limpa o callback
    }
}

// --- Implementação do Stream de Saída (Reprodução) ---

void AudioHandler::startOutputStream() {
    if (m_output_stream) {
        std::cout << "[Audio] Stream de saída já está ativo." << std::endl;
        return;
    }

    PaStreamParameters outputParameters;
    outputParameters.device = Pa_GetDefaultOutputDevice();
    if (outputParameters.device == paNoDevice) {
        throw std::runtime_error("Erro: Nenhum dispositivo de saída de áudio padrão encontrado.");
    }
    outputParameters.channelCount = AUDIO_CHANNELS;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;

    std::cout << "[Audio] Abrindo stream de saída a " << AUDIO_RECEIVE_SAMPLE_RATE << " Hz..." << std::endl;

    CHECK_PA_ERROR(Pa_OpenStream(
        &m_output_stream,
        NULL, // Sem entrada
        &outputParameters,
        AUDIO_RECEIVE_SAMPLE_RATE,
        paFramesPerBufferUnspecified,
        paClipOff,
        NULL, // Usaremos escrita bloqueante, sem callback
        NULL
    ));

    CHECK_PA_ERROR(Pa_StartStream(m_output_stream));
    std::cout << "[Audio] Player de áudio pronto." << std::endl;
}

void AudioHandler::stopOutputStream() {
    if (m_output_stream) {
        std::cout << "[Audio] Parando stream de saída..." << std::endl;
        CHECK_PA_ERROR(Pa_StopStream(m_output_stream));
        CHECK_PA_ERROR(Pa_CloseStream(m_output_stream));
        m_output_stream = nullptr;
    }
}

void AudioHandler::playAudioChunk(const std::vector<float>& audioData) {
    if (!m_output_stream) {
        // std::cerr << "[Audio] Aviso: Tentativa de tocar áudio sem um stream de saída ativo." << std::endl;
        return;
    }
    CHECK_PA_ERROR(Pa_WriteStream(m_output_stream, audioData.data(), audioData.size()));
}


// --- Implementação do Callback C-style ---

int AudioHandler::paInputStreamCallback(
    const void* inputBuffer, void* outputBuffer,
    unsigned long framesPerBuffer,
    const void* timeInfo,
    unsigned long statusFlags,
    void* userData
) {
    // Converte o ponteiro userData de volta para um ponteiro da nossa classe.
    AudioHandler* handler = static_cast<AudioHandler*>(userData);

    if (handler && handler->m_input_callback) {
        // Invoca o callback do usuário com os dados de áudio.
        handler->m_input_callback(
            static_cast<const float*>(inputBuffer),
            framesPerBuffer
        );
    }

    // Sinaliza para a PortAudio continuar o processamento.
    return paContinue;
}

} // namespace trackie::audio