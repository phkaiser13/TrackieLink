/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include <functional>
#include <vector>
#include <string>
#include <atomic>

// <<<<<<< CORREÇÃO: Incluir o cabeçalho real da PortAudio >>>>>>>
// Isso nos dá as definições corretas para PaStream, PaStreamCallbackTimeInfo, etc.
#include <portaudio.h>

// <<<<<<< CORREÇÃO: Remover a declaração "forward" incorreta >>>>>>>
// struct PaStream; // Esta linha causava o erro de redefinição.

namespace trackie::audio {

// --- Constantes de Áudio (espelhando app_config.py) ---
constexpr int AUDIO_SEND_SAMPLE_RATE = 16000;
constexpr int AUDIO_RECEIVE_SAMPLE_RATE = 24000;
constexpr int AUDIO_CHANNELS = 1;
constexpr int AUDIO_CHUNK_SIZE = 1024; // Frames por buffer

/**
 * @brief Callback para receber dados de áudio capturados do microfone.
 * @param inputBuffer Ponteiro para os dados de áudio (intercalados).
 * @param frameCount O número de frames no buffer.
 */
using AudioInputCallback = std::function<void(const float* inputBuffer, unsigned long frameCount)>;

class AudioHandler {
public:
    AudioHandler();
    ~AudioHandler();

    // Desabilita cópia e atribuição.
    AudioHandler(const AudioHandler&) = delete;
    AudioHandler& operator=(const AudioHandler&) = delete;

    /**
     * @brief Inicia o stream de captura de áudio do microfone.
     *
     * Abre o dispositivo de entrada padrão e começa a chamar o callback
     * fornecido com novos dados de áudio assim que eles estiverem disponíveis.
     *
     * @param callback A função a ser chamada com os dados do microfone.
     */
    void startInputStream(const AudioInputCallback& callback);

    /**
     * @brief Para o stream de captura de áudio.
     */
    void stopInputStream();

    /**
     * @brief Inicia o stream de reprodução de áudio.
     *
     * Abre o dispositivo de saída padrão. Retorna imediatamente.
     * Use 'playAudioChunk' para enfileirar dados para reprodução.
     */
    void startOutputStream();

    /**
     * @brief Para o stream de reprodução de áudio.
     */
    void stopOutputStream();

    /**
     * @brief Envia um pedaço de dados de áudio para ser reproduzido.
     *
     * Esta função é bloqueante se o buffer interno do stream de saída
     * estiver cheio.
     *
     * @param audioData Um vetor de floats representando o áudio a ser tocado.
     */
    void playAudioChunk(const std::vector<float>& audioData);

private:
    // --- Métodos de Callback C-style para PortAudio ---
    // <<<<<<< CORREÇÃO: Assinatura do método agora corresponde exatamente ao typedef PaStreamCallback >>>>>>>
    static int paInputStreamCallback(
        const void* inputBuffer,
        void* outputBuffer,
        unsigned long framesPerBuffer,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void* userData
    );

    // --- Estado Interno ---
    void _initialize();
    void _terminate();

    std::atomic<bool> m_is_initialized{false};

    // Ponteiros para os streams da PortAudio
    PaStream* m_input_stream = nullptr;
    PaStream* m_output_stream = nullptr;

    // Callback do usuário para os dados de entrada
    AudioInputCallback m_input_callback;
};

} // namespace trackie::audio