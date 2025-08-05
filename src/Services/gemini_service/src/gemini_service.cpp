/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "gemini_service.hpp"
#include "base64.h" // Supondo que base64.h foi adicionado

namespace trackie::services {

GeminiService::GeminiService(
    std::shared_ptr<api::GeminiClient> client,
    std::shared_ptr<functions::FunctionHandler> funcs,
    std::shared_ptr<audio::AudioHandler> audio)
    : m_client(client), m_func_handler(funcs), m_audio_handler(audio) {}

GeminiService::~GeminiService() {
    stop();
}

void GeminiService::start(const nlohmann::json& config, const std::string& system_prompt) {
    m_config = config;
    m_system_prompt = system_prompt;
    m_stop_flag.store(false);
    m_thread = std::thread(&GeminiService::_communicationLoop, this);
}

void GeminiService::stop() {
    m_stop_flag.store(true);
    m_cv.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void GeminiService::pushUserTextMessage(const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_text_queue.push(message);
    }
    m_cv.notify_one();
}

void GeminiService::pushAudioBuffer(const std::vector<uint8_t>& audio_data) {
    {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_audio_queue.push(audio_data);
    }
    m_cv.notify_one();
}

void GeminiService::pushVideoFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_frame_mutex);
    m_latest_frame = frame.clone();
}

void GeminiService::_communicationLoop() {
    // ... Lógica de comunicação como no application.cpp anterior, mas agora usando os membros desta classe ...
    // Exemplo:
    // 1. Esperar na m_cv
    // 2. Pegar texto/áudio das filas m_text_queue, m_audio_queue
    // 3. Construir a requisição
    // 4. Chamar m_client->sendStreamingRequest
    // 5. No callback, chamar m_func_handler->execute ou m_audio_handler->playAudioChunk
}

} // namespace trackie::services