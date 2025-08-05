/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include "gemini_client.hpp"
#include "function_handler.hpp"
#include "audio_handler.hpp"
#include "logger.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <opencv2/core/mat.hpp>

namespace trackie::services {

    class GeminiService {
    public:
        GeminiService(
            std::shared_ptr<api::GeminiClient> client,
            std::shared_ptr<functions::FunctionHandler> funcs,
            std::shared_ptr<audio::AudioHandler> audio
        );
        ~GeminiService();

        void start(const nlohmann::json& config, const std::string& system_prompt);
        void stop();

        void pushUserTextMessage(const std::string& message);
        void pushAudioBuffer(const std::vector<uint8_t>& audio_data);
        void pushVideoFrame(const cv::Mat& frame);

    private:
        void _communicationLoop();

        std::shared_ptr<api::GeminiClient> m_client;
        std::shared_ptr<functions::FunctionHandler> m_func_handler;
        std::shared_ptr<audio::AudioHandler> m_audio_handler;

        std::thread m_thread;
        std::atomic<bool> m_stop_flag{false};

        std::queue<std::string> m_text_queue;
        std::queue<std::vector<uint8_t>> m_audio_queue;

        std::mutex m_queue_mutex;
        std::condition_variable m_cv;

        std::mutex m_frame_mutex;
        cv::Mat m_latest_frame;

        nlohmann::json m_config;
        std::string m_system_prompt;
    };

} // namespace trackie::services