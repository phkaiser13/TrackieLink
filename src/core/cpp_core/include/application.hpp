/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include "gemini_service.hpp"
#include "audio_handler.hpp"
#include "video_handler.hpp"
#include "yolo_processor.hpp"
#include "logger.hpp"
#include <string>
#include <memory>
#include <atomic>
#include <thread>

namespace trackie::core {

    class Application {
    public:
        Application(std::string mode, bool show_preview);
        ~Application();
        Application(const Application&) = delete;
        Application& operator=(const Application&) = delete;
        int run();

    private:
        void _initialize();
        void _main_loop();
        void _cleanup();
        void _console_input_loop();

        // --- Estado e Configuração ---
        std::string m_mode;
        bool m_show_preview;
        std::atomic<bool> m_stop_flag{false};

        // --- Módulos e Serviços ---
        std::shared_ptr<services::GeminiService> m_gemini_service;
        std::shared_ptr<audio::AudioHandler> m_audio_handler;
        std::shared_ptr<video::VideoHandler> m_video_handler;
        std::shared_ptr<vision::YoloProcessor> m_yolo_processor;

        std::thread m_console_thread;
    };

} // namespace trackie::core