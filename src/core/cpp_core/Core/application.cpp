/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "application.hpp"
#include "gemini_client.hpp"
#include "packloader.hpp"
#include "face_processor.hpp"
#include "midas_processor.hpp"
#include "function_handler.hpp"
#include <nlohmann/json.hpp>

namespace trackie::core {

Application::Application(std::string mode, bool show_preview)
    : m_mode(std::move(mode)), m_show_preview(show_preview) {
    try {
        _initialize();
    } catch (const std::exception& e) {
        log::TLog(log::LogLevel::ERROR, "Falha na inicialização: ", e.what());
        throw;
    }
}

Application::~Application() {
    _cleanup();
}

void Application::_initialize() {
    log::TLog(log::LogLevel::INFO, "Inicializando...");

    // Carregar Config
    auto config = loader::loadMsgpackFile("Data/trckconfig.msgpack");
    auto sys_prompt_data = loader::loadMsgpackFile("Data/ForSystemInstructions/sys.msgpack");
    std::string system_prompt = sys_prompt_data.value("system_instruction", "");

    // Inicializar Módulos de I/O
    m_audio_handler = std::make_shared<audio::AudioHandler>();
    m_video_handler = std::make_shared<video::VideoHandler>();

    // Inicializar Motor de Inferência e Modelos
    InferenceEngine* engine = create_inference_engine();
    InferenceSession *yolo, *face_det, *face_rec, *midas;
    // ... lógica para carregar todos os modelos a partir do config ...

    // Inicializar Módulos de Visão
    std::map<int, std::string> class_names; // ... carregar do config ...
    m_yolo_processor = std::make_shared<vision::YoloProcessor>(yolo, class_names);
    auto face_processor = std::make_shared<vision::FaceProcessor>(face_det, face_rec, config.value("face_db_path", ""));
    auto midas_processor = std::make_shared<vision::MidasProcessor>(midas);

    // Inicializar Módulos de Lógica e Serviços
    auto func_handler = std::make_shared<functions::FunctionHandler>(m_yolo_processor, face_processor, midas_processor);
    auto gemini_client = std::make_shared<api::GeminiClient>(config.value("gemini_api_key", ""));
    m_gemini_service = std::make_shared<services::GeminiService>(gemini_client, func_handler, m_audio_handler);

    // Iniciar Serviços
    m_gemini_service->start(config, system_prompt);
    log::TLog(log::LogLevel::INFO, "Inicialização completa.");
}

void Application::_cleanup() {
    log::TLog(log::LogLevel::INFO, "Finalizando...");
    m_stop_flag.store(true);
    if (m_console_thread.joinable()) m_console_thread.join();
    m_gemini_service->stop();
    m_video_handler->stopCapture();
    // ... limpeza do motor de inferência ...
}

int Application::run() {
    m_console_thread = std::thread(&Application::_console_input_loop, this);
    _main_loop();
    return 0;
}

void Application::_console_input_loop() {
    std::string line;
    while (!m_stop_flag.load()) {
        std::getline(std::cin, line);
        if (line == "q" || !std::cin.good()) {
            m_stop_flag.store(true);
            break;
        }
        m_gemini_service->pushUserTextMessage(line);
    }
}

void Application::_main_loop() {
    m_audio_handler->startInputStream([this](const float* buf, unsigned long count){
        // ... converter para uint8_t e chamar m_gemini_service->pushAudioBuffer(...) ...
    });

    m_video_handler->startCapture([this](const cv::Mat& frame){
        m_gemini_service->pushVideoFrame(frame);
        auto detections = m_yolo_processor->processFrame(frame);
        // ... lógica de desenhar no preview ...
    });

    while(!m_stop_flag.load()) {
        // O loop principal agora só mantém a aplicação viva e a UI responsiva
        // ... lógica do cv::waitKey ...
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

} // namespace trackie::core