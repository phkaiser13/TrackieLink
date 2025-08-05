/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "video_handler.hpp"
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <iostream>

namespace trackie::video {

VideoHandler::VideoHandler(int device_index)
    : m_device_index(device_index) {
    m_capture = std::make_unique<cv::VideoCapture>();
}

VideoHandler::~VideoHandler() {
    stopCapture();
}

void VideoHandler::startCapture(const VideoFrameCallback& callback) {
    if (m_is_running.load()) {
        std::cout << "[Video] Captura já está em execução." << std::endl;
        return;
    }

    if (!m_capture->open(m_device_index)) {
        throw std::runtime_error("Falha ao abrir o dispositivo de vídeo com índice: " + std::to_string(m_device_index));
    }

    if (!m_capture->isOpened()) {
        throw std::runtime_error("Dispositivo de vídeo não pôde ser aberto.");
    }

    m_frame_callback = callback;
    m_is_running.store(true);
    m_capture_thread = std::thread(&VideoHandler::_captureLoop, this);
    std::cout << "[Video] Captura de vídeo iniciada." << std::endl;
}

void VideoHandler::stopCapture() {
    if (!m_is_running.load()) {
        return;
    }

    m_is_running.store(false);
    if (m_capture_thread.joinable()) {
        m_capture_thread.join(); // Aguarda o thread do loop terminar
    }

    if (m_capture && m_capture->isOpened()) {
        m_capture->release();
    }
    std::cout << "[Video] Captura de vídeo parada." << std::endl;
}

bool VideoHandler::isCapturing() const {
    return m_is_running.load();
}

void VideoHandler::_captureLoop() {
    cv::Mat frame;
    while (m_is_running.load()) {
        if (!m_capture->read(frame)) {
            std::cerr << "[Video] Erro: Falha ao ler frame do dispositivo. Parando captura." << std::endl;
            m_is_running.store(false);
            break;
        }

        if (!frame.empty() && m_frame_callback) {
            m_frame_callback(frame);
        }

        // Pequena pausa para não sobrecarregar a CPU, a taxa de quadros
        // da câmera já limita a velocidade.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

} // namespace trackie::video