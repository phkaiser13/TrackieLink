/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include <opencv2/core/mat.hpp>
#include <functional>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

// Forward declaration da classe VideoCapture do OpenCV
namespace cv {
    class VideoCapture;
}

namespace trackie::video {

    /**
     * @brief Callback para receber um novo frame de vídeo capturado.
     * @param frame O frame capturado como um objeto cv::Mat.
     */
    using VideoFrameCallback = std::function<void(const cv::Mat& frame)>;

    class VideoHandler {
    public:
        /**
         * @param device_index O índice do dispositivo de câmera (geralmente 0).
         */
        explicit VideoHandler(int device_index = 0);
        ~VideoHandler();

        VideoHandler(const VideoHandler&) = delete;
        VideoHandler& operator=(const VideoHandler&) = delete;

        /**
         * @brief Inicia o loop de captura de vídeo em um thread separado.
         * @param callback A função a ser chamada para cada novo frame.
         */
        void startCapture(const VideoFrameCallback& callback);

        /**
         * @brief Sinaliza para o loop de captura parar e aguarda o thread finalizar.
         */
        void stopCapture();

        /**
         * @brief Verifica se o loop de captura está atualmente em execução.
         */
        bool isCapturing() const;

    private:
        void _captureLoop();

        std::unique_ptr<cv::VideoCapture> m_capture;
        int m_device_index;

        VideoFrameCallback m_frame_callback;

        // --- Gerenciamento de Thread ---
        std::thread m_capture_thread;
        std::atomic<bool> m_is_running{false};
    };

} // namespace trackie::video