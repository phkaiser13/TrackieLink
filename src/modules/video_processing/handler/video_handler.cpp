#include "video_handler.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp> // For cvtColor
#include <stdexcept>
#include <iostream>

// Includes for CUDA integration
#if defined(WITH_CUDA) // This macro will be defined by CMake if CUDA is found
#include "cuda_core/memory.h"
#include "cuda_core/image_kernels.h"
#endif

namespace trackie::video {

VideoHandler::VideoHandler(int device_index)
    : m_device_index(device_index), m_mode(ProcessMode::CPU) {
    m_capture = std::make_unique<cv::VideoCapture>();
}

VideoHandler::~VideoHandler() {
    stopCapture();
}

void VideoHandler::startCapture(const VideoFrameCallback& callback, ProcessMode mode) {
    if (m_is_running.load()) {
        std::cout << "[Video] Capture is already running." << std::endl;
        return;
    }

    if (!m_capture->open(m_device_index)) {
        throw std::runtime_error("Failed to open video device at index: " + std::to_string(m_device_index));
    }

    m_mode = mode;
    m_frame_callback = callback;
    m_is_running.store(true);
    m_capture_thread = std::thread(&VideoHandler::_captureLoop, this);

    #if defined(WITH_CUDA)
        std::cout << "[Video] Video capture started in " << (mode == ProcessMode::CPU ? "CPU" : "GPU_GRAYSCALE") << " mode." << std::endl;
    #else
        if (mode == ProcessMode::GPU_GRAYSCALE) {
            std::cout << "[Video] WARNING: GPU mode requested but CUDA is not available. Falling back to CPU." << std::endl;
            m_mode = ProcessMode::CPU;
        }
        std::cout << "[Video] Video capture started in CPU mode." << std::endl;
    #endif
}

void VideoHandler::stopCapture() {
    if (!m_is_running.load()) {
        return;
    }
    m_is_running.store(false);
    if (m_capture_thread.joinable()) {
        m_capture_thread.join();
    }
    if (m_capture && m_capture->isOpened()) {
        m_capture->release();
    }
    std::cout << "[Video] Video capture stopped." << std::endl;
}

bool VideoHandler::isCapturing() const {
    return m_is_running.load();
}

#if defined(WITH_CUDA)
// Helper function for the GPU processing path
void processFrameGPU(const cv::Mat& frame, const VideoFrameCallback& callback) {
    if (frame.empty() || !callback) return;

    cv::Mat bgr_frame;
    if (frame.channels() == 1) cv::cvtColor(frame, bgr_frame, cv::COLOR_GRAY2BGR);
    else if (frame.channels() == 4) cv::cvtColor(frame, bgr_frame, cv::COLOR_BGRA2BGR);
    else bgr_frame = frame;

    if (!bgr_frame.isContinuous()) {
        bgr_frame = bgr_frame.clone();
    }

    unsigned char *d_input = nullptr, *d_output = nullptr;
    const size_t rgb_size = bgr_frame.cols * bgr_frame.rows * 3;
    const size_t gray_size = bgr_frame.cols * bgr_frame.rows;

    if (cuda_malloc_device((void**)&d_input, rgb_size) != 0 || cuda_malloc_device((void**)&d_output, gray_size) != 0) {
        std::cerr << "[Video] GPU Error: Failed to allocate device memory." << std::endl;
        cuda_free_device(d_input);
        cuda_free_device(d_output);
        return;
    }

    if (cuda_memcpy_host_to_device(d_input, bgr_frame.data, rgb_size) != 0) {
        std::cerr << "[Video] GPU Error: Failed to copy data to device." << std::endl;
    } else {
        launch_rgb_to_grayscale_kernel(d_input, d_output, bgr_frame.cols, bgr_frame.rows);
        cv::Mat gray_frame(bgr_frame.rows, bgr_frame.cols, CV_8UC1);
        if (cuda_memcpy_device_to_host(gray_frame.data, d_output, gray_size) == 0) {
            callback(gray_frame);
        } else {
            std::cerr << "[Video] GPU Error: Failed to copy data from device." << std::endl;
        }
    }

    cuda_free_device(d_input);
    cuda_free_device(d_output);
}
#endif

void VideoHandler::_captureLoop() {
    cv::Mat frame;
    while (m_is_running.load()) {
        if (!m_capture->read(frame) || frame.empty()) {
            std::cerr << "[Video] Error: Failed to read frame from device. Stopping capture." << std::endl;
            m_is_running.store(false);
            break;
        }

        #if defined(WITH_CUDA)
            if (m_mode == ProcessMode::GPU_GRAYSCALE) {
                processFrameGPU(frame, m_frame_callback);
            } else {
                if (m_frame_callback) m_frame_callback(frame);
            }
        #else
            // If CUDA is not available, always default to CPU path.
            if (m_frame_callback) {
                m_frame_callback(frame);
            }
        #endif

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

} // namespace trackie::video