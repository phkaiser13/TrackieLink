/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "midas_processor.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace {
    std::vector<float> preprocess_midas(const cv::Mat& frame, int width, int height) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(width, height), cv::Scalar(123.675, 116.28, 103.53), true, false, CV_32F);
        return { (float*)blob.datastart, (float*)blob.dataend };
    }
}

namespace trackie::vision {

MidasProcessor::MidasProcessor(InferenceSession* midas_session) : m_midas_session(midas_session) {
    if (!m_midas_session) {
        throw std::runtime_error("MidasProcessor: A sessão de inferência do MiDaS não pode ser nula.");
    }
}

cv::Mat MidasProcessor::getDepthMap(const cv::Mat& frame) {
    auto input_tensor = preprocess_midas(frame, m_input_width, m_input_height);
    const int64_t input_dims[] = {1, 3, m_input_height, m_input_width};

    // A API de inferência precisaria de uma variante para retornar um tensor bruto
    // ... chamada à API de inferência aqui ...

    // Simulação do resultado
    cv::Mat depth_map(m_input_height, m_input_width, CV_32F, cv::Scalar(0.5));

    // Redimensiona o mapa de profundidade para o tamanho original do frame
    cv::Mat resized_depth_map;
    cv::resize(depth_map, resized_depth_map, frame.size());

    // Normaliza para visualização (0-1)
    cv::normalize(resized_depth_map, resized_depth_map, 0, 1, cv::NORM_MINMAX);

    return resized_depth_map;
}

float MidasProcessor::getDepthAt(const cv::Mat& depth_map, int x, int y) {
    if (x < 0 || y < 0 || x >= depth_map.cols || y >= depth_map.rows) {
        return 0.0f;
    }
    // O MiDaS produz profundidade inversa, então um valor maior significa mais perto.
    return depth_map.at<float>(y, x);
}

} // namespace trackie::vision