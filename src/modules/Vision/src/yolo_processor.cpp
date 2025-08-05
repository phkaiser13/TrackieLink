/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "yolo_processor.hpp"
#include <opencv2/dnn.hpp>
#include <vector>

namespace {
    // Função auxiliar para pré-processamento do YOLO
    std::vector<float> preprocess_yolo(const cv::Mat& frame, int size) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(size, size), cv::Scalar(), true, false, CV_32F);
        return { (float*)blob.datastart, (float*)blob.dataend };
    }
}

namespace trackie::vision {

    YoloProcessor::YoloProcessor(InferenceSession* yolo_session, const std::map<int, std::string>& class_names)
        : m_yolo_session(yolo_session), m_class_names(class_names) {
        if (!m_yolo_session) {
            throw std::runtime_error("YoloProcessor: A sessão de inferência do YOLO não pode ser nula.");
        }
    }

    std::vector<DetectionResult> YoloProcessor::processFrame(const cv::Mat& frame) {
        auto input_tensor = preprocess_yolo(frame, m_input_size);
        const int64_t input_dims[] = {1, 3, (int64_t)m_input_size, (int64_t)m_input_size};

        DetectionResult* raw_results = nullptr;
        size_t num_results = 0;

        // Supondo que a API de inferência foi adaptada para ser mais genérica
        run_yolo_inference(m_yolo_session, input_tensor.data(), input_dims, 4, &raw_results, &num_results);

        std::vector<DetectionResult> results;
        if (num_results > 0) {
            results.assign(raw_results, raw_results + num_results);
        }

        free_detection_results(raw_results); // Libera a memória alocada pela API C
        return results;
    }

} // namespace trackie::vision