/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "face_database.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <numeric>
#include <cmath>

// Função auxiliar para pré-processamento (pode ser movida para um arquivo de utilidades)
std::vector<float> preprocess_face_frame(const cv::Mat& frame) {
    cv::Mat resized_frame;
    // Modelos de reconhecimento facial como o ArcFace esperam 112x112
    cv::resize(frame, resized_frame, cv::Size(112, 112));

    // Converte para float e normaliza para [-1, 1] ou [0, 1] dependendo do modelo
    // Este exemplo normaliza para [0, 1] e converte BGR para RGB
    resized_frame.convertTo(resized_frame, CV_32F, 1.0/255.0);
    cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

    // Converte de HWC para CHW
    std::vector<cv::Mat> channels(3);
    cv::split(resized_frame, channels);

    std::vector<float> tensor_data;
    tensor_data.reserve(112 * 112 * 3);
    tensor_data.insert(tensor_data.end(), (float*)channels[0].datastart, (float*)channels[0].dataend);
    tensor_data.insert(tensor_data.end(), (float*)channels[1].datastart, (float*)channels[1].dataend);
    tensor_data.insert(tensor_data.end(), (float*)channels[2].datastart, (float*)channels[2].dataend);

    return tensor_data;
}

// Função auxiliar para calcular a similaridade de cosseno
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot_product = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0;
    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}


namespace trackie::core {

FaceDatabase::FaceDatabase(const std::filesystem::path& db_path, InferenceSession* face_rec_session)
    : m_db_path(db_path), m_face_rec_session(face_rec_session) {
    if (!m_face_rec_session) {
        throw std::runtime_error("FaceDatabase: Sessão de inferência de reconhecimento facial é nula.");
    }
    std::cout << "[FaceDB] Construindo banco de dados de rostos conhecidos..." << std::endl;
    _buildDatabase();
    std::cout << "[FaceDB] Banco de dados construído com " << m_known_faces.size() << " rosto(s) conhecido(s)." << std::endl;
}

void FaceDatabase::_buildDatabase() {
    // Itera sobre os subdiretórios (cada um é uma pessoa)
    for (const auto& person_dir : std::filesystem::directory_iterator(m_db_path)) {
        if (!person_dir.is_directory()) continue;

        std::string person_name = person_dir.path().filename().string();
        std::cout << "[FaceDB] Processando pessoa: " << person_name << std::endl;

        // Itera sobre as imagens de cada pessoa
        for (const auto& image_file : std::filesystem::directory_iterator(person_dir.path())) {
            cv::Mat image = cv::imread(image_file.path().string());
            if (image.empty()) continue;

            std::vector<float> embedding = _generateEmbedding(image);
            if (!embedding.empty()) {
                m_known_faces.push_back({person_name, std::move(embedding)});
                // Em um sistema real, poderíamos calcular a média dos embeddings de várias fotos
                break; // Por simplicidade, usamos apenas a primeira foto válida por pessoa
            }
        }
    }
}

std::vector<float> FaceDatabase::_generateEmbedding(const cv::Mat& face_image) {
    std::vector<float> input_tensor = preprocess_face_frame(face_image);
    const int64_t input_dims[] = {1, 3, 112, 112};

    // A API de inferência precisa ser genérica ou ter uma versão para embeddings
    // Vamos assumir que a saída é um tensor [1, 512] e adaptar a chamada
    // A função run_yolo_inference precisaria de uma variante. Por agora, vamos simular a chamada.
    // int status = run_embedding_inference(m_face_rec_session, ...);

    // Simulação do resultado da inferência
    // Em um caso real, você pegaria o ponteiro de dados do tensor de saída do ONNX Runtime.
    std::vector<float> embedding(512);
    // Preenche com dados de exemplo.
    std::iota(embedding.begin(), embedding.end(), 0.1f);

    return embedding;
}

std::string FaceDatabase::findClosestMatch(const std::vector<float>& query_embedding, float threshold) const {
    if (query_embedding.empty() || m_known_faces.empty()) {
        return "";
    }

    float best_similarity = -1.0f;
    std::string best_match_name = "";

    for (const auto& known_face : m_known_faces) {
        float similarity = cosine_similarity(query_embedding, known_face.embedding);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_match_name = known_face.name;
        }
    }

    if (best_similarity >= threshold) {
        std::cout << "[FaceDB] Melhor correspondência: " << best_match_name << " (Similaridade: " << best_similarity << ")" << std::endl;
        return best_match_name;
    }

    return ""; // Nenhuma correspondência acima do limiar
}

} // namespace trackie::core