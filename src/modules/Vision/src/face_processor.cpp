/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "face_processor.hpp"
#include "logger.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <cmath>
#include <algorithm> // Para std::max_element
#include <chrono>    // Para o timestamp

namespace {
    // Funções auxiliares de pré-processamento e cálculo
    std::vector<float> preprocess_face(const cv::Mat& frame, int size) {
        cv::Mat resized, blob;
        // Modelos de reconhecimento facial como o ArcFace esperam 112x112
        cv::resize(frame, resized, cv::Size(size, size));
        // Normalização para [-1, 1] é comum para modelos faciais
        cv::dnn::blobFromImage(resized, blob, 1.0/127.5, cv::Size(size, size), cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);
        return { (float*)blob.datastart, (float*)blob.dataend };
    }

    // Pré-processamento para o detector de rostos (pode ser diferente)
    std::vector<float> preprocess_face_detector(const cv::Mat& frame, int size) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(size, size), cv::Scalar(), true, false, CV_32F);
        return { (float*)blob.datastart, (float*)blob.dataend };
    }

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) return 0.0f;
        float dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a == 0.0 || norm_b == 0.0) return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
}

namespace trackie::vision {

FaceProcessor::FaceProcessor(InferenceSession* detector, InferenceSession* recognizer, const std::filesystem::path& db_path)
    : m_detector_session(detector), m_recognizer_session(recognizer), m_db_path(db_path) {
    if (!m_detector_session || !m_recognizer_session) {
        throw std::runtime_error("FaceProcessor: As sessões de detecção e reconhecimento são necessárias.");
    }
    _buildDatabase();
}

void FaceProcessor::_buildDatabase() {
    log::TLog(log::LogLevel::INFO, "[FaceDB] Construindo banco de dados de rostos...");
    m_known_faces.clear();
    if (!std::filesystem::exists(m_db_path)) {
        log::TLog(log::LogLevel::WARNING, "[FaceDB] Diretório do banco de dados de rostos não encontrado: ", m_db_path.string());
        return;
    }
    for (const auto& person_dir : std::filesystem::directory_iterator(m_db_path)) {
        if (!person_dir.is_directory()) continue;
        std::string person_name = person_dir.path().filename().string();
        for (const auto& image_file : std::filesystem::directory_iterator(person_dir.path())) {
            cv::Mat image = cv::imread(image_file.path().string());
            if (image.empty()) continue;
            auto embedding = _generateEmbedding(image);
            if (!embedding.empty()) {
                m_known_faces.push_back({person_name, std::move(embedding)});
                break; // Uma imagem por pessoa por simplicidade
            }
        }
    }
    log::TLog(log::LogLevel::INFO, "[FaceDB] Banco de dados construído com ", m_known_faces.size(), " rosto(s).");
}

std::vector<DetectionResult> FaceProcessor::_detectFaces(const cv::Mat& frame) {
    auto input_tensor = preprocess_face_detector(frame, 320);
    const int64_t input_dims[] = {1, 3, 320, 320};
    DetectionResult* raw_results = nullptr;
    size_t num_results = 0;
    run_yolo_inference(m_detector_session, input_tensor.data(), input_dims, 4, &raw_results, &num_results);
    std::vector<DetectionResult> results;
    if (num_results > 0) results.assign(raw_results, raw_results + num_results);
    free_detection_results(raw_results);
    return results;
}

// <<<<<<< ESTA É A MUDANÇA PRINCIPAL >>>>>>>
std::vector<float> FaceProcessor::_generateEmbedding(const cv::Mat& face_image) {
    auto input_tensor = preprocess_face(face_image, 112);
    const int64_t input_dims[] = {1, 3, 112, 112};

    float* raw_embedding = nullptr;
    size_t embedding_size = 0;

    int status = run_embedding_inference(
        m_recognizer_session,
        input_tensor.data(),
        input_dims,
        4,
        &raw_embedding,
        &embedding_size
    );

    std::vector<float> embedding;
    if (status == 0 && embedding_size > 0) {
        embedding.assign(raw_embedding, raw_embedding + embedding_size);
    } else {
        log::TLog(log::LogLevel::ERROR, "Falha ao gerar embedding facial.");
    }

    // Libera a memória alocada pela API C
    if (raw_embedding) {
        free(raw_embedding);
    }

    return embedding;
}

Identity FaceProcessor::identifyPerson(const cv::Mat& frame, float threshold) {
    auto faces = _detectFaces(frame);
    if (faces.empty()) return {"", 0.0f};

    auto largest_face_it = std::max_element(faces.begin(), faces.end(),
        [](const auto& a, const auto& b){
            return (a.x2 - a.x1) * (a.y2 - a.y1) < (b.x2 - b.x1) * (b.y2 - b.y1);
        });

    // Garante que as coordenadas estão dentro dos limites do frame
    cv::Rect face_roi(
        static_cast<int>(largest_face_it->x1), static_cast<int>(largest_face_it->y1),
        static_cast<int>(largest_face_it->x2 - largest_face_it->x1), static_cast<int>(largest_face_it->y2 - largest_face_it->y1)
    );
    face_roi &= cv::Rect(0, 0, frame.cols, frame.rows); // Interseção para garantir que está dentro
    if (face_roi.width <= 0 || face_roi.height <= 0) return {"", 0.0f};

    cv::Mat face_image = frame(face_roi);

    auto query_embedding = _generateEmbedding(face_image);
    if (query_embedding.empty()) return {"", 0.0f};

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
        Identity id;
        strncpy(id.name, best_match_name.c_str(), sizeof(id.name) - 1);
        id.name[sizeof(id.name) - 1] = '\0'; // Garante terminação nula
        id.similarity = best_similarity;
        return id;
    }
    return {"", 0.0f};
}

bool FaceProcessor::saveNewFace(const cv::Mat& frame, const std::string& person_name) {
    auto faces = _detectFaces(frame);
    if (faces.empty()) return false;

    auto largest_face_it = std::max_element(faces.begin(), faces.end(),
        [](const auto& a, const auto& b){
            return (a.x2 - a.x1) * (a.y2 - a.y1) < (b.x2 - b.x1) * (b.y2 - b.y1);
        });

    cv::Rect face_roi(
        static_cast<int>(largest_face_it->x1), static_cast<int>(largest_face_it->y1),
        static_cast<int>(largest_face_it->x2 - largest_face_it->x1), static_cast<int>(largest_face_it->y2 - largest_face_it->y1)
    );
    face_roi &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (face_roi.width <= 0 || face_roi.height <= 0) return false;

    cv::Mat face_image = frame(face_roi);

    std::filesystem::path person_dir = m_db_path / person_name;
    std::filesystem::create_directories(person_dir);

    long long timestamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string filename = person_name + "_" + std::to_string(timestamp) + ".jpg";

    bool success = cv::imwrite((person_dir / filename).string(), face_image);
    if (success) {
        _buildDatabase(); // Reconstrói o banco de dados em memória com o novo rosto
    }
    return success;
}

} // namespace trackie::vision