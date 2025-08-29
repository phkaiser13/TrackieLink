/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 */

#include "face_processor.hpp"
#include "logger.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace {
    // Helper functions remain the same
    std::vector<float> preprocess_face(const cv::Mat& frame, int size) {
        cv::Mat resized, blob;
        cv::resize(frame, resized, cv::Size(size, size));
        cv::dnn::blobFromImage(resized, blob, 1.0/127.5, cv::Size(size, size), cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);
        return { (float*)blob.datastart, (float*)blob.dataend };
    }

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

FaceProcessor::FaceProcessor(
    std::shared_ptr<trackie::inference::InferenceEngine> detector,
    std::shared_ptr<trackie::inference::InferenceEngine> recognizer,
    const std::filesystem::path& db_path)
    : m_detector_engine(detector), m_recognizer_engine(recognizer), m_db_path(db_path) {
    if (!m_detector_engine || !m_recognizer_engine) {
        throw std::runtime_error("FaceProcessor: Detector and recognizer engines are required.");
    }
    _buildDatabase();
}

void FaceProcessor::_buildDatabase() {
    log::TLog(log::LogLevel::INFO, "[FaceDB] Building face database...");
    m_known_faces.clear();
    if (!std::filesystem::exists(m_db_path)) {
        log::TLog(log::LogLevel::WARNING, "[FaceDB] Face database directory not found: ", m_db_path.string());
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
                break;
            }
        }
    }
    log::TLog(log::LogLevel::INFO, "[FaceDB] Database built with ", m_known_faces.size(), " face(s).");
}

std::vector<DetectionResult> FaceProcessor::_detectFaces(const cv::Mat& frame) {
    auto input_tensor = preprocess_face_detector(frame, 320);
    const std::vector<int64_t> input_dims = {1, 3, 320, 320};
    // The new C++ engine's runYolo is a placeholder, so this will return an empty vector.
    // This is acceptable for now as the goal is to make the code compile and run.
    return m_detector_engine->runYolo(input_tensor, input_dims);
}

std::vector<float> FaceProcessor::_generateEmbedding(const cv::Mat& face_image) {
    auto input_tensor = preprocess_face(face_image, 112);
    const std::vector<int64_t> input_dims = {1, 3, 112, 112};

    std::vector<float> embedding = m_recognizer_engine->runEmbedding(input_tensor, input_dims);

    if (embedding.empty()) {
        log::TLog(log::LogLevel::ERROR, "Failed to generate face embedding.");
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

    cv::Rect face_roi(
        static_cast<int>(largest_face_it->x1), static_cast<int>(largest_face_it->y1),
        static_cast<int>(largest_face_it->x2 - largest_face_it->x1), static_cast<int>(largest_face_it->y2 - largest_face_it->y1)
    );
    face_roi &= cv::Rect(0, 0, frame.cols, frame.rows);
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
        id.name[sizeof(id.name) - 1] = '\0';
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
        _buildDatabase();
    }
    return success;
}

} // namespace trackie::vision