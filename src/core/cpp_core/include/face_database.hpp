/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include "inference.h" // Para usar InferenceSession

// Forward declaration
namespace cv { class Mat; }

namespace trackie::core {

// Representa um único rosto conhecido no banco de dados
struct KnownFace {
    std::string name;
    std::vector<float> embedding; // Vetor de 512 floats, por exemplo
};

class FaceDatabase {
public:
    /**
     * @param db_path Caminho para o diretório raiz dos rostos conhecidos (ex: "Data/known_faces").
     * @param face_rec_session Uma sessão de inferência já carregada com o modelo de reconhecimento facial.
     */
    FaceDatabase(const std::filesystem::path& db_path, InferenceSession* face_rec_session);

    /**
     * @brief Encontra a correspondência mais próxima para um dado embedding.
     * @param query_embedding O embedding do rosto a ser identificado.
     * @param threshold O limiar de distância (cosseno) para considerar uma correspondência.
     * @return O nome da pessoa correspondente, ou uma string vazia se nenhuma correspondência for encontrada.
     */
    std::string findClosestMatch(const std::vector<float>& query_embedding, float threshold) const;

private:
    void _buildDatabase();
    std::vector<float> _generateEmbedding(const cv::Mat& face_image);

    std::filesystem::path m_db_path;
    InferenceSession* m_face_rec_session; // Não possui, apenas usa a referência
    std::vector<KnownFace> m_known_faces;
};

} // namespace trackie::core