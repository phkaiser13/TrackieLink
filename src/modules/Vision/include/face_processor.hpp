/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include "inference.h"
#include "vision_types.h"
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <filesystem>

namespace trackie::vision {

    class FaceProcessor {
    public:
        FaceProcessor(InferenceSession* detector, InferenceSession* recognizer, const std::filesystem::path& db_path);

        Identity identifyPerson(const cv::Mat& frame, float threshold);
        bool saveNewFace(const cv::Mat& frame, const std::string& person_name);

    private:
        struct KnownFace {
            std::string name;
            std::vector<float> embedding;
        };

        void _buildDatabase();
        std::vector<DetectionResult> _detectFaces(const cv::Mat& frame);
        std::vector<float> _generateEmbedding(const cv::Mat& face_image);

        InferenceSession* m_detector_session;
        InferenceSession* m_recognizer_session;
        std::filesystem::path m_db_path;
        std::vector<KnownFace> m_known_faces;
    };

} // namespace trackie::vision