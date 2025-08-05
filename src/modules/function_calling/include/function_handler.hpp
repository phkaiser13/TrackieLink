/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include "yolo_processor.hpp"
#include "face_processor.hpp"
#include "midas_processor.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <opencv2/core/mat.hpp>

namespace trackie::functions {

    class FunctionHandler {
    public:
        FunctionHandler(
            std::shared_ptr<vision::YoloProcessor> yolo,
            std::shared_ptr<vision::FaceProcessor> face,
            std::shared_ptr<vision::MidasProcessor> midas
        );

        std::string execute(const std::string& function_name, const nlohmann::json& args, const cv::Mat& current_frame);

    private:
        std::string _handle_identify_person(const cv::Mat& frame);
        std::string _handle_save_face(const nlohmann::json& args, const cv::Mat& frame);
        std::string _handle_find_object(const nlohmann::json& args, const cv::Mat& frame);

        std::shared_ptr<vision::YoloProcessor> m_yolo_processor;
        std::shared_ptr<vision::FaceProcessor> m_face_processor;
        std::shared_ptr<vision::MidasProcessor> m_midas_processor;
    };

} // namespace trackie::functions