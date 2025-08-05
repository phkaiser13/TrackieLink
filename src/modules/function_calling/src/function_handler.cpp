/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "function_handler.hpp"
#include "logger.hpp"

namespace trackie::functions {

FunctionHandler::FunctionHandler(
    std::shared_ptr<vision::YoloProcessor> yolo,
    std::shared_ptr<vision::FaceProcessor> face,
    std::shared_ptr<vision::MidasProcessor> midas)
    : m_yolo_processor(yolo), m_face_processor(face), m_midas_processor(midas) {}

std::string FunctionHandler::execute(const std::string& function_name, const nlohmann::json& args, const cv::Mat& current_frame) {
    log::TLog(log::LogLevel::INFO, "Executando Function Call: ", function_name);
    if (function_name == "identify_person_in_front") {
        return _handle_identify_person(current_frame);
    }
    if (function_name == "save_known_face") {
        return _handle_save_face(args, current_frame);
    }
    if (function_name == "find_object_and_estimate_distance") {
        return _handle_find_object(args, current_frame);
    }
    return "{\"status\": \"error\", \"reason\": \"Função desconhecida.\"}";
}

std::string FunctionHandler::_handle_identify_person(const cv::Mat& frame) {
    auto identity = m_face_processor->identifyPerson(frame, 0.6f);
    nlohmann::json result;
    result["status"] = "success";
    if (identity.similarity > 0) {
        result["person_identified"] = true;
        result["name"] = identity.name;
        result["message"] = std::string("A pessoa na sua frente parece ser ") + identity.name;
    } else {
        result["person_identified"] = false;
        result["message"] = "Não reconheci a pessoa na sua frente.";
    }
    return result.dump();
}

std::string FunctionHandler::_handle_save_face(const nlohmann::json& args, const cv::Mat& frame) {
    std::string name = args.value("person_name", "");
    if (name.empty()) return "{\"status\": \"error\", \"reason\": \"Nome não fornecido.\"}";

    bool success = m_face_processor->saveNewFace(frame, name);
    nlohmann::json result;
    if (success) {
        result["status"] = "success";
        result["message"] = "Rosto de " + name + " salvo com sucesso.";
    } else {
        result["status"] = "error";
        result["reason"] = "Não foi possível detectar um rosto para salvar.";
    }
    return result.dump();
}

std::string FunctionHandler::_handle_find_object(const nlohmann::json& args, const cv::Mat& frame) {
    std::string object_name = args.value("object_name", "");
    if (object_name.empty()) return "{\"status\": \"error\", \"reason\": \"Nome do objeto não fornecido.\"}";

    auto detections = m_yolo_processor->processFrame(frame);
    DetectionResult best_match = {};
    bool found = false;

    for (const auto& det : detections) {
        // ... lógica para mapear det.class_id para object_name ...
        // if (class_name == object_name && det.confidence > best_match.confidence) {
        //     best_match = det;
        //     found = true;
        // }
    }

    if (!found) {
        return "{\"status\": \"success\", \"found\": false, \"message\": \"Não encontrei um(a) " + object_name + ".\"}";
    }

    auto depth_map = m_midas_processor->getDepthMap(frame);
    int center_x = best_match.x1 + (best_match.x2 - best_match.x1) / 2;
    int center_y = best_match.y1 + (best_match.y2 - best_match.y1) / 2;
    float depth_value = m_midas_processor->getDepthAt(depth_map, center_x, center_y);

    // Converter depth_value para metros (altamente empírico)
    float distance_m = 1.0f / (depth_value * 10 + 0.1f);

    nlohmann::json result;
    result["status"] = "success";
    result["found"] = true;
    result["message"] = "O objeto " + object_name + " está a aproximadamente " + std::to_string(distance_m).substr(0,3) + " metros.";
    return result.dump();
}

} // namespace trackie::functions