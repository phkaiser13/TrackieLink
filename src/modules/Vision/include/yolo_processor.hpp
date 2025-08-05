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
#include <vector>
#include <string>
#include <map>

namespace trackie::vision {

    class YoloProcessor {
    public:
        YoloProcessor(InferenceSession* yolo_session, const std::map<int, std::string>& class_names);

        std::vector<DetectionResult> processFrame(const cv::Mat& frame);

    private:
        InferenceSession* m_yolo_session; // Não possui, apenas usa
        std::map<int, std::string> m_class_names;
        int m_input_size = 640;
    };

} // namespace trackie::vision