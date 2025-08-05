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

namespace trackie::vision {

    class MidasProcessor {
    public:
        MidasProcessor(InferenceSession* midas_session);

        // Retorna um mapa de profundidade normalizado (0.0 a 1.0)
        cv::Mat getDepthMap(const cv::Mat& frame);
        float getDepthAt(const cv::Mat& depth_map, int x, int y);

    private:
        InferenceSession* m_midas_session;
        int m_input_width = 256;
        int m_input_height = 256;
    };

} // namespace trackie::vision