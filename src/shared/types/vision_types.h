/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#ifndef TRACKIE_VISION_TYPES_H
#define TRACKIE_VISION_TYPES_H

#include <stddef.h>

// Estrutura para um resultado de detecção genérico (usado por YOLO e detectores de rosto)
typedef struct {
    float x1, y1, x2, y2; // Coordenadas da caixa delimitadora (bounding box)
    int class_id;
    float confidence;
} DetectionResult;

// Estrutura para um resultado de reconhecimento facial
typedef struct {
    char name[64]; // Nome da pessoa identificada (tamanho fixo)
    float similarity; // Similaridade da correspondência
} Identity;

#endif // TRACKIE_VISION_TYPES_H