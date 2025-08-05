/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#ifndef TRACKIE_COMMON_TYPES_H
#define TRACKIE_COMMON_TYPES_H

#include <stddef.h> // Para size_t

// Este arquivo define estruturas de dados que são usadas em múltiplos
// módulos, incluindo a interface do nosso motor de inferência em C.
// Mantê-lo separado garante que não tenhamos dependências circulares
// e que tenhamos uma única fonte de verdade para nossos tipos de dados.

/**
 * @struct DetectionResult
 * @brief Representa uma única detecção de objeto de um modelo como o YOLO.
 *
 * Esta estrutura é preenchida pelo nosso módulo 'c_inference' e consumida
 * pelo 'cpp_core'. É uma estrutura C pura para máxima compatibilidade.
 */
typedef struct {
    float x1, y1, x2, y2; // Coordenadas da caixa delimitadora (bounding box)
    int class_id;         // ID da classe detectada
    float confidence;     // Confiança da detecção (0.0 a 1.0)
} DetectionResult;


// Poderíamos adicionar outras estruturas compartilhadas aqui no futuro.
// Por exemplo, uma estrutura para representar um embedding facial:
/*
typedef struct {
    float embedding[512]; // Tamanho fixo para um modelo específico
} FaceEmbedding;
*/

#endif // TRACKIE_COMMON_TYPES_H