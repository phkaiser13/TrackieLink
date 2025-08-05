/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#ifndef TRACKIE_INFERENCE_H
#define TRACKIE_INFERENCE_H

#include <stddef.h> // Para size_t
#include <stdint.h> // <<<<<<< CORREÇÃO: Incluir para int64_t

// --- Guarda para compatibilidade C/C++ ---
#ifdef __cplusplus
extern "C" {
#endif

// --- Tipos Compartilhados ---
// Incluímos os tipos de visão que esta API usa.
// Agora, DetectionResult vem deste arquivo.
#include "vision_types.h" // <<<<<<< CORREÇÃO: Incluir a fonte única dos tipos

// --- Tipos Opacos (Opaque Pointers) ---
typedef struct InferenceEngine InferenceEngine;
typedef struct InferenceSession InferenceSession;

// --- Estrutura de Dados para Resultados ---
// typedef struct { ... } DetectionResult; // <<<<<<< CORREÇÃO: REMOVIDO DESTE ARQUIVO

// --- Ciclo de Vida do Motor de Inferência ---
InferenceEngine* create_inference_engine();
void destroy_inference_engine(InferenceEngine* engine);

// --- Ciclo de Vida da Sessão de Inferência ---
InferenceSession* load_inference_session(InferenceEngine* engine, const char* model_path);
void destroy_inference_session(InferenceSession* session);

// --- Execução da Inferência ---

/**
 * @brief Executa a inferência em um modelo de detecção (como YOLO).
 *
 * @param session A sessão de inferência carregada.
 * @param input_data Ponteiro para os dados da imagem de entrada (float*).
 * @param input_dims As dimensões dos dados de entrada (ex: {1, 3, 640, 640}).
 * @param num_input_dims O número de dimensões na entrada.
 * @param results Ponteiro para um array de DetectionResult. A função alocará
 *                memória para este array. O chamador DEVE liberar esta memória
 *                usando a função free_detection_results.
 * @param num_results Ponteiro para uma variável size_t que receberá o número
 *                    de detecções encontradas.
 * @return 0 em caso de sucesso, -1 em caso de falha.
 */
int run_yolo_inference(
    InferenceSession* session,
    const float* input_data,
    const int64_t* input_dims,
    size_t num_input_dims,
    DetectionResult** results,
    size_t* num_results
);

/**
 * @brief Libera a memória alocada para o array de resultados da detecção.
 * @param results O ponteiro para o array de resultados a ser liberado.
 */
void free_detection_results(DetectionResult* results);

/**
 * @brief Executa a inferência em um modelo que retorna um único vetor de embedding (ex: ArcFace).
 *
 * @param session A sessão de inferência carregada.
 * @param input_data Ponteiro para os dados da imagem de entrada (float*).
 * @param input_dims As dimensões dos dados de entrada (ex: {1, 3, 112, 112}).
 * @param num_input_dims O número de dimensões na entrada.
 * @param embedding_out Ponteiro para um array de float. A função alocará memória
 *                      para este array. O chamador DEVE liberar esta memória com free().
 * @param embedding_size Ponteiro para uma variável size_t que receberá o tamanho do embedding.
 * @return 0 em caso de sucesso, -1 em caso de falha.
 */
int run_embedding_inference(
    InferenceSession* session,
    const float* input_data,
    const int64_t* input_dims,
    size_t num_input_dims,
    float** embedding_out,
    size_t* embedding_size
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRACKIE_INFERENCE_H