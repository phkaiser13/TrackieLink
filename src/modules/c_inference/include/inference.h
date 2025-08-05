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

// --- Guarda para compatibilidade C/C++ ---
// Isso permite que este cabeçalho seja incluído tanto por compiladores C quanto C++.
// O compilador C++ não fará "name mangling" nos nomes das funções.
#ifdef __cplusplus
extern "C" {
#endif

// --- Tipos Opacos (Opaque Pointers) ---
// Escondemos a implementação interna. O código cliente (C++) só conhecerá
// ponteiros para essas estruturas, não seu conteúdo.
typedef struct InferenceEngine InferenceEngine;
typedef struct InferenceSession InferenceSession;

// --- Estrutura de Dados para Resultados ---
// Define uma estrutura clara para os resultados de detecção do YOLO.
typedef struct {
    float x1, y1, x2, y2; // Coordenadas da caixa delimitadora (bounding box)
    int class_id;         // ID da classe detectada
    float confidence;     // Confiança da detecção
} DetectionResult;


// --- Ciclo de Vida do Motor de Inferência ---

/**
 * @brief Cria e inicializa o ambiente global do motor de inferência (ONNX Runtime).
 * @return Um ponteiro para o motor de inferência ou NULL em caso de falha.
 *         O chamador é responsável por destruir o objeto com destroy_inference_engine.
 */
InferenceEngine* create_inference_engine();

/**
 * @brief Libera todos os recursos associados ao motor de inferência.
 * @param engine O ponteiro para o motor a ser destruído.
 */
void destroy_inference_engine(InferenceEngine* engine);


// --- Ciclo de Vida da Sessão de Inferência ---

/**
 * @brief Carrega um modelo ONNX do disco e cria uma sessão de inferência.
 * @param engine O motor de inferência global.
 * @param model_path O caminho para o arquivo .onnx do modelo.
 * @return Um ponteiro para a sessão de inferência ou NULL em caso de falha.
 *         O chamador é responsável por destruir o objeto com destroy_inference_session.
 */
InferenceSession* load_inference_session(InferenceEngine* engine, const char* model_path);

/**
 * @brief Libera todos os recursos associados a uma sessão de inferência.
 * @param session O ponteiro para a sessão a ser destruída.
 */
void destroy_inference_session(InferenceSession* session);


// --- Execução da Inferência ---

/**
 * @brief Executa a inferência em um modelo YOLOv8.
 *
 * @param session A sessão de inferência carregada.
 * @param input_data Um ponteiro para os dados da imagem de entrada (float*),
 *                   normalizados e no formato esperado pelo modelo (ex: NCHW).
 * @param input_dims As dimensões dos dados de entrada (ex: {1, 3, 640, 640}).
 * @param num_input_dims O número de dimensões na entrada.
 * @param results Um ponteiro para um array de DetectionResult. A função alocará
 *                memória para este array. O chamador DEVE liberar esta memória
 *                usando a função free_detection_results.
 * @param num_results Um ponteiro para uma variável size_t que receberá o número
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


#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRACKIE_INFERENCE_H