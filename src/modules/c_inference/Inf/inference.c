/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root,
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "inference.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// --- Estruturas de Implementação (Ocultas do Cliente) ---
struct InferenceEngine {
    const OrtApi* ort_api;
    OrtEnv* env;
};

struct InferenceSession {
    OrtSession* ort_session;
    OrtAllocator* allocator;
    char* input_name;
    char* output_name;
};

// --- Macro Auxiliar para Tratamento de Erros ---
#define CHECK_ORT_STATUS(api, status) \
    if (status != NULL) { \
        const char* msg = api->GetErrorMessage(status); \
        fprintf(stderr, "ONNX Runtime error: %s\n", msg); \
        api->ReleaseStatus(status); \
        return -1; \
    }

// --- Implementação do Ciclo de Vida (sem alterações) ---
InferenceEngine* create_inference_engine() { /* ... como antes ... */ }
void destroy_inference_engine(InferenceEngine* engine) { /* ... como antes ... */ }
InferenceSession* load_inference_session(InferenceEngine* engine, const char* model_path) { /* ... como antes ... */ }
void destroy_inference_session(InferenceSession* session) { /* ... como antes ... */ }

// --- Implementação da Execução da Inferência (YOLO, sem alterações) ---
int run_yolo_inference(
    InferenceSession* session,
    const float* input_data,
    const int64_t* input_dims,
    size_t num_input_dims,
    DetectionResult** results,
    size_t* num_results
) {
    /* ... como antes ... */
}

void free_detection_results(DetectionResult* results) {
    if (results) {
        free(results);
    }
}

// --- <<<<<<< NOVA FUNÇÃO ADICIONADA >>>>>>> ---
// --- Implementação da Inferência de Embedding ---

int run_embedding_inference(
    InferenceSession* session,
    const float* input_data,
    const int64_t* input_dims,
    size_t num_input_dims,
    float** embedding_out,
    size_t* embedding_size
) {
    if (!session || !input_data || !input_dims || !embedding_out || !embedding_size) return -1;

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    *embedding_out = NULL;
    *embedding_size = 0;

    // 1. Criar o tensor de entrada (mesma lógica do YOLO)
    OrtMemoryInfo* memory_info;
    CHECK_ORT_STATUS(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    size_t input_tensor_size = 1;
    for(size_t i = 0; i < num_input_dims; ++i) input_tensor_size *= input_dims[i];

    OrtValue* input_tensor = NULL;
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        memory_info, (void*)input_data, input_tensor_size * sizeof(float),
        input_dims, num_input_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor
    );
    api->ReleaseMemoryInfo(memory_info);
    CHECK_ORT_STATUS(api, status);

    // 2. Executar o modelo (mesma lógica do YOLO)
    OrtValue* output_tensor = NULL;
    const char* input_names[] = { session->input_name };
    const char* output_names[] = { session->output_name };

    status = api->Run(session->ort_session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
    api->ReleaseValue(input_tensor);
    CHECK_ORT_STATUS(api, status);

    // 3. Processar o tensor de saída (lógica diferente)
    OrtTensorTypeAndShapeInfo* type_info;
    CHECK_ORT_STATUS(api, api->GetTensorTypeAndShapeInfo(output_tensor, &type_info));

    size_t num_output_dims;
    CHECK_ORT_STATUS(api, api->GetDimensionsCount(type_info, &num_output_dims));

    int64_t output_dims[num_output_dims];
    CHECK_ORT_STATUS(api, api->GetDimensions(type_info, output_dims, num_output_dims));

    api->ReleaseTensorTypeAndShapeInfo(type_info);

    // Calcula o tamanho total do tensor de saída (ex: 1 * 512 = 512)
    size_t output_tensor_size = 1;
    for(size_t i = 0; i < num_output_dims; ++i) output_tensor_size *= output_dims[i];

    float* output_data;
    CHECK_ORT_STATUS(api, api->GetTensorMutableData(output_tensor, (void**)&output_data));

    // 4. Alocar memória para o embedding de saída e copiar os dados
    *embedding_size = output_tensor_size;
    *embedding_out = (float*)malloc(output_tensor_size * sizeof(float));
    if (!*embedding_out) {
        fprintf(stderr, "Failed to allocate memory for output embedding\n");
        api->ReleaseValue(output_tensor);
        return -1;
    }
    memcpy(*embedding_out, output_data, output_tensor_size * sizeof(float));

    // 5. Liberar o tensor de saída do ONNX
    api->ReleaseValue(output_tensor);

    return 0; // Sucesso
}