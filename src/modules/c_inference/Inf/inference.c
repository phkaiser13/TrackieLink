/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
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
// Esta macro verifica o status retornado por uma chamada da API ONNX.
// Se houver um erro, imprime a mensagem e retorna um código de erro.
#define CHECK_ORT_STATUS(api, status) \
    if (status != NULL) { \
        const char* msg = api->GetErrorMessage(status); \
        fprintf(stderr, "ONNX Runtime error: %s\n", msg); \
        api->ReleaseStatus(status); \
        return -1; \
    }

// --- Implementação do Ciclo de Vida do Motor ---

InferenceEngine* create_inference_engine() {
    InferenceEngine* engine = (InferenceEngine*)malloc(sizeof(InferenceEngine));
    if (!engine) {
        fprintf(stderr, "Failed to allocate memory for InferenceEngine\n");
        return NULL;
    }

    engine->ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!engine->ort_api) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        free(engine);
        return NULL;
    }

    OrtStatus* status = engine->ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Trackie-Inference-Engine", &engine->env);
    if (status != NULL) {
        fprintf(stderr, "Failed to create ONNX Runtime environment\n");
        engine->ort_api->ReleaseStatus(status);
        free(engine);
        return NULL;
    }

    return engine;
}

void destroy_inference_engine(InferenceEngine* engine) {
    if (engine) {
        if (engine->env) {
            engine->ort_api->ReleaseEnv(engine->env);
        }
        free(engine);
    }
}

// --- Implementação do Ciclo de Vida da Sessão ---

InferenceSession* load_inference_session(InferenceEngine* engine, const char* model_path) {
    if (!engine || !model_path) return NULL;

    InferenceSession* session = (InferenceSession*)malloc(sizeof(InferenceSession));
    if (!session) {
        fprintf(stderr, "Failed to allocate memory for InferenceSession\n");
        return NULL;
    }
    memset(session, 0, sizeof(InferenceSession)); // Zera a estrutura

    OrtSessionOptions* session_options;
    CHECK_ORT_STATUS(engine->ort_api, engine->ort_api->CreateSessionOptions(&session_options));

    // Para melhor performance, pode-se habilitar otimizações aqui.
    // engine->ort_api->SetIntraOpNumThreads(session_options, 1);
    // engine->ort_api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);

    OrtStatus* status = engine->ort_api->CreateSession(engine->env, model_path, session_options, &session->ort_session);

    engine->ort_api->ReleaseSessionOptions(session_options); // Libera as opções após o uso

    if (status != NULL) {
        const char* msg = engine->ort_api->GetErrorMessage(status);
        fprintf(stderr, "Failed to create ONNX session for model %s: %s\n", model_path, msg);
        engine->ort_api->ReleaseStatus(status);
        free(session);
        return NULL;
    }

    // Obter o alocador padrão
    CHECK_ORT_STATUS(engine->ort_api, engine->ort_api->GetAllocatorWithDefaultOptions(&session->allocator));

    // Obter nome do nó de entrada (assumindo 1 entrada)
    CHECK_ORT_STATUS(engine->ort_api, engine->ort_api->SessionGetInputName(session->ort_session, 0, session->allocator, &session->input_name));

    // Obter nome do nó de saída (assumindo 1 saída)
    CHECK_ORT_STATUS(engine->ort_api, engine->ort_api->SessionGetOutputName(session->ort_session, 0, session->allocator, &session->output_name));

    return session;
}

void destroy_inference_session(InferenceSession* session) {
    if (session) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (session->allocator) {
            if (session->input_name) session->allocator->Free(session->allocator, session->input_name);
            if (session->output_name) session->allocator->Free(session->allocator, session->output_name);
        }
        if (session->ort_session) {
            api->ReleaseSession(session->ort_session);
        }
        free(session);
    }
}

// --- Implementação da Execução da Inferência ---

int run_yolo_inference(
    InferenceSession* session,
    const float* input_data,
    const int64_t* input_dims,
    size_t num_input_dims,
    DetectionResult** results,
    size_t* num_results
) {
    if (!session || !input_data || !input_dims || !results || !num_results) return -1;

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    *results = NULL;
    *num_results = 0;

    OrtMemoryInfo* memory_info;
    CHECK_ORT_STATUS(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* input_tensor = NULL;
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        memory_info, (void*)input_data, input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3] * sizeof(float),
        input_dims, num_input_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor
    );
    api->ReleaseMemoryInfo(memory_info);
    CHECK_ORT_STATUS(api, status);

    OrtValue* output_tensor = NULL;
    const char* input_names[] = { session->input_name };
    const char* output_names[] = { session->output_name };

    status = api->Run(session->ort_session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
    api->ReleaseValue(input_tensor);
    CHECK_ORT_STATUS(api, status);

    // Processar o tensor de saída
    OrtTensorTypeAndShapeInfo* type_info;
    CHECK_ORT_STATUS(api, api->GetTensorTypeAndShapeInfo(output_tensor, &type_info));

    size_t num_output_dims;
    CHECK_ORT_STATUS(api, api->GetDimensionsCount(type_info, &num_output_dims));

    int64_t output_dims[num_output_dims];
    CHECK_ORT_STATUS(api, api->GetDimensions(type_info, output_dims, num_output_dims));

    api->ReleaseTensorTypeAndShapeInfo(type_info);

    float* output_data;
    CHECK_ORT_STATUS(api, api->GetTensorMutableData(output_tensor, (void**)&output_data));

    // A saída do YOLOv8 é [batch, 4+num_classes, num_detections], ex: [1, 84, 8400]
    // Os dados são transpostos.
    int num_classes = output_dims[1] - 4;
    int num_detections = output_dims[2];

    // Lista temporária para armazenar resultados válidos
    DetectionResult* temp_results = (DetectionResult*)malloc(num_detections * sizeof(DetectionResult));
    if (!temp_results) { api->ReleaseValue(output_tensor); return -1; }
    size_t valid_detections = 0;

    for (int i = 0; i < num_detections; ++i) {
        // Encontrar a classe com a maior pontuação para esta detecção
        float max_score = 0.0f;
        int class_id = -1;
        for (int j = 0; j < num_classes; ++j) {
            // Acessar o score: output_data[classe_offset + detection_index]
            float score = output_data[(4 + j) * num_detections + i];
            if (score > max_score) {
                max_score = score;
                class_id = j;
            }
        }

        // Usar um limiar de confiança (hard-coded por simplicidade, idealmente viria da config)
        if (max_score > 0.40f) {
            float cx = output_data[0 * num_detections + i];
            float cy = output_data[1 * num_detections + i];
            float w = output_data[2 * num_detections + i];
            float h = output_data[3 * num_detections + i];

            temp_results[valid_detections].x1 = cx - w / 2;
            temp_results[valid_detections].y1 = cy - h / 2;
            temp_results[valid_detections].x2 = cx + w / 2;
            temp_results[valid_detections].y2 = cy + h / 2;
            temp_results[valid_detections].class_id = class_id;
            temp_results[valid_detections].confidence = max_score;
            valid_detections++;
        }
    }

    api->ReleaseValue(output_tensor);

    // Alocar o buffer final com o tamanho exato e copiar os resultados
    if (valid_detections > 0) {
        *results = (DetectionResult*)malloc(valid_detections * sizeof(DetectionResult));
        if (!*results) { free(temp_results); return -1; }
        memcpy(*results, temp_results, valid_detections * sizeof(DetectionResult));
        *num_results = valid_detections;
    }

    free(temp_results);
    return 0; // Sucesso
}

void free_detection_results(DetectionResult* results) {
    if (results) {
        free(results);
    }
}