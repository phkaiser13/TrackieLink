/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "gemini_client.hpp"
#include <curl/curl.h>
#include <stdexcept>
#include <iostream>
#include <variant> // Para std::visit

// Adicionar a inclusão da biblioteca base64 (supõe que foi adicionada ao projeto)
#include "base64.h" // Supõe-se a existência de uma biblioteca/função para Base64

namespace trackie::api {

// --- Função de Callback para a libcurl ---

/**
 * @brief Função C-style que a libcurl invocará para cada pedaço de dados recebido.
 * (Implementação sem alterações)
 */
static size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    const size_t real_size = size * nmemb;
    auto* callback = static_cast<GeminiClient::StreamCallback*>(userp);
    if (callback) {
        (*callback)(std::string_view(static_cast<char*>(contents), real_size), false);
    }
    return real_size;
}


// --- Implementação dos Métodos da Classe GeminiClient ---

GeminiClient::GeminiClient(std::string api_key)
    : m_api_key(std::move(api_key)) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

void GeminiClient::sendStreamingRequest(
    const std::string& model_id,
    const std::vector<RequestPart>& parts,
    const StreamCallback& on_data_received,
    const nlohmann::json& generation_config,
    const nlohmann::json& safety_settings
) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Falha ao inicializar a libcurl (curl_easy_init).");
    }

    struct curl_slist* headers = nullptr;
    auto cleanup = [&]() {
        if (headers) curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    };

    try {
        // --- Construção do Corpo da Requisição ---
        nlohmann::json request_body;
        nlohmann::json json_parts = nlohmann::json::array();

        for (const auto& part : parts) {
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, TextPart>) {
                    json_parts.push_back({{"text", arg.text}});
                } else if constexpr (std::is_same_v<T, AudioPart>) {
                    // Áudio precisa ser codificado em Base64 para ser enviado em JSON
                    std::string encoded_audio = base64_encode(arg.audio_data.data(), arg.audio_data.size());
                    json_parts.push_back({{"inlineData", {
                        {"mimeType", arg.mime_type},
                        {"data", encoded_audio}
                    }}});
                }
            }, part);
        }

        // Monta a estrutura principal do JSON
        request_body["contents"] = {{ {"role", "user"}, {"parts", json_parts} }};

        // Adiciona configurações opcionais se não estiverem vazias
        if (!generation_config.is_null()) {
            request_body["generationConfig"] = generation_config;
        }
        if (!safety_settings.is_null()) {
            request_body["safetySettings"] = safety_settings;
        }

        // A URL para requisições de texto usa o endpoint :streamGenerateContent
        std::string url = std::string(GEMINI_API_BASE_URL) + model_id +
                          ":streamGenerateContent?key=" + m_api_key;

        std::string json_payload = request_body.dump();

        // --- Envio da Requisição ---
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &on_data_received);

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::string error_msg = "curl_easy_perform() falhou: ";
            error_msg += curl_easy_strerror(res);
            // Invoca o callback com a mensagem de erro
            on_data_received(error_msg, true);
        }

    } catch (const std::exception& e) {
        cleanup();
        // Re-lança a exceção para que o chamador possa tratá-la
        throw;
    }

    // Garante a limpeza dos recursos da libcurl
    cleanup();
}

} // namespace trackie::api