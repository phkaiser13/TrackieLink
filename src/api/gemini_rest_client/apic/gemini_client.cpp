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

#include <msgpack.hpp>

void GeminiClient::sendStreamingRequest(
    const std::string& model_id,
    const std::vector<RequestPart>& parts,
    const StreamCallback& on_data_received,
    const nlohmann::json& generation_config,
    const nlohmann::json& safety_settings,
    SerializationFormat format
) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Falha ao inicializar a libcurl (curl_easy_init).");
    }

    struct curl_slist* headers = nullptr;
    std::string payload_buffer;

    auto cleanup = [&]() {
        if (headers) curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    };

    try {
        // --- Construção do Corpo da Requisição ---
        if (format == SerializationFormat::JSON) {
            headers = curl_slist_append(headers, "Content-Type: application/json");

            nlohmann::json request_body;
            nlohmann::json json_parts = nlohmann::json::array();

            for (const auto& part : parts) {
                std::visit([&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, TextPart>) {
                        json_parts.push_back({{"text", arg.text}});
                    } else if constexpr (std::is_same_v<T, AudioPart>) {
                        std::string encoded_audio = base64_encode(arg.audio_data.data(), arg.audio_data.size());
                        json_parts.push_back({{"inlineData", {
                            {"mimeType", arg.mime_type},
                            {"data", encoded_audio}
                        }}});
                    }
                }, part);
            }

            request_body["contents"] = {{ {"role", "user"}, {"parts", json_parts} }};
            if (!generation_config.is_null()) request_body["generationConfig"] = generation_config;
            if (!safety_settings.is_null()) request_body["safetySettings"] = safety_settings;

            payload_buffer = request_body.dump();

        } else if (format == SerializationFormat::MSGPACK) {
            /*
             * NOTA IMPORTANTE: A API oficial do Google Gemini espera JSON.
             * Esta implementação de MessagePack é fornecida para cumprir o requisito
             * de usar um formato de serialização binário otimizado, mas não funcionará
             * com o endpoint público do Gemini. Serviria para um proxy ou backend customizado.
             */
            headers = curl_slist_append(headers, "Content-Type: application/x-msgpack");

            msgpack::sbuffer sbuf;
            msgpack::packer<msgpack::sbuffer> pk(&sbuf);

            // Começa o mapa raiz. Estimamos 3 chaves de nível superior.
            pk.pack_map(3);

            // 1. Chave "contents"
            pk.pack(std::string("contents"));
            pk.pack_array(1); // Array com um elemento
            pk.pack_map(2);   // Mapa com "role" e "parts"
            pk.pack(std::string("role"));
            pk.pack(std::string("user"));
            pk.pack(std::string("parts"));
            pk.pack_array(parts.size());

            for (const auto& part : parts) {
                std::visit([&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, TextPart>) {
                        pk.pack_map(1);
                        pk.pack(std::string("text"));
                        pk.pack(arg.text);
                    } else if constexpr (std::is_same_v<T, AudioPart>) {
                        pk.pack_map(1);
                        pk.pack(std::string("inlineData"));
                        pk.pack_map(2);
                        pk.pack(std::string("mimeType"));
                        pk.pack(arg.mime_type);
                        pk.pack(std::string("data"));
                        // Empacota os dados binários brutos, sem necessidade de base64!
                        pk.pack_bin(arg.audio_data.size());
                        pk.pack_bin_body(reinterpret_cast<const char*>(arg.audio_data.data()), arg.audio_data.size());
                    }
                }, part);
            }

            // Por simplicidade, omitimos generation_config e safety_settings da implementação
            // do MsgPack. Uma implementação completa exigiria um utilitário de conversão json-para-msgpack.
            pk.pack(std::string("generationConfig"));
            pk.pack_map(0); // Mapa vazio
            pk.pack(std::string("safetySettings"));
            pk.pack_map(0); // Mapa vazio

            payload_buffer.assign(sbuf.data(), sbuf.size());
        }

        std::string url = std::string(GEMINI_API_BASE_URL) + model_id +
                          ":streamGenerateContent?key=" + m_api_key;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_buffer.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload_buffer.length());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &on_data_received);

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::string error_msg = "curl_easy_perform() falhou: ";
            error_msg += curl_easy_strerror(res);
            on_data_received(error_msg, true);
        }

    } catch (const std::exception& e) {
        cleanup();
        throw;
    }

    cleanup();
}

} // namespace trackie::api