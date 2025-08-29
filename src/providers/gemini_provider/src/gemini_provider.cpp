/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "providers/gemini_provider/include/gemini_provider.hpp"
#include "shared/utils/base64.h" // For encoding audio data
#include <curl/curl.h>
#include <stdexcept>
#include <iostream>
#include <variant>
#include <sstream>

namespace trackie::providers {

// --- cURL Callback Implementation ---

// Callback function for libcurl to handle incoming data from a stream
static size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    const size_t real_size = size * nmemb;
    auto* callback = static_cast<GeminiProvider::StreamCallback*>(userp);
    if (callback) {
        // The callback receives the data chunk and a boolean indicating if it's an error
        (*callback)(std::string_view(static_cast<char*>(contents), real_size), false);
    }
    return real_size;
}

// --- GeminiProvider Method Implementations ---

GeminiProvider::GeminiProvider(std::string api_key)
    : m_api_key(std::move(api_key)) {
    // Initialize libcurl globally. This is thread-safe since version 7.84.0.
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

GeminiProvider::~GeminiProvider() {
    // Cleanup libcurl resources
    curl_global_cleanup();
}

core::AIResponse GeminiProvider::generate(const core::AIRequest& request) {
    std::stringstream response_stream;
    bool has_error = false;
    std::string error_message;

    // Define the callback for the streaming request
    auto stream_callback = [&](std::string_view chunk, bool is_error) {
        if (is_error) {
            has_error = true;
            error_message.append(chunk);
        } else {
            response_stream << chunk;
        }
    };

    // For this simple generate call, we only handle the text part of the request
    std::vector<GeminiRequestPart> parts = {TextPart{request.prompt}};

    // Use default, empty JSON for config and safety settings for this basic implementation
    nlohmann::json empty_json;

    // Call the streaming function and let the lambda capture the output
    sendStreamingRequest("gemini-1.5-flash", parts, stream_callback, empty_json, empty_json);

    if (has_error) {
        // In case of an error during the stream, we can return it in the response
        return core::AIResponse{error_message, true};
    }

    // TODO: The raw response from Gemini is JSON. This should be parsed to extract
    // the actual text content instead of returning the full JSON.
    // For now, we return the concatenated JSON chunks.
    return core::AIResponse{response_stream.str(), true};
}

void GeminiProvider::sendStreamingRequest(
    const std::string& model_id,
    const std::vector<GeminiRequestPart>& parts,
    const StreamCallback& on_data_received,
    const nlohmann::json& generation_config,
    const nlohmann::json& safety_settings
) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl (curl_easy_init).");
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Auto-cleanup resources
    auto cleanup = [&]() {
        if (headers) curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    };

    try {
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

        const std::string payload_buffer = request_body.dump();
        const std::string url = std::string(GEMINI_API_BASE_URL) + model_id +
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
            std::string error_msg = "curl_easy_perform() failed: ";
            error_msg += curl_easy_strerror(res);
            on_data_received(error_msg, true);
        }

    } catch (const std::exception& e) {
        cleanup();
        throw; // Re-throw after cleanup
    }

    cleanup();
}

} // namespace trackie::providers
