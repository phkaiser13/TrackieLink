/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include "core/ai_core/ai_provider.hpp" // Inclui a nova interface
#include <string>
#include <functional>
#include <vector>
#include <variant>
#include <nlohmann/json.hpp>

namespace trackie::providers {

// Tipos de partes específicos para o Gemini
struct TextPart { std::string text; };
struct AudioPart {
    std::vector<uint8_t> audio_data;
    std::string mime_type = "audio/l16";
};

using GeminiRequestPart = std::variant<TextPart, AudioPart>;

/**
 * @class GeminiProvider
 * @brief Implementação do AIProvider para o Google Gemini.
 *
 * Adapta a comunicação com a API REST do Gemini para se conformar à
 * interface AIProvider, permitindo que a aplicação o utilize de forma agnóstica.
 */
class GeminiProvider : public core::AIProvider {
public:
    using StreamCallback = std::function<void(std::string_view chunk, bool is_error)>;

    explicit GeminiProvider(std::string api_key);
    ~GeminiProvider();

    /**
     * @brief Implementação do método `generate` da interface AIProvider.
     *
     * Este método faz uma chamada síncrona (bloqueante) para a API Gemini,
     * agregando a resposta de um stream internamente.
     */
    core::AIResponse generate(const core::AIRequest& request) override;

    // O método de streaming original pode ser mantido como parte da API pública
    // específica deste provedor, se necessário.
    void sendStreamingRequest(
        const std::string& model_id,
        const std::vector<GeminiRequestPart>& parts,
        const StreamCallback& on_data_received,
        const nlohmann::json& generation_config,
        const nlohmann::json& safety_settings
    ) const;

private:
    std::string m_api_key;
    static constexpr const char* GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/";
};

} // namespace trackie::providers