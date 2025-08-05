/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include <string>
#include <functional>
#include <string_view>
#include <vector>
#include <variant>

// --- Dependência Externa ---
// nlohmann/json é usado na implementação, mas não exposto na interface pública.

namespace trackie::api {

// Definimos os tipos de partes que podemos enviar
struct TextPart { std::string text; };
struct AudioPart {
    std::vector<uint8_t> audio_data;
    std::string mime_type = "audio/l16"; // PCM Linear 16-bit
};

using RequestPart = std::variant<TextPart, AudioPart>;

/**
 * @class GeminiClient
 * @brief Gerencia a comunicação com a API REST do Google Gemini.
 *
 * Esta classe fornece uma interface de alto nível para enviar requisições
 * para a API Gemini. Foi refatorada para usar uma única função `send`
 * que pode lidar com diferentes tipos de conteúdo (partes) como texto e áudio.
 */
class GeminiClient {
public:
    /**
     * @brief Callback para receber pedaços (chunks) de dados da resposta em streaming.
     * @param chunk Um pedaço de dados recebido do servidor.
     * @param is_error Sinaliza se o chunk representa uma mensagem de erro.
     */
    using StreamCallback = std::function<void(std::string_view chunk, bool is_error)>;

    /**
     * @brief Construtor do GeminiClient.
     * @param api_key A chave da API do Google Gemini.
     */
    explicit GeminiClient(std::string api_key);

    /**
     * @brief Destrutor do GeminiClient.
     */
    ~GeminiClient();

    /**
     * @brief Envia uma ou mais partes para o endpoint de streaming do Gemini.
     *
     * Constrói uma requisição JSON com as partes fornecidas. Esta função é
     * versátil e pode ser usada para enviar apenas texto, apenas áudio, ou
     * uma combinação de ambos em uma única requisição multimodal.
     *
     * @param model_id O identificador do modelo a ser usado (ex: "gemini-1.5-flash").
     * @param parts Um vetor de `RequestPart` contendo os dados a serem enviados.
     * @param on_data_received O callback que será invocado para cada pedaço de dados da resposta.
     */
    void sendStreamingRequest(
        const std::string& model_id,
        const std::vector<RequestPart>& parts,
        const StreamCallback& on_data_received
    ) const;

private:
    // A chave da API para autenticação.
    std::string m_api_key;

    // A URL base da API do Gemini.
    static constexpr const char* GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/";
};

} // namespace trackie::api