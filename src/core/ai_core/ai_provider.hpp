#pragma once

#include <string>
#include <vector>
#include <any>

namespace trackie {
namespace core {

// A generic structure for AI model responses
struct AIResponse {
    std::string content;
    bool is_final;
    // Can be extended with more metadata, e.g., token usage, error info
};

// A generic structure for AI model requests
struct AIRequest {
    std::string prompt;
    // Can be extended with model parameters, e.g., temperature, max_tokens
};

/**
 * @class AIProvider
 * @brief An abstract interface for interacting with various AI language models.
 *
 * This class defines a common contract for all AI providers, ensuring that
 * the core application can interact with them in a uniform way.
 */
class AIProvider {
public:
    virtual ~AIProvider() = default;

    /**
     * @brief Sends a request to the AI model and gets a response.
     * @param request The AIRequest object containing the prompt and other parameters.
     * @return An AIResponse object with the model's output.
     */
    virtual AIResponse generate(const AIRequest& request) = 0;

    // In the future, a streaming version could be added:
    // virtual void generateStream(const AIRequest& request,
    //                           std::function<void(const AIResponse&)> callback) = 0;
};

} // namespace core
} // namespace trackie
