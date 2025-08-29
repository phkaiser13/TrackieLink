#include "providers/openai_provider/include/openai_provider.hpp"

namespace trackie::providers {

OpenAIProvider::OpenAIProvider(std::string api_key) : m_api_key(std::move(api_key)) {}

OpenAIProvider::~OpenAIProvider() = default;

core::AIResponse OpenAIProvider::generate(const core::AIRequest& request) {
    // Placeholder implementation
    return core::AIResponse{"Response from OpenAI for prompt: " + request.prompt, true};
}

} // namespace trackie::providers
