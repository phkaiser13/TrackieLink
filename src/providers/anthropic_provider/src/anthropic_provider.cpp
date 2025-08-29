#include "providers/anthropic_provider/include/anthropic_provider.hpp"

namespace trackie::providers {

AnthropicProvider::AnthropicProvider(std::string api_key) : m_api_key(std::move(api_key)) {}

AnthropicProvider::~AnthropicProvider() = default;

core::AIResponse AnthropicProvider::generate(const core::AIRequest& request) {
    // Placeholder implementation
    return core::AIResponse{"Response from Anthropic for prompt: " + request.prompt, true};
}

} // namespace trackie::providers
