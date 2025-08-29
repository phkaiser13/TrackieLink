#pragma once

#include "core/ai_core/ai_provider.hpp"
#include <string>

namespace trackie::providers {

/**
 * @class AnthropicProvider
 * @brief Implementação do AIProvider para as APIs da Anthropic (Claude).
 *
 * (Implementação de placeholder)
 */
class AnthropicProvider : public core::AIProvider {
public:
    explicit AnthropicProvider(std::string api_key);
    ~AnthropicProvider();

    core::AIResponse generate(const core::AIRequest& request) override;

private:
    std::string m_api_key;
};

} // namespace trackie::providers
