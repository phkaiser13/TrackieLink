#pragma once

#include "core/ai_core/ai_provider.hpp"
#include <string>

namespace trackie::providers {

/**
 * @class OpenAIProvider
 * @brief Implementação do AIProvider para as APIs da OpenAI (GPT).
 *
 * (Implementação de placeholder)
 */
class OpenAIProvider : public core::AIProvider {
public:
    explicit OpenAIProvider(std::string api_key);
    ~OpenAIProvider();

    core::AIResponse generate(const core::AIRequest& request) override;

private:
    std::string m_api_key;
};

} // namespace trackie::providers
