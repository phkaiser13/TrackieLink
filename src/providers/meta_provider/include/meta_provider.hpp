#pragma once

#include "core/ai_core/ai_provider.hpp"
#include <string>

namespace trackie::providers {

/**
 * @class MetaProvider
 * @brief Implementação do AIProvider para as APIs da Meta (Llama).
 *
 * (Implementação de placeholder)
 */
class MetaProvider : public core::AIProvider {
public:
    explicit MetaProvider(std::string api_key);
    ~MetaProvider();

    core::AIResponse generate(const core::AIRequest& request) override;

private:
    std::string m_api_key;
};

} // namespace trackie::providers
