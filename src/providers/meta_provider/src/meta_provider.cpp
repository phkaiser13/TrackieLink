#include "providers/meta_provider/include/meta_provider.hpp"

namespace trackie::providers {

MetaProvider::MetaProvider(std::string api_key) : m_api_key(std::move(api_key)) {}

MetaProvider::~MetaProvider() = default;

core::AIResponse MetaProvider::generate(const core::AIRequest& request) {
    // Placeholder implementation
    return core::AIResponse{"Response from Meta for prompt: " + request.prompt, true};
}

} // namespace trackie::providers
