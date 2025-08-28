#include "function_handler.hpp"
#include <stdexcept>

namespace trackie::functions {

void FunctionHandler::registerFunction(const RegisteredFunction& func) {
    if (func.name.empty()) {
        // In a real application, you might throw an exception or log an error.
        return;
    }
    m_functions[func.name] = func;
}

json FunctionHandler::execute(const std::string& function_name, const json& args) {
    auto it = m_functions.find(function_name);
    if (it == m_functions.end()) {
        return {
            {"error", "Function not found"},
            {"function_name", function_name}
        };
    }

    // A production-ready implementation should perform JSON Schema validation here
    // to ensure the 'args' object matches the schema defined in it->second.parameters.
    // This is omitted for brevity but is a critical step.

    try {
        // Execute the stored std::function
        return it->second.function(args);
    } catch (const std::exception& e) {
        return {
            {"error", "Exception caught during function execution"},
            {"function_name", function_name},
            {"details", e.what()}
        };
    }
}

json FunctionHandler::getToolSchemas() const {
    if (m_functions.empty()) {
        return nullptr; // Return null if there are no tools.
    }

    json function_declarations = json::array();
    for (const auto& [name, func] : m_functions) {
        function_declarations.push_back({
            {"name", func.name},
            {"description", func.description},
            {"parameters", func.parameters}
        });
    }

    // The Gemini API expects the tool configuration to be wrapped in this structure.
    return json::array({
        {
            {"function_declarations", function_declarations}
        }
    });
}

} // namespace trackie::functions