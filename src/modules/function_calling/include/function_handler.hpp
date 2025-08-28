#pragma once

#include <string>
#include <functional>
#include <nlohmann/json.hpp>
#include <map>
#include <memory>

namespace trackie::functions {

using json = nlohmann::json;

// A generic function that takes JSON arguments and returns a JSON result.
// This provides a universal signature for all callable functions.
using GenericFunction = std::function<json(const json&)>;

// Represents a function that can be called by the AI model.
// It contains all the metadata needed to describe the function to the AI.
struct RegisteredFunction {
    std::string name;
    std::string description;
    json parameters; // JSON Schema object describing the function's parameters.
    GenericFunction function;
};

/**
 * @class FunctionHandler
 * @brief A generic, extensible handler for AI function calling (tools).
 *
 * This class manages a registry of functions that can be executed by the AI.
 * It provides methods to register new functions at runtime and to execute them
 * based on name and arguments provided by the model. It can also generate
 * the tool schemas required by the AI API.
 */
class FunctionHandler {
public:
    FunctionHandler() = default;

    /**
     * @brief Registers a new function, making it available for the AI to call.
     * @param func The function to register.
     */
    void registerFunction(const RegisteredFunction& func);

    /**
     * @brief Executes a function by name with the given arguments.
     * @param function_name The name of the function to execute.
     * @param args A JSON object containing the arguments.
     * @return A JSON object with the result of the function call.
     */
    json execute(const std::string& function_name, const json& args);

    /**
     * @brief Gets the JSON schemas for all registered tools.
     * @return A JSON array formatted for the 'tools' field of the Gemini API.
     */
    json getToolSchemas() const;

private:
    // The registry of all functions, keyed by their unique name.
    std::map<std::string, RegisteredFunction> m_functions;
};

} // namespace trackie::functions