#ifndef CPP_INFERENCE_ENGINE_HPP
#define CPP_INFERENCE_ENGINE_HPP

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

// Forward declaration for vision types
struct DetectionResult;

namespace trackie {
namespace inference {

class InferenceEngine {
public:
    InferenceEngine(bool use_gpu_if_available = true);
    ~InferenceEngine();

    bool loadModel(const std::string& model_path);

    std::vector<float> runEmbedding(const std::vector<float>& input_data, const std::vector<int64_t>& input_dims);

    // Placeholder for YOLO
    std::vector<DetectionResult> runYolo(const std::vector<float>& input_data, const std::vector<int64_t>& input_dims);

private:
    Ort::Env m_env;
    Ort::Session m_session;
    Ort::AllocatorWithDefaultOptions m_allocator;

    std::string m_input_name;
    std::vector<std::string> m_output_names;
};

} // namespace inference
} // namespace trackie

#endif // CPP_INFERENCE_ENGINE_HPP
