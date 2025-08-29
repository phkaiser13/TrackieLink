#include "inference_engine.hpp"
#include "vision_types.h" // For DetectionResult
#include <iostream>
#include <vector>

namespace trackie {
namespace inference {

InferenceEngine::InferenceEngine(bool use_gpu_if_available)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "TrackieCppInference"),
      m_session(nullptr) {

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_gpu_if_available) {
#if defined(WITH_CUDA)
        std::cout << "INFO: Attempting to use CUDA Execution Provider." << std::endl;
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
#elif defined(WITH_ROCM)
        std::cout << "INFO: Attempting to use ROCm Execution Provider." << std::endl;
        OrtROCMProviderOptions rocm_options{};
        session_options.AppendExecutionProvider_ROCM(rocm_options);
#else
        std::cout << "INFO: No GPU provider compiled. Using CPU." << std::endl;
#endif
    }

    // The C++ API doesn't have a direct way to create the session in the constructor
    // because the model path isn't known yet. We will create it in loadModel.
    // We store the configured options for later.
    // For simplicity in this example, we will re-create options in loadModel.
}

InferenceEngine::~InferenceEngine() {}

bool InferenceEngine::loadModel(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
#if defined(WITH_CUDA)
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif
        m_session = Ort::Session(m_env, model_path.c_str(), session_options);

        // Get input and output names
        Ort::AllocatedStringPtr input_name_ptr = m_session.GetInputNameAllocated(0, m_allocator);
        m_input_name = input_name_ptr.get();

        size_t num_output_nodes = m_session.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr output_name_ptr = m_session.GetOutputNameAllocated(i, m_allocator);
            m_output_names.push_back(output_name_ptr.get());
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "ERROR loading model: " << e.what() << std::endl;
        return false;
    }
    return true;
}

std::vector<float> InferenceEngine::runEmbedding(const std::vector<float>& input_data, const std::vector<int64_t>& input_dims) {
    std::vector<float> results;
    if (!m_session) {
        return results;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_dims.data(),
        input_dims.size()
    );

    std::vector<const char*> input_names = {m_input_name.c_str()};
    std::vector<const char*> output_names;
    for(const auto& name : m_output_names) {
        output_names.push_back(name.c_str());
    }

    try {
        auto output_tensors = m_session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        results.assign(output_data, output_data + output_size);

    } catch (const Ort::Exception& e) {
        std::cerr << "ERROR running embedding inference: " << e.what() << std::endl;
    }

    return results;
}

std::vector<DetectionResult> InferenceEngine::runYolo(const std::vector<float>& input_data, const std::vector<int64_t>& input_dims) {
    std::cout << "WARNING: runYolo is a placeholder and does not produce real detections." << std::endl;
    return {};
}

} // namespace inference
} // namespace trackie
