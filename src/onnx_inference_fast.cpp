#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "tokenizers/tokenizers.h"
#include <nlohmann/json.hpp>
#include <unordered_map>

using json = nlohmann::json;

// Constants
const std::vector<int64_t> DEFAULT_STOP_TOKENS = {151643, 151645};
const int DEFAULT_MAX_SEQ_LEN = 4096;

class FastOnnxInference {
private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    
    // GPU memory info - reuse across iterations
    Ort::MemoryInfo cuda_memory_info_;
    
    // Model parameters
    int num_key_value_heads_;
    int head_dim_;
    int num_layers_;
    bool use_cuda_;
    
    // Pre-allocated persistent data
    std::vector<int32_t> current_tokens_;
    std::vector<int64_t> history_len_data_;
    std::vector<int64_t> ids_len_data_;
    std::vector<int8_t> attention_mask_data_;
    
    // Pre-built token vocabulary for fast decoding
    std::vector<std::string> id_to_token_;
    
    void build_vocab_map(const std::string& model_dir) {
        std::string tokenizer_json_path = model_dir + "/tokenizer.json";
        std::ifstream f(tokenizer_json_path);
        if (!f.is_open()) {
            std::cerr << "Warning: Could not open tokenizer.json for vocab" << std::endl;
            return;
        }
        
        json j;
        f >> j;
        
        if (j.contains("model") && j["model"].contains("vocab")) {
            auto vocab = j["model"]["vocab"];
            int max_id = -1;
            for (auto it = vocab.begin(); it != vocab.end(); ++it) {
                int id = it.value().get<int>();
                if (id > max_id) max_id = id;
            }
            id_to_token_.assign(max_id + 1, std::string());
            for (auto it = vocab.begin(); it != vocab.end(); ++it) {
                const std::string token = it.key();
                int id = it.value().get<int>();
                if (id >= 0 && id < (int)id_to_token_.size()) id_to_token_[id] = token;
            }
        }
        
        if (j.contains("added_tokens")) {
            for (const auto &t : j["added_tokens"]) {
                if (t.contains("id") && t.contains("content")) {
                    int id = t["id"].get<int>();
                    std::string content = t["content"].get<std::string>();
                    if (id >= 0) {
                        if (id >= (int)id_to_token_.size()) id_to_token_.resize(id + 1);
                        id_to_token_[id] = content;
                    }
                }
            }
        }
        
        std::cout << "Loaded vocab with " << id_to_token_.size() << " tokens" << std::endl;
    }
    
    json load_json(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + path);
        }
        json j;
        file >> j;
        return j;
    }

    std::unique_ptr<Ort::Session> build_session(const std::string& onnx_path) {
        Ort::SessionOptions session_opts;
        
        // Minimal logging
        session_opts.SetLogSeverityLevel(4);

        // Aggressive optimization
        session_opts.SetInterOpNumThreads(0);
        session_opts.SetIntraOpNumThreads(0);
        session_opts.EnableCpuMemArena();
        session_opts.SetExecutionMode(ORT_SEQUENTIAL);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // GPU provider with optimizations
        use_cuda_ = false;
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        if (std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") 
            != available_providers.end()) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                // Performance optimizations
                cuda_options.arena_extend_strategy = 1;  // Extend by doubling
                cuda_options.gpu_mem_limit = SIZE_MAX;   // No limit
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
                cuda_options.do_copy_in_default_stream = 1;
                
                session_opts.AppendExecutionProvider_CUDA(cuda_options);
                use_cuda_ = true;
                std::cout << "Using optimized CUDA execution provider" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "CUDA provider failed: " << e.what() << std::endl;
                use_cuda_ = false;
            }
        }
        
        if (!use_cuda_) {
            std::cout << "Using CPU execution provider" << std::endl;
        }

        return std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_opts);
    }

    void precompute_token_strings() {
        // We'll build this from the actual tokenizer.json vocab instead
        // For now, just create a simple fallback
        std::cout << "Loading token vocabulary..." << std::endl;
    }

public:
    FastOnnxInference(const std::string& model_dir, 
                     const std::string& onnx_path,
                     const std::string& config_path) 
        : env_(ORT_LOGGING_LEVEL_ERROR, "fast_onnx_inference"),
          cuda_memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)) {
        
        // Load model config
        auto config = load_json(config_path);
        num_key_value_heads_ = config["num_key_value_heads"];
        head_dim_ = config["head_dim"];
        num_layers_ = config["num_hidden_layers"];

        // Initialize tokenizer
        tokenizer_ = std::make_unique<tokenizers::Tokenizer>(model_dir + "/tokenizer.json");
        if (!tokenizer_->valid()) {
            throw std::runtime_error("Failed to load tokenizer from: " + model_dir + "/tokenizer.json");
        }

        // Build ONNX session
        session_ = build_session(onnx_path);
        
        // Build vocabulary map for fast token decoding
        build_vocab_map(model_dir);
        
        // Pre-allocate memory for persistent data
        current_tokens_.reserve(4096);
        history_len_data_.resize(1);
        ids_len_data_.resize(1);
        attention_mask_data_.resize(1);
        
        // Pre-compute token strings
        precompute_token_strings();
        
        // Create GPU memory info after CUDA is initialized  
        if (use_cuda_) {
            // For GPU, we'll create tensors on CPU and let ONNX Runtime handle GPU transfer
            cuda_memory_info_ = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        }
    }

    void run_inference(const std::string& prompt, int max_decode = 64, bool short_answer = false) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Encode tokens and setup initial state
        auto ids_vec = tokenizer_->encode(prompt, true);
        current_tokens_.assign(ids_vec.begin(), ids_vec.end());
        
        history_len_data_[0] = 0;
        ids_len_data_[0] = static_cast<int64_t>(current_tokens_.size());
        attention_mask_data_[0] = 1;
        
        // Use CPU memory info - ONNX Runtime will handle GPU transfers automatically
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // Get input/output names (cached for performance)
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        static std::vector<std::string> input_names_storage;
        static std::vector<std::string> output_names_storage;
        static std::vector<const char*> input_names;
        static std::vector<const char*> output_names;
        
        if (input_names.empty()) {
            input_names_storage.reserve(num_input_nodes);
            output_names_storage.reserve(num_output_nodes);
            input_names.reserve(num_input_nodes);
            output_names.reserve(num_output_nodes);
            
            for (size_t i = 0; i < num_input_nodes; i++) {
                auto name_ptr = session_->GetInputNameAllocated(i, allocator);
                input_names_storage.emplace_back(name_ptr.get());
                input_names.push_back(input_names_storage.back().c_str());
            }
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
                output_names_storage.emplace_back(name_ptr.get());
                output_names.push_back(output_names_storage.back().c_str());
            }
        }

        // Pre-allocate input tensors vector
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(num_input_nodes);

        // Create empty KV cache tensors (GPU memory)
        std::vector<float> empty_data;
        std::vector<int64_t> past_keys_shape = {num_key_value_heads_, 1, head_dim_, 0};
        std::vector<int64_t> past_values_shape = {num_key_value_heads_, 1, 0, head_dim_};
        
        for (int i = 0; i < num_layers_; i++) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, empty_data.data(), 0,
                past_keys_shape.data(), past_keys_shape.size()));
        }
        for (int i = 0; i < num_layers_; i++) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, empty_data.data(), 0,
                past_values_shape.data(), past_values_shape.size()));
        }

        // Add remaining inputs
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(current_tokens_.size())};
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info, current_tokens_.data(), current_tokens_.size(), 
            input_shape.data(), input_shape.size()));

        std::vector<int64_t> scalar_shape = {1};
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, history_len_data_.data(), 1, scalar_shape.data(), scalar_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, ids_len_data_.data(), 1, scalar_shape.data(), scalar_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
            memory_info, attention_mask_data_.data(), 1, scalar_shape.data(), scalar_shape.size()));

        // Decode loop
        int num_decoded = 0;
        std::string decoded_text;
        decoded_text.reserve(max_decode * 4);  // Pre-allocate string
        
        auto first_token_time = std::chrono::high_resolution_clock::now();
        
        while (num_decoded < max_decode) {
            // Run inference
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names.data(), input_tensors.data(), input_tensors.size(),
                output_names.data(), output_names.size());

            // Get token from output
            auto& token_tensor = output_tensors[output_tensors.size() - 2];
            auto type_info = token_tensor.GetTensorTypeAndShapeInfo().GetElementType();
            
            int64_t token_id = 0;
            if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                token_id = *token_tensor.GetTensorData<int64_t>();
            } else if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                token_id = static_cast<int64_t>(*token_tensor.GetTensorData<int32_t>());
            } else {
                std::cerr << "Unexpected token data type" << std::endl;
                break;
            }
            
            num_decoded++;

            // Check stop token
            if (std::find(DEFAULT_STOP_TOKENS.begin(), DEFAULT_STOP_TOKENS.end(), token_id) 
                != DEFAULT_STOP_TOKENS.end()) {
                break;
            }

            // Fast token decode using vocab map
            std::string token_str;
            if (token_id >= 0 && token_id < (int)id_to_token_.size() && !id_to_token_[token_id].empty()) {
                token_str = id_to_token_[token_id];
                // Replace occurrences of UTF-8 bytes for 'Ġ' (0xC4 0xA0) with space
                size_t pos = 0;
                while (true) {
                    auto it = token_str.find("\xC4\xA0", pos);
                    if (it == std::string::npos) break;
                    token_str.replace(it, 2, " ");
                    pos = it + 1;
                }
            } else {
                token_str = "[" + std::to_string(token_id) + "]";
            }

            std::cout << token_str << std::flush;
            decoded_text += token_str;

            // Short answer check
            if (short_answer && decoded_text.find_first_of(".\n!?") != std::string::npos) {
                break;
            }

            // Prepare next iteration - FAST path
            input_tensors.clear();
            
            // Move KV caches (zero-copy)
            for (size_t i = 0; i < num_layers_ * 2; i++) {
                input_tensors.push_back(std::move(output_tensors[i]));
            }
            
            // Update token for next iteration
            current_tokens_.assign(1, static_cast<int32_t>(token_id));
            
            std::vector<int64_t> token_shape = {1, 1};
            input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
                memory_info, current_tokens_.data(), 1, token_shape.data(), token_shape.size()));
            
            // Update state
            history_len_data_[0] += ids_len_data_[0];
            ids_len_data_[0] = 1;
            attention_mask_data_[0] = 0;
            
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, history_len_data_.data(), 1, scalar_shape.data(), scalar_shape.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, ids_len_data_.data(), 1, scalar_shape.data(), scalar_shape.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
                memory_info, attention_mask_data_.data(), 1, scalar_shape.data(), scalar_shape.size()));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto first_token_duration = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        
        float rate = num_decoded / (total_duration.count() / 1000.0f);
        
        std::cout << "\n\n🚀 FAST GPU Inference:" << std::endl;
        std::cout << "Generated " << num_decoded << " tokens" << std::endl;
        std::cout << "First token: " << first_token_duration.count() << "ms" << std::endl;
        std::cout << "Total time: " << total_duration.count() << "ms" << std::endl;
        std::cout << "Speed: " << rate << " tokens/sec" << std::endl;
        std::cout << "Device: " << (use_cuda_ ? "CUDA GPU" : "CPU") << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        std::string device = "gpu";
        int max_tokens = 64;  // Default to match Python test
        std::vector<std::string> positional;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--device=", 0) == 0) {
                device = arg.substr(9);
            } else if (arg == "--device") {
                if (i + 1 < argc) device = argv[++i];
            } else if (arg.rfind("--max-tokens=", 0) == 0) {
                max_tokens = std::stoi(arg.substr(13));
            } else if (arg == "--max-tokens") {
                if (i + 1 < argc) max_tokens = std::stoi(argv[++i]);
            } else if (arg[0] != '-') {
                positional.push_back(arg);
            }
        }

        // Load configuration
        std::string config_path = "./config.json";
        auto config = nlohmann::json::parse(std::ifstream(config_path));
        
        std::string model_dir = config["paths"]["model_path"];
        std::string onnx_file = config["paths"]["onnx_file"];
        std::string model_config = config["paths"]["model_config"];

        // Create fast inference object
        FastOnnxInference inference(model_dir, onnx_file, model_config);

        // Run inference
        std::string prompt = "What is 2+2?";
        if (!positional.empty()) {
            prompt = "";
            for (size_t i = 0; i < positional.size(); ++i) {
                if (i > 0) prompt += " ";
                prompt += positional[i];
            }
        }

        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Fast ONNX Answering:\n" << std::endl;
        
        inference.run_inference(prompt, max_tokens, false);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}