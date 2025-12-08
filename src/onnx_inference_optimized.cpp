#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "tokenizers/tokenizers.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Constants
const std::vector<int64_t> DEFAULT_STOP_TOKENS = {151643, 151645};
const int DEFAULT_MAX_SEQ_LEN = 4096;

class OptimizedOnnxInference {
private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    std::vector<std::string> id_to_token_;
    int num_key_value_heads_;
    int head_dim_;
    int num_layers_;
    bool use_cuda_;
    
    // Pre-allocated GPU memory info for CUDA
    Ort::MemoryInfo gpu_memory_info_;
    Ort::MemoryInfo cpu_memory_info_;
    
    // Pre-allocated tensor shapes (avoid recomputation)
    std::vector<int64_t> past_keys_shape_;
    std::vector<int64_t> past_values_shape_;
    std::vector<int64_t> token_shape_;
    std::vector<int64_t> scalar_shape_;

public:
    OptimizedOnnxInference(const std::string& model_dir, 
                          const std::string& onnx_path,
                          const std::string& config_path,
                          bool prefer_cuda = true) 
        : env_(ORT_LOGGING_LEVEL_WARNING, "optimized_onnx_inference"),
          gpu_memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)),
          cpu_memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)) {
        
        // Load model config
        auto config = load_json(config_path);
        num_key_value_heads_ = config["num_key_value_heads"];
        head_dim_ = config["head_dim"];
        num_layers_ = config["num_hidden_layers"];

        // Pre-compute tensor shapes
        past_keys_shape_ = {num_key_value_heads_, 1, head_dim_, 0};
        past_values_shape_ = {num_key_value_heads_, 1, 0, head_dim_};
        token_shape_ = {1, 1};
        scalar_shape_ = {1};

        // Initialize tokenizer
        tokenizer_ = std::make_unique<tokenizers::Tokenizer>(model_dir + "/tokenizer.json");
        if (!tokenizer_->valid()) {
            throw std::runtime_error("Failed to load tokenizer");
        }

        // Build id->token map
        id_to_token_ = build_id_to_token_map(model_dir + "/tokenizer.json");

        // Build ONNX session with optimizations
        session_ = build_optimized_session(onnx_path, prefer_cuda);
        
        // Setup GPU memory info if using CUDA
        if (use_cuda_) {
            OrtCUDAProviderOptions cuda_opts;
            // Note: In a real implementation, you'd want to get the actual CUDA memory info
            // For now, we'll use CPU memory info but this could be optimized further
        }
    }

private:
    json load_json(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + path);
        }
        json j;
        file >> j;
        return j;
    }

    std::unique_ptr<Ort::Session> build_optimized_session(const std::string& onnx_path, bool prefer_cuda) {
        Ort::SessionOptions session_opts;
        
        // Aggressive optimization settings
        session_opts.SetLogSeverityLevel(4);  // Minimal logging
        session_opts.SetInterOpNumThreads(0);
        session_opts.SetIntraOpNumThreads(0);
        
        // Memory optimizations
        session_opts.EnableCpuMemArena();
        session_opts.SetExecutionMode(ORT_SEQUENTIAL);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Additional performance configs
        session_opts.AddConfigEntry("session.set_denormal_as_zero", "1");
        session_opts.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        session_opts.AddConfigEntry("session.use_device_allocator_for_initializers", "1");
        
        use_cuda_ = false;
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        if (prefer_cuda && 
            std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") 
            != available_providers.end()) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 1;  // Extend by doubling
                cuda_options.gpu_mem_limit = SIZE_MAX;   // No memory limit
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = 1;
                
                session_opts.AppendExecutionProvider_CUDA(cuda_options);
                use_cuda_ = true;
                std::cout << "✓ Using optimized CUDA execution provider" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "CUDA failed, using CPU: " << e.what() << std::endl;
                use_cuda_ = false;
            }
        }
        
        if (!use_cuda_) {
            std::cout << "Using CPU execution provider" << std::endl;
        }

        return std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_opts);
    }

    std::vector<std::string> build_id_to_token_map(const std::string& tokenizer_json_path) {
        std::ifstream f(tokenizer_json_path);
        if (!f.is_open()) return std::vector<std::string>{};
        
        json j;
        f >> j;
        std::vector<std::string> id_to_token;
        
        if (j.contains("model") && j["model"].contains("vocab")) {
            auto vocab = j["model"]["vocab"];
            int max_id = -1;
            for (auto it = vocab.begin(); it != vocab.end(); ++it) {
                int id = it.value().get<int>();
                if (id > max_id) max_id = id;
            }
            id_to_token.assign(max_id + 1, std::string());
            for (auto it = vocab.begin(); it != vocab.end(); ++it) {
                const std::string token = it.key();
                int id = it.value().get<int>();
                if (id >= 0 && id < (int)id_to_token.size()) id_to_token[id] = token;
            }
        }
        if (j.contains("added_tokens")) {
            for (const auto &t : j["added_tokens"]) {
                if (t.contains("id") && t.contains("content")) {
                    int id = t["id"].get<int>();
                    std::string content = t["content"].get<std::string>();
                    if (id >= 0) {
                        if (id >= (int)id_to_token.size()) id_to_token.resize(id + 1);
                        id_to_token[id] = content;
                    }
                }
            }
        }
        return id_to_token;
    }

public:
    void run_inference(const std::string& prompt, int max_decode = 512, bool short_answer = false) {
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Encode tokens
        auto ids_vec = tokenizer_->encode(prompt, true);
        std::vector<int32_t> current_tokens(ids_vec.begin(), ids_vec.end());
        
        // Pre-allocate persistent data
        std::vector<int64_t> history_len_data = {0};
        std::vector<int64_t> ids_len_data = {static_cast<int64_t>(current_tokens.size())};
        std::vector<int8_t> attention_mask_data = {1};
        
        // Get input/output names (do this once)
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
        std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            input_name_ptrs.push_back(session_->GetInputNameAllocated(i, allocator));
            input_names.push_back(input_name_ptrs.back().get());
        }
        for (size_t i = 0; i < num_output_nodes; i++) {
            output_name_ptrs.push_back(session_->GetOutputNameAllocated(i, allocator));
            output_names.push_back(output_name_ptrs.back().get());
        }

        // Create initial input tensors
        std::vector<Ort::Value> input_tensors;
        setup_initial_tensors(input_tensors, current_tokens, history_len_data, ids_len_data, attention_mask_data);

        // Performance tracking
        int num_decoded = 0;
        std::string decoded_text;
        auto first_token_time = std::chrono::high_resolution_clock::now();
        bool first_token_measured = false;
        
        while (num_decoded < max_decode) {
            // Time individual inference calls
            auto inference_start = std::chrono::high_resolution_clock::now();
            
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names.data(), input_tensors.data(), input_tensors.size(),
                output_names.data(), output_names.size());
                
            auto inference_end = std::chrono::high_resolution_clock::now();

            // Extract token efficiently
            auto& token_tensor = output_tensors[output_tensors.size() - 2];
            int64_t token_id = extract_token_id(token_tensor);
            
            if (!first_token_measured) {
                auto first_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - first_token_time);
                std::cout << "\n⚡ First token: " << first_duration.count() << "ms" << std::endl;
                first_token_measured = true;
            }
            
            num_decoded++;

            // Check stop token
            if (std::find(DEFAULT_STOP_TOKENS.begin(), DEFAULT_STOP_TOKENS.end(), token_id) 
                != DEFAULT_STOP_TOKENS.end()) {
                break;
            }

            // Fast token decoding (minimal string processing)
            std::string token_str = decode_token_fast(token_id);
            std::cout << token_str << std::flush;
            decoded_text += token_str;

            // Short answer check
            if (short_answer && should_stop_short_answer(decoded_text)) {
                break;
            }

            // Efficiently prepare next iteration
            prepare_next_iteration(input_tensors, output_tensors, token_id, 
                                 current_tokens, history_len_data, ids_len_data, attention_mask_data);
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - first_token_time);
        
        float total_rate = (num_decoded > 0) ? num_decoded / (total_duration.count() / 1000.0f) : 0;
        float decode_rate = (num_decoded > 1) ? (num_decoded - 1) / (decode_duration.count() / 1000.0f) : 0;
        
        std::cout << "\n\n🚀 Performance Summary:" << std::endl;
        std::cout << "   Total tokens: " << num_decoded << std::endl;
        std::cout << "   Total time: " << total_duration.count() << "ms" << std::endl;
        std::cout << "   Overall rate: " << total_rate << " tokens/sec" << std::endl;
        std::cout << "   Decode rate: " << decode_rate << " tokens/sec (excluding first token)" << std::endl;
        std::cout << "   Device: " << (use_cuda_ ? "CUDA" : "CPU") << std::endl;
    }

private:
    void setup_initial_tensors(std::vector<Ort::Value>& input_tensors,
                              std::vector<int32_t>& current_tokens,
                              std::vector<int64_t>& history_len_data,
                              std::vector<int64_t>& ids_len_data,
                              std::vector<int8_t>& attention_mask_data) {
        
        // Empty KV caches for first iteration
        std::vector<float> empty_data;
        
        // Past keys and values (empty initially)
        for (int i = 0; i < num_layers_ * 2; i++) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                cpu_memory_info_, empty_data.data(), 0,
                (i < num_layers_) ? past_keys_shape_.data() : past_values_shape_.data(),
                (i < num_layers_) ? past_keys_shape_.size() : past_values_shape_.size()));
        }

        // Input tokens
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(current_tokens.size())};
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            cpu_memory_info_, current_tokens.data(), current_tokens.size(), 
            input_shape.data(), input_shape.size()));

        // Metadata tensors
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            cpu_memory_info_, history_len_data.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            cpu_memory_info_, ids_len_data.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
            cpu_memory_info_, attention_mask_data.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
    }

    int64_t extract_token_id(Ort::Value& token_tensor) {
        auto type_info = token_tensor.GetTensorTypeAndShapeInfo().GetElementType();
        
        if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return token_tensor.GetTensorMutableData<int64_t>()[0];
        } else if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            return static_cast<int64_t>(token_tensor.GetTensorMutableData<int32_t>()[0]);
        }
        return 0; // fallback
    }

    std::string decode_token_fast(int64_t token_id) {
        if (token_id >= 0 && token_id < (int)id_to_token_.size() && !id_to_token_[token_id].empty()) {
            std::string token = id_to_token_[token_id];
            // Fast single-pass replacement
            if (token.find("\xC4\xA0") != std::string::npos) {
                std::string result;
                result.reserve(token.size());
                for (size_t i = 0; i < token.size(); ) {
                    if (i + 1 < token.size() && token[i] == '\xC4' && token[i + 1] == '\xA0') {
                        result += ' ';
                        i += 2;
                    } else {
                        result += token[i];
                        i++;
                    }
                }
                return result;
            }
            return token;
        }
        return "[" + std::to_string(token_id) + "]";
    }

    bool should_stop_short_answer(const std::string& text) {
        return text.find_first_of(".\n!?") != std::string::npos && text.length() < 128;
    }

    void prepare_next_iteration(std::vector<Ort::Value>& input_tensors,
                               std::vector<Ort::Value>& output_tensors,
                               int64_t token_id,
                               std::vector<int32_t>& current_tokens,
                               std::vector<int64_t>& history_len_data,
                               std::vector<int64_t>& ids_len_data,
                               std::vector<int8_t>& attention_mask_data) {
        
        input_tensors.clear();
        
        // Move KV caches from outputs
        for (size_t i = 0; i < num_layers_ * 2; i++) {
            input_tensors.push_back(std::move(output_tensors[i]));
        }
        
        // Update token data
        current_tokens.clear();
        current_tokens.push_back(static_cast<int32_t>(token_id));
        
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            cpu_memory_info_, current_tokens.data(), 1, token_shape_.data(), token_shape_.size()));
        
        // Update metadata
        history_len_data[0] += ids_len_data[0];
        ids_len_data[0] = 1;
        attention_mask_data[0] = 0;
        
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            cpu_memory_info_, history_len_data.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            cpu_memory_info_, ids_len_data.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
            cpu_memory_info_, attention_mask_data.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
    }
};

// Main function to test the optimized version
int main(int argc, char* argv[]) {
    try {
        std::string device = "gpu";
        std::vector<std::string> positional;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--device=", 0) == 0) {
                device = arg.substr(9);
            } else if (arg == "--device" || arg == "-d") {
                if (i + 1 < argc) device = argv[++i];
            } else if (arg.size() > 0 && arg[0] != '-') {
                positional.push_back(arg);
            }
        }

        bool prefer_cuda = (device != "cpu" && device != "CPU");

        // Load configuration
        auto config = json::parse(std::ifstream("./config.json"));
        std::string model_dir = config["paths"]["model_path"];
        std::string onnx_file = config["paths"]["onnx_file"];
        std::string model_config = config["paths"]["model_config"];

        // Create optimized inference object
        OptimizedOnnxInference inference(model_dir, onnx_file, model_config, prefer_cuda);

        std::string prompt = "Hello there, how are u?";
        if (!positional.empty()) {
            prompt = "";
            for (size_t i = 0; i < positional.size(); ++i) {
                if (i > 0) prompt += " ";
                prompt += positional[i];
            }
        }

        std::cout << "\nPrompt: " << prompt << std::endl;
        std::cout << "Optimized Qwen Answering:\n" << std::endl;
        
        inference.run_inference(prompt, 64);  // Match Python test: 64 tokens

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}