#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <cstring>


#include <nlohmann/json.hpp>
#include "tokenizers/tokenizers.h"
#include <onnxruntime_cxx_api.h>

using json = nlohmann::json;

// Constants
const std::vector<int64_t> DEFAULT_STOP_TOKENS = {151643, 151645};
const int DEFAULT_MAX_SEQ_LEN = 4096;

class UltraFastOnnxInference {
private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    
    // Model parameters
    int num_key_value_heads_;
    int head_dim_;
    int num_layers_;
    bool use_cuda_;
    
    // Pre-allocated persistent buffers - avoid reallocations
    std::vector<int32_t> current_tokens_;
    std::vector<int64_t> history_len_data_;
    std::vector<int64_t> ids_len_data_;
    std::vector<int8_t> attention_mask_data_;
    
    // Pre-allocated tensor shapes - avoid repeated vector creation
    std::vector<int64_t> past_keys_shape_;
    std::vector<int64_t> past_values_shape_;
    std::vector<int64_t> scalar_shape_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> token_shape_;
    
    // Cached input/output names and memory info
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    Ort::MemoryInfo memory_info_;
    
    // Token decoding will use tokenizer decode API directly
    

    
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
        
        // Maximum performance settings
        session_opts.SetLogSeverityLevel(4);  // No logging overhead
        session_opts.SetInterOpNumThreads(0);  // Auto detect
        session_opts.SetIntraOpNumThreads(0);  // Auto detect
        
        // Aggressive memory and compute optimizations
        session_opts.EnableCpuMemArena();
        session_opts.EnableMemPattern();
        session_opts.SetExecutionMode(ORT_SEQUENTIAL);  // No threading overhead
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Advanced optimization configs
        session_opts.AddConfigEntry("session.set_denormal_as_zero", "1");
        session_opts.AddConfigEntry("session.intra_op.allow_spinning", "1");
        session_opts.AddConfigEntry("session.inter_op.allow_spinning", "1");
        session_opts.AddConfigEntry("session.enable_quant_qdq_cleanup", "1");
        session_opts.AddConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        session_opts.AddConfigEntry("optimization.enable_gelu_approximation", "1");
        session_opts.AddConfigEntry("session.use_device_allocator_for_initializers", "1");
        session_opts.AddConfigEntry("session.disable_prepacking", "0");
        session_opts.AddConfigEntry("optimization.minimal_build_optimizations", "");

        // Ultra-optimized CUDA provider
        use_cuda_ = false;
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        if (std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") 
            != available_providers.end()) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                
                // Maximum GPU performance settings
                cuda_options.arena_extend_strategy = 1;  // kSameAsRequested - fastest
                cuda_options.gpu_mem_limit = SIZE_MAX;   // No memory limit
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;  // Fastest conv
                cuda_options.do_copy_in_default_stream = 1;  // Avoid stream sync overhead
                cuda_options.tunable_op_enable = 1;  // Enable tunable ops
                cuda_options.tunable_op_tuning_enable = 1;  // Auto-tune for this specific model
                
                session_opts.AppendExecutionProvider_CUDA(cuda_options);
                use_cuda_ = true;
                std::cout << "🚀 Using ULTRA-OPTIMIZED CUDA execution provider" << std::endl;
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



    void pre_allocate_all_tensors() {
        // Pre-allocate all tensor shapes to avoid vector reallocations
        // Model now supports history_len=0 for initial call with TorchScript export
        past_keys_shape_ = {num_key_value_heads_, 1, head_dim_, 0};  // history_len=0 for initial
        past_values_shape_ = {num_key_value_heads_, 1, 0, head_dim_};  // history_len=0 for initial
        scalar_shape_ = {1};
        input_shape_.reserve(2);
        token_shape_ = {1, 1};
        
        // Cache input/output names to avoid repeated allocations
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        input_names_storage_.reserve(num_input_nodes);
        output_names_storage_.reserve(num_output_nodes);
        input_names_.reserve(num_input_nodes);
        output_names_.reserve(num_output_nodes);
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name_ptr = session_->GetInputNameAllocated(i, allocator);
            input_names_storage_.emplace_back(name_ptr.get());
            input_names_.push_back(input_names_storage_.back().c_str());
        }
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
            output_names_storage_.emplace_back(name_ptr.get());
            output_names_.push_back(output_names_storage_.back().c_str());
        }
    }

public:
    UltraFastOnnxInference(const std::string& model_dir, 
                          const std::string& onnx_path,
                          const std::string& config_path) 
        : env_(ORT_LOGGING_LEVEL_ERROR, "ultra_fast_onnx"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)) {
        
        // Load model config
        auto config = load_json(config_path);
        num_key_value_heads_ = config["num_key_value_heads"];
        head_dim_ = config["head_dim"];
        num_layers_ = config["num_hidden_layers"];

        // Initialize tokenizer with config (for chat template support)
        std::string tokenizer_path = model_dir + "/tokenizer.json";
        std::string tokenizer_config_path = model_dir + "/tokenizer_config.json";
        tokenizer_ = std::make_unique<tokenizers::Tokenizer>(tokenizer_path, tokenizer_config_path);
        if (!tokenizer_->valid()) {
            throw std::runtime_error("Failed to load tokenizer from: " + tokenizer_path);
        }
        
        std::cout << "📝 Tokenizer loaded (vocab size: " << tokenizer_->vocab_size() << ")" << std::endl;
        if (tokenizer_->has_chat_template()) {
            std::cout << "💬 Chat template available" << std::endl;
        }

        // Build ONNX session
        session_ = build_session(onnx_path);
        
        // Pre-allocate ALL data structures
        current_tokens_.reserve(4096);
        history_len_data_.resize(1);
        ids_len_data_.resize(1);
        attention_mask_data_.resize(1);
        
        // Pre-allocate tensor metadata
        pre_allocate_all_tensors();
        
        std::cout << "⚡ ULTRA-FAST inference engine ready!" << std::endl;
    }

    void run_inference(const std::string& prompt, int max_decode = 64, bool short_answer = false) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Apply chat template using tokenizer's built-in support
        std::string formatted_prompt;
        if (tokenizer_->has_chat_template()) {
            std::vector<tokenizers::ChatMessage> messages = {
                {"user", prompt}
            };
            formatted_prompt = tokenizer_->apply_chat_template(messages, true);
        } else {
            // Fallback: manual Qwen3 format
            formatted_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        }
        
        // Encode tokens
        auto ids_vec = tokenizer_->encode(formatted_prompt, true);
        current_tokens_.assign(ids_vec.begin(), ids_vec.end());
        
        // Setup initial state - model supports history_len=0 for first call
        history_len_data_[0] = 0;
        ids_len_data_[0] = static_cast<int64_t>(current_tokens_.size());
        attention_mask_data_[0] = 1;
        
        // Pre-allocate input tensors vector with exact size
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names_.size());

        // Create empty KV cache tensors for first iteration (history_len=0)
        std::vector<float> empty_data;
        for (int i = 0; i < num_layers_; i++) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, empty_data.data(), 0,
                past_keys_shape_.data(), past_keys_shape_.size()));
        }
        for (int i = 0; i < num_layers_; i++) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, empty_data.data(), 0,
                past_values_shape_.data(), past_values_shape_.size()));
        }

        // Add input_ids tensor
        input_shape_.clear();
        input_shape_.push_back(1);
        input_shape_.push_back(static_cast<int64_t>(current_tokens_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, current_tokens_.data(), current_tokens_.size(), 
            input_shape_.data(), input_shape_.size()));

        // Add remaining scalar inputs
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, history_len_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ids_len_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
            memory_info_, attention_mask_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));

        // Pre-allocate decode output string
        std::string decoded_text;
        decoded_text.reserve(max_decode * 4);
        
        // Ultra-fast decode loop
        int num_decoded = 0;
        auto first_token_time = std::chrono::high_resolution_clock::now();
        
        while (num_decoded < max_decode) {
            // Run inference - this is the hot path
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_.data(), input_tensors.data(), input_tensors.size(),
                output_names_.data(), output_names_.size());

            // Extract token ID with minimal overhead
            auto& token_tensor = output_tensors[output_tensors.size() - 2];
            auto type_info = token_tensor.GetTensorTypeAndShapeInfo().GetElementType();
            
            int64_t token_id;
            if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                token_id = *token_tensor.GetTensorData<int64_t>();
            } else {
                token_id = static_cast<int64_t>(*token_tensor.GetTensorData<int32_t>());
            }
            
            num_decoded++;

            // Fast stop token check
            if (token_id == 151643 || token_id == 151645) break;

            // Decode token using tokenizer's decode API
            std::vector<int32_t> single_token = {static_cast<int32_t>(token_id)};
            std::string token_str = tokenizer_->decode(single_token, false);
            
            std::cout << token_str << std::flush;
            decoded_text += token_str;

            // Fast short answer check
            if (short_answer && decoded_text.find_first_of(".\n!?") != std::string::npos) {
                break;
            }

            // Prepare next iteration with MINIMAL allocations
            input_tensors.clear();
            
            // Move KV caches (zero-copy transfer)
            for (size_t i = 0; i < num_layers_ * 2; i++) {
                input_tensors.push_back(std::move(output_tensors[i]));
            }
            
            // Update token data in-place
            current_tokens_.clear();
            current_tokens_.push_back(static_cast<int32_t>(token_id));
            
            input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
                memory_info_, current_tokens_.data(), 1, token_shape_.data(), token_shape_.size()));
            
            // Update state efficiently
            history_len_data_[0] += ids_len_data_[0];
            ids_len_data_[0] = 1;
            attention_mask_data_[0] = 0;
            
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info_, history_len_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info_, ids_len_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
                memory_info_, attention_mask_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto first_token_duration = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        
        float rate = num_decoded / (total_duration.count() / 1000.0f);
        
        std::cout << "\n\n⚡ ULTRA-FAST GPU Inference:" << std::endl;
        std::cout << "Generated " << num_decoded << " tokens" << std::endl;
        std::cout << "First token: " << first_token_duration.count() << "ms" << std::endl;
        std::cout << "Total time: " << total_duration.count() << "ms" << std::endl;
        std::cout << "🏎️ SPEED: " << rate << " tokens/sec" << std::endl;
        std::cout << "Device: " << (use_cuda_ ? "ULTRA-CUDA GPU" : "CPU") << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        std::string device = "gpu";
        int max_tokens = 64;
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

        // Create ultra-fast inference object
        UltraFastOnnxInference inference(model_dir, onnx_file, model_config);

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
        std::cout << "Ultra-Fast ONNX Answering:\n" << std::endl;
        
        inference.run_inference(prompt, max_tokens, false);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}