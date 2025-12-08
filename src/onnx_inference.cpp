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

class OnnxInference {
private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    std::vector<std::string> id_to_token_;
    int num_key_value_heads_;
    int head_dim_;
    int num_layers_;
    bool use_cuda_;
    
    json load_json(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + path);
        }
        json j;
        file >> j;
        return j;
    }

    std::unique_ptr<Ort::Session> build_session(const std::string& onnx_path, bool prefer_cuda = true) {
        Ort::SessionOptions session_opts;
        
        // Logging - set to 3 for warnings only
        session_opts.SetLogSeverityLevel(3);

        // Threading
        session_opts.SetInterOpNumThreads(0);
        session_opts.SetIntraOpNumThreads(0);

        // Memory / optimization options
        session_opts.EnableCpuMemArena();
        session_opts.SetExecutionMode(ORT_SEQUENTIAL);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Don't add config entries that might cause issues
        // Comment out potentially problematic entries
        /*
        session_opts.AddConfigEntry("session.set_denormal_as_zero", "1");
        session_opts.AddConfigEntry("session.intra_op.allow_spinning", "1");
        session_opts.AddConfigEntry("session.inter_op.allow_spinning", "1");
        session_opts.AddConfigEntry("session.enable_quant_qdq_cleanup", "1");
        session_opts.AddConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        session_opts.AddConfigEntry("optimization.enable_gelu_approximation", "1");
        session_opts.AddConfigEntry("disable_synchronize_execution_providers", "1");
        session_opts.AddConfigEntry("optimization.minimal_build_optimizations", "");
        session_opts.AddConfigEntry("session.use_device_allocator_for_initializers", "1");
        */

        // Provider options - try CUDA first
        use_cuda_ = false;
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        if (prefer_cuda && 
            std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") 
            != available_providers.end()) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;  // Use GPU 0 (RTX A6000)
                // cuda_options.device_id = 1;  // Uncomment to use GPU 1 (TITAN X)
                session_opts.AppendExecutionProvider_CUDA(cuda_options);
                use_cuda_ = true;
                std::cout << "Using CUDA execution provider on GPU " << cuda_options.device_id << std::endl;
            } catch (const std::exception& e) {
                std::cout << "CUDA provider failed, falling back to CPU: " << e.what() << std::endl;
                use_cuda_ = false;
            }
        }
        
        if (!use_cuda_) {
            std::cout << "Using CPU execution provider" << std::endl;
        }

        return std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_opts);
    }

public:
    OnnxInference(const std::string& model_dir, 
                 const std::string& onnx_path,
                 const std::string& config_path,
                 bool prefer_cuda = true) : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_inference") {
        
        // Load model config
        auto config = load_json(config_path);
        num_key_value_heads_ = config["num_key_value_heads"];
        head_dim_ = config["head_dim"];
        num_layers_ = config["num_hidden_layers"];

        // Initialize tokenizer (HuggingFace Tokenizers C++ binding)
        // Expects a `tokenizer.json` produced by the HF Tokenizers / Transformers
        tokenizer_ = std::make_unique<tokenizers::Tokenizer>(model_dir + "/tokenizer.json");
        if (!tokenizer_->valid()) {
            throw std::runtime_error("Failed to load tokenizer from: " + model_dir + "/tokenizer.json");
        }

        // Build id->token map from tokenizer.json for naive decoding
        auto build_id_to_token_map = [&](const std::string &tokenizer_json_path) {
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
        };

        id_to_token_ = build_id_to_token_map(model_dir + "/tokenizer.json");

        // Build ONNX session
        session_ = build_session(onnx_path, prefer_cuda);
    }

    void run_inference(const std::string& prompt, int max_decode = 512, bool short_answer = false) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Apply chat template for Qwen3 models
        // Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        std::string formatted_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        
        // Encode tokens first and keep them alive
        auto ids_vec = tokenizer_->encode(formatted_prompt, true);
        std::vector<int32_t> current_tokens(ids_vec.begin(), ids_vec.end());
        
        // These need to persist across iterations
        std::vector<int64_t> history_len_data = {0};
        std::vector<int64_t> ids_len_data = {static_cast<int64_t>(current_tokens.size())};
        std::vector<int8_t> attention_mask_data = {1};
        
        // Create input tensors with data that stays in scope
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;

        // Add past_keys for each layer (num_layers_ layers)
        std::vector<float> past_keys_empty;  // Empty tensor
        std::vector<int64_t> past_keys_shape = {num_key_value_heads_, 1, head_dim_, 0};
        for (int i = 0; i < num_layers_; i++) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, past_keys_empty.data(), 0,
                past_keys_shape.data(), past_keys_shape.size()));
        }

        // Add past_values for each layer (num_layers_ layers)
        std::vector<float> past_values_empty;  // Empty tensor
        std::vector<int64_t> past_values_shape = {num_key_value_heads_, 1, 0, head_dim_};
        for (int i = 0; i < num_layers_; i++) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, past_values_empty.data(), 0,
                past_values_shape.data(), past_values_shape.size()));
        }

        // input_ids - use int32_t
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(current_tokens.size())};
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info, current_tokens.data(), current_tokens.size(), 
            input_shape.data(), input_shape.size()));

        // history_len
        std::vector<int64_t> history_len_shape = {1};
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, history_len_data.data(), history_len_data.size(),
            history_len_shape.data(), history_len_shape.size()));

        // ids_len
        std::vector<int64_t> ids_len_shape = {1};
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, ids_len_data.data(), ids_len_data.size(),
            ids_len_shape.data(), ids_len_shape.size()));

        // attention_mask
        std::vector<int64_t> attention_mask_shape = {1};
        input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
            memory_info, attention_mask_data.data(), attention_mask_data.size(),
            attention_mask_shape.data(), attention_mask_shape.size()));
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        // Store allocated strings so they don't go out of scope
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

        // Decode loop
        int num_decoded = 0;
        std::string decoded_text;
        
        while (num_decoded < max_decode) {
            // Run inference
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names.data(), input_tensors.data(), input_tensors.size(),
                output_names.data(), output_names.size());

            // Get token from output (use reference to avoid copy)
            auto& token_tensor = output_tensors[output_tensors.size() - 2];
            auto token_info = token_tensor.GetTensorTypeAndShapeInfo();
            
            // Check if we got int32 or int64
            auto type_info = token_info.GetElementType();
            int64_t token_id = 0;
            if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                int64_t* token_data = token_tensor.GetTensorMutableData<int64_t>();
                token_id = token_data[0];
            } else if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                int32_t* token_data = token_tensor.GetTensorMutableData<int32_t>();
                token_id = static_cast<int64_t>(token_data[0]);
            } else {
                std::cerr << "Unexpected token data type: " << type_info << std::endl;
                break;
            }
            
            num_decoded++;

            // Check stop token
            if (std::find(DEFAULT_STOP_TOKENS.begin(), DEFAULT_STOP_TOKENS.end(), token_id) 
                != DEFAULT_STOP_TOKENS.end()) {
                break;
            }

            // Decode and print token
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

            // Short answer checks
            if (short_answer) {
                if (decoded_text.find_first_of(".\n!?") != std::string::npos && 
                    decoded_text.length() < 128) {
                    break;
                }
            }

            // Prepare inputs for next iteration
            // The outputs are: out_key_0..out_key_35, out_value_0..out_value_35, max_logit_id, kv_seq_len
            // We need to construct: in_key_0..in_key_35, in_value_0..in_value_35, input_ids, history_len, ids_len, attention_mask
            
            // Clear and rebuild input_tensors for next iteration
            input_tensors.clear();
            
            // Copy KV caches from outputs (first 72 tensors)
            for (size_t i = 0; i < num_layers_ * 2; i++) {
                input_tensors.push_back(std::move(output_tensors[i]));
            }
            
            // Update current_tokens with the new token for next iteration
            current_tokens.clear();
            current_tokens.push_back(static_cast<int32_t>(token_id));
            
            std::vector<int64_t> token_shape = {1, 1};
            input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
                memory_info, current_tokens.data(), current_tokens.size(),
                token_shape.data(), token_shape.size()));
            
            // history_len - increment each time
            history_len_data[0] = history_len_data[0] + ids_len_data[0];
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, history_len_data.data(), history_len_data.size(),
                history_len_shape.data(), history_len_shape.size()));
            
            // ids_len - always 1 after first iteration
            ids_len_data[0] = 1;
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, ids_len_data.data(), ids_len_data.size(),
                ids_len_shape.data(), ids_len_shape.size()));
            
            // attention_mask - 0 after first iteration
            attention_mask_data[0] = 0;
            input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
                memory_info, attention_mask_data.data(), attention_mask_data.size(),
                attention_mask_shape.data(), attention_mask_shape.size()));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        float rate = num_decoded / (duration.count() / 1000.0f);
        
        std::cout << "\n\nDecode: " << rate << " token/s" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
        // Parse CLI flags: --device cpu|gpu (default: gpu)
        std::string device = "gpu";
        std::vector<std::string> positional;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--device=", 0) == 0) {
                device = arg.substr(9);
            } else if (arg == "--device" || arg == "-d") {
                if (i + 1 < argc) {
                    device = argv[++i];
                }
            } else if (arg.size() > 0 && arg[0] != '-') {
                positional.push_back(arg);
            }
        }

        bool prefer_cuda = (device != "cpu" && device != "CPU");

        // Load configuration
        std::string config_path = "./config.json";
        auto config = nlohmann::json::parse(std::ifstream(config_path));
        
        std::string model_dir = config["paths"]["model_path"];
        std::string onnx_file = config["paths"]["onnx_file"];
        std::string model_config = config["paths"]["model_config"];

        // Create inference object with device preference
        OnnxInference inference(model_dir, onnx_file, model_config, prefer_cuda);

        // Run inference
        std::string prompt = "Hello there, how are u?";
        if (!positional.empty()) {
            prompt = "";
            for (size_t i = 0; i < positional.size(); ++i) {
                if (i > 0) prompt += " ";
                prompt += positional[i];
            }
        }

        std::cout << "\nPrompt: " << prompt << "\nQwen Answering (device=" << device << "):\n" << std::endl;
        inference.run_inference(prompt);

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}