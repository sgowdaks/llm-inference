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
        
        // Logging
        session_opts.SetLogSeverityLevel(4);
        session_opts.SetLogVerbosityLevel(4);

        // Threading
        session_opts.SetInterOpNumThreads(0);
        session_opts.SetIntraOpNumThreads(0);

        // Memory / optimization options
        session_opts.EnableCpuMemArena();
        session_opts.SetExecutionMode(ORT_SEQUENTIAL);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Session config entries
        session_opts.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
        session_opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");
        session_opts.AddSessionConfigEntry("session.inter_op.allow_spinning", "1");
        session_opts.AddSessionConfigEntry("session.enable_quant_qdq_cleanup", "1");
        session_opts.AddSessionConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        session_opts.AddSessionConfigEntry("optimization.enable_gelu_approximation", "1");
        session_opts.AddSessionConfigEntry("disable_synchronize_execution_providers", "1");
        session_opts.AddSessionConfigEntry("optimization.minimal_build_optimizations", "");
        session_opts.AddSessionConfigEntry("session.use_device_allocator_for_initializers", "1");

        // Provider options
        use_cuda_ = false;
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        if (prefer_cuda && 
            std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") 
            != available_providers.end()) {
            std::vector<const char*> providers = {"CUDAExecutionProvider", "CPUExecutionProvider"};
            session_opts.SetProviders(providers.size(), providers.data());
            use_cuda_ = true;
        } else {
            std::vector<const char*> providers = {"CPUExecutionProvider"};
            session_opts.SetProviders(providers.size(), providers.data());
        }

        return std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_opts);
    }

    std::vector<Ort::Value> prepare_inputs(const std::string& prompt) {
            // Encode using HuggingFace Tokenizers C++ binding
            auto ids_vec = tokenizer_->encode(prompt, true); // with special tokens
            std::vector<int> tokens(ids_vec.begin(), ids_vec.end());
            std::vector<int64_t> tokens_i64(tokens.begin(), tokens.end());
        
        // Create input tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;

        // input_ids
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, tokens_i64.data(), tokens_i64.size(), 
            input_shape.data(), input_shape.size()));

        // ids_len
        std::vector<int64_t> ids_len = {static_cast<int64_t>(tokens.size())};
        std::vector<int64_t> ids_len_shape = {1};
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, ids_len.data(), ids_len.size(),
            ids_len_shape.data(), ids_len_shape.size()));

        // history_len
        std::vector<int64_t> history_len = {0};
        std::vector<int64_t> history_len_shape = {1};
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, history_len.data(), history_len.size(),
            history_len_shape.data(), history_len_shape.size()));

        // attention_mask
        std::vector<int8_t> attention_mask = {1};
        std::vector<int64_t> attention_mask_shape = {1};
        input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
            memory_info, attention_mask.data(), attention_mask.size(),
            attention_mask_shape.data(), attention_mask_shape.size()));

        // past_keys
        std::vector<float> past_keys(num_key_value_heads_ * head_dim_ * 0);
        std::vector<int64_t> past_keys_shape = {num_key_value_heads_, 1, head_dim_, 0};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, past_keys.data(), past_keys.size(),
            past_keys_shape.data(), past_keys_shape.size()));

        // past_values
        std::vector<float> past_values(num_key_value_heads_ * 0 * head_dim_);
        std::vector<int64_t> past_values_shape = {num_key_value_heads_, 1, 0, head_dim_};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, past_values.data(), past_values.size(),
            past_values_shape.data(), past_values_shape.size()));

        return input_tensors;
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
        
        // Prepare inputs
        auto input_tensors = prepare_inputs(prompt);
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            input_names.push_back(session_->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < num_output_nodes; i++) {
            output_names.push_back(session_->GetOutputNameAllocated(i, allocator).get());
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

            // Get token from output
            auto token_tensor = output_tensors[output_tensors.size() - 2];
            auto token_info = token_tensor.GetTensorTypeAndShapeInfo();
            int64_t* token_data = token_tensor.GetTensorMutableData<int64_t>();
            int64_t token_id = token_data[0];
            
            num_decoded++;

            // Check stop token
            if (std::find(DEFAULT_STOP_TOKENS.begin(), DEFAULT_STOP_TOKENS.end(), token_id) 
                != DEFAULT_STOP_TOKENS.end()) {
                break;
            }

            // Update inputs for next iteration
            // Ort::Value is move-only; some std::vector move-assign implementations
            // may attempt to copy elements which fails because copy ctor is deleted.
            // Swap the vectors to transfer ownership of the underlying buffers
            // without invoking copies.
            input_tensors.swap(output_tensors);

            // Reset certain inputs after first token
            if (num_decoded < 2) {
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
                
                std::vector<int8_t> attention_mask = {0};
                std::vector<int64_t> attention_mask_shape = {1};
                input_tensors[3] = Ort::Value::CreateTensor<int8_t>(
                    memory_info, attention_mask.data(), attention_mask.size(),
                    attention_mask_shape.data(), attention_mask_shape.size());

                std::vector<int64_t> ids_len = {1};
                std::vector<int64_t> ids_len_shape = {1};
                input_tensors[1] = Ort::Value::CreateTensor<int64_t>(
                    memory_info, ids_len.data(), ids_len.size(),
                    ids_len_shape.data(), ids_len_shape.size());
            }

            // Decode and print token
            // naive decode: lookup token string from id_to_token_ and do small post-processing
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
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        float rate = num_decoded / (duration.count() / 1000.0f);
        
        std::cout << "\n\nDecode: " << rate << " token/s" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
        // Load configuration
        std::string config_path = "./config.json";
        auto config = nlohmann::json::parse(std::ifstream(config_path));
        
        std::string model_dir = config["paths"]["model_path"];
        std::string onnx_file = config["paths"]["onnx_file"];
        std::string model_config = config["paths"]["model_config"];

        // Create inference object
        OnnxInference inference(model_dir, onnx_file, model_config);

        // Run inference
        std::string prompt = "Hello there, how are u?";
        if (argc > 1) {
            prompt = argv[1];
        }

        std::cout << "\nPrompt: " << prompt << "\nQwen Answering:\n" << std::endl;
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