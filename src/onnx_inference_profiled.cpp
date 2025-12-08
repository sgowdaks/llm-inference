#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "tokenizers/tokenizers.h"
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <iomanip>

using json = nlohmann::json;

// Constants
const std::vector<int64_t> DEFAULT_STOP_TOKENS = {151643, 151645};
const int DEFAULT_MAX_SEQ_LEN = 4096;

// Timing utility
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    std::string name;
    
    Timer(const std::string& n) : name(n) {
        start = std::chrono::high_resolution_clock::now();
    }
    
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "⏱️  " << name << ": " << duration.count() / 1000.0 << "ms" << std::endl;
    }
};

class ProfiledOnnxInference {
private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    
    int num_key_value_heads_;
    int head_dim_;
    int num_layers_;
    bool use_cuda_;
    
    std::vector<int32_t> current_tokens_;
    std::vector<int64_t> history_len_data_;
    std::vector<int64_t> ids_len_data_;
    std::vector<int8_t> attention_mask_data_;
    
    std::vector<int64_t> past_keys_shape_;
    std::vector<int64_t> past_values_shape_;
    std::vector<int64_t> scalar_shape_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> token_shape_;
    
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    Ort::MemoryInfo memory_info_;
    
    std::vector<std::string> id_to_token_;
    std::unordered_map<int64_t, std::string> fast_token_cache_;
    
    // Profiling data
    double total_encode_time_ = 0;
    double total_inference_time_ = 0;
    double total_decode_time_ = 0;
    double total_tensor_prep_time_ = 0;
    double total_token_extract_time_ = 0;
    int inference_calls_ = 0;
    
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
        
        session_opts.SetLogSeverityLevel(4);
        session_opts.SetInterOpNumThreads(0);
        session_opts.SetIntraOpNumThreads(0);
        
        session_opts.EnableCpuMemArena();
        session_opts.EnableMemPattern();
        session_opts.SetExecutionMode(ORT_SEQUENTIAL);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Save optimized model for future use
        session_opts.SetOptimizedModelFilePath("/tmp/qwen_optimized_extended.onnx");
        
        session_opts.AddConfigEntry("session.set_denormal_as_zero", "1");
        session_opts.AddConfigEntry("session.intra_op.allow_spinning", "1");
        session_opts.AddConfigEntry("session.inter_op.allow_spinning", "1");
        session_opts.AddConfigEntry("session.enable_quant_qdq_cleanup", "1");
        session_opts.AddConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        session_opts.AddConfigEntry("optimization.enable_gelu_approximation", "1");
        session_opts.AddConfigEntry("session.use_device_allocator_for_initializers", "1");

        use_cuda_ = false;
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        if (std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") 
            != available_providers.end()) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 1;
                cuda_options.gpu_mem_limit = SIZE_MAX;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
                cuda_options.do_copy_in_default_stream = 1;
                cuda_options.tunable_op_enable = 1;
                cuda_options.tunable_op_tuning_enable = 1;
                
                session_opts.AppendExecutionProvider_CUDA(cuda_options);
                use_cuda_ = true;
                std::cout << "🚀 Using PROFILED CUDA execution provider" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "CUDA provider failed: " << e.what() << std::endl;
                use_cuda_ = false;
            }
        }

        return std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_opts);
    }

    void build_vocab(const std::string& model_dir) {
        Timer t("Vocabulary Build");
        
        std::string tokenizer_json_path = model_dir + "/tokenizer.json";
        std::ifstream f(tokenizer_json_path);
        if (!f.is_open()) {
            std::cerr << "Warning: Could not open tokenizer.json" << std::endl;
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
                std::string token = it.key();
                int id = it.value().get<int>();
                
                size_t pos = 0;
                while (true) {
                    auto found = token.find("\xC4\xA0", pos);
                    if (found == std::string::npos) break;
                    token.replace(found, 2, " ");
                    pos = found + 1;
                }
                
                if (id >= 0 && id < (int)id_to_token_.size()) {
                    id_to_token_[id] = std::move(token);
                }
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
        
        for (int i = 0; i < std::min(10000, (int)id_to_token_.size()); ++i) {
            if (!id_to_token_[i].empty()) {
                fast_token_cache_[i] = id_to_token_[i];
            }
        }
    }

    void pre_allocate_all_tensors() {
        Timer t("Tensor Metadata Pre-allocation");
        
        past_keys_shape_ = {num_key_value_heads_, 1, head_dim_, 0};
        past_values_shape_ = {num_key_value_heads_, 1, 0, head_dim_};
        scalar_shape_ = {1};
        input_shape_.reserve(2);
        token_shape_ = {1, 1};
        
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
    ProfiledOnnxInference(const std::string& model_dir, 
                         const std::string& onnx_path,
                         const std::string& config_path) 
        : env_(ORT_LOGGING_LEVEL_ERROR, "profiled_onnx"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)) {
        
        std::cout << "\n🔧 INITIALIZATION PROFILING:\n" << std::endl;
        
        {
            Timer t("Config Load");
            auto config = load_json(config_path);
            num_key_value_heads_ = config["num_key_value_heads"];
            head_dim_ = config["head_dim"];
            num_layers_ = config["num_hidden_layers"];
        }

        {
            Timer t("Tokenizer Load");
            tokenizer_ = std::make_unique<tokenizers::Tokenizer>(model_dir + "/tokenizer.json");
            if (!tokenizer_->valid()) {
                throw std::runtime_error("Failed to load tokenizer");
            }
        }

        {
            Timer t("ONNX Session Build");
            session_ = build_session(onnx_path);
        }
        
        build_vocab(model_dir);
        
        {
            Timer t("Buffer Pre-allocation");
            current_tokens_.reserve(4096);
            history_len_data_.resize(1);
            ids_len_data_.resize(1);
            attention_mask_data_.resize(1);
        }
        
        pre_allocate_all_tensors();
        
        std::cout << "\n✅ Initialization complete!\n" << std::endl;
    }

    void run_inference(const std::string& prompt, int max_decode = 64, bool short_answer = false) {
        auto total_start = std::chrono::high_resolution_clock::now();
        
        std::cout << "\n📊 DETAILED PROFILING:\n" << std::endl;
        
        // ENCODE PHASE
        auto encode_start = std::chrono::high_resolution_clock::now();
        auto ids_vec = tokenizer_->encode(prompt, true);
        current_tokens_.assign(ids_vec.begin(), ids_vec.end());
        auto encode_end = std::chrono::high_resolution_clock::now();
        double encode_time = std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start).count() / 1000.0;
        std::cout << "⏱️  Tokenization (encode): " << encode_time << "ms" << std::endl;
        
        history_len_data_[0] = 0;
        ids_len_data_[0] = static_cast<int64_t>(current_tokens_.size());
        attention_mask_data_[0] = 1;
        
        // INITIAL TENSOR PREP
        auto prep_start = std::chrono::high_resolution_clock::now();
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names_.size());

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

        input_shape_.clear();
        input_shape_.push_back(1);
        input_shape_.push_back(static_cast<int64_t>(current_tokens_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, current_tokens_.data(), current_tokens_.size(), 
            input_shape_.data(), input_shape_.size()));

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, history_len_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ids_len_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
            memory_info_, attention_mask_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
        
        auto prep_end = std::chrono::high_resolution_clock::now();
        double prep_time = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_start).count() / 1000.0;
        std::cout << "⏱️  Initial tensor prep: " << prep_time << "ms" << std::endl;

        int num_decoded = 0;
        std::string decoded_text;
        decoded_text.reserve(max_decode * 4);
        
        // Track per-iteration timings
        std::vector<double> inference_times;
        std::vector<double> decode_times;
        std::vector<double> tensor_prep_times;
        
        auto first_token_start = std::chrono::high_resolution_clock::now();
        bool first_token_done = false;
        
        while (num_decoded < max_decode) {
            // INFERENCE
            auto inf_start = std::chrono::high_resolution_clock::now();
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_.data(), input_tensors.data(), input_tensors.size(),
                output_names_.data(), output_names_.size());
            auto inf_end = std::chrono::high_resolution_clock::now();
            double inf_time = std::chrono::duration_cast<std::chrono::microseconds>(inf_end - inf_start).count() / 1000.0;
            inference_times.push_back(inf_time);

            // TOKEN EXTRACTION
            auto extract_start = std::chrono::high_resolution_clock::now();
            auto& token_tensor = output_tensors[output_tensors.size() - 2];
            auto type_info = token_tensor.GetTensorTypeAndShapeInfo().GetElementType();
            
            int64_t token_id;
            if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                token_id = *token_tensor.GetTensorData<int64_t>();
            } else {
                token_id = static_cast<int64_t>(*token_tensor.GetTensorData<int32_t>());
            }
            auto extract_end = std::chrono::high_resolution_clock::now();
            double extract_time = std::chrono::duration_cast<std::chrono::microseconds>(extract_end - extract_start).count() / 1000.0;
            
            num_decoded++;
            
            if (!first_token_done) {
                auto first_token_end = std::chrono::high_resolution_clock::now();
                double first_token_time = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_end - first_token_start).count();
                std::cout << "⏱️  FIRST TOKEN latency: " << first_token_time << "ms" << std::endl;
                first_token_done = true;
            }

            if (token_id == 151643 || token_id == 151645) break;

            // TOKEN DECODE
            auto decode_start = std::chrono::high_resolution_clock::now();
            const std::string* token_str = nullptr;
            auto cache_it = fast_token_cache_.find(token_id);
            if (cache_it != fast_token_cache_.end()) {
                token_str = &cache_it->second;
            } else if (token_id >= 0 && token_id < (int)id_to_token_.size() && !id_to_token_[token_id].empty()) {
                token_str = &id_to_token_[token_id];
            }
            
            if (token_str) {
                std::cout << *token_str << std::flush;
                decoded_text += *token_str;
            }
            auto decode_end = std::chrono::high_resolution_clock::now();
            double decode_time = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count() / 1000.0;
            decode_times.push_back(decode_time);

            if (short_answer && decoded_text.find_first_of(".\n!?") != std::string::npos) {
                break;
            }

            // TENSOR PREP FOR NEXT ITERATION
            auto next_prep_start = std::chrono::high_resolution_clock::now();
            input_tensors.clear();
            
            for (size_t i = 0; i < num_layers_ * 2; i++) {
                input_tensors.push_back(std::move(output_tensors[i]));
            }
            
            current_tokens_.clear();
            current_tokens_.push_back(static_cast<int32_t>(token_id));
            
            input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
                memory_info_, current_tokens_.data(), 1, token_shape_.data(), token_shape_.size()));
            
            history_len_data_[0] += ids_len_data_[0];
            ids_len_data_[0] = 1;
            attention_mask_data_[0] = 0;
            
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info_, history_len_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info_, ids_len_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<int8_t>(
                memory_info_, attention_mask_data_.data(), 1, scalar_shape_.data(), scalar_shape_.size()));
            
            auto next_prep_end = std::chrono::high_resolution_clock::now();
            double next_prep_time = std::chrono::duration_cast<std::chrono::microseconds>(next_prep_end - next_prep_start).count() / 1000.0;
            tensor_prep_times.push_back(next_prep_time);
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        
        // Calculate statistics
        double avg_inference = 0, min_inference = 1e9, max_inference = 0;
        double avg_decode = 0, avg_prep = 0;
        
        for (double t : inference_times) {
            avg_inference += t;
            min_inference = std::min(min_inference, t);
            max_inference = std::max(max_inference, t);
        }
        avg_inference /= inference_times.size();
        
        for (double t : decode_times) avg_decode += t;
        avg_decode /= decode_times.size();
        
        for (double t : tensor_prep_times) avg_prep += t;
        if (!tensor_prep_times.empty()) avg_prep /= tensor_prep_times.size();
        
        double total_inference = 0;
        for (double t : inference_times) total_inference += t;
        
        double total_decode = 0;
        for (double t : decode_times) total_decode += t;
        
        double total_prep = 0;
        for (double t : tensor_prep_times) total_prep += t;
        
        float rate = num_decoded / (total_duration.count() / 1000.0f);
        
        std::cout << "\n\n📈 PERFORMANCE BREAKDOWN:\n" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cout << "Total tokens generated: " << num_decoded << std::endl;
        std::cout << "Total time: " << total_duration.count() << "ms" << std::endl;
        std::cout << "Overall speed: " << std::fixed << std::setprecision(2) << rate << " tok/s" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        
        std::cout << "\n⏱️  TIME BREAKDOWN:" << std::endl;
        std::cout << "  Initial encode:     " << std::fixed << std::setprecision(2) << encode_time << "ms" << std::endl;
        std::cout << "  Initial prep:       " << prep_time << "ms" << std::endl;
        std::cout << "  Total inference:    " << total_inference << "ms (" 
                  << (total_inference / total_duration.count() * 100) << "%)" << std::endl;
        std::cout << "  Total decode:       " << total_decode << "ms (" 
                  << (total_decode / total_duration.count() * 100) << "%)" << std::endl;
        std::cout << "  Total tensor prep:  " << total_prep << "ms (" 
                  << (total_prep / total_duration.count() * 100) << "%)" << std::endl;
        
        std::cout << "\n📊 PER-ITERATION STATS:" << std::endl;
        std::cout << "  Avg inference:      " << avg_inference << "ms/token" << std::endl;
        std::cout << "  Min inference:      " << min_inference << "ms/token" << std::endl;
        std::cout << "  Max inference:      " << max_inference << "ms/token" << std::endl;
        std::cout << "  Avg decode:         " << avg_decode << "ms/token" << std::endl;
        std::cout << "  Avg tensor prep:    " << avg_prep << "ms/token" << std::endl;
        
        std::cout << "\n🎯 BOTTLENECK ANALYSIS:" << std::endl;
        if (total_inference > total_duration.count() * 0.8) {
            std::cout << "  ⚠️  INFERENCE is the bottleneck (>80% of time)" << std::endl;
            std::cout << "      → Model computation dominates" << std::endl;
            std::cout << "      → Consider: quantization, smaller model, better GPU" << std::endl;
        } else if (total_prep > total_duration.count() * 0.1) {
            std::cout << "  ⚠️  TENSOR PREP is significant (>10% of time)" << std::endl;
            std::cout << "      → Tensor creation/movement overhead" << std::endl;
        }
        
        if (avg_inference > 50) {
            std::cout << "  ⚠️  High per-token latency (" << avg_inference << "ms)" << std::endl;
            std::cout << "      → Expected: 10-30ms for 8B model on modern GPU" << std::endl;
            std::cout << "      → Check: GPU utilization, CUDA streams, batch size" << std::endl;
        }
        
        std::cout << "\nDevice: " << (use_cuda_ ? "CUDA GPU" : "CPU") << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
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

        std::string config_path = "./config.json";
        auto config = nlohmann::json::parse(std::ifstream(config_path));
        
        std::string model_dir = config["paths"]["model_path"];
        std::string onnx_file = config["paths"]["onnx_file"];
        std::string model_config = config["paths"]["model_config"];

        ProfiledOnnxInference inference(model_dir, onnx_file, model_config);

        std::string prompt = "What is 2+2?";
        if (!positional.empty()) {
            prompt = "";
            for (size_t i = 0; i < positional.size(); ++i) {
                if (i > 0) prompt += " ";
                prompt += positional[i];
            }
        }

        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Profiled ONNX Answering:\n" << std::endl;
        
        inference.run_inference(prompt, max_tokens, false);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}