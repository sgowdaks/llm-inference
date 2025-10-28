#include <iostream>
#include "tokenizers/tokenizers.h"
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

static std::vector<std::string> build_id_to_token_map(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path);
    if (!f.is_open()) return {};
    json j;
    f >> j;

    // model.vocab is a mapping token->id
    std::vector<std::string> id_to_token;
    if (j.contains("model") && j["model"].contains("vocab")) {
        auto vocab = j["model"]["vocab"];
        // find max id
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

    // added_tokens (special tokens) may be present
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

int main() {
    try {
        tokenizers::Tokenizer tokenizer("/home/shivani/work/llm-inference/Qwen3-8B/tokenizer.json");
        if (!tokenizer.valid()) {
            std::cerr << "Failed to load tokenizer\n";
            return 1;
        }
        
        std::string text = "Hello, how are you?";
        auto ids = tokenizer.encode(text, true);
        std::cout << "Encoded ids: ";
        for (const auto& id : ids) {
            std::cout << id << " ";
        }
        std::cout << "\n";
        
        // naive decode using tokenizer.json vocab -> token strings
        auto id_to_token = build_id_to_token_map("/home/shivani/work/llm-inference/Qwen3-8B/tokenizer.json");
        std::string decoded;
        for (auto id : ids) {
            if (id >= 0 && id < (int)id_to_token.size() && !id_to_token[id].empty()) {
                std::string tok = id_to_token[id];
                // replace occurrences of UTF-8 bytes for 'Ġ' (0xC4 0xA0) with space
                size_t pos = 0;
                while (true) {
                    auto it = tok.find("\xC4\xA0", pos);
                    if (it == std::string::npos) break;
                    tok.replace(it, 2, " ");
                    pos = it + 1;
                }
                decoded += tok;
            } else {
                decoded += "[" + std::to_string(id) + "]";
            }
        }
        std::cout << "Decoded text (naive): " << decoded << "\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}