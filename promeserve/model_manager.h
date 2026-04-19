#pragma once

// ============================================================================
// PromeServe — Model Manager
//
// Scans Ollama model directory, lists available models, manages loading/unloading.
// ============================================================================

#include "torch/io/gguf_model.h"
#include "torch/io/ollama.h"

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace promeserve {

namespace fs = std::filesystem;

struct ModelInfo {
    std::string name;         // e.g., "qwen3:4b"
    std::string family;       // e.g., "qwen3"
    std::string tag;          // e.g., "4b"
    std::string gguf_path;    // resolved blob path
    int64_t size_bytes = 0;   // file size
    std::string digest;       // sha256 digest from manifest

    // Populated after loading
    std::string architecture;
    int64_t param_count = 0;
    int64_t context_length = 0;

    std::string size_str() const {
        if (size_bytes >= 1024LL * 1024 * 1024) {
            return std::to_string(size_bytes / (1024LL * 1024 * 1024)) + " GB";
        } else if (size_bytes >= 1024LL * 1024) {
            return std::to_string(size_bytes / (1024LL * 1024)) + " MB";
        }
        return std::to_string(size_bytes) + " B";
    }
};

class ModelManager {
public:
    ModelManager() : use_cuda_(false) {}

    void set_device(const std::string& device) {
        use_cuda_ = (device == "cuda" || device == "gpu");
    }

    // ========================================================================
    // Scan Ollama model directory for available models
    // ========================================================================

    void scan_models() {
        std::lock_guard<std::mutex> lock(mutex_);
        models_.clear();

        std::string ollama_home;
        try {
            ollama_home = torch::io::ollama::get_ollama_home();
        } catch (...) {
            std::cerr << "[ModelManager] Could not determine Ollama home directory" << std::endl;
            return;
        }

#ifdef _WIN32
        std::string manifests_dir = ollama_home + "\\manifests\\registry.ollama.ai\\library";
#else
        std::string manifests_dir = ollama_home + "/manifests/registry.ollama.ai/library";
#endif

        if (!fs::exists(manifests_dir)) {
            std::cerr << "[ModelManager] Ollama manifests directory not found: " << manifests_dir << std::endl;
            return;
        }

        // Iterate model families (directories under library/)
        for (auto& family_entry : fs::directory_iterator(manifests_dir)) {
            if (!family_entry.is_directory()) continue;
            std::string family_name = family_entry.path().filename().string();

            // Iterate tags (files under family/)
            for (auto& tag_entry : fs::directory_iterator(family_entry.path())) {
                if (!tag_entry.is_regular_file()) continue;
                std::string tag_name = tag_entry.path().filename().string();
                std::string full_name = family_name + ":" + tag_name;

                ModelInfo info;
                info.name = full_name;
                info.family = family_name;
                info.tag = tag_name;

                // Try to resolve GGUF path
                try {
                    info.gguf_path = torch::io::ollama::resolve_model(full_name);

                    // Get file size
                    if (fs::exists(info.gguf_path)) {
                        info.size_bytes = static_cast<int64_t>(fs::file_size(info.gguf_path));
                    }

                    // Read digest from manifest
                    std::ifstream mf(tag_entry.path().string());
                    if (mf) {
                        std::stringstream ss;
                        ss << mf.rdbuf();
                        info.digest = torch::io::ollama::find_model_digest(ss.str());
                    }
                } catch (...) {
                    // Skip models we can't resolve
                    continue;
                }

                models_[full_name] = info;

                // Add alias: family:latest -> family:latest (already there)
                // Also map bare family name to latest if tag is "latest"
                if (tag_name == "latest") {
                    models_[family_name] = info;
                    models_[family_name].name = family_name;
                }
            }
        }

        std::cout << "[ModelManager] Found " << list_models_locked().size() << " models" << std::endl;
    }

    // Internal: list with lock already held by caller.
    std::vector<ModelInfo> list_models_locked() const {
        std::vector<ModelInfo> result;
        std::map<std::string, bool> seen;
        for (auto& kv : models_) {
            if (kv.second.gguf_path.empty()) continue;
            if (seen.count(kv.second.gguf_path)) continue;
            seen[kv.second.gguf_path] = true;
            result.push_back(kv.second);
        }
        return result;
    }

    // ========================================================================
    // List available models (for GET /api/tags)
    // ========================================================================

    std::vector<ModelInfo> list_models() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<ModelInfo> result;
        // Deduplicate by gguf_path
        std::map<std::string, bool> seen;
        for (auto& kv : models_) {
            if (kv.second.gguf_path.empty()) continue;
            if (seen.count(kv.second.gguf_path)) continue;
            seen[kv.second.gguf_path] = true;
            result.push_back(kv.second);
        }
        return result;
    }

    // ========================================================================
    // Get model info
    // ========================================================================

    bool has_model(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return models_.count(name) > 0;
    }

    ModelInfo get_model_info(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = models_.find(name);
        if (it != models_.end()) return it->second;
        return ModelInfo{};
    }

    // ========================================================================
    // Load / unload model
    // ========================================================================

    torch::io::GGUFModel* get_loaded_model() {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_model_.get();
    }

    std::string loaded_model_name() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_model_name_;
    }

    bool load_model(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Already loaded?
        if (loaded_model_ && loaded_model_name_ == name) {
            return true;
        }

        // Resolve path
        std::string gguf_path;
        auto it = models_.find(name);
        if (it != models_.end()) {
            gguf_path = it->second.gguf_path;
        } else {
            // Try direct resolution
            try {
                gguf_path = torch::io::ollama::resolve_model(name);
            } catch (...) {
                // Try as file path
                if (fs::exists(name)) {
                    gguf_path = name;
                } else {
                    std::cerr << "[ModelManager] Model not found: " << name << std::endl;
                    return false;
                }
            }
        }

        std::cout << "[ModelManager] Loading model: " << name << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();

        // Unload previous model
        loaded_model_.reset();
        loaded_model_name_.clear();

        try {
            auto model = std::make_unique<torch::io::GGUFModel>();
#ifdef PT_DEBUG_HTTP
            std::cerr << "[ModelManager] Loading GGUF: " << gguf_path << std::endl;
#endif
            model->load(gguf_path);
#ifdef PT_DEBUG_HTTP
            std::cerr << "[ModelManager] GGUF loaded. use_cuda_=" << use_cuda_
                      << " model->use_cuda_=" << model->use_cuda_
                      << " model->use_quant_gemv_=" << model->use_quant_gemv_
                      << " arch=" << model->config.architecture
                      << " layers=" << model->config.num_layers
                      << " hidden=" << model->config.hidden_size
                      << " vocab=" << model->config.vocab_size << std::endl;
#endif

#ifdef PT_USE_CUDA
            if (use_cuda_) {
                model->to_cuda();
                model->load_quantized_to_cuda();
                std::cout << "[ModelManager] Quantized weights loaded to GPU" << std::endl;
            }
#endif

            loaded_model_name_ = name;
            loaded_model_ = std::move(model);

            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << "[ModelManager] Model loaded in " << (ms / 1000.0) << "s" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[ModelManager] Failed to load model: " << e.what() << std::endl;
            return false;
        }
    }

    void unload_model() {
        std::lock_guard<std::mutex> lock(mutex_);
        loaded_model_.reset();
        loaded_model_name_.clear();
    }

private:
    std::map<std::string, ModelInfo> models_;
    std::unique_ptr<torch::io::GGUFModel> loaded_model_;
    std::string loaded_model_name_;
    bool use_cuda_;
    mutable std::mutex mutex_;
};

}  // namespace promeserve
