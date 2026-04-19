#pragma once

// ============================================================================
// PromeServe — Ollama-Compatible LLM Inference Server
//
// Built on PromeTorch. Drop-in replacement for Ollama's REST API.
// Existing clients (Open WebUI, LangChain, etc.) work out of the box.
//
// Usage:
//   PromeServe server;
//   server.start(11434, "cuda");
// ============================================================================

#include "promeserve/http_server.h"
#include "promeserve/model_manager.h"
#include "promeserve/api_handlers.h"

#include <string>
#include <iostream>

namespace promeserve {

class PromeServe {
public:
    PromeServe() : handlers_(models_) {}

    // Configure thread pool, queue depth, request timeout.
    void set_config(const ServerConfig& cfg) { config_ = cfg; }
    const ServerConfig& config() const { return config_; }

    // ========================================================================
    // Start the server
    // ========================================================================

    void start(int port = 11434, const std::string& device = "cuda",
               const std::string& preload_model = "") {

        std::cout << "==========================================" << std::endl;
        std::cout << "  PromeServe v0.1.0" << std::endl;
        std::cout << "  PromeTorch LLM Inference Server" << std::endl;
        std::cout << "  Ollama-compatible REST API" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << std::endl;
        std::cout << "  Device:  " << device << std::endl;
        std::cout << "  Port:    " << port << std::endl;

        // Configure device
        models_.set_device(device);

        // Scan for available models
        std::cout << std::endl;
        models_.scan_models();

        auto model_list = models_.list_models();
        if (!model_list.empty()) {
            std::cout << "\nAvailable models:" << std::endl;
            for (auto& m : model_list) {
                std::cout << "  - " << m.name << " (" << m.size_str() << ")" << std::endl;
            }
        } else {
            std::cout << "\nNo Ollama models found. Pull models with: ollama pull <model>" << std::endl;
        }

        // Pre-load a model if specified
        if (!preload_model.empty()) {
            std::cout << "\nPre-loading model: " << preload_model << std::endl;
            if (!models_.load_model(preload_model)) {
                std::cerr << "Warning: Failed to pre-load model " << preload_model << std::endl;
            }
        }

        // Apply server config (thread pool, queue, timeout)
        server_.set_config(config_);

        // Register API routes (handlers pick up timeout from server)
        handlers_.register_routes(server_);

        // Start listening (blocking)
        std::cout << std::endl;
        server_.start(port);
    }

    void stop() {
        server_.stop();
    }

    ModelManager& models() { return models_; }

private:
    HttpServer server_;
    ModelManager models_;
    ApiHandlers handlers_;
    ServerConfig config_;
};

}  // namespace promeserve
