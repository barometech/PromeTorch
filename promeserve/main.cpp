// ============================================================================
// PromeServe — CLI Entry Point
//
// Ollama-compatible LLM inference server built on PromeTorch.
//
// Usage:
//   promeserve [--port 11434] [--device cuda|cpu] [--model qwen3:4b]
//
// Examples:
//   promeserve                              # Start on default port, CUDA
//   promeserve --port 8080 --device cpu     # CPU mode, custom port
//   promeserve --model qwen3:4b             # Pre-load a model at startup
//
// API endpoints (Ollama-compatible):
//   POST /api/generate  — text completion (streaming NDJSON)
//   POST /api/chat      — chat completion (streaming NDJSON)
//   GET  /api/tags      — list available models
//   POST /api/show      — model info
//   GET  /api/version   — server version
//   GET  /              — health check
// ============================================================================

#include "promeserve/promeserve.h"

#include <iostream>
#include <string>
#include <cstring>
#include <csignal>

static promeserve::PromeServe* g_server = nullptr;

void signal_handler(int sig) {
    (void)sig;
    std::cout << "\n[PromeServe] Shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void print_usage(const char* argv0) {
    std::cout << "PromeServe — PromeTorch LLM Inference Server" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << argv0 << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --port PORT      Listen port (default: 11434)" << std::endl;
    std::cout << "  --device DEVICE  Device: cuda, cpu (default: cuda)" << std::endl;
    std::cout << "  --model MODEL    Pre-load model at startup (e.g., qwen3:4b)" << std::endl;
    std::cout << "  --help           Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "API endpoints (Ollama-compatible):" << std::endl;
    std::cout << "  POST /api/generate  — text completion" << std::endl;
    std::cout << "  POST /api/chat      — chat completion" << std::endl;
    std::cout << "  GET  /api/tags      — list models" << std::endl;
    std::cout << "  POST /api/show      — model info" << std::endl;
    std::cout << "  GET  /api/version   — version" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << argv0 << " --port 11434 --device cuda --model qwen3:4b" << std::endl;
    std::cout << std::endl;
    std::cout << "Then use with any Ollama client:" << std::endl;
    std::cout << "  curl http://localhost:11434/api/generate -d '{\"model\":\"qwen3:4b\",\"prompt\":\"Hello\"}'" << std::endl;
}

int main(int argc, char* argv[]) {
    int port = 11434;
    std::string device = "cuda";
    std::string model;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--cpu") {
            device = "cpu";
        } else if (arg == "--cuda" || arg == "--gpu") {
            device = "cuda";
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Register signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    promeserve::PromeServe server;
    g_server = &server;

    server.start(port, device, model);

    return 0;
}
