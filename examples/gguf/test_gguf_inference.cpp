// ============================================================================
// PromeTorch GGUF Inference Test
//
// Load a GGUF model (from Ollama or file path) and generate text.
//
// Usage:
//   test_gguf_inference gemma3:4b "What is 2+2?"
//   test_gguf_inference /path/to/model.gguf "Hello world"
//   test_gguf_inference gemma3:4b --info    (print metadata only)
// ============================================================================

#include "torch/io/gguf_model.h"
#include <iostream>
#include <string>
#include <cstring>

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " <model> [options] [prompt]" << std::endl;
    std::cout << std::endl;
    std::cout << "Model can be:" << std::endl;
    std::cout << "  Ollama name:  gemma3:4b, qwen3:4b, deepseek-r1:8b" << std::endl;
    std::cout << "  File path:    /path/to/model.gguf" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --info           Print model metadata and exit" << std::endl;
    std::cout << "  --tensors        Print tensor list and exit" << std::endl;
    std::cout << "  --max-tokens N   Maximum tokens to generate (default: 128)" << std::endl;
    std::cout << "  --temp T         Temperature (default: 0.7)" << std::endl;
    std::cout << "  --top-k K        Top-k sampling (default: 40)" << std::endl;
    std::cout << "  --top-p P        Top-p sampling (default: 0.9)" << std::endl;
    std::cout << "  --device cuda    Run on GPU (CUDA)" << std::endl;
    std::cout << "  --cuda           Same as --device cuda" << std::endl;
    std::cout << "  --greedy         Greedy decoding (temp=0)" << std::endl;
    std::cout << "  --chat           Apply chat template (recommended for questions)" << std::endl;
    std::cout << "  --raw            No chat template (default, good for completions)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt;
    bool info_only = false;
    bool tensors_only = false;
    bool use_cuda = false;
    bool use_chat = false;
    int max_tokens = 128;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--info") {
            info_only = true;
        } else if (arg == "--tensors") {
            tensors_only = true;
        } else if (arg == "--device" && i + 1 < argc) {
            std::string dev = argv[++i];
            if (dev == "cuda" || dev == "gpu") use_cuda = true;
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--greedy") {
            temperature = 0.0f;
        } else if (arg == "--chat") {
            use_chat = true;
        } else if (arg == "--raw") {
            use_chat = false;
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (arg[0] != '-') {
            prompt = arg;
        }
    }

    try {
        // ================================================================
        // Info/tensors only mode — just parse GGUF, don't load weights
        // ================================================================
        if (info_only || tensors_only) {
            torch::io::gguf::GGUFReader reader;

            // Check if it's an Ollama model name or file path
            std::string gguf_path;
            if (model_path.find('/') != std::string::npos ||
                model_path.find('\\') != std::string::npos ||
                model_path.find(".gguf") != std::string::npos) {
                gguf_path = model_path;
            } else {
                gguf_path = torch::io::ollama::resolve_model(model_path);
            }

            reader.open(gguf_path);

            if (info_only) {
                reader.print_metadata();

                // Also parse and print config
                torch::io::TransformerConfig config;
                config.parse(reader);
                config.print();
            }

            if (tensors_only) {
                reader.print_tensors();
            }

            return 0;
        }

        // ================================================================
        // Full model load + inference
        // ================================================================

        if (prompt.empty()) {
            prompt = "Hello, how are you?";
            std::cout << "[Using default prompt: \"" << prompt << "\"]" << std::endl;
        }

        std::cout << "============================================" << std::endl;
        std::cout << " PromeTorch GGUF Inference" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << "Max tokens: " << max_tokens << std::endl;
        std::cout << "Temperature: " << temperature << std::endl;
        std::cout << "Top-k: " << top_k << std::endl;
        std::cout << "Top-p: " << top_p << std::endl;
        std::cout << "Device: " << (use_cuda ? "CUDA" : "CPU") << std::endl;
        std::cout << "Mode: " << (use_chat ? "Chat" : "Completion") << std::endl;
        std::cout << "============================================" << std::endl;

        // Load model
        torch::io::GGUFModel model;

        if (model_path.find('/') != std::string::npos ||
            model_path.find('\\') != std::string::npos ||
            model_path.find(".gguf") != std::string::npos) {
            model.load(model_path);
        } else {
            model.load_ollama(model_path);
        }

        // Move to CUDA if requested
        if (use_cuda) {
            model.to_cuda();
        }

        // Generate
        std::cout << "\n--- Generation ---" << std::endl;
        std::string response;
        if (use_chat) {
            response = model.chat(prompt, max_tokens, temperature, top_k, top_p);
        } else {
            response = model.generate(prompt, max_tokens, temperature, top_k, top_p);
        }

        std::cout << "\n--- Full Response ---" << std::endl;
        std::cout << response << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
