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
#include "torch/distributed/ddp.h"
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
    std::cout << "  --profile        Enable GPU profiling (timing breakdown)" << std::endl;
    std::cout << "  --fp16-weights   Dequant Q4_K -> FP16 at load; cuBLAS HGEMV decode (CUDA only)" << std::endl;
    std::cout << "  --llama-gemv     Use llama.cpp-style Q4_K GEMV v2 kernel (CUDA only)" << std::endl;
    std::cout << std::endl;
    std::cout << "Multi-process tensor-parallel (CPU, Elbrus NUMA):" << std::endl;
    std::cout << "  --nprocs N       Number of tensor-parallel processes (default 1)" << std::endl;
    std::cout << "  --rank R         This process's rank in [0, N) (default 0)" << std::endl;
    std::cout << "  --master-addr A  Master address for rank-0 hub (default 127.0.0.1)" << std::endl;
    std::cout << "  --master-port P  Master port (default 29500)" << std::endl;
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
    bool use_profile = false;
    bool use_fp16_weights = false;
    bool use_llama_gemv = false;
    int max_tokens = 128;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;

    // Tensor-parallel / multi-process config
    int tp_nprocs = 1;
    int tp_rank = 0;
    std::string tp_master_addr = "127.0.0.1";
    int tp_master_port = 29500;

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
        } else if (arg == "--profile") {
            use_profile = true;
        } else if (arg == "--fp16-weights" || arg == "--fp16_weights") {
            use_fp16_weights = true;
        } else if (arg == "--llama-gemv" || arg == "--llama_gemv") {
            use_llama_gemv = true;
        } else if ((arg == "--max-tokens" || arg == "--max_tokens") && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if ((arg == "--temp" || arg == "--temperature") && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--nprocs" && i + 1 < argc) {
            tp_nprocs = std::atoi(argv[++i]);
        } else if (arg == "--rank" && i + 1 < argc) {
            tp_rank = std::atoi(argv[++i]);
        } else if ((arg == "--master-addr" || arg == "--master_addr") && i + 1 < argc) {
            tp_master_addr = argv[++i];
        } else if ((arg == "--master-port" || arg == "--master_port") && i + 1 < argc) {
            tp_master_port = std::atoi(argv[++i]);
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

        // --- Multi-process tensor-parallel init (CPU-only) ---
        bool tp_mode = (tp_nprocs > 1);
        if (tp_mode) {
            if (use_cuda) {
                std::cerr << "Error: --nprocs > 1 is CPU-only (no CUDA multi-process yet)"
                          << std::endl;
                return 1;
            }
            if (tp_rank < 0 || tp_rank >= tp_nprocs) {
                std::cerr << "Error: --rank must be in [0, " << tp_nprocs << ")" << std::endl;
                return 1;
            }
            torch::distributed::DDPConfig cfg;
            cfg.rank        = tp_rank;
            cfg.world_size  = tp_nprocs;
            cfg.master_addr = tp_master_addr;
            cfg.master_port = tp_master_port;
            cfg.timeout_sec = 300;
            std::cout << "[TP] rank " << tp_rank << "/" << tp_nprocs
                      << " connecting to " << tp_master_addr << ":" << tp_master_port << std::endl;
            torch::distributed::init_process_group(cfg);
        }

        // Load model
        torch::io::GGUFModel model;

        if (model_path.find('/') != std::string::npos ||
            model_path.find('\\') != std::string::npos ||
            model_path.find(".gguf") != std::string::npos) {
            model.load(model_path);
        } else {
            model.load_ollama(model_path);
        }

        if (tp_mode) {
            if (!model.init_tensor_parallel(tp_rank, tp_nprocs)) {
                std::cerr << "Error: init_tensor_parallel failed for rank " << tp_rank << std::endl;
                torch::distributed::destroy_process_group();
                return 1;
            }
        }

        // Move to CUDA if requested
        if (use_cuda) {
            model.to_cuda();
            // Load quantized weights for fast decode GEMV
            model.load_quantized_to_cuda();

            // Optional: dequant Q4_K -> FP16 for cuBLAS HGEMV decode
            if (use_fp16_weights) {
                bool ok = model.dequant_all_to_fp16();
                std::cout << "[Model] FP16 weights path: "
                          << (ok ? "ENABLED" : "FAILED / using quant fallback") << std::endl;
            }

            // Optional: route Q4_K GEMV through llama.cpp-style v2 kernel
            if (use_llama_gemv) {
                model.use_llama_gemv_ = true;
                std::cout << "[Model] llama-style Q4_K GEMV v2: ENABLED" << std::endl;
            }
        }

        // Enable profiling if requested
        if (use_profile) {
            model.profiler.enable();
            std::cout << "[Profile] GPU profiling enabled" << std::endl;
        }

        // Generate
        if (tp_mode) {
            if (tp_rank == 0) {
                std::cout << "\n--- Generation (tensor-parallel, nprocs=" << tp_nprocs
                          << ") ---" << std::endl;
            }
            std::string response = model.generate_tp(prompt, max_tokens, temperature);
            if (tp_rank == 0) {
                std::cout << "\n--- Full Response ---" << std::endl;
                std::cout << response << std::endl;
            }
            torch::distributed::destroy_process_group();
        } else {
            std::cout << "\n--- Generation ---" << std::endl;
            std::string response;
            if (use_chat) {
                response = model.chat(prompt, max_tokens, temperature, top_k, top_p);
            } else {
                response = model.generate(prompt, max_tokens, temperature, top_k, top_p);
            }

            std::cout << "\n--- Full Response ---" << std::endl;
            std::cout << response << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (torch::distributed::is_initialized()) {
            torch::distributed::destroy_process_group();
        }
        return 1;
    }

    return 0;
}
