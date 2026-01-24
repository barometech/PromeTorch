// ============================================================================
// PIR 270M Training Example
// ============================================================================
// Character-level language modeling with PIR architecture
//
// Usage:
//   ./train_pir <text_file> [--vocab_size N] [--n_embd N] [--n_layers N]
//
// Example:
//   ./train_pir shakespeare.txt --vocab_size 256 --n_embd 128 --n_layers 4
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "c10/util/MemoryDebug.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <random>
#include <map>
#include <set>

// Global device setting
static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

// Helper to move tensor to device
inline at::Tensor to_device(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) {
        return at::to_cuda(t);  // at::to_cuda, not at::cuda::to_cuda
    }
#endif
    return t;
}

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

// ============================================================================
// Character-Level Tokenizer
// ============================================================================

class CharTokenizer {
public:
    CharTokenizer() = default;

    // Build vocabulary from text
    void build_vocab(const std::string& text) {
        std::set<char> unique_chars(text.begin(), text.end());

        int idx = 0;
        for (char c : unique_chars) {
            char_to_idx_[c] = idx;
            idx_to_char_[idx] = c;
            idx++;
        }

        vocab_size_ = static_cast<int64_t>(unique_chars.size());
        std::cout << "Vocabulary size: " << vocab_size_ << std::endl;
    }

    // Encode string to token indices
    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> tokens;
        tokens.reserve(text.size());
        for (char c : text) {
            auto it = char_to_idx_.find(c);
            if (it != char_to_idx_.end()) {
                tokens.push_back(it->second);
            }
        }
        return tokens;
    }

    // Decode token indices to string
    std::string decode(const std::vector<int64_t>& tokens) const {
        std::string text;
        text.reserve(tokens.size());
        for (int64_t idx : tokens) {
            auto it = idx_to_char_.find(idx);
            if (it != idx_to_char_.end()) {
                text += it->second;
            }
        }
        return text;
    }

    int64_t vocab_size() const { return vocab_size_; }

private:
    std::map<char, int> char_to_idx_;
    std::map<int, char> idx_to_char_;
    int64_t vocab_size_ = 0;
};

// ============================================================================
// Data Loader
// ============================================================================

class TextDataLoader {
public:
    TextDataLoader(
        const std::vector<int64_t>& data,
        int64_t batch_size,
        int64_t block_size
    ) : data_(data)
      , batch_size_(batch_size)
      , block_size_(block_size)
      , pos_(0)
    {
        // Random number generator
        std::random_device rd;
        gen_ = std::mt19937(rd());
        dist_ = std::uniform_int_distribution<int64_t>(
            0, static_cast<int64_t>(data_.size()) - block_size_ - 1
        );
    }

    // Get a random batch
    std::pair<Tensor, Tensor> get_batch() {
        Tensor inputs = at::empty({batch_size_, block_size_});
        Tensor targets = at::empty({batch_size_, block_size_});

        float* in_data = inputs.mutable_data_ptr<float>();
        float* tgt_data = targets.mutable_data_ptr<float>();

        for (int64_t b = 0; b < batch_size_; ++b) {
            int64_t start = dist_(gen_);
            for (int64_t t = 0; t < block_size_; ++t) {
                in_data[b * block_size_ + t] = static_cast<float>(data_[start + t]);
                tgt_data[b * block_size_ + t] = static_cast<float>(data_[start + t + 1]);
            }
        }

        // Move to device (GPU if configured)
        return {to_device(inputs), to_device(targets)};
    }

private:
    std::vector<int64_t> data_;
    int64_t batch_size_;
    int64_t block_size_;
    int64_t pos_;
    std::mt19937 gen_;
    std::uniform_int_distribution<int64_t> dist_;
};

// ============================================================================
// Training Loop
// ============================================================================

void train(
    PIR270M& model,
    TextDataLoader& train_loader,
    AdamW& optimizer,
    int64_t num_iterations,
    int64_t log_interval,
    int64_t eval_interval
) {
    std::cout << "\n=== Training PIR 270M ===" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    float running_loss = 0.0f;
    int loss_count = 0;

    for (int64_t iter = 1; iter <= num_iterations; ++iter) {
        // Reset debug counters at start of each iteration
        torch::autograd::g_nodes_created = 0;
        torch::autograd::g_nodes_destroyed = 0;
        torch::autograd::g_nodes_released = 0;
        torch::autograd::g_accum_grad_created = 0;
        torch::autograd::g_accum_grad_destroyed = 0;
        torch::autograd::g_weak_ptr_hit = 0;
        torch::autograd::g_weak_ptr_miss = 0;
        torch::autograd::g_weak_ptr_upgrade = 0;
        torch::autograd::g_meta_create_count = 0;
        torch::autograd::g_meta_upgrade_count = 0;

        // Memory checkpoint at start of each iteration
#ifdef PT_USE_CUDA
        if (g_device.is_cuda()) {
            size_t cuda_used = 0, cuda_total = 0;
            cudaMemGetInfo(&cuda_used, &cuda_total);
            cuda_used = cuda_total - cuda_used;  // GetInfo returns free, not used
            std::cout << "[MEM iter " << iter << "] CUDA: " << (cuda_used / 1048576.0) << " MB used" << std::endl;
        }
#endif
        std::cout << "DBG: iter " << iter << " - get_batch" << std::endl;
        std::cout.flush();
        // Get batch
        auto [inputs, targets] = train_loader.get_batch();
        std::cout << "DBG: inputs shape [" << inputs.size(0) << ", " << inputs.size(1) << "]" << std::endl;
        std::cout.flush();

        // Zero gradients
        std::cout << "DBG: zero_grad" << std::endl;
        std::cout.flush();
        optimizer.zero_grad();

        // Forward pass
        std::cout << "DBG: forward_with_loss" << std::endl;
        std::cout.flush();
        auto [logits, loss] = model.forward_with_loss(inputs, targets);
        std::cout << "DBG: forward done" << std::endl;
        std::cout.flush();

        // Copy loss to CPU for reading
        std::cout << "DBG: copy loss to CPU" << std::endl;
        std::cout.flush();
        Tensor loss_cpu = loss;
#ifdef PT_USE_CUDA
        if (loss.is_cuda()) {
            loss_cpu = at::to_cpu(loss);
        }
#endif
        std::cout << "DBG: read loss val" << std::endl;
        std::cout.flush();
        float loss_val = loss_cpu.data_ptr<float>()[0];
        running_loss += loss_val;
        loss_count++;
        std::cout << "DBG: loss_val=" << loss_val << std::endl;
        std::cout.flush();

        // Backward pass
        std::cout << "DBG: backward" << std::endl;
        std::cout.flush();
        torch::autograd::backward({loss});
        std::cout << "DBG: backward done" << std::endl;
        std::cout.flush();

        // CRITICAL: Clear autograd graph to prevent memory leak!
        // Without this, backward functions accumulate and cause heap corruption
        torch::autograd::clear_grad_fn(loss);
        torch::autograd::clear_grad_fn(logits);

        // Also clear the tensors themselves to release graph references
        loss = Tensor();
        logits = Tensor();

        // Also clear inputs and targets to release any graph references
        inputs = Tensor();
        targets = Tensor();

        // Gradient clipping
        std::cout << "DBG: clip_grad_norm_" << std::endl;
        std::cout.flush();
        clip_grad_norm_(model, 1.0);
        std::cout << "DBG: clip_grad_norm_ done" << std::endl;
        std::cout.flush();

        // Update weights
        std::cout << "DBG: optimizer.step()" << std::endl;
        std::cout.flush();
        optimizer.step();
        std::cout << "DBG: optimizer.step() done" << std::endl;
        std::cout.flush();

#ifdef PT_USE_CUDA
        // Synchronize CUDA and free cached memory periodically
        if (g_device.is_cuda()) {
            c10::cuda::cuda_synchronize();
            // Free cached memory EVERY iteration to prevent heap corruption
            c10::cuda::CUDACachingAllocator::get().empty_cache();
        }
#endif

        // DEBUG: Print node statistics every iteration
        torch::autograd::print_node_stats();
        torch::autograd::print_weak_ptr_stats();
        torch::autograd::print_meta_stats();

        // DEBUG: Check if parameter gradients have requires_grad
        int grads_with_grad = 0;
        for (auto* param : model.parameters()) {
            if (param->grad().defined() && param->grad().requires_grad()) {
                grads_with_grad++;
            }
            // Also check if grad has grad_fn (non-leaf)
            auto* meta = torch::autograd::get_autograd_meta(param->grad());
            if (meta && meta->grad_fn) {
                std::cout << "[LEAK] Gradient has grad_fn!" << std::endl;
            }
        }
        if (grads_with_grad > 0) {
            std::cout << "[WARN] " << grads_with_grad << " gradients have requires_grad=true" << std::endl;
        }

        // Logging
        if (iter % log_interval == 0) {
            float avg_loss = running_loss / loss_count;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time
            ).count();

            std::cout << "Iter " << iter << "/" << num_iterations
                      << " | Loss: " << avg_loss
                      << " | Time: " << elapsed << "s";
#ifdef PT_USE_CUDA
            if (g_device.is_cuda()) {
                auto& alloc = c10::cuda::CUDACachingAllocator::get();
                std::cout << " | GPU alloc: " << (alloc.get_allocated_memory() / 1048576.0) << " MB"
                          << " cached: " << (alloc.get_cached_memory() / 1048576.0) << " MB";
            }
#endif
            std::cout << std::endl;

            running_loss = 0.0f;
            loss_count = 0;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time
    ).count();

    std::cout << "\nTraining completed in " << total_time << " seconds" << std::endl;
}

// ============================================================================
// Text Generation
// ============================================================================

void generate_text(
    PIR270M& model,
    const CharTokenizer& tokenizer,
    const std::string& prompt,
    int64_t max_tokens = 200
) {
    std::cout << "\n=== Generating Text ===" << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "\nGenerated:\n" << std::string(40, '-') << std::endl;

    std::vector<int64_t> tokens = tokenizer.encode(prompt);

    if (tokens.empty()) {
        std::cout << "(Empty prompt after tokenization)" << std::endl;
        return;
    }

    std::vector<int64_t> generated = model.generate(
        tokens,
        max_tokens,
        0.85,    // temperature
        40,      // top_k
        0.92,    // top_p
        -1,      // eos_token (none for char-level)
        1.15     // repetition_penalty
    );

    std::string output = tokenizer.decode(generated);
    std::cout << output << std::endl;
    std::cout << std::string(40, '-') << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // CRITICAL: Register AutogradMetaImpl factory BEFORE any tensor creation!
    // This prevents the expensive upgrade from base AutogradMeta to AutogradMetaImpl
    c10::set_autograd_meta_factory(&torch::autograd::create_autograd_meta_impl);

    // Default configuration
    std::string text_file = "";
    std::string device_str = "cpu";
    int64_t batch_size = 8;
    int64_t block_size = 256;
    int64_t n_embd = 256;
    int64_t n_layers = 6;
    int64_t n_pir_layers = 3;
    int64_t num_iterations = 1000;
    int64_t log_interval = 50;
    float learning_rate = 3e-4f;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoll(argv[++i]);
        } else if (arg == "--block_size" && i + 1 < argc) {
            block_size = std::stoll(argv[++i]);
        } else if (arg == "--n_embd" && i + 1 < argc) {
            n_embd = std::stoll(argv[++i]);
        } else if (arg == "--n_layers" && i + 1 < argc) {
            n_layers = std::stoll(argv[++i]);
        } else if (arg == "--n_pir_layers" && i + 1 < argc) {
            n_pir_layers = std::stoll(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            num_iterations = std::stoll(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            learning_rate = std::stof(argv[++i]);
        } else if (arg == "--log_interval" && i + 1 < argc) {
            log_interval = std::stoll(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg[0] != '-') {
            text_file = arg;
        }
    }

    // Set device
    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Using CUDA device" << std::endl;
#else
        std::cerr << "Warning: CUDA not available, using CPU" << std::endl;
        g_device = c10::Device(c10::DeviceType::CPU);
#endif
    } else {
        g_device = c10::Device(c10::DeviceType::CPU);
        std::cout << "Using CPU device" << std::endl;
    }

    if (text_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " <text_file> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --batch_size N      Batch size (default: 8)" << std::endl;
        std::cerr << "  --block_size N      Context length (default: 256)" << std::endl;
        std::cerr << "  --n_embd N          Embedding dimension (default: 256)" << std::endl;
        std::cerr << "  --n_layers N        Number of layers (default: 6)" << std::endl;
        std::cerr << "  --n_pir_layers N    PIR layers per block (default: 3)" << std::endl;
        std::cerr << "  --iterations N      Training iterations (default: 1000)" << std::endl;
        std::cerr << "  --lr F              Learning rate (default: 3e-4)" << std::endl;
        std::cerr << "  --log_interval N    Logging interval (default: 50)" << std::endl;
        std::cerr << "  --device D          Device: cpu or cuda (default: cpu)" << std::endl;
        return 1;
    }

    // Load text file
    std::cout << "Loading text from: " << text_file << std::endl;
    std::ifstream file(text_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << text_file << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    std::cout << "Text length: " << text.size() << " characters" << std::endl;

    // Build tokenizer
    CharTokenizer tokenizer;
    tokenizer.build_vocab(text);

    // Encode text
    std::vector<int64_t> data = tokenizer.encode(text);
    std::cout << "Encoded length: " << data.size() << " tokens" << std::endl;

    // Create model
    std::cout << "\n=== Model Configuration ===" << std::endl;
    std::cout << "  Embedding dim: " << n_embd << std::endl;
    std::cout << "  Layers: " << n_layers << std::endl;
    std::cout << "  PIR layers: " << n_pir_layers << std::endl;
    std::cout << "  Block size: " << block_size << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;

    PIR270MConfig config;
    config.vocab_size = tokenizer.vocab_size();
    config.n_embd = n_embd;
    config.n_layers = n_layers;
    config.n_pir_layers = n_pir_layers;
    config.block_size = block_size;
    config.dropout = 0.0;
    config.tie_weights = true;

    PIR270M model(config);

    // Move model to device (GPU if configured)
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        std::cout << "Moving model to CUDA..." << std::endl;
        model.to(g_device);
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    // Create data loader
    TextDataLoader train_loader(data, batch_size, block_size);

    // Create optimizer
    AdamWOptions opts(learning_rate);
    opts.betas(0.9, 0.95);
    opts.weight_decay_(0.1);
    AdamW optimizer(model.parameters(), opts);

    // Train
    train(model, train_loader, optimizer, num_iterations, log_interval, 0);

    // Generate sample text
    std::string prompt = text.substr(0, std::min(static_cast<size_t>(50), text.size()));
    generate_text(model, tokenizer, prompt, 200);

    // Try generating from a simple prompt
    std::string simple_prompt = text.substr(0, 10);
    generate_text(model, tokenizer, simple_prompt, 300);

    std::cout << "\n=== Training Complete ===" << std::endl;

    return 0;
}
