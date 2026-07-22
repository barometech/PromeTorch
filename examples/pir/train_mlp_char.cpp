// ============================================================================
// MLP Character Model - For GPU Utilization Comparison
// ============================================================================
// Simple MLP-based character-level language model to compare against PIR
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <map>
#include <set>

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;
using at::Tensor;

// Global device setting
static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

// Helper for device-aware relu
inline Tensor device_relu(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) {
        return at::cuda_ops::relu(t);
    }
#endif
    return at::native::relu(t);
}

inline Tensor to_device(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) {
        return at::to_cuda(t);
    }
#endif
    return t;
}

// ============================================================================
// Simple MLP Language Model
// ============================================================================

class MLPCharModel : public Module {
public:
    MLPCharModel(int64_t vocab_size, int64_t block_size, int64_t n_embd, int64_t n_hidden)
        : vocab_size_(vocab_size), block_size_(block_size), n_embd_(n_embd) {

        // Token embedding
        embedding_ = std::make_shared<Embedding>(vocab_size, n_embd);
        register_module("embedding", embedding_);

        // Simple MLP: flatten context -> hidden -> output
        int64_t input_size = block_size * n_embd;

        fc1_ = std::make_shared<Linear>(input_size, n_hidden, true);
        register_module("fc1", fc1_);
        fc2_ = std::make_shared<Linear>(n_hidden, n_hidden, true);
        register_module("fc2", fc2_);
        fc3_ = std::make_shared<Linear>(n_hidden, vocab_size, true);
        register_module("fc3", fc3_);

        // Count parameters
        int64_t total_params = 0;
        for (auto& p : parameters()) {
            total_params += p->numel();
        }
        std::cout << "MLP Char Model initialized: " << (total_params / 1e6) << "M parameters" << std::endl;
    }

    Tensor forward(const Tensor& idx) {
        // idx: [batch, block_size]
        int64_t B = idx.size(0);

        // Embed tokens: [batch, block_size, n_embd]
        Tensor x = embedding_->forward(idx);

        // Flatten: [batch, block_size * n_embd]
        x = x.view({B, -1});

        // MLP layers with ReLU (device_relu dispatches to CUDA)
        x = device_relu(fc1_->forward(x));
        x = device_relu(fc2_->forward(x));

        // Output logits: [batch, vocab_size]
        Tensor logits = fc3_->forward(x);

        return logits;
    }

    Tensor forward_with_loss(const Tensor& idx, const Tensor& targets) {
        Tensor logits = forward(idx);  // [batch, vocab_size]

        int64_t B = logits.size(0);
        int64_t V = vocab_size_;
        bool is_cuda = logits.is_cuda();

        // Move to CPU for cross-entropy (same as PIR)
        Tensor logits_cpu = logits;
        Tensor targets_cpu = targets;
#ifdef PT_USE_CUDA
        if (is_cuda) {
            logits_cpu = at::to_cpu(logits);
            targets_cpu = at::to_cpu(targets);
        }
#endif

        const float* logits_data = logits_cpu.data_ptr<float>();
        const int64_t* targets_data = targets_cpu.data_ptr<int64_t>();

        // Compute softmax and loss on CPU
        Tensor softmax = at::empty({B, V});
        float* softmax_data = softmax.mutable_data_ptr<float>();

        float total_loss = 0.0f;
        int64_t count = 0;

        for (int64_t i = 0; i < B; ++i) {
            int64_t offset = i * V;
            int64_t target_idx = targets_data[i];

            // Compute softmax with numerical stability
            float max_logit = logits_data[offset];
            for (int64_t v = 1; v < V; ++v) {
                max_logit = std::max(max_logit, logits_data[offset + v]);
            }

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                float exp_val = std::exp(logits_data[offset + v] - max_logit);
                softmax_data[offset + v] = exp_val;
                sum_exp += exp_val;
            }

            for (int64_t v = 0; v < V; ++v) {
                softmax_data[offset + v] /= sum_exp;
            }

            // Loss for valid targets
            if (target_idx >= 0 && target_idx < V) {
                float log_prob = std::log(softmax_data[offset + target_idx] + 1e-10f);
                total_loss -= log_prob;
                count++;
            }
        }

        // Create loss tensor
        Tensor loss = at::empty({});
        if (count > 0) {
            loss.mutable_data_ptr<float>()[0] = total_loss / static_cast<float>(count);
        } else {
            loss.mutable_data_ptr<float>()[0] = 0.0f;
        }

        // Move loss back to GPU
#ifdef PT_USE_CUDA
        if (is_cuda) {
            loss = at::to_cuda(loss);
        }
#endif

        // Set up autograd (same as PIR)
        if (logits.requires_grad()) {
            // Convert targets to float for backward compatibility
            Tensor targets_float = at::empty({B});
            float* tf_data = targets_float.mutable_data_ptr<float>();
            for (int64_t i = 0; i < B; ++i) {
                tf_data[i] = static_cast<float>(targets_data[i]);
            }

            auto backward_fn = std::make_shared<torch::autograd::CrossEntropyBackward>(
                softmax, targets_float, -100, V, count, is_cuda
            );
            backward_fn->add_input_metadata(logits);

            auto* meta = torch::autograd::ensure_autograd_meta_impl(loss);
            meta->grad_fn = backward_fn;
            meta->output_nr_ = 0;
            meta->is_leaf_ = false;
            meta->requires_grad_ = true;
        }

        return loss;
    }

private:
    int64_t vocab_size_;
    int64_t block_size_;
    int64_t n_embd_;
    std::shared_ptr<Embedding> embedding_;
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<Linear> fc3_;
};

// ============================================================================
// Training Loop
// ============================================================================

void train_mlp(
    MLPCharModel& model,
    const std::vector<int64_t>& data,
    int64_t block_size,
    int64_t batch_size,
    int64_t iterations,
    float lr
) {
    Adam optimizer(model.parameters(), AdamOptions(lr));

    std::random_device rd;
    std::mt19937 gen(rd());
    int64_t data_len = static_cast<int64_t>(data.size());
    std::uniform_int_distribution<int64_t> dist(0, data_len - block_size - 2);

    auto start_time = std::chrono::high_resolution_clock::now();
    float running_loss = 0.0f;
    int loss_count = 0;

    for (int64_t iter = 1; iter <= iterations; iter++) {
        // Create batch
        std::vector<int64_t> input_data(batch_size * block_size);
        std::vector<int64_t> target_data(batch_size);

        for (int64_t b = 0; b < batch_size; b++) {
            int64_t start = dist(gen);
            for (int64_t t = 0; t < block_size; t++) {
                input_data[b * block_size + t] = data[start + t];
            }
            // Target is the next character after the context
            target_data[b] = data[start + block_size];
        }

        // Create tensors manually
        Tensor inputs = at::empty({batch_size, block_size}, at::TensorOptions().dtype(c10::ScalarType::Long));
        Tensor targets = at::empty({batch_size}, at::TensorOptions().dtype(c10::ScalarType::Long));

        // Copy data
        int64_t* inp_ptr = inputs.mutable_data_ptr<int64_t>();
        int64_t* tgt_ptr = targets.mutable_data_ptr<int64_t>();
        std::memcpy(inp_ptr, input_data.data(), batch_size * block_size * sizeof(int64_t));
        std::memcpy(tgt_ptr, target_data.data(), batch_size * sizeof(int64_t));

        inputs = to_device(inputs);
        targets = to_device(targets);

        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        std::cout << "iter " << iter << ": fwd..." << std::flush;
        Tensor loss = model.forward_with_loss(inputs, targets);

        // Get loss value
        Tensor loss_cpu = loss;
#ifdef PT_USE_CUDA
        if (g_device.is_cuda()) {
            loss_cpu = at::to_cpu(loss);
        }
#endif
        float loss_val = loss_cpu.data_ptr<float>()[0];
        running_loss += loss_val;
        loss_count++;
        std::cout << " loss=" << loss_val << " bwd..." << std::flush;

        // Backward pass
        torch::autograd::backward({loss});
        std::cout << " clip..." << std::flush;

        // CRITICAL: Clear autograd graph (same as PIR)
        torch::autograd::clear_grad_fn(loss);

        // Clear tensors to release graph references
        loss = Tensor();
        inputs = Tensor();
        targets = Tensor();

        // Gradient clipping
        clip_grad_norm_(model, 1.0f);
        std::cout << " step..." << std::flush;

        // Update weights
        optimizer.step();
        std::cout << " done" << std::endl;

        // Logging
        if (iter % 10 == 0) {
            float avg_loss = running_loss / loss_count;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            std::cout << "Iter " << iter << "/" << iterations
                      << " | Loss: " << avg_loss
                      << " | Time: " << elapsed << "s";
#ifdef PT_USE_CUDA
            if (g_device.is_cuda()) {
                auto& alloc = c10::cuda::CUDACachingAllocator::get();
                std::cout << " | GPU: " << (alloc.get_allocated_memory() / 1048576.0) << " MB";
            }
#endif
            std::cout << std::endl;

            running_loss = 0.0f;
            loss_count = 0;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "\nTraining completed in " << total_time << " seconds" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_file = "shakespeare.txt";
    int64_t batch_size = 32;
    int64_t block_size = 64;
    int64_t n_embd = 128;
    int64_t n_hidden = 512;
    int64_t iterations = 100;
    float lr = 0.001f;
    std::string device_str = "cpu";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoll(argv[++i]);
        } else if (arg == "--block_size" && i + 1 < argc) {
            block_size = std::stoll(argv[++i]);
        } else if (arg == "--n_embd" && i + 1 < argc) {
            n_embd = std::stoll(argv[++i]);
        } else if (arg == "--n_hidden" && i + 1 < argc) {
            n_hidden = std::stoll(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoll(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        }
    }

    // Set device
    if (device_str == "cuda") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Using CUDA device" << std::endl;
#else
        std::cout << "CUDA not available, using CPU" << std::endl;
#endif
    } else {
        std::cout << "Using CPU device" << std::endl;
    }

    // Load data
    std::cout << "Loading text from: " << data_file << std::endl;
    std::ifstream file(data_file);
    if (!file) {
        std::cerr << "Error: Cannot open file " << data_file << std::endl;
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    std::cout << "Text length: " << text.size() << " characters" << std::endl;

    // Build vocabulary
    std::set<char> unique_chars(text.begin(), text.end());
    std::map<char, int64_t> char_to_idx;
    int64_t idx = 0;
    for (char c : unique_chars) {
        char_to_idx[c] = idx++;
    }
    int64_t vocab_size = static_cast<int64_t>(unique_chars.size());
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    // Encode text
    std::vector<int64_t> data;
    data.reserve(text.size());
    for (char c : text) {
        data.push_back(char_to_idx[c]);
    }
    std::cout << "Encoded length: " << data.size() << " tokens" << std::endl;

    // Create model
    std::cout << "\n=== Model Configuration ===" << std::endl;
    std::cout << "  Block size: " << block_size << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Embedding dim: " << n_embd << std::endl;
    std::cout << "  Hidden dim: " << n_hidden << std::endl;

    MLPCharModel model(vocab_size, block_size, n_embd, n_hidden);

    // Move to device
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        std::cout << "Moving model to CUDA..." << std::endl;
        model.to(g_device);
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    // Train
    std::cout << "\n=== Training MLP Char Model ===" << std::endl;
    std::cout << std::flush;
    train_mlp(model, data, block_size, batch_size, iterations, lr);

    std::cout << "\n=== Training Complete ===" << std::endl;

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        c10::cuda::CUDACachingAllocator::get().shutdown();
    }
#endif

    return 0;
}
