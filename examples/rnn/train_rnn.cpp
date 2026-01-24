// ============================================================================
// RNN Training Example - Character-level Language Model
// ============================================================================
// Simple RNN/LSTM for predicting next character
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_map>
#include <algorithm>

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

inline at::Tensor to_device(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) {
        return at::to_cuda(t);
    }
#endif
    return t;
}

inline at::Tensor move_to_cpu(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) {
        return at::to_cpu(t);
    }
#endif
    return t;
}

// ============================================================================
// Simple RNN Cell (manual implementation)
// ============================================================================

class RNNCell : public Module {
public:
    RNNCell(int64_t input_size, int64_t hidden_size)
        : Module("RNNCell"), hidden_size_(hidden_size) {
        // W_ih * x + W_hh * h + b
        W_ih = std::make_shared<Linear>(input_size, hidden_size);
        W_hh = std::make_shared<Linear>(hidden_size, hidden_size, false);  // no bias

        register_module("W_ih", W_ih);
        register_module("W_hh", W_hh);
    }

    // Forward: returns new hidden state
    Tensor forward_cell(const Tensor& x, const Tensor& h) {
        // h_new = tanh(W_ih * x + W_hh * h)
        Tensor h_new = W_ih->forward(x).add(W_hh->forward(h)).tanh();
        return h_new;
    }

    Tensor forward(const Tensor& x) override {
        // x: [B, T, input_size]
        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        // Initialize hidden state
        Tensor h = to_device(at::zeros({B, hidden_size_}));

        // Output container
        std::vector<Tensor> outputs;

        for (int64_t t = 0; t < T; ++t) {
            // Get x[:, t, :]
            Tensor x_t = x.select(1, t);  // [B, input_size]
            h = forward_cell(x_t, h);
            outputs.push_back(h.unsqueeze(1));  // [B, 1, hidden_size]
        }

        // Stack outputs: [B, T, hidden_size]
        return at::cat(outputs, 1);
    }

    int64_t hidden_size() const { return hidden_size_; }

private:
    int64_t hidden_size_;
    std::shared_ptr<Linear> W_ih, W_hh;
};

// ============================================================================
// LSTM Cell (manual implementation)
// ============================================================================

class LSTMCell : public Module {
public:
    LSTMCell(int64_t input_size, int64_t hidden_size)
        : Module("LSTMCell"), hidden_size_(hidden_size) {
        // Gates: input, forget, cell, output
        // Combined projection for efficiency
        W_ih = std::make_shared<Linear>(input_size, 4 * hidden_size);
        W_hh = std::make_shared<Linear>(hidden_size, 4 * hidden_size, false);

        register_module("W_ih", W_ih);
        register_module("W_hh", W_hh);
    }

    // Forward: returns (h_new, c_new)
    std::pair<Tensor, Tensor> forward_cell(const Tensor& x, const Tensor& h, const Tensor& c) {
        // gates = W_ih * x + W_hh * h
        Tensor gates = W_ih->forward(x).add(W_hh->forward(h));  // [B, 4*H]

        // Split into 4 gates
        auto sizes = gates.sizes();
        int64_t B = sizes[0];

        // Manual split (since we don't have chunk for CUDA yet)
        Tensor i_gate = gates.narrow(1, 0, hidden_size_).sigmoid();
        Tensor f_gate = gates.narrow(1, hidden_size_, hidden_size_).sigmoid();
        Tensor g_gate = gates.narrow(1, 2 * hidden_size_, hidden_size_).tanh();
        Tensor o_gate = gates.narrow(1, 3 * hidden_size_, hidden_size_).sigmoid();

        // Cell state update
        Tensor c_new = f_gate.mul(c).add(i_gate.mul(g_gate));

        // Hidden state
        Tensor h_new = o_gate.mul(c_new.tanh());

        return {h_new, c_new};
    }

    Tensor forward(const Tensor& x) override {
        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        // Initialize states
        Tensor h = to_device(at::zeros({B, hidden_size_}));
        Tensor c = to_device(at::zeros({B, hidden_size_}));

        std::vector<Tensor> outputs;

        for (int64_t t = 0; t < T; ++t) {
            Tensor x_t = x.select(1, t);
            auto [h_new, c_new] = forward_cell(x_t, h, c);
            h = h_new;
            c = c_new;
            outputs.push_back(h.unsqueeze(1));
        }

        return at::cat(outputs, 1);
    }

    int64_t hidden_size() const { return hidden_size_; }

private:
    int64_t hidden_size_;
    std::shared_ptr<Linear> W_ih, W_hh;
};

// ============================================================================
// Character-level Language Model
// ============================================================================

class CharRNN : public Module {
public:
    CharRNN(int64_t vocab_size, int64_t embed_size, int64_t hidden_size, bool use_lstm = false)
        : Module("CharRNN"), vocab_size_(vocab_size), use_lstm_(use_lstm) {

        embedding = std::make_shared<Embedding>(vocab_size, embed_size);

        if (use_lstm) {
            lstm = std::make_shared<LSTMCell>(embed_size, hidden_size);
            register_module("lstm", lstm);
        } else {
            rnn = std::make_shared<RNNCell>(embed_size, hidden_size);
            register_module("rnn", rnn);
        }

        fc = std::make_shared<Linear>(hidden_size, vocab_size);

        register_module("embedding", embedding);
        register_module("fc", fc);
    }

    Tensor forward(const Tensor& x) override {
        // x: [B, T] integer indices
        Tensor embedded = embedding->forward(x);  // [B, T, embed_size]

        Tensor hidden;
        if (use_lstm_) {
            hidden = lstm->forward(embedded);  // [B, T, hidden_size]
        } else {
            hidden = rnn->forward(embedded);
        }

        // Project to vocab
        // Reshape to [B*T, hidden_size] -> [B*T, vocab_size] -> [B, T, vocab_size]
        auto sizes = hidden.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];
        int64_t H = sizes[2];

        Tensor flat = hidden.reshape({B * T, H});
        Tensor logits = fc->forward(flat);

        return logits.reshape({B, T, vocab_size_});
    }

private:
    int64_t vocab_size_;
    bool use_lstm_;
    std::shared_ptr<Embedding> embedding;
    std::shared_ptr<RNNCell> rnn;
    std::shared_ptr<LSTMCell> lstm;
    std::shared_ptr<Linear> fc;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_path = "data/shakespeare.txt";
    std::string device_str = "cpu";
    int64_t seq_len = 32;
    int64_t batch_size = 32;
    int64_t hidden_size = 128;
    int64_t embed_size = 64;
    int64_t iterations = 1000;
    float lr = 0.002f;
    bool use_lstm = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg == "--seq_len" && i + 1 < argc) {
            seq_len = std::stoll(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoll(argv[++i]);
        } else if (arg == "--hidden" && i + 1 < argc) {
            hidden_size = std::stoll(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoll(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        } else if (arg == "--lstm") {
            use_lstm = true;
        }
    }

    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Using CUDA" << std::endl;
#else
        std::cerr << "CUDA not available, using CPU" << std::endl;
#endif
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    // Load text
    std::cout << "Loading text from: " << data_path << std::endl;
    std::ifstream file(data_path);
    if (!file) {
        std::cerr << "Cannot open file: " << data_path << std::endl;
        return 1;
    }
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    file.close();

    std::cout << "Text length: " << text.size() << " characters" << std::endl;

    // Build vocabulary
    std::unordered_map<char, int64_t> char_to_idx;
    std::vector<char> idx_to_char;

    for (char c : text) {
        if (char_to_idx.find(c) == char_to_idx.end()) {
            int64_t idx = idx_to_char.size();
            char_to_idx[c] = idx;
            idx_to_char.push_back(c);
        }
    }

    int64_t vocab_size = idx_to_char.size();
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    // Convert text to indices
    std::vector<int64_t> data(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        data[i] = char_to_idx[text[i]];
    }

    // Create model
    auto model = std::make_shared<CharRNN>(vocab_size, embed_size, hidden_size, use_lstm);
    std::cout << (use_lstm ? "LSTM" : "RNN") << " Model created" << std::endl;
    std::cout << "  Vocab: " << vocab_size << ", Embed: " << embed_size
              << ", Hidden: " << hidden_size << std::endl;

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        model->to(g_device);
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    // Optimizer
    AdamOptions opts(lr);
    Adam optimizer(model->parameters(), opts);

    // Loss
    CrossEntropyLoss criterion;

    // Training
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, data.size() - seq_len - 2);

    std::cout << "\n=== RNN Training ===" << std::endl;
    std::cout << "Iterations: " << iterations << ", Seq len: " << seq_len
              << ", Batch: " << batch_size << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int64_t iter = 1; iter <= iterations; ++iter) {
        model->train();

        // Create batch
        Tensor inputs = at::empty({batch_size, seq_len});
        Tensor targets = at::empty({batch_size, seq_len});

        float* in_ptr = inputs.mutable_data_ptr<float>();
        float* tgt_ptr = targets.mutable_data_ptr<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t start = dist(gen);
            for (int64_t t = 0; t < seq_len; ++t) {
                in_ptr[b * seq_len + t] = static_cast<float>(data[start + t]);
                tgt_ptr[b * seq_len + t] = static_cast<float>(data[start + t + 1]);
            }
        }

        inputs = to_device(inputs);
        targets = to_device(targets);

        optimizer.zero_grad();

        // Forward
        Tensor logits = model->forward(inputs);  // [B, T, vocab_size]

        // Reshape for cross entropy: [B*T, vocab_size] and [B*T]
        logits = logits.reshape({batch_size * seq_len, vocab_size});
        targets = targets.reshape({batch_size * seq_len});

        Tensor loss = criterion.forward(logits, targets);

        // Backward
        torch::autograd::backward({loss});
        optimizer.step();

        if (iter % 100 == 0 || iter == 1) {
            Tensor loss_cpu = move_to_cpu(loss);
            float loss_val = loss_cpu.data_ptr<float>()[0];

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

            std::cout << "Iter " << iter << "/" << iterations
                      << " Loss: " << loss_val
                      << " Time: " << elapsed << "ms" << std::endl;
        }
    }

    // Generate sample text
    std::cout << "\n=== Generating Sample ===" << std::endl;
    model->eval();

    std::string seed = "The ";
    std::string generated = seed;

    // Convert seed to tensor
    Tensor input = at::empty({1, 1});
    float* in_ptr = input.mutable_data_ptr<float>();

    // Generate 200 characters
    char last_char = seed.back();
    for (int i = 0; i < 200; ++i) {
        in_ptr[0] = static_cast<float>(char_to_idx[last_char]);
        Tensor input_dev = to_device(input);

        Tensor logits = model->forward(input_dev);  // [1, 1, vocab_size]
        Tensor logits_cpu = move_to_cpu(logits);

        // Get last timestep, squeeze to [vocab_size]
        const float* log_ptr = logits_cpu.data_ptr<float>();

        // Find argmax
        int best = 0;
        float best_val = log_ptr[0];
        for (int j = 1; j < vocab_size; ++j) {
            if (log_ptr[j] > best_val) {
                best_val = log_ptr[j];
                best = j;
            }
        }

        last_char = idx_to_char[best];
        generated += last_char;
    }

    std::cout << "Generated: " << generated << std::endl;

    std::cout << "\n=== RNN Training Complete ===" << std::endl;
    return 0;
}
