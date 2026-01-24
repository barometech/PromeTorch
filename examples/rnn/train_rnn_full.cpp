// ============================================================================
// RNN Training - All Variants (RNN, LSTM, GRU, RWKV-like)
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "../common/profiler.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_map>

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

inline at::Tensor to_device(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) return at::to_cuda(t);
#endif
    return t;
}

inline at::Tensor move_to_cpu(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) return at::to_cpu(t);
#endif
    return t;
}

// ============================================================================
// GRU Cell
// ============================================================================

class GRUCell : public Module {
public:
    GRUCell(int64_t input_size, int64_t hidden_size)
        : Module("GRUCell"), hidden_size_(hidden_size) {
        LOG_INFO("Creating GRUCell: input=" + std::to_string(input_size) +
                 ", hidden=" + std::to_string(hidden_size));

        // Reset and update gates
        W_ir = std::make_shared<Linear>(input_size, hidden_size);   // input to reset
        W_hr = std::make_shared<Linear>(hidden_size, hidden_size, false);
        W_iz = std::make_shared<Linear>(input_size, hidden_size);   // input to update
        W_hz = std::make_shared<Linear>(hidden_size, hidden_size, false);
        W_in = std::make_shared<Linear>(input_size, hidden_size);   // input to new
        W_hn = std::make_shared<Linear>(hidden_size, hidden_size, false);

        register_module("W_ir", W_ir);
        register_module("W_hr", W_hr);
        register_module("W_iz", W_iz);
        register_module("W_hz", W_hz);
        register_module("W_in", W_in);
        register_module("W_hn", W_hn);
    }

    std::pair<Tensor, Tensor> forward_cell(const Tensor& x, const Tensor& h) {
        LOG_FORWARD("GRUCell");

        // Reset gate: r = sigmoid(W_ir*x + W_hr*h)
        Tensor r = W_ir->forward(x).add(W_hr->forward(h)).sigmoid();

        // Update gate: z = sigmoid(W_iz*x + W_hz*h)
        Tensor z = W_iz->forward(x).add(W_hz->forward(h)).sigmoid();

        // New gate: n = tanh(W_in*x + r * W_hn*h)
        Tensor n = W_in->forward(x).add(r.mul(W_hn->forward(h))).tanh();

        // h_new = (1 - z) * n + z * h
        Tensor one_minus_z = z.neg().add(at::Scalar(1.0f));
        Tensor h_new = one_minus_z.mul(n).add(z.mul(h));

        return {h_new, h_new};  // GRU has h=c conceptually
    }

    Tensor forward(const Tensor& x) override {
        PROFILE_SCOPE("GRU forward");
        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        Tensor h = to_device(at::zeros({B, hidden_size_}));
        std::vector<Tensor> outputs;

        for (int64_t t = 0; t < T; ++t) {
            Tensor x_t = x.select(1, t);
            auto [h_new, _] = forward_cell(x_t, h);
            h = h_new;
            outputs.push_back(h.unsqueeze(1));
        }

        return at::cat(outputs, 1);
    }

    int64_t hidden_size() const { return hidden_size_; }

private:
    int64_t hidden_size_;
    std::shared_ptr<Linear> W_ir, W_hr, W_iz, W_hz, W_in, W_hn;
};

// ============================================================================
// LSTM Cell (updated with logging)
// ============================================================================

class LSTMCellFull : public Module {
public:
    LSTMCellFull(int64_t input_size, int64_t hidden_size)
        : Module("LSTMCell"), hidden_size_(hidden_size) {
        LOG_INFO("Creating LSTMCell: input=" + std::to_string(input_size) +
                 ", hidden=" + std::to_string(hidden_size));

        W_ii = std::make_shared<Linear>(input_size, hidden_size);
        W_hi = std::make_shared<Linear>(hidden_size, hidden_size, false);
        W_if = std::make_shared<Linear>(input_size, hidden_size);
        W_hf = std::make_shared<Linear>(hidden_size, hidden_size, false);
        W_ig = std::make_shared<Linear>(input_size, hidden_size);
        W_hg = std::make_shared<Linear>(hidden_size, hidden_size, false);
        W_io = std::make_shared<Linear>(input_size, hidden_size);
        W_ho = std::make_shared<Linear>(hidden_size, hidden_size, false);

        register_module("W_ii", W_ii);
        register_module("W_hi", W_hi);
        register_module("W_if", W_if);
        register_module("W_hf", W_hf);
        register_module("W_ig", W_ig);
        register_module("W_hg", W_hg);
        register_module("W_io", W_io);
        register_module("W_ho", W_ho);
    }

    std::pair<Tensor, Tensor> forward_cell(const Tensor& x, const Tensor& h, const Tensor& c) {
        LOG_FORWARD("LSTMCell");

        Tensor i = W_ii->forward(x).add(W_hi->forward(h)).sigmoid();
        Tensor f = W_if->forward(x).add(W_hf->forward(h)).sigmoid();
        Tensor g = W_ig->forward(x).add(W_hg->forward(h)).tanh();
        Tensor o = W_io->forward(x).add(W_ho->forward(h)).sigmoid();

        Tensor c_new = f.mul(c).add(i.mul(g));
        Tensor h_new = o.mul(c_new.tanh());

        return {h_new, c_new};
    }

    Tensor forward(const Tensor& x) override {
        PROFILE_SCOPE("LSTM forward");
        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

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
    std::shared_ptr<Linear> W_ii, W_hi, W_if, W_hf, W_ig, W_hg, W_io, W_ho;
};

// ============================================================================
// RWKV-like Cell (simplified linear attention)
// ============================================================================

class RWKVCell : public Module {
public:
    RWKVCell(int64_t input_size, int64_t hidden_size)
        : Module("RWKVCell"), hidden_size_(hidden_size) {
        LOG_INFO("Creating RWKVCell: input=" + std::to_string(input_size) +
                 ", hidden=" + std::to_string(hidden_size));

        // Time mixing
        W_k = std::make_shared<Linear>(input_size, hidden_size);
        W_v = std::make_shared<Linear>(input_size, hidden_size);
        W_r = std::make_shared<Linear>(input_size, hidden_size);
        W_o = std::make_shared<Linear>(hidden_size, hidden_size);

        // Learned decay
        decay = at::randn({hidden_size}).mul(at::Scalar(0.1f));

        register_module("W_k", W_k);
        register_module("W_v", W_v);
        register_module("W_r", W_r);
        register_module("W_o", W_o);
    }

    Tensor forward(const Tensor& x) override {
        PROFILE_SCOPE("RWKV forward");
        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        // Compute K, V, R for all time steps
        Tensor k = W_k->forward(x);  // [B, T, H]
        Tensor v = W_v->forward(x);
        Tensor r = W_r->forward(x).sigmoid();  // receptance

        // State: weighted sum of past k*v
        Tensor state = to_device(at::zeros({B, hidden_size_}));
        Tensor decay_dev = to_device(decay);
        Tensor exp_decay = decay_dev.neg().exp();  // e^(-decay)

        std::vector<Tensor> outputs;

        for (int64_t t = 0; t < T; ++t) {
            Tensor k_t = k.select(1, t);  // [B, H]
            Tensor v_t = v.select(1, t);
            Tensor r_t = r.select(1, t);

            // Update state with exponential decay
            state = state.mul(exp_decay).add(k_t.mul(v_t));

            // Output
            Tensor out = r_t.mul(state);
            out = W_o->forward(out);
            outputs.push_back(out.unsqueeze(1));
        }

        return at::cat(outputs, 1);
    }

    int64_t hidden_size() const { return hidden_size_; }

private:
    int64_t hidden_size_;
    std::shared_ptr<Linear> W_k, W_v, W_r, W_o;
    Tensor decay;
};

// ============================================================================
// Unified RNN Model
// ============================================================================

enum class RNNType { RNN, LSTM, GRU, RWKV };

class CharRNNFull : public Module {
public:
    CharRNNFull(int64_t vocab_size, int64_t embed_size, int64_t hidden_size, RNNType type)
        : Module("CharRNNFull"), vocab_size_(vocab_size), type_(type) {

        LOG_INFO("Creating CharRNN: vocab=" + std::to_string(vocab_size) +
                 ", embed=" + std::to_string(embed_size) +
                 ", hidden=" + std::to_string(hidden_size));

        embedding = std::make_shared<Embedding>(vocab_size, embed_size);
        register_module("embedding", embedding);

        int64_t rnn_hidden = hidden_size;

        switch (type) {
            case RNNType::RNN: {
                LOG_INFO("Using vanilla RNN");
                W_ih = std::make_shared<Linear>(embed_size, hidden_size);
                W_hh = std::make_shared<Linear>(hidden_size, hidden_size, false);
                register_module("W_ih", W_ih);
                register_module("W_hh", W_hh);
                break;
            }
            case RNNType::LSTM: {
                LOG_INFO("Using LSTM");
                lstm = std::make_shared<LSTMCellFull>(embed_size, hidden_size);
                register_module("lstm", lstm);
                break;
            }
            case RNNType::GRU: {
                LOG_INFO("Using GRU");
                gru = std::make_shared<GRUCell>(embed_size, hidden_size);
                register_module("gru", gru);
                break;
            }
            case RNNType::RWKV: {
                LOG_INFO("Using RWKV");
                rwkv = std::make_shared<RWKVCell>(embed_size, hidden_size);
                register_module("rwkv", rwkv);
                break;
            }
        }

        fc = std::make_shared<Linear>(hidden_size, vocab_size);
        register_module("fc", fc);
    }

    Tensor forward(const Tensor& x) override {
        PROFILE_SCOPE("CharRNN forward");
        LOG_TENSOR("input", x);

        Tensor embedded = embedding->forward(x);
        LOG_TENSOR("embedded", embedded);

        Tensor hidden;
        switch (type_) {
            case RNNType::RNN:
                hidden = forward_rnn(embedded);
                break;
            case RNNType::LSTM:
                hidden = lstm->forward(embedded);
                break;
            case RNNType::GRU:
                hidden = gru->forward(embedded);
                break;
            case RNNType::RWKV:
                hidden = rwkv->forward(embedded);
                break;
        }
        LOG_TENSOR("hidden", hidden);

        auto sizes = hidden.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];
        int64_t H = sizes[2];

        Tensor flat = hidden.reshape({B * T, H});
        Tensor logits = fc->forward(flat);
        logits = logits.reshape({B, T, vocab_size_});

        LOG_TENSOR("logits", logits);
        return logits;
    }

private:
    Tensor forward_rnn(const Tensor& x) {
        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        Tensor h = to_device(at::zeros({B, W_hh->out_features()}));
        std::vector<Tensor> outputs;

        for (int64_t t = 0; t < T; ++t) {
            Tensor x_t = x.select(1, t);
            h = W_ih->forward(x_t).add(W_hh->forward(h)).tanh();
            outputs.push_back(h.unsqueeze(1));
        }

        return at::cat(outputs, 1);
    }

    int64_t vocab_size_;
    RNNType type_;
    std::shared_ptr<Embedding> embedding;
    std::shared_ptr<Linear> W_ih, W_hh;
    std::shared_ptr<LSTMCellFull> lstm;
    std::shared_ptr<GRUCell> gru;
    std::shared_ptr<RWKVCell> rwkv;
    std::shared_ptr<Linear> fc;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_path = "data/shakespeare.txt";
    std::string device_str = "cpu";
    std::string rnn_type_str = "lstm";
    int64_t seq_len = 32;
    int64_t batch_size = 32;
    int64_t hidden_size = 128;
    int64_t embed_size = 64;
    int64_t iterations = 500;
    float lr = 0.002f;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) data_path = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--type" && i + 1 < argc) rnn_type_str = argv[++i];
        else if (arg == "--seq_len" && i + 1 < argc) seq_len = std::stoll(argv[++i]);
        else if (arg == "--batch_size" && i + 1 < argc) batch_size = std::stoll(argv[++i]);
        else if (arg == "--hidden" && i + 1 < argc) hidden_size = std::stoll(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoll(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--verbose" || arg == "-v") verbose = true;
    }

    // Set log level
    if (verbose) {
        profiler::Logger::instance().set_level(profiler::LogLevel::TRACE);
    } else {
        profiler::Logger::instance().set_level(profiler::LogLevel::INFO);
    }

    LOG_INFO("=== RNN Training (All Variants) ===");

    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        LOG_INFO("Using CUDA");
        LOG_MEMORY("Initial");
#else
        LOG_WARN("CUDA not available, using CPU");
#endif
    } else {
        LOG_INFO("Using CPU");
    }

    // Parse RNN type
    RNNType rnn_type;
    if (rnn_type_str == "rnn") rnn_type = RNNType::RNN;
    else if (rnn_type_str == "lstm") rnn_type = RNNType::LSTM;
    else if (rnn_type_str == "gru") rnn_type = RNNType::GRU;
    else if (rnn_type_str == "rwkv") rnn_type = RNNType::RWKV;
    else {
        LOG_ERROR("Unknown RNN type: " + rnn_type_str);
        return 1;
    }

    // Load text
    LOG_INFO("Loading text from: " + data_path);
    std::ifstream file(data_path);
    if (!file) {
        LOG_ERROR("Cannot open: " + data_path);
        return 1;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    LOG_INFO("Text length: " + std::to_string(text.size()) + " chars");

    // Build vocab
    std::unordered_map<char, int64_t> char_to_idx;
    std::vector<char> idx_to_char;
    for (char c : text) {
        if (char_to_idx.find(c) == char_to_idx.end()) {
            char_to_idx[c] = idx_to_char.size();
            idx_to_char.push_back(c);
        }
    }
    int64_t vocab_size = idx_to_char.size();
    LOG_INFO("Vocab size: " + std::to_string(vocab_size));

    std::vector<int64_t> data(text.size());
    for (size_t i = 0; i < text.size(); ++i) data[i] = char_to_idx[text[i]];

    // Create model
    auto model = std::make_shared<CharRNNFull>(vocab_size, embed_size, hidden_size, rnn_type);
    LOG_MEMORY("After model creation");

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        PROFILE_SCOPE("Model to CUDA");
        model->to(g_device);
        LOG_MEMORY("After model.to(CUDA)");
    }
#endif

    AdamOptions opts(lr);
    Adam optimizer(model->parameters(), opts);
    CrossEntropyLoss criterion;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, data.size() - seq_len - 2);

    profiler::Stats loss_stats;
    profiler::Stats time_stats;

    LOG_INFO("Starting training: " + std::to_string(iterations) + " iterations");

    for (int64_t iter = 1; iter <= iterations; ++iter) {
        profiler::Timer iter_timer;
        iter_timer.start();

        model->train();

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

        Tensor logits;
        {
            PROFILE_SCOPE("Forward pass");
            logits = model->forward(inputs);
        }

        logits = logits.reshape({batch_size * seq_len, vocab_size});
        targets = targets.reshape({batch_size * seq_len});

        Tensor loss = criterion.forward(logits, targets);

        {
            PROFILE_SCOPE("Backward pass");
            torch::autograd::backward({loss});
        }

        {
            PROFILE_SCOPE("Optimizer step");
            optimizer.step();
        }

        double iter_time = iter_timer.stop();
        time_stats.add(iter_time);

        Tensor loss_cpu = move_to_cpu(loss);
        float loss_val = loss_cpu.data_ptr<float>()[0];
        loss_stats.add(loss_val);

        if (iter % 50 == 0 || iter == 1) {
            LOG_INFO("Iter " + std::to_string(iter) + "/" + std::to_string(iterations) +
                     " | Loss: " + std::to_string(loss_val) +
                     " | Time: " + std::to_string(iter_time) + "ms");
            LOG_MEMORY("Iter " + std::to_string(iter));
        }
    }

    LOG_INFO("\n=== Training Statistics ===");
    loss_stats.print("Loss");
    time_stats.print("Iteration time (ms)");

    LOG_INFO("=== RNN Training Complete ===");
    return 0;
}
