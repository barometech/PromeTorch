#pragma once

#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/csrc/autograd/autograd.h"
#include <cmath>
#include <vector>

namespace torch {
namespace nn {

using at::Tensor;
// Note: Do NOT use 'using' for autograd functions here — pir.h defines
// duplicate names in torch::nn, which causes ambiguity. Use fully qualified calls.

// ============================================================================
// RNNCellImpl — h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)
// ============================================================================

class RNNCellImpl : public Module {
public:
    RNNCellImpl(int64_t input_size, int64_t hidden_size, bool bias = true)
        : Module("RNNCell")
        , input_size_(input_size)
        , hidden_size_(hidden_size)
    {
        ih_ = std::make_shared<Linear>(input_size, hidden_size, bias);
        hh_ = std::make_shared<Linear>(hidden_size, hidden_size, bias);
        register_module("ih", ih_);
        register_module("hh", hh_);
    }

    // Forward: x [batch, input_size], h [batch, hidden_size] -> h' [batch, hidden_size]
    Tensor forward(const Tensor& input, const Tensor& hx) {
        Tensor gates = torch::autograd::add_autograd((*ih_)(input), (*hh_)(hx));
        return torch::autograd::tanh_autograd(gates);
    }

    // Convenience: forward with zero hidden state
    Tensor forward(const Tensor& input) override {
        Tensor hx = at::zeros({input.size(0), hidden_size_});
        return forward(input, hx);
    }

    int64_t input_size() const { return input_size_; }
    int64_t hidden_size() const { return hidden_size_; }

private:
    int64_t input_size_, hidden_size_;
    std::shared_ptr<Linear> ih_, hh_;
};

// ============================================================================
// LSTMCellImpl — 4 gates: i, f, g, o
// ============================================================================

class LSTMCellImpl : public Module {
public:
    LSTMCellImpl(int64_t input_size, int64_t hidden_size, bool bias = true)
        : Module("LSTMCell")
        , input_size_(input_size)
        , hidden_size_(hidden_size)
    {
        // Combined gates: [4*hidden, input] and [4*hidden, hidden]
        ih_ = std::make_shared<Linear>(input_size, 4 * hidden_size, bias);
        hh_ = std::make_shared<Linear>(hidden_size, 4 * hidden_size, bias);
        register_module("ih", ih_);
        register_module("hh", hh_);
    }

    // Forward: x [batch, input], (h, c) -> (h', c')
    std::pair<Tensor, Tensor> forward_lstm(const Tensor& input, const Tensor& hx, const Tensor& cx) {
        // gates = x @ W_ih^T + h @ W_hh^T  [batch, 4*hidden]
        Tensor gates = torch::autograd::add_autograd((*ih_)(input), (*hh_)(hx));

        // Split into 4 gates along last dim (use narrow_autograd for gradient flow!)
        int64_t H = hidden_size_;
        Tensor gates_c = gates.contiguous();

        Tensor i_gate = torch::autograd::sigmoid_autograd(torch::autograd::narrow_autograd(gates_c, 1, 0, H));
        Tensor f_gate = torch::autograd::sigmoid_autograd(torch::autograd::narrow_autograd(gates_c, 1, H, H));
        Tensor g_gate = torch::autograd::tanh_autograd(torch::autograd::narrow_autograd(gates_c, 1, 2 * H, H));
        Tensor o_gate = torch::autograd::sigmoid_autograd(torch::autograd::narrow_autograd(gates_c, 1, 3 * H, H));

        // c' = f * c + i * g
        Tensor c_new = torch::autograd::add_autograd(
            torch::autograd::mul_autograd(f_gate, cx),
            torch::autograd::mul_autograd(i_gate, g_gate));
        // h' = o * tanh(c')
        Tensor h_new = torch::autograd::mul_autograd(o_gate, torch::autograd::tanh_autograd(c_new));

        return {h_new, c_new};
    }

    // Convenience: single input forward
    Tensor forward(const Tensor& input) override {
        Tensor hx = at::zeros({input.size(0), hidden_size_});
        Tensor cx = at::zeros({input.size(0), hidden_size_});
        auto [h, c] = forward_lstm(input, hx, cx);
        return h;
    }

    int64_t input_size() const { return input_size_; }
    int64_t hidden_size() const { return hidden_size_; }

private:
    int64_t input_size_, hidden_size_;
    std::shared_ptr<Linear> ih_, hh_;
};

// ============================================================================
// GRUCellImpl — reset, update, new gates
// ============================================================================

class GRUCellImpl : public Module {
public:
    GRUCellImpl(int64_t input_size, int64_t hidden_size, bool bias = true)
        : Module("GRUCell")
        , input_size_(input_size)
        , hidden_size_(hidden_size)
    {
        ih_ = std::make_shared<Linear>(input_size, 3 * hidden_size, bias);
        hh_ = std::make_shared<Linear>(hidden_size, 3 * hidden_size, bias);
        register_module("ih", ih_);
        register_module("hh", hh_);
    }

    // Forward: x [batch, input], h [batch, hidden] -> h' [batch, hidden]
    Tensor forward(const Tensor& input, const Tensor& hx) {
        int64_t H = hidden_size_;

        Tensor gi = (*ih_)(input);   // [batch, 3*H]
        Tensor gh = (*hh_)(hx);      // [batch, 3*H]

        Tensor gi_c = gi.contiguous();
        Tensor gh_c = gh.contiguous();

        // r = sigmoid(gi[:, 0:H] + gh[:, 0:H])
        Tensor r = torch::autograd::sigmoid_autograd(torch::autograd::add_autograd(
            torch::autograd::narrow_autograd(gi_c, 1, 0, H),
            torch::autograd::narrow_autograd(gh_c, 1, 0, H)));
        // z = sigmoid(gi[:, H:2H] + gh[:, H:2H])
        Tensor z = torch::autograd::sigmoid_autograd(torch::autograd::add_autograd(
            torch::autograd::narrow_autograd(gi_c, 1, H, H),
            torch::autograd::narrow_autograd(gh_c, 1, H, H)));
        // n = tanh(gi[:, 2H:3H] + r * gh[:, 2H:3H])
        Tensor n = torch::autograd::tanh_autograd(torch::autograd::add_autograd(
            torch::autograd::narrow_autograd(gi_c, 1, 2 * H, H),
            torch::autograd::mul_autograd(r, torch::autograd::narrow_autograd(gh_c, 1, 2 * H, H))));

        // h' = (1 - z) * n + z * h
        Tensor ones = at::ones(z.sizes());
        Tensor one_minus_z = torch::autograd::sub_autograd(ones, z);
        Tensor h_new = torch::autograd::add_autograd(
            torch::autograd::mul_autograd(one_minus_z, n),
            torch::autograd::mul_autograd(z, hx));

        return h_new;
    }

    Tensor forward(const Tensor& input) override {
        Tensor hx = at::zeros({input.size(0), hidden_size_});
        return forward(input, hx);
    }

    int64_t input_size() const { return input_size_; }
    int64_t hidden_size() const { return hidden_size_; }

private:
    int64_t input_size_, hidden_size_;
    std::shared_ptr<Linear> ih_, hh_;
};

// ============================================================================
// RNN — Multi-layer RNN with optional bidirectional
// ============================================================================
// Input: [seq_len, batch, input_size] (or [batch, seq_len, input_size] if batch_first)
// Output: (output, h_n)
//   output: [seq_len, batch, num_directions * hidden_size]
//   h_n:    [num_layers * num_directions, batch, hidden_size]

class RNN : public Module {
public:
    RNN(int64_t input_size, int64_t hidden_size,
        int64_t num_layers = 1, bool bias = true,
        bool batch_first = false, double dropout = 0.0,
        bool bidirectional = false)
        : Module("RNN")
        , input_size_(input_size)
        , hidden_size_(hidden_size)
        , num_layers_(num_layers)
        , batch_first_(batch_first)
        , dropout_(dropout)
        , bidirectional_(bidirectional)
    {
        int64_t num_directions = bidirectional ? 2 : 1;
        for (int64_t layer = 0; layer < num_layers; ++layer) {
            int64_t in_size = (layer == 0) ? input_size : hidden_size * num_directions;
            for (int64_t dir = 0; dir < num_directions; ++dir) {
                auto cell = std::make_shared<RNNCellImpl>(in_size, hidden_size, bias);
                std::string name = "cell_" + std::to_string(layer) + "_" + std::to_string(dir);
                register_module(name, cell);
                cells_.push_back(cell);
            }
        }
    }

    // Returns (output, h_n)
    std::pair<Tensor, Tensor> forward_rnn(const Tensor& input, const Tensor& h0 = Tensor()) {
        int64_t num_directions = bidirectional_ ? 2 : 1;

        // Handle batch_first
        Tensor x = batch_first_ ? input.transpose(0, 1).contiguous() : input;
        // x: [seq_len, batch, features]
        int64_t seq_len = x.size(0);
        int64_t batch = x.size(1);

        // Initialize hidden states
        Tensor hx;
        if (h0.defined()) {
            hx = h0;
        } else {
            hx = at::zeros({num_layers_ * num_directions, batch, hidden_size_});
        }

        // Collect final hidden states
        std::vector<Tensor> h_n_list;

        // Current input for each layer
        Tensor layer_input = x;

        for (int64_t layer = 0; layer < num_layers_; ++layer) {
            std::vector<Tensor> layer_outputs;

            // Forward direction
            int64_t cell_idx = layer * num_directions;
            auto& fwd_cell = cells_[cell_idx];
            Tensor h_fwd = at::native::select(hx, 0, layer * num_directions);

            for (int64_t t = 0; t < seq_len; ++t) {
                Tensor xt = at::native::select(layer_input, 0, t);
                h_fwd = std::dynamic_pointer_cast<RNNCellImpl>(fwd_cell)->forward(xt, h_fwd);
                layer_outputs.push_back(h_fwd.unsqueeze(0));
            }
            h_n_list.push_back(h_fwd.unsqueeze(0));

            if (bidirectional_) {
                // Backward direction
                auto& bwd_cell = cells_[cell_idx + 1];
                Tensor h_bwd = at::native::select(hx, 0, layer * num_directions + 1);
                std::vector<Tensor> bwd_outputs(seq_len);

                for (int64_t t = seq_len - 1; t >= 0; --t) {
                    Tensor xt = at::native::select(layer_input, 0, t);
                    h_bwd = std::dynamic_pointer_cast<RNNCellImpl>(bwd_cell)->forward(xt, h_bwd);
                    bwd_outputs[t] = h_bwd.unsqueeze(0);
                }
                h_n_list.push_back(h_bwd.unsqueeze(0));

                // Concatenate forward and backward outputs
                for (int64_t t = 0; t < seq_len; ++t) {
                    layer_outputs[t] = at::native::cat({layer_outputs[t], bwd_outputs[t]}, 2);
                }
            }

            // Stack outputs: [seq_len, batch, hidden * num_directions]
            layer_input = at::native::cat(layer_outputs, 0);
        }

        // Stack h_n: [num_layers * num_directions, batch, hidden]
        Tensor h_n = at::native::cat(h_n_list, 0);

        Tensor output = batch_first_ ? layer_input.transpose(0, 1).contiguous() : layer_input;
        return {output, h_n};
    }

    Tensor forward(const Tensor& input) override {
        auto [output, h_n] = forward_rnn(input);
        return output;
    }

    int64_t input_size() const { return input_size_; }
    int64_t hidden_size() const { return hidden_size_; }
    int64_t num_layers() const { return num_layers_; }

private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;
    double dropout_;
    std::vector<ModulePtr> cells_;
};

// ============================================================================
// LSTM — Multi-layer LSTM
// ============================================================================
// Returns (output, (h_n, c_n))

class LSTM : public Module {
public:
    LSTM(int64_t input_size, int64_t hidden_size,
         int64_t num_layers = 1, bool bias = true,
         bool batch_first = false, double dropout = 0.0,
         bool bidirectional = false)
        : Module("LSTM")
        , input_size_(input_size)
        , hidden_size_(hidden_size)
        , num_layers_(num_layers)
        , batch_first_(batch_first)
        , dropout_(dropout)
        , bidirectional_(bidirectional)
    {
        int64_t num_directions = bidirectional ? 2 : 1;
        for (int64_t layer = 0; layer < num_layers; ++layer) {
            int64_t in_size = (layer == 0) ? input_size : hidden_size * num_directions;
            for (int64_t dir = 0; dir < num_directions; ++dir) {
                auto cell = std::make_shared<LSTMCellImpl>(in_size, hidden_size, bias);
                std::string name = "cell_" + std::to_string(layer) + "_" + std::to_string(dir);
                register_module(name, cell);
                cells_.push_back(cell);
            }
        }
    }

    // Returns (output, h_n, c_n)
    struct LSTMOutput {
        Tensor output;
        Tensor h_n;
        Tensor c_n;
    };

    LSTMOutput forward_lstm(const Tensor& input,
                            const Tensor& h0 = Tensor(),
                            const Tensor& c0 = Tensor()) {
        int64_t num_directions = bidirectional_ ? 2 : 1;

        Tensor x = batch_first_ ? input.transpose(0, 1).contiguous() : input;
        int64_t seq_len = x.size(0);
        int64_t batch = x.size(1);

        Tensor hx = h0.defined() ? h0 : at::zeros({num_layers_ * num_directions, batch, hidden_size_});
        Tensor cx = c0.defined() ? c0 : at::zeros({num_layers_ * num_directions, batch, hidden_size_});

        std::vector<Tensor> h_n_list, c_n_list;
        Tensor layer_input = x;

        for (int64_t layer = 0; layer < num_layers_; ++layer) {
            std::vector<Tensor> layer_outputs;
            int64_t cell_idx = layer * num_directions;

            // Forward direction
            auto fwd_cell = std::dynamic_pointer_cast<LSTMCellImpl>(cells_[cell_idx]);
            Tensor h_fwd = at::native::select(hx, 0, layer * num_directions);
            Tensor c_fwd = at::native::select(cx, 0, layer * num_directions);

            for (int64_t t = 0; t < seq_len; ++t) {
                Tensor xt = at::native::select(layer_input, 0, t);
                auto [h_new, c_new] = fwd_cell->forward_lstm(xt, h_fwd, c_fwd);
                h_fwd = h_new;
                c_fwd = c_new;
                layer_outputs.push_back(h_fwd.unsqueeze(0));
            }
            h_n_list.push_back(h_fwd.unsqueeze(0));
            c_n_list.push_back(c_fwd.unsqueeze(0));

            if (bidirectional_) {
                auto bwd_cell = std::dynamic_pointer_cast<LSTMCellImpl>(cells_[cell_idx + 1]);
                Tensor h_bwd = at::native::select(hx, 0, layer * num_directions + 1);
                Tensor c_bwd = at::native::select(cx, 0, layer * num_directions + 1);
                std::vector<Tensor> bwd_outputs(seq_len);

                for (int64_t t = seq_len - 1; t >= 0; --t) {
                    Tensor xt = at::native::select(layer_input, 0, t);
                    auto [h_new, c_new] = bwd_cell->forward_lstm(xt, h_bwd, c_bwd);
                    h_bwd = h_new;
                    c_bwd = c_new;
                    bwd_outputs[t] = h_bwd.unsqueeze(0);
                }
                h_n_list.push_back(h_bwd.unsqueeze(0));
                c_n_list.push_back(c_bwd.unsqueeze(0));

                for (int64_t t = 0; t < seq_len; ++t) {
                    layer_outputs[t] = at::native::cat({layer_outputs[t], bwd_outputs[t]}, 2);
                }
            }

            layer_input = at::native::cat(layer_outputs, 0);
        }

        Tensor h_n = at::native::cat(h_n_list, 0);
        Tensor c_n = at::native::cat(c_n_list, 0);
        Tensor output = batch_first_ ? layer_input.transpose(0, 1).contiguous() : layer_input;

        return {output, h_n, c_n};
    }

    Tensor forward(const Tensor& input) override {
        auto result = forward_lstm(input);
        return result.output;
    }

    int64_t input_size() const { return input_size_; }
    int64_t hidden_size() const { return hidden_size_; }
    int64_t num_layers() const { return num_layers_; }

private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;
    double dropout_;
    std::vector<ModulePtr> cells_;
};

// ============================================================================
// GRU — Multi-layer GRU
// ============================================================================

class GRU : public Module {
public:
    GRU(int64_t input_size, int64_t hidden_size,
        int64_t num_layers = 1, bool bias = true,
        bool batch_first = false, double dropout = 0.0,
        bool bidirectional = false)
        : Module("GRU")
        , input_size_(input_size)
        , hidden_size_(hidden_size)
        , num_layers_(num_layers)
        , batch_first_(batch_first)
        , dropout_(dropout)
        , bidirectional_(bidirectional)
    {
        int64_t num_directions = bidirectional ? 2 : 1;
        for (int64_t layer = 0; layer < num_layers; ++layer) {
            int64_t in_size = (layer == 0) ? input_size : hidden_size * num_directions;
            for (int64_t dir = 0; dir < num_directions; ++dir) {
                auto cell = std::make_shared<GRUCellImpl>(in_size, hidden_size, bias);
                std::string name = "cell_" + std::to_string(layer) + "_" + std::to_string(dir);
                register_module(name, cell);
                cells_.push_back(cell);
            }
        }
    }

    std::pair<Tensor, Tensor> forward_gru(const Tensor& input, const Tensor& h0 = Tensor()) {
        int64_t num_directions = bidirectional_ ? 2 : 1;

        Tensor x = batch_first_ ? input.transpose(0, 1).contiguous() : input;
        int64_t seq_len = x.size(0);
        int64_t batch = x.size(1);

        Tensor hx = h0.defined() ? h0 : at::zeros({num_layers_ * num_directions, batch, hidden_size_});

        std::vector<Tensor> h_n_list;
        Tensor layer_input = x;

        for (int64_t layer = 0; layer < num_layers_; ++layer) {
            std::vector<Tensor> layer_outputs;
            int64_t cell_idx = layer * num_directions;

            // Forward direction
            auto fwd_cell = std::dynamic_pointer_cast<GRUCellImpl>(cells_[cell_idx]);
            Tensor h_fwd = at::native::select(hx, 0, layer * num_directions);

            for (int64_t t = 0; t < seq_len; ++t) {
                Tensor xt = at::native::select(layer_input, 0, t);
                h_fwd = fwd_cell->forward(xt, h_fwd);
                layer_outputs.push_back(h_fwd.unsqueeze(0));
            }
            h_n_list.push_back(h_fwd.unsqueeze(0));

            if (bidirectional_) {
                auto bwd_cell = std::dynamic_pointer_cast<GRUCellImpl>(cells_[cell_idx + 1]);
                Tensor h_bwd = at::native::select(hx, 0, layer * num_directions + 1);
                std::vector<Tensor> bwd_outputs(seq_len);

                for (int64_t t = seq_len - 1; t >= 0; --t) {
                    Tensor xt = at::native::select(layer_input, 0, t);
                    h_bwd = bwd_cell->forward(xt, h_bwd);
                    bwd_outputs[t] = h_bwd.unsqueeze(0);
                }
                h_n_list.push_back(h_bwd.unsqueeze(0));

                for (int64_t t = 0; t < seq_len; ++t) {
                    layer_outputs[t] = at::native::cat({layer_outputs[t], bwd_outputs[t]}, 2);
                }
            }

            layer_input = at::native::cat(layer_outputs, 0);
        }

        Tensor h_n = at::native::cat(h_n_list, 0);
        Tensor output = batch_first_ ? layer_input.transpose(0, 1).contiguous() : layer_input;

        return {output, h_n};
    }

    Tensor forward(const Tensor& input) override {
        auto [output, h_n] = forward_gru(input);
        return output;
    }

    int64_t input_size() const { return input_size_; }
    int64_t hidden_size() const { return hidden_size_; }
    int64_t num_layers() const { return num_layers_; }

private:
    int64_t input_size_, hidden_size_, num_layers_;
    bool batch_first_, bidirectional_;
    double dropout_;
    std::vector<ModulePtr> cells_;
};

} // namespace nn
} // namespace torch
