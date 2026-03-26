// ============================================================================
// PIR 250M Training on Elbrus-8SV (32 cores, NUMA-aware)
// ============================================================================
// Port of PIR 270M.py to PromeTorch C++
// Architecture: Pure PIR (no attention) — parallel scan + SwiGLU FFN
//
// Usage:
//   ./train_pir_elbrus --data tiny_shakespeare.txt [--n_layers 4] [--n_embd 256]
//
// Elbrus NUMA: 4 nodes × 8 cores. OMP_PLACES=cores OMP_PROC_BIND=close
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "c10/core/Allocator.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <iomanip>
#include <numeric>
#include <string>
#include <sstream>
#include <cassert>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;
using at::Tensor;

// ============================================================================
// Configuration
// ============================================================================

struct PIRConfig {
    int64_t vocab_size   = 256;     // ASCII char-level tokenizer
    int64_t n_embd       = 256;     // Embedding dimension (small for testing)
    int64_t n_layers     = 4;       // Number of transformer blocks (small for testing)
    int64_t n_pir_layers = 4;       // PIR layers per block (multi-scale decay)
    int64_t block_size   = 512;     // Context window (small for testing)
    float   ffn_mult     = 3.5f;    // SwiGLU hidden multiplier

    // Multi-scale decay ranges
    // L0: words/morphology, L1: phrases, L2: paragraphs, L3: document sections
    float decay_min[4] = {0.80f, 0.95f, 0.99f, 0.998f};
    float decay_max[4] = {0.95f, 0.99f, 0.998f, 0.9995f};
};

struct TrainConfig {
    int64_t batch_size    = 8;
    int64_t max_steps     = 10000;
    int64_t warmup_steps  = 200;
    float   learning_rate = 6e-4f;
    float   min_lr        = 6e-5f;
    float   weight_decay  = 0.1f;
    float   grad_clip     = 1.0f;
    float   beta1         = 0.9f;
    float   beta2         = 0.95f;

    int64_t log_interval  = 10;
    int64_t eval_interval = 200;
    int64_t gen_interval  = 500;
    int64_t gen_tokens    = 200;

    std::string data_path = "tiny_shakespeare.txt";
};

// ============================================================================
// Parallel Scan — O(T) via cumsum trick
// ============================================================================
// PARALLEL SCAN with custom backward (not autograd chain!)
// ============================================================================
// Forward: out[t] = sum_{s<=t} prod_{s<k<=t}(gate[k]) * x[s]
// Backward: d_x[t] = sum_{s>=t} prod_{t<k<=s}(gate[k]) * d_out[s]  (reverse scan)
//           d_gate[t] = sum_{s>=t} out[s-1] * d_out[s] * prod_{...}
//
// The autograd chain (log→cumsum→clamp→exp→mul) kills gradients because
// clamp(-20,20) on cumsum_log zeroes gradient when clamped. Instead we
// compute forward/backward directly with scalar loops.

// Forward: sequential scan (no autograd, just computation)
Tensor parallel_scan_forward(const Tensor& gates, const Tensor& x) {
    // gates, x: [B, T, D]
    int64_t B = x.size(0), T = x.size(1), D = x.size(2);
    auto out = at::zeros_like(x);
    float* g_ptr = gates.data_ptr<float>();
    float* x_ptr = x.data_ptr<float>();
    float* o_ptr = out.data_ptr<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int64_t d = 0; d < D; d++) {
            float h = 0.0f;
            for (int64_t t = 0; t < T; t++) {
                int64_t idx = b * T * D + t * D + d;
                h = g_ptr[idx] * h + x_ptr[idx];
                o_ptr[idx] = h;
            }
        }
    }
    return out;
}

// Backward: reverse scan for d_x, d_gates
std::pair<Tensor, Tensor> parallel_scan_backward(
    const Tensor& gates, const Tensor& x, const Tensor& out, const Tensor& d_out
) {
    int64_t B = x.size(0), T = x.size(1), D = x.size(2);
    auto d_x = at::zeros_like(x);
    auto d_gates = at::zeros_like(gates);
    float* g_ptr = gates.data_ptr<float>();
    float* x_ptr = x.data_ptr<float>();
    float* o_ptr = out.data_ptr<float>();
    float* do_ptr = d_out.data_ptr<float>();
    float* dx_ptr = d_x.data_ptr<float>();
    float* dg_ptr = d_gates.data_ptr<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int64_t d = 0; d < D; d++) {
            float dh = 0.0f;
            for (int64_t t = T - 1; t >= 0; t--) {
                int64_t idx = b * T * D + t * D + d;
                dh += do_ptr[idx];
                dx_ptr[idx] = dh;
                // d_gate[t] = dh * h[t-1]  (h[t] = gate[t]*h[t-1] + x[t])
                if (t > 0) {
                    dg_ptr[idx] = dh * o_ptr[b * T * D + (t-1) * D + d];
                } else {
                    dg_ptr[idx] = 0.0f;
                }
                dh = dh * g_ptr[idx];  // propagate through gate
            }
        }
    }
    return {d_x, d_gates};
}

// Custom autograd node for parallel scan
struct ParallelScanBackward : public torch::autograd::Node {
    Tensor saved_gates, saved_x, saved_out;

    ParallelScanBackward(const Tensor& gates, const Tensor& x, const Tensor& out)
        : saved_gates(gates), saved_x(x), saved_out(out) {}

    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override {
        auto d_out = grads[0];
        if (!d_out.defined()) return {Tensor(), Tensor()};
        auto result = parallel_scan_backward(saved_gates, saved_x, saved_out, d_out);
        return {result.first, result.second};  // d_gates, d_x
    }

    std::string name() const override { return "ParallelScanBackward"; }
    void release_saved_tensors() override {
        saved_gates = Tensor(); saved_x = Tensor(); saved_out = Tensor();
    }
};

// Autograd-wrapped parallel scan
Tensor parallel_scan(const Tensor& gates, const Tensor& x) {
    auto gates_c = gates.contiguous();
    auto x_c = x.contiguous();
    auto out = parallel_scan_forward(gates_c, x_c);

    // Wire autograd
    if (gates.requires_grad() || x.requires_grad()) {
        auto node = std::make_shared<ParallelScanBackward>(gates_c, x_c, out);
        node->add_input_metadata(gates);
        node->add_input_metadata(x);
        torch::autograd::set_grad_fn(out, node);
    }
    return out;
}

// Helper: create a scalar tensor for autograd-tracked scalar multiply
inline Tensor scalar_tensor(float val) {
    Tensor t = at::zeros({});
    t.mutable_data_ptr<float>()[0] = val;
    return t;
}

// Dynamic parallel scan with gating modulation
Tensor dynamic_parallel_scan(const Tensor& x, const Tensor& gate_logits,
                              const Tensor& base_decay) {
    // x: [B, T, D], gate_logits: [B, T, D], base_decay: [D]
    // modulation = tanh(gate_logits) * 0.1
    auto tanh_gl = torch::autograd::tanh_autograd(gate_logits);
    // Multiply by 0.1: create a constant tensor filled with 0.1 for mul_autograd
    auto scale_01 = at::empty(tanh_gl.sizes());
    {
        float* sd = scale_01.mutable_data_ptr<float>();
        for (int64_t i = 0; i < scale_01.numel(); i++) sd[i] = 0.1f;
    }
    auto modulation = torch::autograd::mul_autograd(tanh_gl, scale_01);
    // ones_mod = 1 + modulation
    auto ones_like_mod = at::ones(modulation.sizes());
    auto ones_mod = torch::autograd::add_autograd(ones_like_mod, modulation);

    // gates = base_decay * (1 + modulation), clamped to [0.5, 0.9999]
    // base_decay is [D], unsqueeze to [1, 1, D] for broadcast
    auto gates = torch::autograd::mul_autograd(ones_mod, base_decay.unsqueeze(0).unsqueeze(0));
    gates = torch::autograd::clamp_autograd(gates, at::Scalar(0.5f), at::Scalar(0.9999f));

    return parallel_scan(gates, x);
}

// ============================================================================
// RMSNorm
// ============================================================================

class RMSNorm : public Module {
    float eps_;
public:
    RMSNorm(int64_t dim, float eps = 1e-6f)
        : Module("RMSNorm"), eps_(eps) {
        register_parameter("weight", Parameter(at::ones({dim})));
    }

    Tensor forward(const Tensor& input) override {
        // RMSNorm: x * rsqrt(mean(x^2, dim=-1, keepdim=true) + eps) * weight
        // All ops through autograd for gradient flow

        // x^2
        auto x_sq = torch::autograd::mul_autograd(input, input);
        // mean(x^2, dim=-1, keepdim=true)
        auto mean_sq = torch::autograd::mean_autograd(x_sq, -1, /*keepdim=*/true);
        // mean + eps (eps is constant, non-autograd add is fine — grad flows through mean_sq)
        auto mean_plus_eps = mean_sq.add(at::Scalar(eps_));
        // rsqrt via pow(-0.5) with autograd tracking
        auto rms_scale = torch::autograd::pow_autograd(mean_plus_eps, at::Scalar(-0.5f));
        // x * rsqrt(...)
        auto normed = torch::autograd::mul_autograd(input, rms_scale);
        // * weight (broadcast [D] -> [B, T, D])
        auto weight_param = get_parameter("weight")->data();
        return torch::autograd::mul_autograd(normed, weight_param);
    }
};

// ============================================================================
// Rotary Embedding (RoPE) — precomputed cos/sin tables
// ============================================================================

class RotaryEmbedding {
    Tensor cos_cached_;  // [max_seq_len, dim/2]
    Tensor sin_cached_;
    int64_t dim_;

public:
    RotaryEmbedding() : dim_(0) {}

    RotaryEmbedding(int64_t dim, int64_t max_seq_len = 2048, float base = 10000.0f)
        : dim_(dim) {
        // inv_freq = 1 / (base^(2i/dim))  for i = 0..dim/2
        int64_t half_dim = dim / 2;
        Tensor inv_freq = at::empty({half_dim});
        float* ifp = inv_freq.mutable_data_ptr<float>();
        for (int64_t i = 0; i < half_dim; i++) {
            ifp[i] = 1.0f / std::pow(base, 2.0f * i / dim);
        }

        // freqs[t, i] = t * inv_freq[i]
        // emb = [freqs, freqs] along last dim -> [max_seq_len, dim]
        cos_cached_ = at::empty({max_seq_len, dim});
        sin_cached_ = at::empty({max_seq_len, dim});
        float* cos_data = cos_cached_.mutable_data_ptr<float>();
        float* sin_data = sin_cached_.mutable_data_ptr<float>();

        for (int64_t t = 0; t < max_seq_len; t++) {
            for (int64_t i = 0; i < half_dim; i++) {
                float angle = t * ifp[i];
                float c = std::cos(angle);
                float s = std::sin(angle);
                cos_data[t * dim + i] = c;
                cos_data[t * dim + half_dim + i] = c;
                sin_data[t * dim + i] = s;
                sin_data[t * dim + half_dim + i] = s;
            }
        }
    }

    // Returns cos/sin for first seq_len positions: [1, seq_len, dim]
    std::pair<Tensor, Tensor> get(int64_t seq_len) const {
        auto cos_slice = cos_cached_.narrow(0, 0, seq_len).unsqueeze(0);
        auto sin_slice = sin_cached_.narrow(0, 0, seq_len).unsqueeze(0);
        return {cos_slice, sin_slice};
    }
};

// Apply RoPE: rotate first half of embedding, keep second half unchanged
// x: [B, T, D]   cos,sin: [1, T, D/2] (but stored as [1, T, D] with repeated halves)
Tensor apply_rotary_pos_emb(const Tensor& x, const Tensor& cos_t, const Tensor& sin_t) {
    // x1, x2 = x.chunk(2, dim=-1)
    auto chunks = x.chunk(2, -1);
    Tensor x1 = chunks[0];  // [B, T, D/2]
    Tensor x2 = chunks[1];  // [B, T, D/2]

    // For x1: apply rotation
    // rotate_half(x1): [-x1_second_half, x1_first_half]
    int64_t half = x1.size(-1) / 2;
    // But cos/sin are [1, T, D], first D/2 entries are for the rotation
    // cos_for_x1 = cos[:, :, :D/2]
    auto cos_half = cos_t.narrow(-1, 0, x1.size(-1));
    auto sin_half = sin_t.narrow(-1, 0, x1.size(-1));

    // rotate_half(x1): split x1 into two halves, negate+swap
    auto x1_chunks = x1.chunk(2, -1);
    Tensor x1a = x1_chunks[0];
    Tensor x1b = x1_chunks[1];
    auto rotated = at::native::cat({torch::autograd::neg_autograd(x1b), x1a}, -1);

    // x1_rotated = x1 * cos + rotate_half(x1) * sin
    auto x1_cos = torch::autograd::mul_autograd(x1, cos_half);
    auto rot_sin = torch::autograd::mul_autograd(rotated, sin_half);
    auto x1_rotated = torch::autograd::add_autograd(x1_cos, rot_sin);

    return at::native::cat({x1_rotated, x2}, -1);
}

// ============================================================================
// PIR Layer — single-scale concept compression
// ============================================================================

class PIRLayer : public Module {
    int64_t n_embd_;
    std::shared_ptr<Linear> gate_proj_, value_proj_, out_proj_;
    std::shared_ptr<RMSNorm> norm_;

public:
    PIRLayer(int64_t n_embd, float decay_min, float decay_max, int layer_idx = 0)
        : Module("PIRLayer"), n_embd_(n_embd) {

        // base_decay: linspace(decay_min, decay_max, n_embd)
        Tensor base_decay = at::linspace(decay_min, decay_max, n_embd);
        register_buffer("base_decay", Buffer(base_decay));

        gate_proj_ = std::make_shared<Linear>(n_embd, n_embd, /*bias=*/false);
        value_proj_ = std::make_shared<Linear>(n_embd, n_embd, /*bias=*/false);
        out_proj_ = std::make_shared<Linear>(n_embd, n_embd, /*bias=*/false);
        norm_ = std::make_shared<RMSNorm>(n_embd);

        register_module("gate_proj", gate_proj_);
        register_module("value_proj", value_proj_);
        register_module("out_proj", out_proj_);
        register_module("norm", norm_);

        // Init weights (orthogonal-like: scaled normal)
        init_weights();
    }

    void init_weights() {
        auto& gw = gate_proj_->get_parameter("weight")->data();
        init::normal_(gw, 0.0, 0.1 / std::sqrt((double)n_embd_));

        auto& vw = value_proj_->get_parameter("weight")->data();
        init::normal_(vw, 0.0, 1.0 / std::sqrt((double)n_embd_));

        auto& ow = out_proj_->get_parameter("weight")->data();
        init::normal_(ow, 0.0, 0.5 / std::sqrt((double)n_embd_));
    }

    Tensor forward(const Tensor& input) override {
        // gate_logits = gate_proj(x)
        auto gate_logits = gate_proj_->forward(input);
        // values = value_proj(x)
        auto values = value_proj_->forward(input);
        // value_gate = sigmoid(gate_logits)
        auto value_gate = torch::autograd::sigmoid_autograd(gate_logits);
        // gated_values = values * value_gate
        auto gated_values = torch::autograd::mul_autograd(values, value_gate);

        // Get base_decay buffer
        auto* buf = get_buffer("base_decay");
        Tensor base_decay = buf->data();

        // scanned = dynamic_parallel_scan(gated_values, gate_logits, base_decay)
        auto scanned = dynamic_parallel_scan(gated_values, gate_logits, base_decay);

        // out = out_proj(scanned)
        auto out = out_proj_->forward(scanned);
        // norm(out)
        return norm_->forward(out);
    }
};

// ============================================================================
// PIR Block — 4 PIR layers with different decay scales
// ============================================================================

class PIRBlock : public Module {
    std::vector<std::shared_ptr<PIRLayer>> layers_;
    std::shared_ptr<Linear> mix_proj_;
    std::shared_ptr<RMSNorm> norm_;

public:
    PIRBlock(int64_t n_embd, int64_t n_pir_layers, const PIRConfig& cfg)
        : Module("PIRBlock") {
        for (int64_t i = 0; i < n_pir_layers; i++) {
            int idx = i % 4;
            auto layer = std::make_shared<PIRLayer>(
                n_embd, cfg.decay_min[idx], cfg.decay_max[idx], i
            );
            layers_.push_back(layer);
            register_module("pir_" + std::to_string(i), layer);
        }

        mix_proj_ = std::make_shared<Linear>(n_embd, n_embd, /*bias=*/false);
        norm_ = std::make_shared<RMSNorm>(n_embd);
        register_module("mix_proj", mix_proj_);
        register_module("norm", norm_);

        // Init mix_proj
        auto& mw = mix_proj_->get_parameter("weight")->data();
        init::normal_(mw, 0.0, 0.5 / std::sqrt((double)n_embd));
    }

    Tensor forward(const Tensor& input) override {
        Tensor h = input;
        for (auto& layer : layers_) {
            // h = h + pir_layer(h)  — residual
            auto out = layer->forward(h);
            h = torch::autograd::add_autograd(h, out);
        }
        auto mixed = mix_proj_->forward(h);
        return norm_->forward(mixed);
    }
};

// ============================================================================
// SwiGLU Feed-Forward Network
// ============================================================================

class SwiGLUFFN : public Module {
    std::shared_ptr<Linear> w1_, w2_, w3_;

public:
    SwiGLUFFN(int64_t n_embd, float mult = 3.5f)
        : Module("SwiGLUFFN") {
        int64_t hidden = static_cast<int64_t>(n_embd * mult * 2.0f / 3.0f);
        hidden = ((hidden + 63) / 64) * 64;  // Round up to 64

        w1_ = std::make_shared<Linear>(n_embd, hidden, /*bias=*/false);
        w2_ = std::make_shared<Linear>(hidden, n_embd, /*bias=*/false);
        w3_ = std::make_shared<Linear>(n_embd, hidden, /*bias=*/false);

        register_module("w1", w1_);
        register_module("w2", w2_);
        register_module("w3", w3_);

        // Init weights
        auto& w1d = w1_->get_parameter("weight")->data();
        init::normal_(w1d, 0.0, 1.0 / std::sqrt((double)n_embd));
        auto& w2d = w2_->get_parameter("weight")->data();
        init::normal_(w2d, 0.0, 0.5 / std::sqrt((double)hidden));
        auto& w3d = w3_->get_parameter("weight")->data();
        init::normal_(w3d, 0.0, 1.0 / std::sqrt((double)n_embd));
    }

    Tensor forward(const Tensor& input) override {
        // SwiGLU: w2(silu(w1(x)) * w3(x))
        auto h1 = w1_->forward(input);
        auto h1_silu = torch::autograd::silu_autograd(h1);
        auto h3 = w3_->forward(input);
        auto gated = torch::autograd::mul_autograd(h1_silu, h3);
        return w2_->forward(gated);
    }
};

// ============================================================================
// Transformer Block = PIRBlock + SwiGLU FFN (no attention!)
// ============================================================================

class TransformerBlock : public Module {
    std::shared_ptr<PIRBlock> pir_;
    std::shared_ptr<SwiGLUFFN> ffn_;
    std::shared_ptr<RMSNorm> norm1_, norm2_;

public:
    TransformerBlock(int64_t n_embd, int64_t n_pir_layers, float ffn_mult,
                     const PIRConfig& cfg)
        : Module("TransformerBlock") {
        pir_ = std::make_shared<PIRBlock>(n_embd, n_pir_layers, cfg);
        ffn_ = std::make_shared<SwiGLUFFN>(n_embd, ffn_mult);
        norm1_ = std::make_shared<RMSNorm>(n_embd);
        norm2_ = std::make_shared<RMSNorm>(n_embd);

        register_module("pir", pir_);
        register_module("ffn", ffn_);
        register_module("norm1", norm1_);
        register_module("norm2", norm2_);
    }

    Tensor forward(const Tensor& input) override {
        // x = x + pir(norm1(x))
        auto normed1 = norm1_->forward(input);
        auto pir_out = pir_->forward(normed1);
        auto x = torch::autograd::add_autograd(input, pir_out);

        // x = x + ffn(norm2(x))
        auto normed2 = norm2_->forward(x);
        auto ffn_out = ffn_->forward(normed2);
        return torch::autograd::add_autograd(x, ffn_out);
    }
};

// ============================================================================
// PIR 250M Language Model
// ============================================================================

class PIR250M : public Module {
    PIRConfig config_;
    std::shared_ptr<Embedding> tok_emb_;
    RotaryEmbedding rope_;
    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
    std::shared_ptr<RMSNorm> norm_out_;
    std::shared_ptr<Linear> lm_head_;

public:
    PIR250M(const PIRConfig& cfg)
        : Module("PIR250M"), config_(cfg) {

        // Token embedding
        tok_emb_ = std::make_shared<Embedding>(cfg.vocab_size, cfg.n_embd);
        register_module("tok_emb", tok_emb_);

        // Initialize embedding weights
        auto& emb_w = tok_emb_->get_parameter("weight")->data();
        init::normal_(emb_w, 0.0, 0.02);

        // RoPE (precomputed, not a module — no parameters)
        rope_ = RotaryEmbedding(cfg.n_embd / 2, cfg.block_size);

        // Transformer blocks
        for (int64_t i = 0; i < cfg.n_layers; i++) {
            auto block = std::make_shared<TransformerBlock>(
                cfg.n_embd, cfg.n_pir_layers, cfg.ffn_mult, cfg
            );
            blocks_.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }

        // Output norm and LM head
        norm_out_ = std::make_shared<RMSNorm>(cfg.n_embd);
        register_module("norm_out", norm_out_);

        lm_head_ = std::make_shared<Linear>(cfg.n_embd, cfg.vocab_size, /*bias=*/false);
        register_module("lm_head", lm_head_);

        // Weight tying: lm_head shares embedding weight
        // Copy embedding data to lm_head (they diverge during training, but
        // for true weight tying we make lm_head point to same parameter)
        // PromeTorch doesn't have native weight tying, so we manually sync
        auto& lm_w = lm_head_->get_parameter("weight")->data();
        init::normal_(lm_w, 0.0, 0.02);

        // Count params
        int64_t total = count_parameters(*this);
        std::cout << "PIR model initialized: " << (total / 1e6) << "M parameters"
                  << " | n_layers=" << cfg.n_layers
                  << " | n_embd=" << cfg.n_embd
                  << " | block_size=" << cfg.block_size
                  << std::endl;
    }

    Tensor forward(const Tensor& input) override {
        // input: [B, T] integer token indices
        int64_t T = input.size(1);

        // Token embedding: [B, T] -> [B, T, D]
        auto x = tok_emb_->forward(input);

        // Apply RoPE
        auto [cos_t, sin_t] = rope_.get(T);
        x = apply_rotary_pos_emb(x, cos_t, sin_t);

        // Transformer blocks
        for (auto& block : blocks_) {
            x = block->forward(x);
        }

        // Output norm + LM head
        x = norm_out_->forward(x);
        auto logits = lm_head_->forward(x);  // [B, T, vocab_size]

        return logits;
    }

    // Compute loss: cross-entropy over shifted predictions
    // logits: [B, T, V], targets: [B, T]
    // We predict next token: logits[:, :-1, :] vs targets[:, 1:]
    Tensor compute_loss(const Tensor& logits, const Tensor& targets) {
        int64_t B = logits.size(0);
        int64_t T = logits.size(1);
        int64_t V = logits.size(2);

        // Shift: predict position t+1 from position t
        // logits_shift = logits[:, :-1, :]  -> [B, T-1, V]
        // targets_shift = targets[:, 1:]    -> [B, T-1]
        auto logits_shift = logits.narrow(1, 0, T - 1).contiguous();  // [B, T-1, V]
        auto targets_shift = targets.narrow(1, 1, T - 1).contiguous(); // [B, T-1]

        // Flatten to [B*(T-1), V] and [B*(T-1)]
        auto logits_flat = logits_shift.reshape({B * (T - 1), V});
        auto targets_flat = targets_shift.reshape({B * (T - 1)});

        // Cross-entropy loss using the PromeTorch loss module
        CrossEntropyLoss loss_fn;
        return loss_fn.forward(logits_flat, targets_flat);
    }

    // Simple greedy generation
    std::string generate(const std::string& prompt, int64_t max_tokens,
                         float temperature = 0.8f) {
        std::string result = prompt;
        std::mt19937 rng(42);

        // Build input tensor from prompt
        std::vector<float> input_data;
        for (char c : prompt) {
            input_data.push_back(static_cast<float>(static_cast<unsigned char>(c)));
        }

        for (int64_t i = 0; i < max_tokens; i++) {
            int64_t seq_len = static_cast<int64_t>(input_data.size());
            if (seq_len > config_.block_size) {
                // Truncate to block_size
                input_data.erase(input_data.begin(),
                                 input_data.begin() + (seq_len - config_.block_size));
                seq_len = config_.block_size;
            }

            // Create input tensor [1, seq_len]
            Tensor input_t = at::empty({1, seq_len});
            float* inp = input_t.mutable_data_ptr<float>();
            for (int64_t j = 0; j < seq_len; j++) {
                inp[j] = input_data[j];
            }

            // Forward pass (no grad)
            auto logits = this->forward(input_t);  // [1, seq_len, V]

            // Get last position logits: [V]
            int64_t V = config_.vocab_size;
            const float* last_logits = logits.data_ptr<float>() +
                                       (seq_len - 1) * V;

            // Apply temperature and sample
            std::vector<float> probs(V);
            float max_logit = *std::max_element(last_logits, last_logits + V);
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; v++) {
                probs[v] = std::exp((last_logits[v] - max_logit) / temperature);
                sum_exp += probs[v];
            }
            for (int64_t v = 0; v < V; v++) {
                probs[v] /= sum_exp;
            }

            // Weighted random sampling
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int next_token = dist(rng);

            result += static_cast<char>(next_token);
            input_data.push_back(static_cast<float>(next_token));
        }

        return result;
    }
};

// ============================================================================
// Data Loading — char-level tokenizer + batch preparation
// ============================================================================

class TextDataset {
    std::vector<uint8_t> data_;
    int64_t block_size_;
    std::mt19937 rng_;

public:
    TextDataset(const std::string& path, int64_t block_size)
        : block_size_(block_size), rng_(42) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "ERROR: Cannot open data file: " << path << std::endl;
            return;
        }
        data_ = std::vector<uint8_t>(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
        std::cout << "Loaded " << data_.size() << " bytes from " << path << std::endl;
    }

    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }

    // Get a random batch: returns (input, target) each [B, T]
    std::pair<Tensor, Tensor> get_batch(int64_t batch_size) {
        int64_t T = block_size_;
        Tensor input = at::empty({batch_size, T});
        Tensor target = at::empty({batch_size, T});
        float* inp = input.mutable_data_ptr<float>();
        float* tgt = target.mutable_data_ptr<float>();

        int64_t max_start = static_cast<int64_t>(data_.size()) - T - 1;
        if (max_start <= 0) {
            std::cerr << "Data too short for block_size=" << T << std::endl;
            return {input, target};
        }

        std::uniform_int_distribution<int64_t> dist(0, max_start);

        // Pre-generate random starts (RNG is not thread-safe)
        std::vector<int64_t> starts(batch_size);
        for (int64_t b = 0; b < batch_size; b++) {
            starts[b] = dist(rng_);
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t b = 0; b < batch_size; b++) {
            int64_t start = starts[b];

            for (int64_t t = 0; t < T; t++) {
                inp[b * T + t] = static_cast<float>(data_[start + t]);
                tgt[b * T + t] = static_cast<float>(data_[start + t + 1]);
            }
        }

        return {input, target};
    }
};

// ============================================================================
// Cosine LR schedule with warmup
// ============================================================================

float get_lr(int64_t step, const TrainConfig& cfg) {
    // Warmup phase
    if (step < cfg.warmup_steps) {
        return cfg.learning_rate * static_cast<float>(step) / cfg.warmup_steps;
    }
    // Cosine decay phase
    float progress = static_cast<float>(step - cfg.warmup_steps) /
                     static_cast<float>(cfg.max_steps - cfg.warmup_steps);
    progress = std::min(progress, 1.0f);
    float cosine = 0.5f * (1.0f + std::cos(M_PI * progress));
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cosine;
}

// ============================================================================
// NUMA-aware thread binding for Elbrus
// ============================================================================

void setup_numa_threads(int num_threads = 0) {
#ifdef _OPENMP
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    std::cout << "OpenMP threads: " << num_threads << std::endl;

    // For NUMA-aware execution on Elbrus:
    // Set OMP_PLACES=cores and OMP_PROC_BIND=close before running
    // This ensures each thread stays on its NUMA node
    const char* places = std::getenv("OMP_PLACES");
    const char* bind = std::getenv("OMP_PROC_BIND");
    std::cout << "OMP_PLACES=" << (places ? places : "(not set)")
              << " OMP_PROC_BIND=" << (bind ? bind : "(not set)") << std::endl;

    if (!places || !bind) {
        std::cout << "WARNING: For NUMA-optimal performance on Elbrus, set:\n"
                  << "  export OMP_PLACES=cores\n"
                  << "  export OMP_PROC_BIND=close\n"
                  << "  export OMP_NUM_THREADS=32\n"
                  << std::endl;
    }
#else
    std::cout << "OpenMP not available — single-threaded execution" << std::endl;
#endif
}

// ============================================================================
// Main Training Loop
// ============================================================================

void parse_args(int argc, char** argv, PIRConfig& model_cfg, TrainConfig& train_cfg) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            train_cfg.data_path = argv[++i];
        } else if (arg == "--n_layers" && i + 1 < argc) {
            model_cfg.n_layers = std::atoi(argv[++i]);
        } else if (arg == "--n_embd" && i + 1 < argc) {
            model_cfg.n_embd = std::atoi(argv[++i]);
        } else if (arg == "--block_size" && i + 1 < argc) {
            model_cfg.block_size = std::atoi(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            train_cfg.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--max_steps" && i + 1 < argc) {
            train_cfg.max_steps = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            train_cfg.learning_rate = std::atof(argv[++i]);
        } else if (arg == "--vocab_size" && i + 1 < argc) {
            model_cfg.vocab_size = std::atoi(argv[++i]);
        } else if (arg == "--n_pir_layers" && i + 1 < argc) {
            model_cfg.n_pir_layers = std::atoi(argv[++i]);
        } else if (arg == "--log_interval" && i + 1 < argc) {
            train_cfg.log_interval = std::atoi(argv[++i]);
        } else if (arg == "--eval_interval" && i + 1 < argc) {
            train_cfg.eval_interval = std::atoi(argv[++i]);
        } else if (arg == "--gen_interval" && i + 1 < argc) {
            train_cfg.gen_interval = std::atoi(argv[++i]);
        } else if (arg == "--gen_tokens" && i + 1 < argc) {
            train_cfg.gen_tokens = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            setup_numa_threads(std::atoi(argv[++i]));
        } else if (arg == "--full") {
            // Full 250M config
            model_cfg.vocab_size = 256;
            model_cfg.n_embd = 768;
            model_cfg.n_layers = 16;
            model_cfg.n_pir_layers = 4;
            model_cfg.block_size = 2048;
        } else if (arg == "--help") {
            std::cout << "PIR 250M Training — PromeTorch on Elbrus-8SV\n\n"
                      << "Usage: train_pir_elbrus [OPTIONS]\n\n"
                      << "  --data PATH          Training text file (default: tiny_shakespeare.txt)\n"
                      << "  --n_layers N         Transformer blocks (default: 4)\n"
                      << "  --n_embd N           Embedding dimension (default: 256)\n"
                      << "  --block_size N       Context window (default: 512)\n"
                      << "  --batch_size N       Batch size (default: 8)\n"
                      << "  --max_steps N        Training steps (default: 10000)\n"
                      << "  --lr FLOAT           Learning rate (default: 6e-4)\n"
                      << "  --n_pir_layers N     PIR layers per block (default: 4)\n"
                      << "  --vocab_size N       Vocabulary size (default: 256)\n"
                      << "  --log_interval N     Log every N steps (default: 10)\n"
                      << "  --eval_interval N    Eval every N steps (default: 200)\n"
                      << "  --gen_interval N     Generate text every N steps (default: 500)\n"
                      << "  --gen_tokens N       Tokens to generate (default: 200)\n"
                      << "  --threads N          OpenMP threads (default: auto)\n"
                      << "  --full               Full 250M config (768d, 16 layers, 2048 ctx)\n"
                      << std::endl;
            std::exit(0);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "PIR 250M Training — PromeTorch on Elbrus-8SV" << std::endl;
    std::cout << "============================================" << std::endl;

    // Setup NUMA-aware threading
    setup_numa_threads();

    PIRConfig model_cfg;
    TrainConfig train_cfg;
    parse_args(argc, argv, model_cfg, train_cfg);

    // Print config
    std::cout << "\nModel config:"
              << " vocab=" << model_cfg.vocab_size
              << " embd=" << model_cfg.n_embd
              << " layers=" << model_cfg.n_layers
              << " pir_layers=" << model_cfg.n_pir_layers
              << " block_size=" << model_cfg.block_size
              << " ffn_mult=" << model_cfg.ffn_mult
              << std::endl;

    std::cout << "Train config:"
              << " batch=" << train_cfg.batch_size
              << " steps=" << train_cfg.max_steps
              << " lr=" << train_cfg.learning_rate
              << " warmup=" << train_cfg.warmup_steps
              << " grad_clip=" << train_cfg.grad_clip
              << std::endl;

    // Load data
    TextDataset dataset(train_cfg.data_path, model_cfg.block_size);
    if (dataset.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return 1;
    }

    int64_t tokens_per_step = train_cfg.batch_size * model_cfg.block_size;
    std::cout << "Tokens per step: " << tokens_per_step
              << " | Total tokens: " << (tokens_per_step * train_cfg.max_steps)
              << std::endl;

    // Create model
    auto model = std::make_shared<PIR250M>(model_cfg);

    // Create optimizer: AdamW with weight decay
    auto params = model->parameters();
    AdamWOptions adam_opts(train_cfg.learning_rate);
    adam_opts.betas(train_cfg.beta1, train_cfg.beta2);
    adam_opts.weight_decay_(train_cfg.weight_decay);
    AdamW optimizer(params, adam_opts);

    // Training loop
    std::cout << "\n--- Training started ---\n" << std::endl;

    float running_loss = 0.0f;
    int64_t running_count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();
    auto step_start = total_start;

    for (int64_t step = 1; step <= train_cfg.max_steps; step++) {
        // Update learning rate (cosine schedule with warmup)
        float lr = get_lr(step, train_cfg);
        optimizer.set_lr(lr);

        // Get batch
        auto [input, target] = dataset.get_batch(train_cfg.batch_size);

        // Forward pass
        optimizer.zero_grad();
        auto logits = model->forward(input);

        // Compute loss
        auto loss = model->compute_loss(logits, target);

        // Backward pass
        torch::autograd::backward({loss});

        // Gradient clipping
        float grad_norm = fast_clip_grad_norm_(*model, train_cfg.grad_clip);

        // Optimizer step
        optimizer.step();

        // Track loss
        float loss_val = loss.data_ptr<float>()[0];
        running_loss += loss_val;
        running_count++;

        // Logging
        if (step % train_cfg.log_interval == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(
                now - step_start).count();
            double tok_per_sec = (tokens_per_step * train_cfg.log_interval) /
                                 (elapsed_ms / 1000.0);

            float avg_loss = running_loss / running_count;
            float perplexity = std::exp(std::min(avg_loss, 20.0f));

            std::cout << std::fixed << std::setprecision(4)
                      << "step " << std::setw(6) << step
                      << " | loss " << std::setw(7) << avg_loss
                      << " | ppl " << std::setw(10) << std::setprecision(1) << perplexity
                      << " | lr " << std::setprecision(6) << lr
                      << " | gnorm " << std::setprecision(3) << grad_norm
                      << " | " << std::setprecision(0) << tok_per_sec << " tok/s"
                      << std::endl;

            running_loss = 0.0f;
            running_count = 0;
            step_start = now;
        }

        // Generate sample text
        if (step % train_cfg.gen_interval == 0) {
            std::cout << "\n--- Generation at step " << step << " ---" << std::endl;
            model->eval();

            std::string prompt = "The ";
            std::string generated = model->generate(prompt, train_cfg.gen_tokens, 0.8f);

            // Print with non-printable chars replaced
            std::cout << ">>> ";
            for (char c : generated) {
                if (c >= 32 && c < 127) {
                    std::cout << c;
                } else if (c == '\n') {
                    std::cout << '\n';
                } else {
                    std::cout << '?';
                }
            }
            std::cout << "\n--- end generation ---\n" << std::endl;

            model->train();
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_secs = std::chrono::duration<double>(total_end - total_start).count();
    double total_tokens = static_cast<double>(tokens_per_step) * train_cfg.max_steps;

    std::cout << "\n============================================" << std::endl;
    std::cout << "Training complete!" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(1)
              << total_secs << " seconds" << std::endl;
    std::cout << "Avg throughput: " << std::setprecision(0)
              << (total_tokens / total_secs) << " tok/s" << std::endl;
    std::cout << "============================================" << std::endl;

    // Final generation
    std::cout << "\n--- Final generation ---" << std::endl;
    model->eval();
    std::string final_text = model->generate("To be or not to be, ", 500, 0.7f);
    for (char c : final_text) {
        if (c >= 32 && c < 127) std::cout << c;
        else if (c == '\n') std::cout << '\n';
        else std::cout << '?';
    }
    std::cout << "\n--- end ---" << std::endl;

    return 0;
}
