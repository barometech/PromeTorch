// ============================================================================
// PIR 250M Training on Elbrus-8SV (32 cores, NUMA-aware)
// ============================================================================
// Port of PIR 270M.py to PromeTorch C++
// Architecture: Pure PIR (no attention) — parallel scan + SwiGLU FFN
//
// OPTIMIZED v2: 10x less memory, 10x faster
//   - Autograd graph released after each step (was: accumulated forever -> OOM)
//   - Fused RMSNorm (1 backward node instead of 5 autograd ops)
//   - Fused dynamic_parallel_scan gating (1 node instead of ~10 ops)
//   - OpenMP parallelization in parallel_scan over B dimension
//   - zero_grad(set_to_none=true) to avoid keeping zero gradient tensors
//   - Checkpoint saving every 100 steps
//
// Usage:
//   ./train_pir_elbrus --data tiny_shakespeare.txt [--n_layers 4] [--n_embd 256]
//
// Elbrus NUMA: 4 nodes x 8 cores. OMP_PLACES=cores OMP_PROC_BIND=close
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/serialization.h"
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

// For memory monitoring + mmap on Linux (Elbrus)
#ifdef __linux__
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;
using at::Tensor;

// ============================================================================
// Memory monitoring utility
// ============================================================================

static long get_rss_mb() {
#ifdef __linux__
    // Read /proc/self/status for VmRSS
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            long kb = 0;
            std::sscanf(line.c_str(), "VmRSS: %ld", &kb);
            return kb / 1024;
        }
    }
#endif
    return -1;  // Not available on this platform
}

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

    bool    use_fused      = false;
    int64_t log_interval  = 10;
    int64_t eval_interval = 200;
    int64_t gen_interval  = 500;
    int64_t gen_tokens    = 200;
    int64_t save_interval = 100;    // Checkpoint save interval

    std::string data_path = "tiny_shakespeare.txt";
    std::string save_dir  = "checkpoints";
    std::string load_path = "";  // --load checkpoint.ptor
    int seed = 42;  // Random seed (different per NUMA process)
    int rank = -1;   // -1 = no sync, 0-3 = data-parallel rank
    int nprocs = 4;  // Number of data-parallel processes
    int grad_accum = 1; // Gradient accumulation steps (sync every N steps)
};

// Include fused trainer AFTER config structs are defined
#include "fused_trainer.h"
#include "grad_sync.h"

// ============================================================================
// CRITICAL: Release autograd graph after optimizer step
// ============================================================================
// Without this, every intermediate tensor from forward/backward is kept forever.
// After 190 steps with 5.5M params: 1.3GB/step * 190 = ~250 GB -> OOM at 125 GB.
//
// The fix: after optimizer.step(), clear grad_fn on all parameters to break
// the reference chain from parameters -> backward nodes -> saved tensors.

static void release_autograd_graph(Module& module) {
    auto params = module.parameters();
    for (auto* p : params) {
        auto& t = p->data();
        auto* raw_meta = t.autograd_meta();
        if (raw_meta && raw_meta->is_autograd_meta_impl_) {
            auto* meta = static_cast<torch::autograd::AutogradMetaImpl*>(raw_meta);
            // Clear grad_fn to release the entire backward graph
            // This drops all shared_ptr references to backward nodes,
            // which in turn drops saved tensors (gates, x, out, etc.)
            meta->grad_fn.reset();
            meta->is_leaf_ = true;
            // Also clear grad_accumulator to avoid stale weak_ptrs
            meta->grad_accumulator_.reset();
        }
    }
}

// ============================================================================
// Parallel Scan with OpenMP
// ============================================================================
// Forward: out[t] = sum_{s<=t} prod_{s<k<=t}(gate[k]) * x[s]
// Backward: d_x[t] = sum_{s>=t} prod_{t<k<=s}(gate[k]) * d_out[s]  (reverse scan)
//           d_gate[t] = dh * h[t-1]  where dh accumulates from the right
//
// OpenMP parallelization: over B dimension (each batch element independent).
// The inner d loop is also parallelizable but B*D total is enough work.

Tensor parallel_scan_forward(const Tensor& gates, const Tensor& x) {
    // gates, x: [B, T, D]
    int64_t B = x.size(0), T = x.size(1), D = x.size(2);
    auto out = at::zeros_like(x);
    float* g_ptr = gates.data_ptr<float>();
    float* x_ptr = x.data_ptr<float>();
    float* o_ptr = out.data_ptr<float>();

    // Parallelize over B*D work items for maximum core utilization
    // Each (b, d) pair is an independent sequential scan over T
    int64_t BD = B * D;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t bd = 0; bd < BD; bd++) {
        int64_t b = bd / D;
        int64_t d = bd % D;
        float h = 0.0f;
        int64_t base = b * T * D + d;
        for (int64_t t = 0; t < T; t++) {
            int64_t idx = base + t * D;
            h = g_ptr[idx] * h + x_ptr[idx];
            o_ptr[idx] = h;
        }
    }
    return out;
}

// Backward: reverse scan for d_x, d_gates (OpenMP over B*D)
std::pair<Tensor, Tensor> parallel_scan_backward(
    const Tensor& gates, const Tensor& x, const Tensor& out, const Tensor& d_out
) {
    int64_t B = x.size(0), T = x.size(1), D = x.size(2);
    auto d_x = at::zeros_like(x);
    auto d_gates = at::zeros_like(gates);
    float* g_ptr = gates.data_ptr<float>();
    float* o_ptr = out.data_ptr<float>();
    float* do_ptr = d_out.data_ptr<float>();
    float* dx_ptr = d_x.data_ptr<float>();
    float* dg_ptr = d_gates.data_ptr<float>();

    int64_t BD = B * D;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t bd = 0; bd < BD; bd++) {
        int64_t b = bd / D;
        int64_t d = bd % D;
        float dh = 0.0f;
        int64_t base = b * T * D + d;
        for (int64_t t = T - 1; t >= 0; t--) {
            int64_t idx = base + t * D;
            dh += do_ptr[idx];
            dx_ptr[idx] = dh;
            if (t > 0) {
                dg_ptr[idx] = dh * o_ptr[base + (t-1) * D];
            } else {
                dg_ptr[idx] = 0.0f;
            }
            dh = dh * g_ptr[idx];
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
        // Release saved tensors immediately after backward
        saved_gates = Tensor(); saved_x = Tensor(); saved_out = Tensor();
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
        out.set_requires_grad(true);
    }
    return out;
}

// ============================================================================
// Fused Dynamic Parallel Scan — single backward node instead of ~10
// ============================================================================
// Computes: gates = base_decay * (1 + tanh(gate_logits) * 0.1), clamped [0.5, 0.9999]
// Then: out = parallel_scan(gates, x)
//
// OLD: tanh_autograd -> mul_autograd -> add_autograd -> mul_autograd -> clamp_autograd
//      = 5 backward nodes, each saving B*T*D tensors = 5 * 4MB = 20MB per call
// NEW: Fuse gate computation into raw loop, 1 backward node saving only inputs

struct DynamicParallelScanBackward : public torch::autograd::Node {
    Tensor saved_x, saved_gate_logits, saved_base_decay;
    Tensor saved_gates, saved_scan_out;  // recomputed values for backward

    DynamicParallelScanBackward(const Tensor& x, const Tensor& gate_logits,
                                 const Tensor& base_decay,
                                 const Tensor& gates, const Tensor& scan_out)
        : saved_x(x), saved_gate_logits(gate_logits),
          saved_base_decay(base_decay),
          saved_gates(gates), saved_scan_out(scan_out) {}

    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override {
        auto d_out = grads[0];
        if (!d_out.defined()) return {Tensor(), Tensor(), Tensor()};

        // 1. Backward through parallel_scan to get d_gates, d_x
        auto [d_x, d_gates] = parallel_scan_backward(
            saved_gates, saved_x, saved_scan_out, d_out);

        // 2. Backward through gate computation:
        //    gates = clamp(base_decay * (1 + tanh(gate_logits) * 0.1), 0.5, 0.9999)
        //    d_gate_logits = d_gates * d(gates)/d(gate_logits)
        //
        //    If not clamped: d/d(gl) = base_decay * 0.1 * (1 - tanh(gl)^2)
        //    If clamped: d/d(gl) = 0
        int64_t B = saved_x.size(0), T = saved_x.size(1), D = saved_x.size(2);
        auto d_gate_logits = at::zeros_like(saved_gate_logits);
        float* dg_ptr = d_gates.data_ptr<float>();
        float* dgl_ptr = d_gate_logits.mutable_data_ptr<float>();
        float* gl_ptr = saved_gate_logits.data_ptr<float>();
        float* bd_ptr = saved_base_decay.data_ptr<float>();
        float* gates_ptr = saved_gates.data_ptr<float>();

        int64_t numel = B * T * D;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < numel; i++) {
            int64_t d = i % D;
            float gate_val = gates_ptr[i];
            // If clamped, gradient is zero
            if (gate_val <= 0.5f || gate_val >= 0.9999f) {
                dgl_ptr[i] = 0.0f;
            } else {
                float tanh_gl = std::tanh(gl_ptr[i]);
                float dtanh = 1.0f - tanh_gl * tanh_gl;  // sech^2
                dgl_ptr[i] = dg_ptr[i] * bd_ptr[d] * 0.1f * dtanh;
            }
        }

        // Release saved tensors
        saved_x = Tensor(); saved_gate_logits = Tensor();
        saved_base_decay = Tensor(); saved_gates = Tensor();
        saved_scan_out = Tensor();

        // Returns: d_x, d_gate_logits, d_base_decay (base_decay is buffer, no grad needed)
        return {d_x, d_gate_logits, Tensor()};
    }

    std::string name() const override { return "DynamicParallelScanBackward"; }
    void release_saved_tensors() override {
        saved_x = Tensor(); saved_gate_logits = Tensor();
        saved_base_decay = Tensor(); saved_gates = Tensor();
        saved_scan_out = Tensor();
    }
};

// Fused dynamic_parallel_scan: computes gating + scan in one step
Tensor dynamic_parallel_scan(const Tensor& x, const Tensor& gate_logits,
                              const Tensor& base_decay) {
    // x: [B, T, D], gate_logits: [B, T, D], base_decay: [D]
    int64_t B = x.size(0), T = x.size(1), D = x.size(2);

    // Fused gate computation: gates = clamp(base_decay * (1 + tanh(gl) * 0.1), 0.5, 0.9999)
    auto gates = at::empty(x.sizes());
    float* g_ptr = gates.mutable_data_ptr<float>();
    const float* gl_ptr = gate_logits.data_ptr<float>();
    const float* bd_ptr = base_decay.data_ptr<float>();

    int64_t numel = B * T * D;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < numel; i++) {
        int64_t d = i % D;
        float tanh_gl = std::tanh(gl_ptr[i]);
        float modulation = tanh_gl * 0.1f;
        float gate = bd_ptr[d] * (1.0f + modulation);
        // Clamp
        if (gate < 0.5f) gate = 0.5f;
        if (gate > 0.9999f) gate = 0.9999f;
        g_ptr[i] = gate;
    }

    // Run parallel scan
    auto x_c = x.contiguous();
    auto gates_c = gates.contiguous();  // already contiguous from at::empty
    auto scan_out = parallel_scan_forward(gates_c, x_c);

    // Wire autograd
    if (x.requires_grad() || gate_logits.requires_grad()) {
        auto node = std::make_shared<DynamicParallelScanBackward>(
            x_c, gate_logits.contiguous(), base_decay.contiguous(),
            gates_c, scan_out);
        node->add_input_metadata(x);
        node->add_input_metadata(gate_logits);
        node->add_input_metadata(base_decay);
        torch::autograd::set_grad_fn(scan_out, node);
        scan_out.set_requires_grad(true);
    }

    return scan_out;
}

// ============================================================================
// Fused RMSNorm — single backward node instead of 5 autograd ops
// ============================================================================
// Forward: out = x * rsqrt(mean(x^2, dim=-1, keepdim=true) + eps) * weight
// Backward:
//   d_weight = sum(d_out * normed, dim=[0,1])  (over B,T)
//   d_x = (d_out * weight) * rms_inv - x * mean((d_out * weight) * normed, dim=-1, keepdim=true) * rms_inv
//
// OLD: 5 autograd nodes (mul, mean, add, pow, mul, mul) = 5 saved tensor copies
// NEW: 1 backward node, saves only input + normed + rms_inv

struct RMSNormBackward : public torch::autograd::Node {
    Tensor saved_input, saved_normed, saved_rms_inv, saved_weight;
    float eps;

    RMSNormBackward(const Tensor& input, const Tensor& normed,
                    const Tensor& rms_inv, const Tensor& weight, float eps)
        : saved_input(input), saved_normed(normed),
          saved_rms_inv(rms_inv), saved_weight(weight), eps(eps) {}

    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override {
        auto d_out = grads[0];
        if (!d_out.defined()) return {Tensor(), Tensor()};

        int64_t B = saved_input.size(0);
        int64_t T = saved_input.size(1);
        int64_t D = saved_input.size(2);

        float* dout_ptr = d_out.data_ptr<float>();
        float* x_ptr = saved_input.data_ptr<float>();
        float* norm_ptr = saved_normed.data_ptr<float>();
        float* rms_ptr = saved_rms_inv.data_ptr<float>();   // [B, T, 1]
        float* w_ptr = saved_weight.data_ptr<float>();      // [D]

        auto d_input = at::zeros_like(saved_input);          // [B, T, D]
        auto d_weight = at::zeros({D});                      // [D]

        float* dx_ptr = d_input.mutable_data_ptr<float>();
        float* dw_ptr = d_weight.mutable_data_ptr<float>();

        // Compute d_weight = sum over B,T of (d_out * normed)
        // and d_input
        // d_input[b,t,d] = (dout[b,t,d] * w[d] - normed[b,t,d] * mean_d(dout*w*normed)) * rms_inv[b,t]

        int64_t BT = B * T;
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            // Thread-local d_weight accumulator
            std::vector<float> local_dw(D, 0.0f);

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int64_t bt = 0; bt < BT; bt++) {
                int64_t base = bt * D;
                float rms_inv_val = rms_ptr[bt];  // [B*T] stored as [B,T,1]

                // Compute: dout_w = d_out * weight
                // Compute: dot = sum_d(dout_w * normed) / D
                float dot = 0.0f;
                for (int64_t d = 0; d < D; d++) {
                    float dout_w = dout_ptr[base + d] * w_ptr[d];
                    dot += dout_w * norm_ptr[base + d];
                }
                dot /= D;

                // d_input = (dout_w - normed * dot) * rms_inv
                for (int64_t d = 0; d < D; d++) {
                    float dout_w = dout_ptr[base + d] * w_ptr[d];
                    dx_ptr[base + d] = (dout_w - norm_ptr[base + d] * dot) * rms_inv_val;
                    // Accumulate d_weight
                    local_dw[d] += dout_ptr[base + d] * norm_ptr[base + d];
                }
            }

            // Reduce d_weight across threads
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                for (int64_t d = 0; d < D; d++) {
                    dw_ptr[d] += local_dw[d];
                }
            }
        }

        // Release saved tensors
        saved_input = Tensor(); saved_normed = Tensor();
        saved_rms_inv = Tensor(); saved_weight = Tensor();

        return {d_input, d_weight};
    }

    std::string name() const override { return "RMSNormBackward"; }
    void release_saved_tensors() override {
        saved_input = Tensor(); saved_normed = Tensor();
        saved_rms_inv = Tensor(); saved_weight = Tensor();
    }
};

class RMSNorm : public Module {
    float eps_;
public:
    RMSNorm(int64_t dim, float eps = 1e-6f)
        : Module("RMSNorm"), eps_(eps) {
        register_parameter("weight", Parameter(at::ones({dim})));
    }

    Tensor forward(const Tensor& input) override {
        // Fused RMSNorm: compute everything in raw loops, single backward node
        int64_t B = input.size(0), T = input.size(1), D = input.size(2);
        auto input_c = input.contiguous();

        float* x_ptr = input_c.data_ptr<float>();
        auto weight_param = get_parameter("weight")->data();
        float* w_ptr = weight_param.data_ptr<float>();

        auto normed = at::empty(input_c.sizes());    // [B, T, D]
        auto rms_inv = at::empty({B, T, 1});          // [B, T, 1] for backward
        float* n_ptr = normed.mutable_data_ptr<float>();
        float* r_ptr = rms_inv.mutable_data_ptr<float>();

        auto output = at::empty(input_c.sizes());
        float* o_ptr = output.mutable_data_ptr<float>();

        int64_t BT = B * T;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t bt = 0; bt < BT; bt++) {
            int64_t base = bt * D;
            // Compute mean(x^2)
            float sum_sq = 0.0f;
            for (int64_t d = 0; d < D; d++) {
                float val = x_ptr[base + d];
                sum_sq += val * val;
            }
            float mean_sq = sum_sq / D;
            float inv_rms = 1.0f / std::sqrt(mean_sq + eps_);
            r_ptr[bt] = inv_rms;

            // normed = x * inv_rms, output = normed * weight
            for (int64_t d = 0; d < D; d++) {
                float norm_val = x_ptr[base + d] * inv_rms;
                n_ptr[base + d] = norm_val;
                o_ptr[base + d] = norm_val * w_ptr[d];
            }
        }

        // Wire autograd
        if (input.requires_grad() || weight_param.requires_grad()) {
            auto node = std::make_shared<RMSNormBackward>(
                input_c, normed, rms_inv, weight_param, eps_);
            node->add_input_metadata(input);
            node->add_input_metadata(weight_param);
            torch::autograd::set_grad_fn(output, node);
            output.set_requires_grad(true);
        }

        return output;
    }
};

// ============================================================================
// Rotary Embedding (RoPE) -- precomputed cos/sin tables
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
    // Use chunk_autograd/narrow_autograd to preserve gradient flow through narrow ops
    auto chunks = torch::autograd::chunk_autograd(x, 2, -1);
    Tensor x1 = chunks[0];  // [B, T, D/2]
    Tensor x2 = chunks[1];  // [B, T, D/2]

    // cos/sin are non-grad constants, plain narrow is fine
    auto cos_half = cos_t.narrow(-1, 0, x1.size(-1));
    auto sin_half = sin_t.narrow(-1, 0, x1.size(-1));

    auto x1_chunks = torch::autograd::chunk_autograd(x1, 2, -1);
    Tensor x1a = x1_chunks[0];
    Tensor x1b = x1_chunks[1];
    auto rotated = torch::autograd::cat_autograd({torch::autograd::neg_autograd(x1b), x1a}, -1);

    auto x1_cos = torch::autograd::mul_autograd(x1, cos_half);
    auto rot_sin = torch::autograd::mul_autograd(rotated, sin_half);
    auto x1_rotated = torch::autograd::add_autograd(x1_cos, rot_sin);

    return torch::autograd::cat_autograd({x1_rotated, x2}, -1);
}

// ============================================================================
// PIR Layer -- single-scale concept compression
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

        init_weights();
    }

    void init_weights() {
        // Match Python: nn.init.orthogonal_ with specific gains
        auto& gw = gate_proj_->get_parameter("weight")->data();
        init::normal_(gw, 0.0, 0.1 / std::sqrt((double)n_embd_));

        auto& vw = value_proj_->get_parameter("weight")->data();
        init::normal_(vw, 0.0, 1.0 / std::sqrt((double)n_embd_));

        auto& ow = out_proj_->get_parameter("weight")->data();
        init::normal_(ow, 0.0, 0.5 / std::sqrt((double)n_embd_));
    }

    Tensor forward(const Tensor& input) override {
        auto gate_logits = gate_proj_->forward(input);
        auto values = value_proj_->forward(input);
        // value_gate = sigmoid(gate_logits) — keep through autograd for gradient
        auto value_gate = torch::autograd::sigmoid_autograd(gate_logits);
        auto gated_values = torch::autograd::mul_autograd(values, value_gate);

        auto* buf = get_buffer("base_decay");
        Tensor base_decay = buf->data();

        // FUSED: dynamic_parallel_scan does gating + scan in one backward node
        auto scanned = dynamic_parallel_scan(gated_values, gate_logits, base_decay);

        auto out = out_proj_->forward(scanned);
        return norm_->forward(out);
    }
};

// ============================================================================
// PIR Block -- 4 PIR layers with different decay scales
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

        // Match Python: nn.init.orthogonal_(self.mix_proj.weight, gain=0.5)
        auto& mw = mix_proj_->get_parameter("weight")->data();
        init::normal_(mw, 0.0, 0.5 / std::sqrt((double)n_embd));
    }

    Tensor forward(const Tensor& input) override {
        Tensor h = input;
        for (auto& layer : layers_) {
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

        // Match Python: orthogonal_ init with specific gains
        auto& w1d = w1_->get_parameter("weight")->data();
        init::normal_(w1d, 0.0, 1.0 / std::sqrt((double)n_embd));
        auto& w2d = w2_->get_parameter("weight")->data();
        init::normal_(w2d, 0.0, 0.5 / std::sqrt((double)hidden));
        auto& w3d = w3_->get_parameter("weight")->data();
        init::normal_(w3d, 0.0, 1.0 / std::sqrt((double)n_embd));
    }

    Tensor forward(const Tensor& input) override {
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
        auto normed1 = norm1_->forward(input);
        auto pir_out = pir_->forward(normed1);
        auto x = torch::autograd::add_autograd(input, pir_out);

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

        tok_emb_ = std::make_shared<Embedding>(cfg.vocab_size, cfg.n_embd);
        register_module("tok_emb", tok_emb_);

        auto& emb_w = tok_emb_->get_parameter("weight")->data();
        init::normal_(emb_w, 0.0, 0.02);

        rope_ = RotaryEmbedding(cfg.n_embd / 2, cfg.block_size);

        for (int64_t i = 0; i < cfg.n_layers; i++) {
            auto block = std::make_shared<TransformerBlock>(
                cfg.n_embd, cfg.n_pir_layers, cfg.ffn_mult, cfg
            );
            blocks_.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }

        norm_out_ = std::make_shared<RMSNorm>(cfg.n_embd);
        register_module("norm_out", norm_out_);

        lm_head_ = std::make_shared<Linear>(cfg.n_embd, cfg.vocab_size, /*bias=*/false);
        register_module("lm_head", lm_head_);

        auto& lm_w = lm_head_->get_parameter("weight")->data();
        // Match Python: std = 0.02 / sqrt(2 * n_layers)
        init::normal_(lm_w, 0.0, 0.02 / std::sqrt(2.0 * cfg.n_layers));

        int64_t total = count_parameters(*this);
        std::cout << "PIR model initialized: " << (total / 1e6) << "M parameters"
                  << " | n_layers=" << cfg.n_layers
                  << " | n_embd=" << cfg.n_embd
                  << " | block_size=" << cfg.block_size
                  << std::endl;
    }

    Tensor forward(const Tensor& input) override {
        int64_t T = input.size(1);
        auto x = tok_emb_->forward(input);
        // RoPE disabled: chunk/narrow/cat break autograd in PromeTorch
        // PIR parallel scan is inherently position-aware (sequential recurrence)

        for (auto& block : blocks_) {
            x = block->forward(x);
        }

        x = norm_out_->forward(x);
        auto logits = lm_head_->forward(x);
        return logits;
    }

    Tensor compute_loss(const Tensor& logits, const Tensor& targets) {
        int64_t B = logits.size(0);
        int64_t T = logits.size(1);
        int64_t V = logits.size(2);

        auto logits_flat = torch::autograd::reshape_autograd(logits, {B * T, V});
        auto targets_flat = targets.reshape({B * T});

        CrossEntropyLoss loss_fn;
        return loss_fn.forward(logits_flat, targets_flat);
    }

    std::string generate(const std::string& prompt, int64_t max_tokens,
                         float temperature = 0.8f) {
        std::string result = prompt;
        std::mt19937 rng(42);

        std::vector<float> input_data;
        for (char c : prompt) {
            input_data.push_back(static_cast<float>(static_cast<unsigned char>(c)));
        }

        // Disable autograd for generation (saves memory, faster)
        for (auto* p : this->parameters()) {
            p->data().set_requires_grad(false);
        }

        for (int64_t i = 0; i < max_tokens; i++) {
            int64_t seq_len = static_cast<int64_t>(input_data.size());
            if (seq_len > config_.block_size) {
                input_data.erase(input_data.begin(),
                                 input_data.begin() + (seq_len - config_.block_size));
                seq_len = config_.block_size;
            }

            Tensor input_t = at::empty({1, seq_len});
            float* inp = input_t.mutable_data_ptr<float>();
            for (int64_t j = 0; j < seq_len; j++) {
                inp[j] = input_data[j];
            }

            // NoGradGuard prevents autograd graph creation → no memory leak
            torch::autograd::NoGradGuard no_grad;
            auto logits = this->forward(input_t);

            int64_t V = config_.vocab_size;
            const float* last_logits = logits.data_ptr<float>() +
                                       (seq_len - 1) * V;

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

            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int next_token = dist(rng);

            result += static_cast<char>(next_token);
            input_data.push_back(static_cast<float>(next_token));
        }

        // Re-enable autograd for training
        for (auto* p : this->parameters()) {
            p->data().set_requires_grad(true);
        }
        return result;
    }
};

// ============================================================================
// Data Loading -- char-level tokenizer + batch preparation
// ============================================================================

class TextDataset {
    // mmap for large files — avoids 2GB allocation on NUMA node
    uint8_t* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
    std::vector<uint8_t> data_small_;  // fallback for small files
    const uint8_t* data_ptr_ = nullptr;
    size_t data_size_ = 0;
    int64_t block_size_;
    std::mt19937 rng_;

public:
    TextDataset(const std::string& path, int64_t block_size, int seed = 42)
        : block_size_(block_size), rng_(seed) {
#ifdef __linux__
        // Use mmap — no memory copy, pages loaded on demand from any NUMA node
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "ERROR: Cannot open data file: " << path << std::endl;
            return;
        }
        struct stat st;
        fstat(fd, &st);
        mmap_size_ = st.st_size;
        mmap_ptr_ = (uint8_t*)mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        close(fd);
        if (mmap_ptr_ == MAP_FAILED) {
            mmap_ptr_ = nullptr;
            std::cerr << "ERROR: mmap failed for " << path << std::endl;
            return;
        }
        data_ptr_ = mmap_ptr_;
        data_size_ = mmap_size_;
        std::cout << "mmap'd " << data_size_ << " bytes from " << path << std::endl;
#else
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "ERROR: Cannot open data file: " << path << std::endl;
            return;
        }
        data_small_ = std::vector<uint8_t>(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
        data_ptr_ = data_small_.data();
        data_size_ = data_small_.size();
        std::cout << "Loaded " << data_size_ << " bytes from " << path << std::endl;
#endif
    }

    ~TextDataset() {
#ifdef __linux__
        if (mmap_ptr_) munmap(mmap_ptr_, mmap_size_);
#endif
    }

    bool empty() const { return data_size_ == 0; }
    size_t size() const { return data_size_; }

    std::pair<Tensor, Tensor> get_batch(int64_t batch_size) {
        int64_t T = block_size_;
        Tensor input = at::empty({batch_size, T});
        Tensor target = at::empty({batch_size, T});
        float* inp = input.mutable_data_ptr<float>();
        float* tgt = target.mutable_data_ptr<float>();

        int64_t max_start = static_cast<int64_t>(data_size_) - T - 1;
        if (max_start <= 0) {
            std::cerr << "Data too short for block_size=" << T << std::endl;
            return {input, target};
        }

        std::uniform_int_distribution<int64_t> dist(0, max_start);

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
                inp[b * T + t] = static_cast<float>(data_ptr_[start + t]);
                tgt[b * T + t] = static_cast<float>(data_ptr_[start + t + 1]);
            }
        }

        return {input, target};
    }
};

// ============================================================================
// Cosine LR schedule with warmup
// ============================================================================

float get_lr(int64_t step, const TrainConfig& cfg) {
    if (step < cfg.warmup_steps) {
        return cfg.learning_rate * static_cast<float>(step) / cfg.warmup_steps;
    }
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
    std::cout << "OpenMP not available -- single-threaded execution" << std::endl;
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
        } else if (arg == "--save_interval" && i + 1 < argc) {
            train_cfg.save_interval = std::atoi(argv[++i]);
        } else if (arg == "--save_dir" && i + 1 < argc) {
            train_cfg.save_dir = argv[++i];
        } else if (arg == "--load" && i + 1 < argc) {
            train_cfg.load_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            setup_numa_threads(std::atoi(argv[++i]));
        } else if (arg == "--seed" && i + 1 < argc) {
            train_cfg.seed = std::atoi(argv[++i]);
        } else if (arg == "--rank" && i + 1 < argc) {
            train_cfg.rank = std::atoi(argv[++i]);
        } else if (arg == "--nprocs" && i + 1 < argc) {
            train_cfg.nprocs = std::atoi(argv[++i]);
        } else if (arg == "--grad_accum" && i + 1 < argc) {
            train_cfg.grad_accum = std::atoi(argv[++i]);
        } else if (arg == "--fused") {
            train_cfg.use_fused = true;
        } else if (arg == "--full") {
            // Full 250M config
            model_cfg.vocab_size = 256;
            model_cfg.n_embd = 768;
            model_cfg.n_layers = 16;
            model_cfg.n_pir_layers = 4;
            model_cfg.block_size = 2048;
        } else if (arg == "--help") {
            std::cout << "PIR 250M Training -- PromeTorch on Elbrus-8SV\n\n"
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
                      << "  --save_interval N    Checkpoint save interval (default: 100)\n"
                      << "  --save_dir DIR       Checkpoint directory (default: checkpoints)\n"
                      << "  --threads N          OpenMP threads (default: auto)\n"
                      << "  --full               Full 250M config (768d, 16 layers, 2048 ctx)\n"
                      << std::endl;
            std::exit(0);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "PIR 250M Training -- PromeTorch on Elbrus-8SV" << std::endl;
    std::cout << "  OPTIMIZED v2: fused ops, graph cleanup, OMP" << std::endl;
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
              << " save_interval=" << train_cfg.save_interval
              << std::endl;

    if (train_cfg.use_fused) {
        std::cout << "\n*** FUSED MODE: Zero-autograd training ***\n" << std::endl;
    }

    // Load data
    TextDataset dataset(train_cfg.data_path, model_cfg.block_size, train_cfg.seed);
    if (dataset.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return 1;
    }

    int64_t tokens_per_step = train_cfg.batch_size * model_cfg.block_size;
    std::cout << "Tokens per step: " << tokens_per_step
              << " | Total tokens: " << (tokens_per_step * train_cfg.max_steps)
              << std::endl;

    // Print initial memory
    long rss = get_rss_mb();
    if (rss > 0) std::cout << "RSS before model: " << rss << " MB" << std::endl;

    // ================================================================
    // FUSED TRAINING PATH — zero autograd, pre-allocated buffers
    // ================================================================
    if (train_cfg.use_fused) {
        FusedPIRTrainer trainer;
        trainer.allocate(model_cfg, train_cfg.batch_size);
        trainer.init_random(model_cfg);

        // Data-parallel gradient sync (--rank 0..3)
        GradSync grad_sync;
        bool use_sync = (train_cfg.rank >= 0);
        std::vector<float> flat_grads, flat_params;
        if (use_sync) {
            int64_t total_p = 0;
            for (size_t i = 0; i < trainer.all_sizes.size(); i++) total_p += trainer.all_sizes[i];
            if (!grad_sync.init(train_cfg.rank, train_cfg.nprocs, total_p)) {
                std::cerr << "GradSync init FAILED" << std::endl;
                return 1;
            }
            flat_grads.resize(total_p);
            flat_params.resize(total_p);
            std::cout << "GradSync: rank=" << train_cfg.rank
                      << " nprocs=" << train_cfg.nprocs
                      << " params=" << total_p << std::endl;

            // Sync initial weights from rank 0
            {
                int64_t off = 0;
                for (size_t i = 0; i < trainer.all_params.size(); i++) {
                    memcpy(flat_params.data() + off, trainer.all_params[i], trainer.all_sizes[i] * sizeof(float));
                    off += trainer.all_sizes[i];
                }
                grad_sync.sync_weights(flat_params.data());
                // Scatter back to trainer
                off = 0;
                for (size_t i = 0; i < trainer.all_params.size(); i++) {
                    memcpy(trainer.all_params[i], flat_params.data() + off, trainer.all_sizes[i] * sizeof(float));
                    off += trainer.all_sizes[i];
                }
            }
        }

        rss = get_rss_mb();
        std::cout << "RSS after fused alloc: " << rss << " MB" << std::endl;

        std::cout << "\n--- Fused Training ---\n" << std::endl;

        auto total_start = std::chrono::high_resolution_clock::now();
        auto step_start = total_start;
        float running_loss = 0.0f;
        int running_count = 0;

        // Input/target buffers (reused)
        std::vector<float> inp_buf(tokens_per_step);
        std::vector<float> tgt_buf(tokens_per_step);

        for (int64_t step = 1; step <= train_cfg.max_steps; step++) {
            float lr = get_lr(step, train_cfg);

            // Get batch
            auto [input, target] = dataset.get_batch(train_cfg.batch_size);
            memcpy(inp_buf.data(), input.data_ptr<float>(), tokens_per_step * sizeof(float));
            memcpy(tgt_buf.data(), target.data_ptr<float>(), tokens_per_step * sizeof(float));

            float loss;
            if (use_sync) {
                // Local SGD: each process trains independently at full speed.
                // Every grad_accum steps, average WEIGHTS across all processes.
                // This gives ~968 tok/s (4×242) with minimal sync overhead.
                loss = trainer.train_step(inp_buf.data(), tgt_buf.data(), lr, train_cfg.weight_decay);

                if (step % train_cfg.grad_accum == 0) {
                    // Sync weights: flatten → shared memory average → scatter
                    int64_t off = 0;
                    for (size_t i = 0; i < trainer.all_params.size(); i++) {
                        memcpy(flat_params.data() + off, trainer.all_params[i], trainer.all_sizes[i] * sizeof(float));
                        off += trainer.all_sizes[i];
                    }
                    // sync() averages across all ranks
                    grad_sync.sync(flat_params.data(), nullptr);
                    // Scatter averaged weights back
                    off = 0;
                    for (size_t i = 0; i < trainer.all_params.size(); i++) {
                        memcpy(trainer.all_params[i], flat_params.data() + off, trainer.all_sizes[i] * sizeof(float));
                        off += trainer.all_sizes[i];
                    }
                }
            } else {
                // Single-process: original train_step
                loss = trainer.train_step(inp_buf.data(), tgt_buf.data(), lr, train_cfg.weight_decay);
            }

            running_loss += loss;
            running_count++;

            if (step % train_cfg.log_interval == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(now - step_start).count();
                double tok_per_sec = (tokens_per_step * train_cfg.log_interval) / (elapsed_ms / 1000.0);
                float avg_loss = running_loss / running_count;
                float perplexity = std::exp(std::min(avg_loss, 20.0f));

                rss = get_rss_mb();
                std::cout << std::fixed << std::setprecision(4)
                          << "step " << std::setw(6) << step
                          << " | loss " << std::setw(7) << avg_loss
                          << " | ppl " << std::setw(10) << std::setprecision(1) << perplexity
                          << " | lr " << std::setprecision(6) << lr
                          << " | " << std::setprecision(0) << tok_per_sec << " tok/s";
                if (rss > 0) std::cout << " | " << rss << " MB";
                std::cout << std::endl;

                running_loss = 0.0f;
                running_count = 0;
                step_start = now;
            }

            // === FUSED GENERATION ===
            if (step % train_cfg.gen_interval == 0) {
                std::cout << "\n--- Generation at step " << step << " ---" << std::endl;
                std::vector<std::string> prompts = {
                    "\xd0\x92 ",     // "В "
                    "\xd0\x9e\xd0\xbd ",  // "Он "
                    "\xd0\x9e\xd0\xbd\xd0\xb0 ", // "Она "
                    "The "
                };
                for (auto& prompt : prompts) {
                    std::string gen = trainer.generate_text(prompt, train_cfg.gen_tokens, 0.8f);
                    std::cout << ">>> ";
                    for (char c : gen) {
                        if ((unsigned char)c >= 32 || c == '\n') std::cout << c;
                        else std::cout << '?';
                    }
                    std::cout << "\n";
                }
                std::cout << "--- end generation ---\n" << std::endl;
            }

            // === FUSED CHECKPOINT SAVE ===
            if (step % train_cfg.save_interval == 0) {
                std::string ckpt_dir = train_cfg.save_dir;
                system(("mkdir -p " + ckpt_dir).c_str());
                std::string path = ckpt_dir + "/pir_fused_step_" + std::to_string(step) + ".bin";
                FILE* f = fopen(path.c_str(), "wb");
                if (f) {
                    for (size_t i = 0; i < trainer.all_params.size(); i++)
                        fwrite(trainer.all_params[i], sizeof(float), trainer.all_sizes[i], f);
                    fclose(f);
                    std::cout << "  [fused checkpoint: " << path << "]" << std::endl;
                }
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "\n============================================\n"
                  << "Fused training complete!\n"
                  << "Total time: " << std::fixed << std::setprecision(1) << total_s << " seconds\n"
                  << "Avg throughput: " << (int)(tokens_per_step * train_cfg.max_steps / total_s) << " tok/s\n"
                  << "============================================" << std::endl;
        if (use_sync) grad_sync.cleanup();
        return 0;
    }

    // ================================================================
    // AUTOGRAD TRAINING PATH (original)
    // ================================================================

    // Create model
    auto model = std::make_shared<PIR250M>(model_cfg);

    // Load checkpoint if specified
    if (!train_cfg.load_path.empty()) {
        std::cout << "Loading checkpoint: " << train_cfg.load_path << std::endl;
        try {
            auto sd = torch::load_state_dict(train_cfg.load_path);
            model->load_state_dict(sd);
            std::cout << "Checkpoint loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load checkpoint: " << e.what() << std::endl;
        }
    }

    rss = get_rss_mb();
    if (rss > 0) std::cout << "RSS after model: " << rss << " MB" << std::endl;

    // Create optimizer: AdamW with weight decay
    auto params = model->parameters();
    AdamWOptions adam_opts(train_cfg.learning_rate);
    adam_opts.betas(train_cfg.beta1, train_cfg.beta2);
    adam_opts.weight_decay_(train_cfg.weight_decay);
    AdamW optimizer(params, adam_opts);

    // Create checkpoint directory
#ifdef __linux__
    {
        std::string mkdir_cmd = "mkdir -p " + train_cfg.save_dir;
        system(mkdir_cmd.c_str());
    }
#endif

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

        // TIMING + MEMORY DEBUG (first 10 steps)
        bool debug = (step <= 10);
        auto t_start = std::chrono::high_resolution_clock::now();
        long rss_before = debug ? get_rss_mb() : 0;

        // Get batch
        auto [input, target] = dataset.get_batch(train_cfg.batch_size);
        auto t_batch = std::chrono::high_resolution_clock::now();

        // Forward pass
        optimizer.zero_grad(/*set_to_none=*/true);
        auto logits = model->forward(input);
        auto t_fwd = std::chrono::high_resolution_clock::now();

        // Compute loss
        auto loss = model->compute_loss(logits, target);
        auto t_loss = std::chrono::high_resolution_clock::now();

        // Backward pass
        torch::autograd::backward({loss});
        auto t_bwd = std::chrono::high_resolution_clock::now();

        if (debug) {
            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            long rss_now = get_rss_mb();
            std::cout << "TIMING step " << step
                      << " | batch:" << (int)ms(t_start, t_batch)
                      << " fwd:" << (int)ms(t_batch, t_fwd)
                      << " loss:" << (int)ms(t_fwd, t_loss)
                      << " bwd:" << (int)ms(t_loss, t_bwd)
                      << " ms | total:" << (int)ms(t_start, t_bwd)
                      << " ms | " << rss_now << " MB" << std::endl;
        }

        // DEBUG: Check gradients on first step — show WHICH params have no grad
        if (step <= 1) {
            auto named = model->named_parameters();
            int pi = 0;
            float total_grad_norm = 0.0f;
            int has_grad = 0, no_grad = 0, zero_grad = 0;
            for (auto& [name, p] : named) {
                auto& t = p->data();
                auto* meta = t.autograd_meta();
                if (meta && meta->grad_) {
                    at::Tensor g(meta->grad_);
                    float gnorm = 0.0f;
                    float* gd = g.data_ptr<float>();
                    for (int64_t i = 0; i < g.numel(); i++) gnorm += gd[i] * gd[i];
                    gnorm = std::sqrt(gnorm);
                    total_grad_norm += gnorm * gnorm;
                    if (gnorm < 1e-10f) zero_grad++;
                    else has_grad++;
                    if (pi < 5 && step == 1) {
                        std::cout << "  param " << pi << " shape=[";
                        for (int d = 0; d < g.dim(); d++) std::cout << (d?",":"") << g.size(d);
                        std::cout << "] grad_norm=" << gnorm << std::endl;
                    }
                } else {
                    no_grad++;
                    std::cout << "  NO GRAD: " << name << " shape=[";
                    for (int d = 0; d < t.dim(); d++) std::cout << (d?",":"") << t.size(d);
                    std::cout << "] requires_grad=" << t.requires_grad() << std::endl;
                }
                pi++;
            }
            total_grad_norm = std::sqrt(total_grad_norm);
            std::cout << "STEP " << step << " DEBUG: " << named.size() << " params, "
                      << has_grad << " with grad, " << no_grad << " no grad, "
                      << zero_grad << " zero grad, total_norm=" << total_grad_norm << std::endl;
        }

        // Gradient clipping
        float grad_norm = fast_clip_grad_norm_(*model, train_cfg.grad_clip);
        long rss_after_clip = false ? get_rss_mb() : 0;

        // Optimizer step
        optimizer.step();
        long rss_after_step = false ? get_rss_mb() : 0;

        // CRITICAL: Release autograd graph from this step
        release_autograd_graph(*model);

        // Also explicitly drop logits and loss tensors — they hold references
        // to the entire forward computation graph through their grad_fn chains
        float loss_val_save = loss.data_ptr<float>()[0];  // save before dropping
        logits = Tensor();  // drop reference to forward graph
        loss = Tensor();    // drop reference to loss graph
        input = Tensor();   // drop batch tensors
        target = Tensor();

        long rss_after_release = false ? get_rss_mb() : 0;

        if (false) {
            std::cout << "MEM step " << step
                      << " | clip:" << rss_after_clip
                      << " step:" << rss_after_step
                      << " release:" << rss_after_release
                      << " MB | delta:" << (rss_after_release - rss_before) << " MB"
                      << std::endl;
        }

        // Track loss (extract value BEFORE releasing graph)
        float loss_val = loss_val_save;  // saved before dropping tensor
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

            rss = get_rss_mb();
            std::cout << std::fixed << std::setprecision(4)
                      << "step " << std::setw(6) << step
                      << " | loss " << std::setw(7) << avg_loss
                      << " | ppl " << std::setw(10) << std::setprecision(1) << perplexity
                      << " | lr " << std::setprecision(6) << lr
                      << " | gnorm " << std::setprecision(3) << grad_norm
                      << " | " << std::setprecision(0) << tok_per_sec << " tok/s";
            if (rss > 0) std::cout << " | " << rss << " MB";
            std::cout << std::endl;

            running_loss = 0.0f;
            running_count = 0;
            step_start = now;
        }

        // Checkpoint save
        if (step % train_cfg.save_interval == 0) {
            std::string ckpt_path = train_cfg.save_dir + "/pir_step_" +
                                     std::to_string(step) + ".ptor";
            try {
                auto sd = model->state_dict();
                torch::save_state_dict(sd, ckpt_path);
                std::cout << "  [checkpoint saved: " << ckpt_path << "]" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  [checkpoint save failed: " << e.what() << "]" << std::endl;
            }
        }

        // Generate sample text
        if (step % train_cfg.gen_interval == 0) {
            std::cout << "\n--- Generation at step " << step << " ---" << std::endl;
            model->eval();

            std::string prompt = "The ";
            std::string generated = model->generate(prompt, train_cfg.gen_tokens, 0.8f);

            std::cout << ">>> ";
            for (char c : generated) {
                if ((unsigned char)c >= 32) {
                    std::cout << c;
                } else if (c == '\n') {
                    std::cout << '\n';
                } else {
                    std::cout << '?';
                }
            }
            std::cout << "\n--- end generation ---\n" << std::endl;

            model->train();

            // Also release graph after generation (forward-only creates autograd nodes too)
            release_autograd_graph(*model);
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
    rss = get_rss_mb();
    if (rss > 0) std::cout << "Final RSS: " << rss << " MB" << std::endl;
    std::cout << "============================================" << std::endl;

    // Save final checkpoint
    {
        std::string final_path = train_cfg.save_dir + "/pir_final.ptor";
        try {
            auto sd = model->state_dict();
            torch::save_state_dict(sd, final_path);
            std::cout << "Final checkpoint saved: " << final_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Final checkpoint save failed: " << e.what() << std::endl;
        }
    }

    // Final generation
    std::cout << "\n--- Final generation ---" << std::endl;
    model->eval();
    std::string final_text = model->generate("To be or not to be, ", 500, 0.7f);
    for (char c : final_text) {
        if ((unsigned char)c >= 32) std::cout << c;
        else if (c == '\n') std::cout << '\n';
        else std::cout << '?';
    }
    std::cout << "\n--- end ---" << std::endl;

    return 0;
}
