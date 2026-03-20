#pragma once
// ============================================================================
// PromePile Inference — Zero-overhead LLM decode step compiler
// ============================================================================
// Specialized for autoregressive LLM inference (single-token decode).
// Pre-compiles the entire decode step into a flat sequence of hot:: calls.
//
// Key properties:
//   - ZERO Tensor objects during execution (pure float* -> float*)
//   - ZERO memory allocation (all buffers pre-allocated at compile time)
//   - ZERO dispatch overhead (no dtype/device/contiguity checks)
//   - ZERO autograd (inference only)
//   - Pre-computed weight pointers (no map lookups)
//   - Per-layer buffer reuse (ping-pong pattern)
//
// Architecture support:
//   - Transformer decoder (GPT/Llama/Qwen style)
//   - RMSNorm + RoPE + Grouped Query Attention + SwiGLU FFN
//
// Usage:
//   CompiledDecodeStep decoder;
//   decoder.compile(model, max_seq_len, hidden_dim, n_heads, n_layers, vocab_size);
//   // In decode loop:
//   decoder.decode_token(embedding, kv_cache, position, logits);
// ============================================================================

#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "torch/compile/promepile.h"

#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace torch {
namespace compile {

// ============================================================================
// Pre-compiled layer weights (pointers only, no copies)
// ============================================================================

struct TransformerLayerWeights {
    // Attention
    const float* q_weight = nullptr;  // [n_heads * head_dim, hidden]
    const float* k_weight = nullptr;  // [n_kv_heads * head_dim, hidden]
    const float* v_weight = nullptr;  // [n_kv_heads * head_dim, hidden]
    const float* o_weight = nullptr;  // [hidden, n_heads * head_dim]

    // FFN (SwiGLU: gate_proj, up_proj, down_proj)
    const float* gate_weight = nullptr;  // [ffn_dim, hidden]
    const float* up_weight = nullptr;    // [ffn_dim, hidden]
    const float* down_weight = nullptr;  // [hidden, ffn_dim]

    // Normalization
    const float* attn_norm = nullptr;    // [hidden] RMSNorm weight
    const float* ffn_norm = nullptr;     // [hidden] RMSNorm weight

    // Optional biases (null if not used)
    const float* q_bias = nullptr;
    const float* k_bias = nullptr;
    const float* v_bias = nullptr;
    const float* o_bias = nullptr;
};

// ============================================================================
// CompiledDecodeStep — pre-compiled single-token decode
// ============================================================================

struct CompiledDecodeStep {
    // Model architecture
    int64_t hidden_dim = 0;
    int64_t n_heads = 0;
    int64_t n_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t ffn_dim = 0;
    int64_t n_layers = 0;
    int64_t vocab_size = 0;
    int64_t max_seq_len = 0;
    float rms_eps = 1e-6f;

    // Per-layer weight pointers
    std::vector<TransformerLayerWeights> layer_weights;

    // Final norm + output projection
    const float* final_norm = nullptr;   // [hidden]
    const float* lm_head = nullptr;      // [vocab_size, hidden]

    // Pre-allocated scratch buffers (owned)
    float* buf_norm = nullptr;      // [hidden]
    float* buf_q = nullptr;         // [n_heads * head_dim]
    float* buf_k = nullptr;         // [n_kv_heads * head_dim]
    float* buf_v = nullptr;         // [n_kv_heads * head_dim]
    float* buf_attn_out = nullptr;  // [n_heads * head_dim]
    float* buf_scores = nullptr;    // [n_heads * max_seq_len]
    float* buf_gate = nullptr;      // [ffn_dim]
    float* buf_up = nullptr;        // [ffn_dim]
    float* buf_ffn_out = nullptr;   // [hidden]
    float* buf_residual = nullptr;  // [hidden]
    float* buf_logits = nullptr;    // [vocab_size]

    bool compiled = false;

    // ========================================================================
    // Setup: register all weight pointers
    // ========================================================================

    void setup(int64_t hidden, int64_t heads, int64_t kv_heads,
               int64_t head_d, int64_t ffn, int64_t layers,
               int64_t vocab, int64_t max_seq, float eps = 1e-6f) {
        hidden_dim = hidden;
        n_heads = heads;
        n_kv_heads = kv_heads;
        head_dim = head_d;
        ffn_dim = ffn;
        n_layers = layers;
        vocab_size = vocab;
        max_seq_len = max_seq;
        rms_eps = eps;

        layer_weights.resize(layers);
        allocate_buffers();
        compiled = true;
    }

    // ========================================================================
    // Execute one decode step
    // ========================================================================
    // Input: hidden state for current token [hidden_dim]
    // KV cache: [n_layers][2][max_seq_len][n_kv_heads * head_dim]
    // position: current position in sequence
    // Output: logits [vocab_size]

    void decode_token(const float* embedding,
                      float* kv_cache,       // flat: [n_layers * 2 * max_seq * kv_dim]
                      int64_t position,
                      float* logits) {
        // Copy embedding to residual stream
        std::memcpy(buf_residual, embedding, static_cast<size_t>(hidden_dim) * sizeof(float));

        int64_t kv_dim = n_kv_heads * head_dim;
        int64_t kv_layer_stride = 2 * max_seq_len * kv_dim;

        for (int64_t layer = 0; layer < n_layers; layer++) {
            auto& w = layer_weights[layer];

            // ============================================================
            // 1. Attention norm (RMSNorm)
            // ============================================================
            rmsnorm(buf_residual, w.attn_norm, buf_norm, hidden_dim, rms_eps);

            // ============================================================
            // 2. QKV projections (GEMV: [dim, hidden] @ hidden -> dim)
            // ============================================================
            at::native::hot::sgemv(n_heads * head_dim, hidden_dim,
                1.0f, w.q_weight, hidden_dim, buf_norm, 0.0f, buf_q);
            at::native::hot::sgemv(kv_dim, hidden_dim,
                1.0f, w.k_weight, hidden_dim, buf_norm, 0.0f, buf_k);
            at::native::hot::sgemv(kv_dim, hidden_dim,
                1.0f, w.v_weight, hidden_dim, buf_norm, 0.0f, buf_v);

            // Add biases if present
            if (w.q_bias) {
                at::native::hot::add_inplace(buf_q, w.q_bias, n_heads * head_dim);
            }
            if (w.k_bias) {
                at::native::hot::add_inplace(buf_k, w.k_bias, kv_dim);
            }
            if (w.v_bias) {
                at::native::hot::add_inplace(buf_v, w.v_bias, kv_dim);
            }

            // ============================================================
            // 3. RoPE (in-place on Q and K)
            // ============================================================
            apply_rope(buf_q, position, n_heads, head_dim);
            apply_rope(buf_k, position, n_kv_heads, head_dim);

            // ============================================================
            // 4. Store K,V in cache
            // ============================================================
            float* k_cache = kv_cache + layer * kv_layer_stride + position * kv_dim;
            float* v_cache = kv_cache + layer * kv_layer_stride
                           + max_seq_len * kv_dim + position * kv_dim;
            std::memcpy(k_cache, buf_k, static_cast<size_t>(kv_dim) * sizeof(float));
            std::memcpy(v_cache, buf_v, static_cast<size_t>(kv_dim) * sizeof(float));

            // ============================================================
            // 5. Attention: score, softmax, weighted sum
            // ============================================================
            int64_t seq_len = position + 1;
            int64_t n_rep = n_heads / n_kv_heads;  // GQA repeat factor

            std::memset(buf_attn_out, 0,
                        static_cast<size_t>(n_heads * head_dim) * sizeof(float));

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            float* k_base = kv_cache + layer * kv_layer_stride;
            float* v_base = k_base + max_seq_len * kv_dim;

            for (int64_t h = 0; h < n_heads; h++) {
                int64_t kv_h = h / n_rep;  // which KV head this Q head attends to
                const float* q_head = buf_q + h * head_dim;
                float* scores = buf_scores + h * max_seq_len;

                // Compute attention scores: Q_h . K_t for t in [0, seq_len)
                for (int64_t t = 0; t < seq_len; t++) {
                    const float* k_t = k_base + t * kv_dim + kv_h * head_dim;
                    scores[t] = at::native::hot::sdot(head_dim, q_head, k_t) * scale;
                }

                // Softmax over [0, seq_len)
                float max_score = scores[0];
                for (int64_t t = 1; t < seq_len; t++) {
                    if (scores[t] > max_score) max_score = scores[t];
                }
                float sum_exp = 0.0f;
                for (int64_t t = 0; t < seq_len; t++) {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                float inv_sum = 1.0f / (sum_exp + 1e-10f);
                for (int64_t t = 0; t < seq_len; t++) {
                    scores[t] *= inv_sum;
                }

                // Weighted sum of values
                float* out_head = buf_attn_out + h * head_dim;
                for (int64_t t = 0; t < seq_len; t++) {
                    const float* v_t = v_base + t * kv_dim + kv_h * head_dim;
                    float s = scores[t];
                    for (int64_t d = 0; d < head_dim; d++) {
                        out_head[d] += s * v_t[d];
                    }
                }
            }

            // ============================================================
            // 6. Output projection: o_weight @ attn_out
            // ============================================================
            // o_weight is [hidden, n_heads*head_dim], attn_out is [n_heads*head_dim]
            // result is [hidden]
            at::native::hot::sgemv(hidden_dim, n_heads * head_dim,
                1.0f, w.o_weight, n_heads * head_dim, buf_attn_out,
                0.0f, buf_ffn_out);
            if (w.o_bias) {
                at::native::hot::add_inplace(buf_ffn_out, w.o_bias, hidden_dim);
            }

            // Residual add
            at::native::hot::add_inplace(buf_residual, buf_ffn_out, hidden_dim);

            // ============================================================
            // 7. FFN norm (RMSNorm)
            // ============================================================
            rmsnorm(buf_residual, w.ffn_norm, buf_norm, hidden_dim, rms_eps);

            // ============================================================
            // 8. FFN (SwiGLU): gate, up, silu(gate)*up, down
            // ============================================================
            // gate = gate_weight @ norm  [ffn_dim]
            at::native::hot::sgemv(ffn_dim, hidden_dim,
                1.0f, w.gate_weight, hidden_dim, buf_norm, 0.0f, buf_gate);
            // up = up_weight @ norm  [ffn_dim]
            at::native::hot::sgemv(ffn_dim, hidden_dim,
                1.0f, w.up_weight, hidden_dim, buf_norm, 0.0f, buf_up);

            // SiLU(gate) * up
            for (int64_t j = 0; j < ffn_dim; j++) {
                float g = buf_gate[j];
                float silu = g / (1.0f + std::exp(-g));  // SiLU = x * sigmoid(x)
                buf_gate[j] = silu * buf_up[j];
            }

            // down = down_weight @ gate  [hidden]
            at::native::hot::sgemv(hidden_dim, ffn_dim,
                1.0f, w.down_weight, ffn_dim, buf_gate, 0.0f, buf_ffn_out);

            // Residual add
            at::native::hot::add_inplace(buf_residual, buf_ffn_out, hidden_dim);
        }

        // ================================================================
        // Final norm + LM head
        // ================================================================
        rmsnorm(buf_residual, final_norm, buf_norm, hidden_dim, rms_eps);

        // logits = lm_head @ norm  [vocab_size, hidden] @ [hidden] -> [vocab_size]
        at::native::hot::sgemv(vocab_size, hidden_dim,
            1.0f, lm_head, hidden_dim, buf_norm, 0.0f, logits);
    }

    // ========================================================================
    // KV cache allocation helper
    // ========================================================================
    // Returns flat buffer: [n_layers * 2 * max_seq_len * kv_dim]
    float* alloc_kv_cache() const {
        int64_t kv_dim = n_kv_heads * head_dim;
        size_t bytes = static_cast<size_t>(n_layers * 2 * max_seq_len * kv_dim) * sizeof(float);
        bytes = (bytes + 63) & ~63ULL;
        float* cache = static_cast<float*>(PROMEPILE_ALIGNED_ALLOC(64, bytes));
        if (!cache) throw std::runtime_error("PromePile: KV cache allocation failed");
        std::memset(cache, 0, bytes);
        return cache;
    }

    void free_kv_cache(float* cache) const {
        if (cache) PROMEPILE_ALIGNED_FREE(cache);
    }

    void print_summary() const {
        std::cout << "=== PromePile Compiled Decode Step ===" << std::endl;
        std::cout << "  Hidden: " << hidden_dim << std::endl;
        std::cout << "  Heads: " << n_heads << " (KV: " << n_kv_heads << ")" << std::endl;
        std::cout << "  Head dim: " << head_dim << std::endl;
        std::cout << "  FFN dim: " << ffn_dim << std::endl;
        std::cout << "  Layers: " << n_layers << std::endl;
        std::cout << "  Vocab: " << vocab_size << std::endl;
        std::cout << "  Max seq: " << max_seq_len << std::endl;

        size_t scratch = static_cast<size_t>(
            hidden_dim * 3         // norm, residual, ffn_out
            + n_heads * head_dim   // q
            + n_kv_heads * head_dim * 2  // k, v
            + n_heads * head_dim   // attn_out
            + n_heads * max_seq_len // scores
            + ffn_dim * 2          // gate, up
            + vocab_size           // logits
        ) * sizeof(float);
        std::cout << "  Scratch memory: " << (scratch / 1024) << " KB" << std::endl;

        size_t kv_mem = static_cast<size_t>(
            n_layers * 2 * max_seq_len * n_kv_heads * head_dim
        ) * sizeof(float);
        std::cout << "  KV cache: " << (kv_mem / (1024 * 1024)) << " MB" << std::endl;
    }

    ~CompiledDecodeStep() {
        free_buffers();
    }

    CompiledDecodeStep() = default;
    CompiledDecodeStep(CompiledDecodeStep&& o) noexcept
        : hidden_dim(o.hidden_dim), n_heads(o.n_heads), n_kv_heads(o.n_kv_heads)
        , head_dim(o.head_dim), ffn_dim(o.ffn_dim), n_layers(o.n_layers)
        , vocab_size(o.vocab_size), max_seq_len(o.max_seq_len), rms_eps(o.rms_eps)
        , layer_weights(std::move(o.layer_weights))
        , final_norm(o.final_norm), lm_head(o.lm_head)
        , buf_norm(o.buf_norm), buf_q(o.buf_q), buf_k(o.buf_k), buf_v(o.buf_v)
        , buf_attn_out(o.buf_attn_out), buf_scores(o.buf_scores)
        , buf_gate(o.buf_gate), buf_up(o.buf_up), buf_ffn_out(o.buf_ffn_out)
        , buf_residual(o.buf_residual), buf_logits(o.buf_logits)
        , compiled(o.compiled) {
        o.buf_norm = o.buf_q = o.buf_k = o.buf_v = nullptr;
        o.buf_attn_out = o.buf_scores = nullptr;
        o.buf_gate = o.buf_up = o.buf_ffn_out = nullptr;
        o.buf_residual = o.buf_logits = nullptr;
        o.compiled = false;
    }

private:
    void allocate_buffers() {
        buf_norm = alloc(hidden_dim);
        buf_q = alloc(n_heads * head_dim);
        buf_k = alloc(n_kv_heads * head_dim);
        buf_v = alloc(n_kv_heads * head_dim);
        buf_attn_out = alloc(n_heads * head_dim);
        buf_scores = alloc(n_heads * max_seq_len);
        buf_gate = alloc(ffn_dim);
        buf_up = alloc(ffn_dim);
        buf_ffn_out = alloc(hidden_dim);
        buf_residual = alloc(hidden_dim);
        buf_logits = alloc(vocab_size);
    }

    void free_buffers() {
        auto fr = [](float*& p) { if (p) { PROMEPILE_ALIGNED_FREE(p); p = nullptr; } };
        fr(buf_norm); fr(buf_q); fr(buf_k); fr(buf_v);
        fr(buf_attn_out); fr(buf_scores);
        fr(buf_gate); fr(buf_up); fr(buf_ffn_out);
        fr(buf_residual); fr(buf_logits);
    }

    static float* alloc(int64_t numel) {
        size_t bytes = static_cast<size_t>(numel) * sizeof(float);
        bytes = (bytes + 63) & ~63ULL;
        if (bytes == 0) bytes = 64;
        float* p = static_cast<float*>(PROMEPILE_ALIGNED_ALLOC(64, bytes));
        if (!p) throw std::runtime_error("PromePile: alloc failed");
        std::memset(p, 0, bytes);
        return p;
    }

    // RMSNorm: out[i] = (x[i] / rms) * weight[i]
    static void rmsnorm(const float* x, const float* weight,
                        float* out, int64_t dim, float eps) {
        float ss = 0.0f;
        for (int64_t i = 0; i < dim; i++) ss += x[i] * x[i];
        float rms = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
        for (int64_t i = 0; i < dim; i++) out[i] = x[i] * rms * weight[i];
    }

    // RoPE: apply rotary position embedding to grouped heads
    static void apply_rope(float* x, int64_t position,
                           int64_t n_heads, int64_t head_dim) {
        for (int64_t h = 0; h < n_heads; h++) {
            float* head = x + h * head_dim;
            for (int64_t i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / std::pow(10000.0f,
                    static_cast<float>(i) / static_cast<float>(head_dim));
                float theta = static_cast<float>(position) * freq;
                float cos_t = std::cos(theta);
                float sin_t = std::sin(theta);
                float x0 = head[i];
                float x1 = head[i + 1];
                head[i]     = x0 * cos_t - x1 * sin_t;
                head[i + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
};

} // namespace compile
} // namespace torch
