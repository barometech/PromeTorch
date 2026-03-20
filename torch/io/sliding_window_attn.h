#pragma once

// ============================================================================
// Sliding Window Attention for CPU Decode
//
// During auto-regressive decode, standard attention computes Q @ K^T for ALL
// past KV positions — O(seq_len) per head. For long contexts, this becomes
// the bottleneck.
//
// Optimization: only attend to the last `window_size` positions. For positions
// beyond the window, use a compressed summary vector (running mean of
// discarded K/V vectors, updated incrementally).
//
// This reduces attention from O(seq_len) to O(window_size).
//
// For typical inference (2048 context), with window=512:
//   - Positions 0..1535: summarized into 1 vector per head
//   - Positions 1536..2047: exact attention (512 dot products)
//   - Speedup: ~4x for attention at full context
//
// Quality: for most generative tasks, recent context dominates attention
// weights. The summary vector captures global context adequately.
//
// Usage: set window_size=0 to disable (full attention).
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "c10/util/ThreadPool.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace torch {
namespace io {

// ============================================================================
// Per-layer summary state for sliding window
// ============================================================================
struct SlidingWindowState {
    // Running mean of K vectors per KV head, for positions before the window
    // k_summary[kv_head][head_dim] — mean of discarded K vectors
    // v_summary[kv_head][head_dim] — mean of discarded V vectors
    std::vector<float> k_summary;   // [n_kv_heads * head_dim]
    std::vector<float> v_summary;   // [n_kv_heads * head_dim]
    int64_t n_summarized = 0;       // number of positions summarized

    bool allocated = false;

    void allocate(int64_t n_kv_heads, int64_t head_dim) {
        k_summary.resize(n_kv_heads * head_dim, 0.0f);
        v_summary.resize(n_kv_heads * head_dim, 0.0f);
        n_summarized = 0;
        allocated = true;
    }

    // Update summary with a new position that is being evicted from the window
    void add_to_summary(const float* k_vec, const float* v_vec,
                        int64_t n_kv_heads, int64_t head_dim) {
        // Incremental mean: mean_new = mean_old + (x - mean_old) / (n+1)
        n_summarized++;
        float inv_n = 1.0f / static_cast<float>(n_summarized);

        int64_t kv_dim = n_kv_heads * head_dim;
#ifdef __AVX2__
        __m256 v_inv_n = _mm256_set1_ps(inv_n);
        int64_t j = 0;
        for (; j + 7 < kv_dim; j += 8) {
            // k_summary += (k_vec - k_summary) / n
            __m256 ks = _mm256_loadu_ps(k_summary.data() + j);
            __m256 kv = _mm256_loadu_ps(k_vec + j);
            __m256 diff = _mm256_sub_ps(kv, ks);
            ks = _mm256_fmadd_ps(diff, v_inv_n, ks);
            _mm256_storeu_ps(k_summary.data() + j, ks);

            __m256 vs = _mm256_loadu_ps(v_summary.data() + j);
            __m256 vv = _mm256_loadu_ps(v_vec + j);
            diff = _mm256_sub_ps(vv, vs);
            vs = _mm256_fmadd_ps(diff, v_inv_n, vs);
            _mm256_storeu_ps(v_summary.data() + j, vs);
        }
        for (; j < kv_dim; ++j) {
            k_summary[j] += (k_vec[j] - k_summary[j]) * inv_n;
            v_summary[j] += (v_vec[j] - v_summary[j]) * inv_n;
        }
#else
        for (int64_t j = 0; j < kv_dim; ++j) {
            k_summary[j] += (k_vec[j] - k_summary[j]) * inv_n;
            v_summary[j] += (v_vec[j] - v_summary[j]) * inv_n;
        }
#endif
    }
};

// ============================================================================
// Sliding window attention manager
// ============================================================================
struct SlidingWindowAttention {
    int64_t window_size = 512;      // attend to last N positions exactly
    bool enabled = false;

    // Per-layer state
    std::vector<SlidingWindowState> layer_states;

    void init(int64_t n_layers, int64_t n_kv_heads, int64_t head_dim, int64_t win_size) {
        window_size = win_size;
        if (window_size <= 0) {
            enabled = false;
            return;
        }
        enabled = true;
        layer_states.resize(n_layers);
        for (auto& state : layer_states) {
            state.allocate(n_kv_heads, head_dim);
        }
    }

    // ====================================================================
    // Update: called after KV cache append, before attention computation.
    // If total_seq > window_size, summarize the oldest position.
    // ====================================================================
    void update_window(int64_t layer_idx, int64_t total_seq,
                       const float* k_cache, const float* v_cache,
                       int64_t n_kv_heads, int64_t head_dim) {
        if (!enabled || total_seq <= window_size) return;

        auto& state = layer_states[layer_idx];
        int64_t kv_dim = n_kv_heads * head_dim;

        // The position being evicted is at index (total_seq - window_size - 1)
        // We need to summarize all positions from state.n_summarized to
        // (total_seq - window_size - 1)
        int64_t evict_up_to = total_seq - window_size;
        while (state.n_summarized < evict_up_to) {
            int64_t pos = state.n_summarized;
            state.add_to_summary(
                k_cache + pos * kv_dim,
                v_cache + pos * kv_dim,
                n_kv_heads, head_dim);
        }
    }

    // ====================================================================
    // Compute attention with sliding window
    //
    // q_head: query vector for one head [head_dim]
    // k_cache: full K cache [max_seq, kv_dim]
    // v_cache: full V cache [max_seq, kv_dim]
    // out_head: output [head_dim]
    // total_seq: total number of cached positions
    // kv_h: which KV head this query head maps to
    // head_dim: dimension per head
    // kv_dim: total KV dimension (n_kv_heads * head_dim)
    // layer_idx: layer index
    // scale: 1/sqrt(head_dim)
    //
    // scores_buf: scratch buffer for attention scores [window_size + 1]
    // ====================================================================
    void compute_attention(
            const float* q_head,
            const float* k_cache, const float* v_cache,
            float* out_head,
            int64_t total_seq, int64_t kv_h,
            int64_t head_dim, int64_t kv_dim,
            int64_t layer_idx, float scale,
            float* scores_buf) const {

        if (!enabled || total_seq <= window_size) {
            // Fall through to standard attention
            compute_full_attention(q_head, k_cache, v_cache, out_head,
                                   total_seq, kv_h, head_dim, kv_dim, scale, scores_buf);
            return;
        }

        const auto& state = layer_states[layer_idx];
        int64_t win_start = total_seq - window_size;
        int64_t n_window = window_size;
        // +1 for the summary position
        int64_t n_scores = n_window + (state.n_summarized > 0 ? 1 : 0);

        // Score[0]: summary dot product (if summarized positions exist)
        int64_t score_offset = 0;
        if (state.n_summarized > 0) {
            const float* k_sum = state.k_summary.data() + kv_h * head_dim;
#ifdef __AVX2__
            __m256 dot_acc = _mm256_setzero_ps();
            int64_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                dot_acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(q_head + d),
                    _mm256_loadu_ps(k_sum + d), dot_acc);
            }
            float dot = hsum_avx_sw(dot_acc);
            for (; d < head_dim; ++d) dot += q_head[d] * k_sum[d];
#else
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * k_sum[d];
#endif
            scores_buf[0] = dot * scale;
            score_offset = 1;
        }

        // Scores for window positions
#ifdef __AVX2__
        for (int64_t t = 0; t < n_window; ++t) {
            const float* k_head = k_cache + (win_start + t) * kv_dim + kv_h * head_dim;
            __m256 dot_acc = _mm256_setzero_ps();
            int64_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                dot_acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(q_head + d),
                    _mm256_loadu_ps(k_head + d), dot_acc);
            }
            float dot = hsum_avx_sw(dot_acc);
            for (; d < head_dim; ++d) dot += q_head[d] * k_head[d];
            scores_buf[score_offset + t] = dot * scale;
        }
#else
        for (int64_t t = 0; t < n_window; ++t) {
            const float* k_head = k_cache + (win_start + t) * kv_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * k_head[d];
            scores_buf[score_offset + t] = dot * scale;
        }
#endif

        // Softmax over n_scores
        float max_score = scores_buf[0];
        for (int64_t t = 1; t < n_scores; ++t)
            if (scores_buf[t] > max_score) max_score = scores_buf[t];

        float sum_exp = 0.0f;
        for (int64_t t = 0; t < n_scores; ++t) {
            scores_buf[t] = std::exp(scores_buf[t] - max_score);
            sum_exp += scores_buf[t];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (int64_t t = 0; t < n_scores; ++t) scores_buf[t] *= inv_sum;

        // Weighted sum of V
        std::fill(out_head, out_head + head_dim, 0.0f);

        // Summary V contribution
        if (state.n_summarized > 0) {
            const float* v_sum = state.v_summary.data() + kv_h * head_dim;
            float w = scores_buf[0];
#ifdef __AVX2__
            __m256 vw = _mm256_set1_ps(w);
            int64_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                _mm256_storeu_ps(out_head + d,
                    _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_sum + d),
                        _mm256_loadu_ps(out_head + d)));
            }
            for (; d < head_dim; ++d) out_head[d] += w * v_sum[d];
#else
            for (int64_t d = 0; d < head_dim; ++d) out_head[d] += w * v_sum[d];
#endif
        }

        // Window V contributions
#ifdef __AVX2__
        for (int64_t t = 0; t < n_window; ++t) {
            const float* v_head = v_cache + (win_start + t) * kv_dim + kv_h * head_dim;
            float w = scores_buf[score_offset + t];
            __m256 vw = _mm256_set1_ps(w);
            int64_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                _mm256_storeu_ps(out_head + d,
                    _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_head + d),
                        _mm256_loadu_ps(out_head + d)));
            }
            for (; d < head_dim; ++d) out_head[d] += w * v_head[d];
        }
#else
        for (int64_t t = 0; t < n_window; ++t) {
            const float* v_head = v_cache + (win_start + t) * kv_dim + kv_h * head_dim;
            float w = scores_buf[score_offset + t];
            for (int64_t d = 0; d < head_dim; ++d) out_head[d] += w * v_head[d];
        }
#endif
    }

private:
    // Standard full attention (fallback when window not active)
    static void compute_full_attention(
            const float* q_head,
            const float* k_cache, const float* v_cache,
            float* out_head,
            int64_t total_seq, int64_t kv_h,
            int64_t head_dim, int64_t kv_dim, float scale,
            float* scores_buf) {

#ifdef __AVX2__
        for (int64_t t = 0; t < total_seq; ++t) {
            const float* k_head = k_cache + t * kv_dim + kv_h * head_dim;
            __m256 dot_acc = _mm256_setzero_ps();
            int64_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                dot_acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(q_head + d),
                    _mm256_loadu_ps(k_head + d), dot_acc);
            }
            float dot = hsum_avx_sw(dot_acc);
            for (; d < head_dim; ++d) dot += q_head[d] * k_head[d];
            scores_buf[t] = dot * scale;
        }
#else
        for (int64_t t = 0; t < total_seq; ++t) {
            const float* k_head = k_cache + t * kv_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * k_head[d];
            scores_buf[t] = dot * scale;
        }
#endif

        float max_score = scores_buf[0];
        for (int64_t t = 1; t < total_seq; ++t)
            if (scores_buf[t] > max_score) max_score = scores_buf[t];

        float sum_exp = 0.0f;
        for (int64_t t = 0; t < total_seq; ++t) {
            scores_buf[t] = std::exp(scores_buf[t] - max_score);
            sum_exp += scores_buf[t];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (int64_t t = 0; t < total_seq; ++t) scores_buf[t] *= inv_sum;

        std::fill(out_head, out_head + head_dim, 0.0f);
#ifdef __AVX2__
        for (int64_t t = 0; t < total_seq; ++t) {
            const float* v_head = v_cache + t * kv_dim + kv_h * head_dim;
            __m256 vw = _mm256_set1_ps(scores_buf[t]);
            int64_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                _mm256_storeu_ps(out_head + d,
                    _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_head + d),
                        _mm256_loadu_ps(out_head + d)));
            }
            for (; d < head_dim; ++d) out_head[d] += scores_buf[t] * v_head[d];
        }
#else
        for (int64_t t = 0; t < total_seq; ++t) {
            const float* v_head = v_cache + t * kv_dim + kv_h * head_dim;
            float w = scores_buf[t];
            for (int64_t d = 0; d < head_dim; ++d) out_head[d] += w * v_head[d];
        }
#endif
    }

#ifdef __AVX2__
    static float hsum_avx_sw(__m256 v) {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 sum4 = _mm_add_ps(lo, hi);
        sum4 = _mm_hadd_ps(sum4, sum4);
        sum4 = _mm_hadd_ps(sum4, sum4);
        return _mm_cvtss_f32(sum4);
    }
#endif
};

} // namespace io
} // namespace torch
