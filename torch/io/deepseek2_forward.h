#pragma once
// ============================================================================
// deepseek2_forward.h — MLA + MoE forward kernels for GigaChat3 / DeepSeek-V2/V3
// ============================================================================
//
// Implements Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE)
// forward passes for the deepseek2 architecture. The full GigaChat3.1-10B-A1.8B
// inference uses these in place of the standard llama-family GQA attention and
// dense FFN. Both functions are decode-only (single token, position `pos`).
//
// Layout reminder (GigaChat3.1):
//   hidden_size H = 1536          n_heads = 32        kv_lora_rank = 512
//   key_length_mla = 192          key_length_rope = 64
//   value_length_mla = 192        head_dim_full = 256 (= 192 + 64)
//   expert_count = 64             expert_used_count = 4
//   expert_shared_count = 1       expert_FF = 1280
//
// MLA cache layout (uncompressed, one entry per generated token):
//   K_cache_layer[T, n_heads, 256] — concat of k_nope (192) + k_pe (shared 64)
//   V_cache_layer[T, n_heads, 192] — per-head V
// We could store kv_compressed (512+64) once and re-expand each step, saving
// 4× cache memory; that optimization belongs to a later phase.
//
// Both functions assume scratch is at least:
//   MLA  : n_heads*256 + 576 + n_heads*192 + n_heads*192 + n_heads*192 floats
//   MoE  : 2*expert_FF (gate+up) + H (down out) + expert_count (router) +
//          expert_used_count*H (per-expert outs) + H (shared expert out) floats
//
// References: llama.cpp src/models/deepseek2.cpp (MIT) for tensor shapes &
// math. Kernels here are independent CPU implementations targeting the same
// numerical contract.
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "torch/io/cpu_quant_gemv.h"
#include "torch/io/gguf_model.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"

namespace torch {
namespace io {

// ----------------------------------------------------------------------------
// MLA attention forward — decode (single token at position `pos`)
// ----------------------------------------------------------------------------
// Inputs:
//   x_norm         [H]                — pre-norm hidden (RMSNorm already applied)
//   layer                              — TransformerLayer with MLA tensors
//   config                             — TransformerConfig (is_mla=true expected)
//   K_cache, V_cache                   — pre-allocated [max_seq, n_heads*256/192]
//                                         floats; layer_offset already applied
//   pos                                — current token position (0-indexed)
//   scratch                            — workspace, see size estimate above
// Output:
//   y_out          [H]                 — output after attn_output projection
//
inline void mla_attention_forward_decode(
    const float*               x_norm,
    const TransformerLayer&    layer,
    const TransformerConfig&   config,
    float*                     K_cache_layer,   // [max_seq, cache_stride]
    float*                     V_cache_layer,   // [max_seq, cache_stride]
    int64_t                    pos,
    float*                     y_out,
    float*                     scratch,
    int64_t                    cache_stride = 0  // 0 = use natural layout
) {
    const int64_t H        = config.hidden_size;
    const int64_t n_heads  = config.num_heads;
    const int64_t no_rope  = config.key_length_mla;
    const int64_t rope_dim = config.key_length_rope;
    const int64_t kvla     = config.kv_lora_rank;
    const int64_t v_dim    = config.value_length_mla;
    const int64_t head_full = config.head_dim;            // 256 = no_rope + rope_dim
    const int64_t T         = pos + 1;
    // Logical strides for indexing within one cache row:
    const int64_t k_stride  = n_heads * head_full;
    const int64_t v_stride  = n_heads * v_dim;
    // Physical stride between consecutive cache rows (>= max(k_stride, v_stride)):
    const int64_t k_phys = (cache_stride > 0) ? cache_stride : k_stride;
    const int64_t v_phys = (cache_stride > 0) ? cache_stride : v_stride;

    // Scratch layout (aligned packing):
    //   q_full        [n_heads * head_full]
    //   kv_compressed [kvla + rope_dim]              ← 512 + 64 = 576
    //   k_nope_all    [n_heads * no_rope]            ← 32 * 192 = 6144
    //   v_all         [n_heads * v_dim]              ← 32 * 192 = 6144
    //   attn_out      [n_heads * v_dim]              ← reused for output GEMV input
    float* q_full        = scratch;
    float* kv_compressed = q_full        + n_heads * head_full;
    float* k_pe          = kv_compressed + kvla;            // last rope_dim of kv_compressed
    float* k_nope_all    = kv_compressed + kvla + rope_dim;
    float* v_all         = k_nope_all    + n_heads * no_rope;
    float* attn_out      = v_all         + n_heads * v_dim;

    // 1. Q projection: q_full[n_heads * head_full] = attn_q @ x_norm
    cpu_quant::cpu_quant_gemv(
        layer.q_attn_q.quant_type,
        layer.q_attn_q.cpu_data,
        x_norm, q_full,
        H, n_heads * head_full,
        layer.q_attn_q.row_stride_bytes, nullptr);

    // 2. KV down + RoPE half: kv_compressed[576] = attn_kv_a_mqa @ x_norm
    cpu_quant::cpu_quant_gemv(
        layer.q_attn_kv_a_mqa.quant_type,
        layer.q_attn_kv_a_mqa.cpu_data,
        x_norm, kv_compressed,
        H, kvla + rope_dim,
        layer.q_attn_kv_a_mqa.row_stride_bytes, nullptr);

    // 3. RMSNorm on kv_compressed[:kvla] using attn_kv_a_norm.
    {
        const float* gamma = layer.attn_kv_a_norm.data_ptr<float>();
        double ss = 0;
        for (int64_t j = 0; j < kvla; ++j) ss += (double)kv_compressed[j] * kv_compressed[j];
        float rms = 1.0f / std::sqrt(static_cast<float>(ss / kvla) + config.rms_norm_eps);
        for (int64_t j = 0; j < kvla; ++j) kv_compressed[j] = kv_compressed[j] * rms * gamma[j];
    }

    // 4. K up per head: k_nope_all[h, no_rope] = attn_k_b[h] @ kv_compressed[:kvla]
    //    attn_k_b is 3D [n_heads, no_rope, kvla], slice along dim 0.
    for (int64_t h = 0; h < n_heads; ++h) {
        cpu_quant::cpu_quant_gemv_3d_indexed(
            layer.q_attn_k_b.quant_type,
            layer.q_attn_k_b.cpu_data,
            kv_compressed,
            k_nope_all + h * no_rope,
            kvla, no_rope, n_heads,
            h,
            layer.q_attn_k_b.row_stride_bytes, nullptr);
    }

    // 5. V up per head: v_all[h, v_dim] = attn_v_b[h] @ kv_compressed[:kvla]
    for (int64_t h = 0; h < n_heads; ++h) {
        cpu_quant::cpu_quant_gemv_3d_indexed(
            layer.q_attn_v_b.quant_type,
            layer.q_attn_v_b.cpu_data,
            kv_compressed,
            v_all + h * v_dim,
            kvla, v_dim, n_heads,
            h,
            layer.q_attn_v_b.row_stride_bytes, nullptr);
    }

    // 6. RoPE on k_pe (shared across heads) and on rope-half of each Q head.
    //    YaRN if config.rope_yarn, else plain neox-style RoPE.
    constexpr int64_t MAX_HALF = 64;  // up to head_dim/2 = 32 here, but be safe
    float cos_buf[MAX_HALF];
    float sin_buf[MAX_HALF];
    const int64_t half_rope = rope_dim / 2;
    if (config.rope_yarn) {
        at::native::hot::rope_precompute_yarn(
            cos_buf, sin_buf, pos, rope_dim,
            config.rope_freq_base, config.yarn_factor,
            config.yarn_beta_fast, config.yarn_beta_slow,
            config.yarn_orig_ctx, config.yarn_log_multiplier);
    } else {
        at::native::hot::rope_precompute(
            cos_buf, sin_buf, pos, rope_dim, config.rope_freq_base);
    }

    // Apply NeoX rope to k_pe (one shared copy).
    for (int64_t d = 0; d < half_rope; ++d) {
        float x0 = k_pe[d];
        float x1 = k_pe[d + half_rope];
        k_pe[d]                = x0 * cos_buf[d] - x1 * sin_buf[d];
        k_pe[d + half_rope]    = x0 * sin_buf[d] + x1 * cos_buf[d];
    }
    // Apply NeoX rope to last rope_dim of each Q head.
    for (int64_t h = 0; h < n_heads; ++h) {
        float* q_rope = q_full + h * head_full + no_rope;  // [rope_dim]
        for (int64_t d = 0; d < half_rope; ++d) {
            float x0 = q_rope[d];
            float x1 = q_rope[d + half_rope];
            q_rope[d]              = x0 * cos_buf[d] - x1 * sin_buf[d];
            q_rope[d + half_rope]  = x0 * sin_buf[d] + x1 * cos_buf[d];
        }
    }

    // 7. Cache K/V for current position. k_full[h] = [k_nope[h] ; k_pe (shared)].
    {
        float* K_pos = K_cache_layer + pos * k_phys;
        float* V_pos = V_cache_layer + pos * v_phys;
        for (int64_t h = 0; h < n_heads; ++h) {
            std::memcpy(K_pos + h * head_full,            k_nope_all + h * no_rope, no_rope * sizeof(float));
            std::memcpy(K_pos + h * head_full + no_rope,  k_pe,                      rope_dim * sizeof(float));
            std::memcpy(V_pos + h * v_dim,                v_all      + h * v_dim,    v_dim * sizeof(float));
        }
    }

    // 8. Attention: per-head softmax(Q . K^T * mscale² / sqrt(d)) . V.
    //
    // YaRN attention scale (llama.cpp deepseek2.cpp build):
    //   attn_factor_org = attn_factor * (1 + 0.1 * log(factor))
    //   kq_mscale       = attn_factor_org * (1 + 0.1 * log_mul * log(factor))
    //   kq_scale        = kq_mscale² / sqrt(head_dim)
    // attn_factor defaults to 1.0 (no override key in GigaChat3 metadata).
    {
        float kq_mscale = 1.0f;
        if (config.rope_yarn && config.yarn_factor > 1.0f) {
            const float lf = std::log(config.yarn_factor);
            const float attn_factor_org = 1.0f + 0.1f * lf;
            kq_mscale = attn_factor_org *
                        (1.0f + 0.1f * config.yarn_log_multiplier * lf);
        }
        const float scale = (kq_mscale * kq_mscale) /
                            std::sqrt(static_cast<float>(head_full));
        std::vector<float> scores(T);
        for (int64_t h = 0; h < n_heads; ++h) {
            const float* q_ptr = q_full + h * head_full;
            // Compute Q . K[t][h] for each t.
            for (int64_t t = 0; t < T; ++t) {
                const float* k_ptr = K_cache_layer + t * k_phys + h * head_full;
                float dot = 0;
                for (int64_t d = 0; d < head_full; ++d) dot += q_ptr[d] * k_ptr[d];
                scores[t] = dot * scale;
            }
            // Softmax in-place.
            float mx = scores[0];
            for (int64_t t = 1; t < T; ++t) if (scores[t] > mx) mx = scores[t];
            float sum = 0;
            for (int64_t t = 0; t < T; ++t) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
            const float inv = 1.0f / sum;
            for (int64_t t = 0; t < T; ++t) scores[t] *= inv;
            // Weighted V → attn_out[h].
            float* out_h = attn_out + h * v_dim;
            std::memset(out_h, 0, v_dim * sizeof(float));
            for (int64_t t = 0; t < T; ++t) {
                const float w = scores[t];
                const float* v_ptr = V_cache_layer + t * v_phys + h * v_dim;
                for (int64_t d = 0; d < v_dim; ++d) out_h[d] += w * v_ptr[d];
            }
        }
    }

    // 9. Output projection: y_out[H] = attn_output @ attn_out
    //    attn_output is [H, n_heads * v_dim]: K = n_heads * v_dim, N = H.
    cpu_quant::cpu_quant_gemv(
        layer.q_attn_output.quant_type,
        layer.q_attn_output.cpu_data,
        attn_out, y_out,
        n_heads * v_dim, H,
        layer.q_attn_output.row_stride_bytes, nullptr);
}

// ----------------------------------------------------------------------------
// MoE FFN forward — decode (single token), x_norm pre-FFN-normed
// ----------------------------------------------------------------------------
// Inputs:
//   x_norm         [H]               — pre-norm hidden after attention
//   layer                              — TransformerLayer (is_moe_layer=true)
//   config                             — TransformerConfig
//   scratch                            — workspace
// Output:
//   y_out          [H]                — sum of top-k experts + shared expert
//
inline void moe_ffn_forward_decode(
    const float*               x_norm,
    const TransformerLayer&    layer,
    const TransformerConfig&   config,
    float*                     y_out,
    float*                     scratch
) {
    const int64_t H        = config.hidden_size;
    const int64_t E        = config.expert_count;
    const int64_t TOPK     = config.expert_used_count;
    const int64_t E_FF     = config.expert_feed_forward_length;
    const bool   has_shexp = config.expert_shared_count > 0;

    // Scratch layout:
    //   router_scores  [E]
    //   topk_idx       [TOPK]   (stored as float for simplicity)
    //   topk_weight    [TOPK]
    //   gate_buf       [E_FF]
    //   up_buf         [E_FF]
    //   expert_out_acc [H]      (accumulator for top-k expert outputs)
    //   shared_buf     [E_FF]   (used by shared expert if present)
    //   shared_out     [H]
    float* router_scores  = scratch;
    float* gate_buf       = router_scores + E;
    float* up_buf         = gate_buf       + E_FF;
    float* expert_out_acc = up_buf         + E_FF;
    float* shared_buf     = expert_out_acc + H;
    float* shared_out     = shared_buf     + E_FF;

    // 1. Router: scores[E] = ffn_gate_inp @ x_norm.
    cpu_quant::cpu_quant_gemv(
        layer.q_ffn_gate_inp.quant_type,
        layer.q_ffn_gate_inp.cpu_data,
        x_norm, router_scores,
        H, E,
        layer.q_ffn_gate_inp.row_stride_bytes, nullptr);

    // 2. Apply sigmoid (deepseek2 expert_gating_func=2 is sigmoid + group-limited topk).
    //    For now: simple sigmoid + global top-k. expert_group_count=1 → no grouping.
    for (int64_t e = 0; e < E; ++e) {
        router_scores[e] = 1.0f / (1.0f + std::exp(-router_scores[e]));
    }

    // 3. Top-K selection (partial sort).
    std::vector<std::pair<float,int>> ranked(E);
    for (int64_t e = 0; e < E; ++e) ranked[e] = {router_scores[e], static_cast<int>(e)};
    std::partial_sort(ranked.begin(), ranked.begin() + TOPK, ranked.end(),
                      [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                          return a.first > b.first;
                      });

    // 4. Normalize selected weights (expert_weights_norm = true for GigaChat3).
    float wsum = 0;
    for (int64_t k = 0; k < TOPK; ++k) wsum += ranked[k].first;
    const float inv_wsum = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;

    // 5. Per-token, per-expert: gate, up, down, accumulate.
    std::memset(expert_out_acc, 0, H * sizeof(float));
    for (int64_t k = 0; k < TOPK; ++k) {
        const int e = ranked[k].second;
        const float w = ranked[k].first * inv_wsum;

        // gate_buf[E_FF] = ffn_gate_exps[e] @ x_norm
        cpu_quant::cpu_quant_gemv_3d_indexed(
            layer.q_ffn_gate_exps.quant_type,
            layer.q_ffn_gate_exps.cpu_data,
            x_norm, gate_buf,
            H, E_FF, E, e,
            layer.q_ffn_gate_exps.row_stride_bytes, nullptr);

        // up_buf[E_FF] = ffn_up_exps[e] @ x_norm
        cpu_quant::cpu_quant_gemv_3d_indexed(
            layer.q_ffn_up_exps.quant_type,
            layer.q_ffn_up_exps.cpu_data,
            x_norm, up_buf,
            H, E_FF, E, e,
            layer.q_ffn_up_exps.row_stride_bytes, nullptr);

        // Activated: SiLU(gate) * up   →  in-place into gate_buf
        for (int64_t j = 0; j < E_FF; ++j) {
            float g = gate_buf[j];
            float silu = g / (1.0f + std::exp(-g));
            gate_buf[j] = silu * up_buf[j];
        }

        // expert_out[H] = ffn_down_exps[e] @ activated
        // Reuse up_buf — but we need [H] sized, which is 1536 ≤ E_FF=1280? No, H=1536 > E_FF=1280.
        // Use shared_buf as scratch overlap if needed — actually we have up_buf=1280, but H=1536.
        // For safety, use shared_out as the down output buffer (will be overwritten by shared expert later if any).
        cpu_quant::cpu_quant_gemv_3d_indexed(
            layer.q_ffn_down_exps.quant_type,
            layer.q_ffn_down_exps.cpu_data,
            gate_buf, shared_out,   // tmp scratch: [H]
            E_FF, H, E, e,
            layer.q_ffn_down_exps.row_stride_bytes, nullptr);

        // Accumulate weighted: expert_out_acc += w * shared_out
        for (int64_t j = 0; j < H; ++j) expert_out_acc[j] += w * shared_out[j];
    }

    // 6. Shared expert (always applied alongside top-k):
    if (has_shexp) {
        // gate_shared = ffn_gate_shexp @ x_norm
        cpu_quant::cpu_quant_gemv(
            layer.q_ffn_gate_shexp.quant_type,
            layer.q_ffn_gate_shexp.cpu_data,
            x_norm, gate_buf,
            H, E_FF,
            layer.q_ffn_gate_shexp.row_stride_bytes, nullptr);
        cpu_quant::cpu_quant_gemv(
            layer.q_ffn_up_shexp.quant_type,
            layer.q_ffn_up_shexp.cpu_data,
            x_norm, up_buf,
            H, E_FF,
            layer.q_ffn_up_shexp.row_stride_bytes, nullptr);
        for (int64_t j = 0; j < E_FF; ++j) {
            float g = gate_buf[j];
            float silu = g / (1.0f + std::exp(-g));
            shared_buf[j] = silu * up_buf[j];
        }
        cpu_quant::cpu_quant_gemv(
            layer.q_ffn_down_shexp.quant_type,
            layer.q_ffn_down_shexp.cpu_data,
            shared_buf, shared_out,
            E_FF, H,
            layer.q_ffn_down_shexp.row_stride_bytes, nullptr);
    } else {
        std::memset(shared_out, 0, H * sizeof(float));
    }

    // 7. y_out[H] = expert_out_acc + shared_out (residual added by caller).
    for (int64_t j = 0; j < H; ++j) y_out[j] = expert_out_acc[j] + shared_out[j];
}

// ----------------------------------------------------------------------------
// Dense FFN forward (for leading_dense_block_count==1: only block 0 in
// GigaChat3) — same as standard SiGLU FFN, exposed here for symmetry.
// ----------------------------------------------------------------------------
inline void dense_ffn_forward_decode(
    const float*               x_norm,
    const TransformerLayer&    layer,
    const TransformerConfig&   config,
    float*                     y_out,
    float*                     scratch
) {
    const int64_t H     = config.hidden_size;
    const int64_t INTER = config.intermediate_size;
    float* gate = scratch;
    float* up   = scratch + INTER;
    cpu_quant::cpu_quant_gemv(layer.q_ffn_gate.quant_type, layer.q_ffn_gate.cpu_data,
                   x_norm, gate, H, INTER, layer.q_ffn_gate.row_stride_bytes, nullptr);
    cpu_quant::cpu_quant_gemv(layer.q_ffn_up.quant_type,   layer.q_ffn_up.cpu_data,
                   x_norm, up,   H, INTER, layer.q_ffn_up.row_stride_bytes,   nullptr);
    for (int64_t j = 0; j < INTER; ++j) {
        float g = gate[j];
        float silu = g / (1.0f + std::exp(-g));
        gate[j] = silu * up[j];
    }
    cpu_quant::cpu_quant_gemv(layer.q_ffn_down.quant_type, layer.q_ffn_down.cpu_data,
                   gate, y_out, INTER, H, layer.q_ffn_down.row_stride_bytes, nullptr);
}

}  // namespace io
}  // namespace torch
