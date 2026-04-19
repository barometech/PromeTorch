// ============================================================================
// Attention — top-level dispatcher.
// CPU: standard softmax(Q @ K^T / sqrt(d)) @ V with optional causal mask,
//      optional additive mask (float) or boolean mask, optional dropout,
//      and gradient support via sdpa_forward_cpu_impl (returns probs+mask for bwd).
// CUDA: uses at::cuda::scaled_dot_product_attention (FlashAttention) when
//       tensors are on CUDA and shapes match (head_dim in {64,128}); otherwise
//       falls back to CPU math on the host side of the tensor.
//
// Input shape conventions:
//   Last dim is head_dim (D). Second-to-last is sequence length.
//   Supported ranks:
//     2D:       [N, D]            → single-batch, single-head
//     3D:       [B, N, D]         → multi-batch, single-head
//     4D:       [B, N, H, D]      → multi-batch, multi-head  (PromeTorch/FlashAttention
//                                   convention — NOT PyTorch's [B, H, N, D])
//     (generic) [..., N, H, D]    via flattening leading batch dims
//
// attn_mask ranks: [N_q, N_k], [B, N_q, N_k], or [B, H, N_q, N_k].
// Bool mask: True = keep, False = -inf mask. Float mask: additive.
// is_causal and attn_mask are mutually exclusive (matches PyTorch).
// ============================================================================
#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "c10/core/ScalarType.h"

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/FlashAttention.h"
#endif

#include <cmath>
#include <random>
#include <limits>
#include <vector>

namespace at {
namespace native {

// ============================================================================
// Internal 4D reference kernel.
// Q/K/V: [B, N_q or N_k, H, D]  (contiguous, float32)
// attn_mask: optional; rank 2/3/4 (see below), float or bool.
// dropout_mask (output): [B, H, N_q, N_k] float — Bernoulli keep=1, drop=0,
//                        scaled by 1/(1-p). Only written when dropout_p > 0 and
//                        ptr != nullptr. Needed for backward reproducibility.
// attn_probs   (output): [B, H, N_q, N_k] float — softmax(S) before dropout.
//                        Only written when ptr != nullptr (for backward).
// Returns: out [B, N_q, H, D].
// ============================================================================
inline Tensor sdpa_forward_cpu_impl(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask,
    float dropout_p,
    bool is_causal,
    float softmax_scale,
    bool training,
    uint64_t dropout_seed,
    Tensor* out_attn_probs,   // optional: [B,H,N_q,N_k]
    Tensor* out_dropout_mask) // optional: [B,H,N_q,N_k] — only written if dropout_p>0
{
    const int64_t B   = query.size(0);
    const int64_t N_q = query.size(1);
    const int64_t H   = query.size(2);
    const int64_t D   = query.size(3);
    const int64_t N_k = key.size(1);

    PT_CHECK_MSG(query.dtype() == c10::ScalarType::Float,
                 "scaled_dot_product_attention CPU path requires float32");

    Tensor q_c = query.is_contiguous() ? query : query.contiguous();
    Tensor k_c = key.is_contiguous()   ? key   : key.contiguous();
    Tensor v_c = value.is_contiguous() ? value : value.contiguous();

    Tensor out = empty({B, N_q, H, D},
                       TensorOptions().dtype(query.dtype()).device(query.device()));

    const float* Qd = q_c.data_ptr<float>();
    const float* Kd = k_c.data_ptr<float>();
    const float* Vd = v_c.data_ptr<float>();
    float* Od = out.mutable_data_ptr<float>();

    // ----- Mask handling --------------------------------------------------
    // PyTorch: bool mask, True = participate, False = masked out (-inf).
    // float mask is added directly to attention scores.
    const bool has_mask = attn_mask.defined();
    const bool mask_is_bool = has_mask && attn_mask.dtype() == c10::ScalarType::Bool;

    Tensor mask_c;
    if (has_mask) {
        mask_c = attn_mask.is_contiguous() ? attn_mask : attn_mask.contiguous();
    }
    const float* Md_f  = (has_mask && !mask_is_bool) ? mask_c.data_ptr<float>() : nullptr;
    const bool*  Md_b  = (has_mask &&  mask_is_bool) ? mask_c.data_ptr<bool>()  : nullptr;
    const int64_t mask_dim = has_mask ? attn_mask.dim() : 0;

    auto mask_offset = [&](int64_t b, int64_t h, int64_t i, int64_t j) -> int64_t {
        // Supported ranks:
        //   2: [N_q, N_k]
        //   3: [B, N_q, N_k]
        //   4: [B, H, N_q, N_k]
        if (mask_dim == 2) return i * N_k + j;
        if (mask_dim == 3) return (b * N_q + i) * N_k + j;
        /*mask_dim == 4*/   return ((b * H + h) * N_q + i) * N_k + j;
    };

    // ----- Dropout prep ---------------------------------------------------
    const bool apply_dropout = (dropout_p > 0.0f) && training;
    const float drop_scale = apply_dropout ? (1.0f / (1.0f - dropout_p)) : 1.0f;
    std::mt19937_64 rng(dropout_seed);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);

    float* probs_d = out_attn_probs ? out_attn_probs->mutable_data_ptr<float>() : nullptr;
    float* drop_d  = out_dropout_mask ? out_dropout_mask->mutable_data_ptr<float>() : nullptr;

    // Index helpers (row-major contiguous [B, N, H, D]).
    auto qkv_idx = [&](int64_t b, int64_t n, int64_t h, int64_t d, int64_t N) {
        return ((b * N + n) * H + h) * D + d;
    };

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t i = 0; i < N_q; ++i) {
                // 1) Scores s_j = scale * Q[b,i,h,:] . K[b,j,h,:] + mask
                std::vector<float> scores(N_k);
                float max_s = NEG_INF;
                for (int64_t j = 0; j < N_k; ++j) {
                    float s = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        s += Qd[qkv_idx(b, i, h, d, N_q)] * Kd[qkv_idx(b, j, h, d, N_k)];
                    }
                    s *= softmax_scale;

                    if (is_causal && j > i) s = NEG_INF;

                    if (has_mask) {
                        int64_t moff = mask_offset(b, h, i, j);
                        if (mask_is_bool) {
                            if (!Md_b[moff]) s = NEG_INF;  // False => masked
                        } else {
                            s += Md_f[moff];
                        }
                    }
                    scores[j] = s;
                    if (s > max_s) max_s = s;
                }

                // 2) Softmax (numerically stable)
                float sum_exp = 0.0f;
                for (int64_t j = 0; j < N_k; ++j) {
                    float e = (max_s == NEG_INF) ? 0.0f : std::exp(scores[j] - max_s);
                    scores[j] = e;
                    sum_exp += e;
                }
                float inv = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
                for (int64_t j = 0; j < N_k; ++j) scores[j] *= inv;

                // Save softmax probs (pre-dropout) for backward.
                if (probs_d) {
                    int64_t base = ((b * H + h) * N_q + i) * N_k;
                    for (int64_t j = 0; j < N_k; ++j) probs_d[base + j] = scores[j];
                }

                // 3) Dropout on attention weights (post-softmax, scaled by 1/(1-p)).
                if (apply_dropout) {
                    int64_t base = ((b * H + h) * N_q + i) * N_k;
                    for (int64_t j = 0; j < N_k; ++j) {
                        bool keep = udist(rng) >= dropout_p;
                        float m = keep ? drop_scale : 0.0f;
                        scores[j] *= m;
                        if (drop_d) drop_d[base + j] = m;
                    }
                } else if (drop_d) {
                    // training=false or p=0 → mask is all-1 (pass-through) for bwd.
                    int64_t base = ((b * H + h) * N_q + i) * N_k;
                    for (int64_t j = 0; j < N_k; ++j) drop_d[base + j] = 1.0f;
                }

                // 4) Output = sum_j scores[j] * V[b,j,h,:]
                for (int64_t d = 0; d < D; ++d) {
                    float o = 0.0f;
                    for (int64_t j = 0; j < N_k; ++j) {
                        o += scores[j] * Vd[qkv_idx(b, j, h, d, N_k)];
                    }
                    Od[qkv_idx(b, i, h, d, N_q)] = o;
                }
            }
        }
    }

    return out;
}

// ============================================================================
// Public-facing native::scaled_dot_product_attention
// ----------------------------------------------------------------------------
// Shape rules (mirrors PyTorch):
//   - Last dim = head_dim (D), second-to-last = sequence length.
//   - 3D input [N, D]: treated as (B=1, H=1) with N_q/N_k = N.
//   - 4D input: PyTorch uses [B, H, N, D] (batch_first). Our CPU kernel
//     internally wants [B, N, H, D]; we permute as needed.
//   - Higher ranks: collapse leading dims into B.
//
// is_causal + attn_mask: PyTorch treats these as mutually exclusive and
// raises. We raise as well to match behavior.
// ============================================================================
inline Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask = Tensor(),
    float dropout_p = 0.0f,
    bool is_causal = false,
    float scale = -1.0f)
{
    PT_CHECK_MSG(query.dim() >= 2 && key.dim() >= 2 && value.dim() >= 2,
                 "scaled_dot_product_attention: Q/K/V must have at least 2 dims (..., N, D)");
    PT_CHECK_MSG(query.dim() == key.dim() && key.dim() == value.dim(),
                 "scaled_dot_product_attention: Q/K/V must have the same rank");
    PT_CHECK_MSG(!(is_causal && attn_mask.defined()),
                 "scaled_dot_product_attention: is_causal and attn_mask are mutually exclusive");

    const int64_t D = query.size(query.dim() - 1);
    PT_CHECK_MSG(key.size(key.dim() - 1)   == D, "K head_dim must match Q");
    PT_CHECK_MSG(value.size(value.dim() - 1) == D, "V head_dim must match Q");

    const float softmax_scale = (scale < 0.0f) ? (1.0f / std::sqrt((float)D)) : scale;

#ifdef PT_USE_CUDA
    if (query.device().type() == c10::DeviceType::CUDA &&
        query.dim() == 4 &&
        at::cuda::can_use_flash_attention(query, key, value))
    {
        return at::cuda::scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale);
    }
#endif

    // Reshape to canonical [B, N, H, D] expected by sdpa_forward_cpu_impl.
    //
    // IMPORTANT: we adopt the [B, N, H, D] 4D convention (same as
    // FlashAttention.cu and the previous CPU implementation), NOT PyTorch's
    // [B, H, N, D]. This keeps binary compatibility with existing callers
    // and with the CUDA FlashAttention kernel (which the user explicitly
    // asked us not to modify).
    //
    //   query.dim() == 2: [N_q, D]           → [1, N_q, 1, D]
    //   query.dim() == 3: [B, N_q, D]        → [B, N_q, 1, D]       (single head)
    //   query.dim() == 4: [B, N_q, H, D]     → as-is (canonical)
    //   query.dim() >= 5: flatten leading dims into B.
    auto to_canonical = [](const Tensor& t) -> Tensor {
        const int64_t nd = t.dim();
        if (nd == 2) {
            return t.unsqueeze(0).unsqueeze(2);  // [N, D] → [1, N, 1, D]
        } else if (nd == 3) {
            return t.unsqueeze(2);               // [B, N, D] → [B, N, 1, D]
        } else if (nd == 4) {
            return t;                            // already [B, N, H, D]
        } else {
            // [..., N, H, D] → [B', N, H, D]  (flatten leading-batch dims)
            int64_t D_loc = t.size(nd - 1);
            int64_t H_loc = t.size(nd - 2);
            int64_t N_loc = t.size(nd - 3);
            int64_t Bp = 1;
            for (int64_t i = 0; i < nd - 3; ++i) Bp *= t.size(i);
            return t.reshape({Bp, N_loc, H_loc, D_loc});
        }
    };

    Tensor qc = to_canonical(query);
    Tensor kc = to_canonical(key);
    Tensor vc = to_canonical(value);
    // Contiguous required by the kernel.
    qc = qc.contiguous();
    kc = kc.contiguous();
    vc = vc.contiguous();

    // Use a fresh seed for dropout each call (matches PyTorch stochastic bwd).
    static std::random_device rd;
    uint64_t seed = (uint64_t)rd() | ((uint64_t)rd() << 32);

    Tensor out = sdpa_forward_cpu_impl(
        qc, kc, vc, attn_mask, dropout_p, is_causal, softmax_scale,
        /*training=*/(dropout_p > 0.0f), seed, nullptr, nullptr);

    // Now reshape output back to caller's convention.
    const int64_t nd = query.dim();
    if (nd == 2) {
        return out.squeeze(2).squeeze(0);    // [1, N_q, 1, D] → [N_q, D]
    } else if (nd == 3) {
        return out.squeeze(2);               // [B, N_q, 1, D] → [B, N_q, D]
    } else if (nd == 4) {
        return out;                          // [B, N_q, H, D]
    } else {
        // [B', N_q, H, D] → [..., N_q, H, D]
        std::vector<int64_t> final_shape = query.sizes().vec();
        final_shape[nd - 3] = out.size(1);   // N_q
        final_shape[nd - 2] = out.size(2);   // H
        final_shape[nd - 1] = out.size(3);   // D
        return out.reshape(final_shape);
    }
}

} // namespace native

// Public at:: alias so user code can call at::scaled_dot_product_attention(q,k,v,...)
inline Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask = Tensor(),
    float dropout_p = 0.0f,
    bool is_causal = false,
    float scale = -1.0f)
{
    return native::scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale);
}

} // namespace at
