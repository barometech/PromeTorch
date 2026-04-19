// ============================================================================
// Attention — top-level dispatcher.
// CPU: standard softmax(Q @ K^T / sqrt(d)) @ V with optional causal mask.
// CUDA: uses at::cuda::scaled_dot_product_attention (FlashAttention) when
//       tensors are on CUDA and shapes match (head_dim in {64,128}); otherwise
//       falls back to CPU math on the host side of the tensor.
// ============================================================================
#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "c10/core/ScalarType.h"

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/FlashAttention.h"
#endif

#include <cmath>

namespace at {
namespace native {

// Q: [B, N_q, H, D]
// K: [B, N_k, H, D]
// V: [B, N_k, H, D]
// out: [B, N_q, H, D]
//
// scale: -1 means 1 / sqrt(head_dim).
inline Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask = Tensor(),
    float dropout_p = 0.0f,
    bool is_causal = false,
    float scale = -1.0f)
{
    PT_CHECK_MSG(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                 "scaled_dot_product_attention: Q/K/V must be 4D [B,N,H,D]");
    const int64_t B = query.size(0);
    const int64_t N_q = query.size(1);
    const int64_t H = query.size(2);
    const int64_t D = query.size(3);
    const int64_t N_k = key.size(1);
    PT_CHECK(key.size(0) == B && value.size(0) == B);
    PT_CHECK(key.size(2) == H && value.size(2) == H);
    PT_CHECK(key.size(3) == D && value.size(3) == D);
    PT_CHECK(value.size(1) == N_k);

    const float softmax_scale = (scale < 0.0f) ? (1.0f / std::sqrt((float)D)) : scale;

#ifdef PT_USE_CUDA
    if (query.device().type() == c10::DeviceType::CUDA &&
        at::cuda::can_use_flash_attention(query, key, value))
    {
        return at::cuda::scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale);
    }
#endif

    // CPU reference: O[b,i,h,:] = sum_j softmax(scale * Q[b,i,h,:] . K[b,j,h,:] + mask) * V[b,j,h,:]
    // All buffers on CPU float32. Dropout ignored (inference / training without dropout for now).
    PT_CHECK_MSG(query.dtype() == c10::ScalarType::Float,
                 "scaled_dot_product_attention CPU path requires float32");

    Tensor q_c = query.is_contiguous() ? query : query.contiguous();
    Tensor k_c = key.is_contiguous() ? key : key.contiguous();
    Tensor v_c = value.is_contiguous() ? value : value.contiguous();

    Tensor out = empty({B, N_q, H, D},
                       TensorOptions().dtype(query.dtype()).device(query.device()));

    const float* Qd = q_c.data_ptr<float>();
    const float* Kd = k_c.data_ptr<float>();
    const float* Vd = v_c.data_ptr<float>();
    float* Od = out.mutable_data_ptr<float>();

    const bool has_mask = attn_mask.defined();
    const float* Md = has_mask ? attn_mask.data_ptr<float>() : nullptr;
    const int64_t mask_dim = has_mask ? attn_mask.dim() : 0;

    // Index helpers (row-major contiguous [B, N, H, D]).
    auto qkv_idx = [&](int64_t b, int64_t n, int64_t h, int64_t d, int64_t N) {
        return ((b * N + n) * H + h) * D + d;
    };

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            // Process each query position
            for (int64_t i = 0; i < N_q; ++i) {
                // 1) Scores s_j = scale * Q[b,i,h,:] . K[b,j,h,:]
                std::vector<float> scores(N_k);
                float max_s = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < N_k; ++j) {
                    float s = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        s += Qd[qkv_idx(b, i, h, d, N_q)] * Kd[qkv_idx(b, j, h, d, N_k)];
                    }
                    s *= softmax_scale;
                    // Causal mask: j > i -> -inf
                    if (is_causal && j > i) s = -std::numeric_limits<float>::infinity();
                    // Additive mask
                    if (has_mask) {
                        // Support broadcasting: mask shape may be [N_q,N_k], [B,N_q,N_k],
                        // or [B,H,N_q,N_k]. Otherwise fall back to assuming last two dims.
                        float m = 0.0f;
                        if (mask_dim == 2) {
                            m = Md[i * N_k + j];
                        } else if (mask_dim == 3) {
                            m = Md[(b * N_q + i) * N_k + j];
                        } else if (mask_dim == 4) {
                            m = Md[((b * H + h) * N_q + i) * N_k + j];
                        }
                        s += m;
                    }
                    scores[j] = s;
                    if (s > max_s) max_s = s;
                }
                // 2) Softmax (numerically stable)
                float sum_exp = 0.0f;
                for (int64_t j = 0; j < N_k; ++j) {
                    float e = (max_s == -std::numeric_limits<float>::infinity())
                                  ? 0.0f : std::exp(scores[j] - max_s);
                    scores[j] = e;
                    sum_exp += e;
                }
                float inv = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
                for (int64_t j = 0; j < N_k; ++j) scores[j] *= inv;
                // 3) Output = sum_j scores[j] * V[b,j,h,:]
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
