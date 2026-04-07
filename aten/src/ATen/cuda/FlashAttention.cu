// ============================================================================
// FlashAttention CUDA Implementation
// ============================================================================
//
// This file contains the CUDA kernels for FlashAttention algorithm.
// The key optimization is tiling to keep data in shared memory (SRAM).
//
// ============================================================================

#ifdef PT_USE_CUDA

#include "aten/src/ATen/cuda/FlashAttention.h"
#include "c10/cuda/CUDAAllocator.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

namespace at {
namespace cuda {

// ============================================================================
// Constants
// ============================================================================

// FIX: WARP_SIZE macro for CUDA (32) vs AMD HIP (64)
#ifdef __HIP_PLATFORM_AMD__
constexpr int WARP_SIZE = 64;
constexpr unsigned WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;
#else
constexpr int WARP_SIZE = 32;
constexpr unsigned WARP_MASK = 0xFFFFFFFF;
#endif
constexpr float SOFTMAX_SCALE_UNINIT = -1.0f;

// ============================================================================
// Device Functions
// ============================================================================

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(WARP_MASK, val, offset));
    }
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(WARP_MASK, val, offset);
    }
    return val;
}

// Safe exponential (prevent overflow)
__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(x, 88.0f));  // exp(88) ~ 1.6e38 < FLT_MAX
}

// ============================================================================
// FlashAttention Forward Kernel
// ============================================================================
//
// Grid:  (batch_size * num_heads, ceil(seq_len_q / BLOCK_Q))
// Block: (BLOCK_KV, BLOCK_Q) threads   [BLOCK_Q * BLOCK_KV <= 1024]
//
// tx in [0, BLOCK_KV): used for K/V columns in dot products, and loops over HEAD_DIM
// ty in [0, BLOCK_Q):  one per Q row in the tile
//
// Shared memory layout:
//   Q_shared:  [BLOCK_Q][HEAD_DIM]
//   K_shared:  [BLOCK_KV][HEAD_DIM]
//   V_shared:  [BLOCK_KV][HEAD_DIM]
//   S_shared:  [BLOCK_Q][BLOCK_KV]   (attention scores / probabilities)
//   O_shared:  [BLOCK_Q][HEAD_DIM]   (output accumulator)
//   m_shared:  [BLOCK_Q]             (row max for online softmax)
//   l_shared:  [BLOCK_Q]             (row sum for online softmax)

template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,     // [B, N_q, H, D]
    const float* __restrict__ K,     // [B, N_k, H, D]
    const float* __restrict__ V,     // [B, N_k, H, D]
    float* __restrict__ O,           // [B, N_q, H, D]
    float* __restrict__ L,           // [B, H, N_q] logsumexp for backward
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float softmax_scale,
    const bool is_causal
) {
    extern __shared__ float smem[];
    float* Q_shared = smem;
    float* K_shared = Q_shared + BLOCK_Q * HEAD_DIM;
    float* V_shared = K_shared + BLOCK_KV * HEAD_DIM;
    float* S_shared = V_shared + BLOCK_KV * HEAD_DIM;
    float* O_shared = S_shared + BLOCK_Q * BLOCK_KV;
    float* m_shared = O_shared + BLOCK_Q * HEAD_DIM;
    float* l_shared = m_shared + BLOCK_Q;

    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int q_block_idx = blockIdx.y;

    const int tx = threadIdx.x;  // 0..BLOCK_KV-1
    const int ty = threadIdx.y;  // 0..BLOCK_Q-1

    const int q_row = q_block_idx * BLOCK_Q + ty;
    const bool valid_q = (q_row < seq_len_q);

    const int q_batch_offset = batch_idx * seq_len_q * num_heads * HEAD_DIM;
    const int kv_batch_offset = batch_idx * seq_len_k * num_heads * HEAD_DIM;

    // Load Q tile: use loop since HEAD_DIM may exceed BLOCK_KV
    for (int d = tx; d < HEAD_DIM; d += BLOCK_KV) {
        if (valid_q) {
            Q_shared[ty * HEAD_DIM + d] = Q[q_batch_offset +
                q_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM + d];
        } else {
            Q_shared[ty * HEAD_DIM + d] = 0.0f;
        }
    }

    // Initialize output accumulator
    for (int d = tx; d < HEAD_DIM; d += BLOCK_KV) {
        O_shared[ty * HEAD_DIM + d] = 0.0f;
    }

    if (tx == 0) {
        m_shared[ty] = -INFINITY;
        l_shared[ty] = 0.0f;
    }
    __syncthreads();

    int num_kv_blocks = (seq_len_k + BLOCK_KV - 1) / BLOCK_KV;
    if (is_causal && valid_q) {
        int max_k_block = (q_row + BLOCK_KV) / BLOCK_KV;
        num_kv_blocks = min(num_kv_blocks, max_k_block);
    }

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int k_col_base = kv_block * BLOCK_KV;

        // Cooperative load of K and V tiles
        {
            const int total_threads = BLOCK_Q * BLOCK_KV;
            const int total_elems = BLOCK_KV * HEAD_DIM;
            const int tid = ty * BLOCK_KV + tx;
            for (int idx = tid; idx < total_elems; idx += total_threads) {
                int kr = idx / HEAD_DIM;
                int kd = idx % HEAD_DIM;
                int global_kr = k_col_base + kr;
                if (global_kr < seq_len_k) {
                    int g_idx = kv_batch_offset + global_kr * num_heads * HEAD_DIM +
                                head_idx * HEAD_DIM + kd;
                    K_shared[kr * HEAD_DIM + kd] = K[g_idx];
                    V_shared[kr * HEAD_DIM + kd] = V[g_idx];
                } else {
                    K_shared[kr * HEAD_DIM + kd] = 0.0f;
                    V_shared[kr * HEAD_DIM + kd] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute S[ty][tx] = dot(Q_shared[ty], K_shared[tx]) * scale
        {
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q_shared[ty * HEAD_DIM + d] * K_shared[tx * HEAD_DIM + d];
            }
            score *= softmax_scale;

            int k_col = k_col_base + tx;
            if ((is_causal && k_col > q_row) || k_col >= seq_len_k || !valid_q) {
                score = -INFINITY;
            }
            S_shared[ty * BLOCK_KV + tx] = score;
        }
        __syncthreads();

        // Online softmax update (thread tx==0 handles the full row for ty)
        // This is serial over BLOCK_KV but BLOCK_KV is small (32).
        if (tx == 0) {
            // Save old max BEFORE computing new max
            float m_old = m_shared[ty];

            // Find max of current tile
            float m_tile = -INFINITY;
            for (int i = 0; i < BLOCK_KV; ++i) {
                m_tile = fmaxf(m_tile, S_shared[ty * BLOCK_KV + i]);
            }

            // New global max
            float m_new = fmaxf(m_old, m_tile);
            m_shared[ty] = m_new;

            // Rescale factor for previous accumulator
            float scale_old = safe_exp(m_old - m_new);

            // Compute exp(score - m_new) for this tile, store back in S_shared
            float sum_tile = 0.0f;
            for (int i = 0; i < BLOCK_KV; ++i) {
                float e = safe_exp(S_shared[ty * BLOCK_KV + i] - m_new);
                S_shared[ty * BLOCK_KV + i] = e;
                sum_tile += e;
            }

            // Rescale previous output
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_shared[ty * HEAD_DIM + d] *= scale_old;
            }

            // Update running sum
            l_shared[ty] = scale_old * l_shared[ty] + sum_tile;
        }
        __syncthreads();

        // Accumulate P @ V into O_shared
        // Each thread handles a subset of HEAD_DIM
        for (int d = tx; d < HEAD_DIM; d += BLOCK_KV) {
            float o_acc = 0.0f;
            for (int k = 0; k < BLOCK_KV; ++k) {
                o_acc += S_shared[ty * BLOCK_KV + k] * V_shared[k * HEAD_DIM + d];
            }
            O_shared[ty * HEAD_DIM + d] += o_acc;
        }
        __syncthreads();
    }

    // Final normalization: O = O / l
    if (valid_q) {
        float l_final = l_shared[ty];
        float inv_l = 1.0f / (l_final + 1e-6f);

        for (int d = tx; d < HEAD_DIM; d += BLOCK_KV) {
            int o_idx = q_batch_offset + q_row * num_heads * HEAD_DIM +
                        head_idx * HEAD_DIM + d;
            O[o_idx] = O_shared[ty * HEAD_DIM + d] * inv_l;
        }

        if (tx == 0) {
            int l_idx = batch_idx * num_heads * seq_len_q +
                        head_idx * seq_len_q + q_row;
            L[l_idx] = m_shared[ty] + logf(l_final + 1e-6f);
        }
    }
}

// ============================================================================
// FlashAttention Forward Host Function
// ============================================================================

std::tuple<Tensor, Tensor, Tensor> flash_attention_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const FlashAttentionConfig& config
) {
    PT_CHECK(query.dim() == 4, "Query must be 4D [B, N_q, H, D]");
    PT_CHECK(key.dim() == 4, "Key must be 4D [B, N_k, H, D]");
    PT_CHECK(value.dim() == 4, "Value must be 4D [B, N_k, H, D]");
    PT_CHECK(query.device().type() == c10::DeviceType::CUDA, "Query must be on CUDA");

    const int batch_size = query.size(0);
    const int seq_len_q = query.size(1);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);
    const int seq_len_k = key.size(1);

    PT_CHECK(key.size(0) == batch_size && value.size(0) == batch_size, "Batch size mismatch");
    PT_CHECK(key.size(2) == num_heads && value.size(2) == num_heads, "Head count mismatch");
    PT_CHECK(key.size(3) == head_dim && value.size(3) == head_dim, "Head dim mismatch");
    PT_CHECK(key.size(1) == value.size(1), "K and V sequence length mismatch");

    float softmax_scale = config.softmax_scale;
    if (softmax_scale < 0) {
        softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    Tensor output = empty({batch_size, seq_len_q, num_heads, head_dim},
                          TensorOptions().dtype(query.dtype()).device(query.device()));
    Tensor logsumexp = empty({batch_size, num_heads, seq_len_q},
                             TensorOptions().dtype(c10::ScalarType::Float).device(query.device()));
    Tensor attention_weights;

    // Block sizes: BLOCK_Q * BLOCK_KV <= 1024
    // head_dim=64:  BQ=32, BKV=32 -> 1024 threads
    // head_dim=128: BQ=16, BKV=32 -> 512 threads

    auto launch = [&](auto bq_tag, auto bkv_tag, auto hd_tag) {
        constexpr int BQ = decltype(bq_tag)::value;
        constexpr int BKV = decltype(bkv_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;

        size_t smem_size = (BQ * HD +       // Q_shared
                            BKV * HD +      // K_shared
                            BKV * HD +      // V_shared
                            BQ * BKV +      // S_shared
                            BQ * HD +       // O_shared
                            BQ +            // m_shared
                            BQ) * sizeof(float);

        dim3 grid(batch_size * num_heads, (seq_len_q + BQ - 1) / BQ);
        dim3 block(BKV, BQ);

        flash_attention_forward_kernel<BQ, BKV, HD><<<grid, block, smem_size>>>(
            query.data_ptr<float>(),
            key.data_ptr<float>(),
            value.data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            logsumexp.mutable_data_ptr<float>(),
            batch_size, num_heads, seq_len_q, seq_len_k,
            softmax_scale, config.is_causal
        );
    };

    if (head_dim == 64) {
        launch(std::integral_constant<int, 32>{},
               std::integral_constant<int, 32>{},
               std::integral_constant<int, 64>{});
    } else if (head_dim == 128) {
        launch(std::integral_constant<int, 16>{},
               std::integral_constant<int, 32>{},
               std::integral_constant<int, 128>{});
    } else {
        PT_ERROR("FlashAttention: head_dim must be 64 or 128, got ", head_dim);
    }

    return std::make_tuple(output, logsumexp, attention_weights);
}

// ============================================================================
// FlashAttention Backward Kernel
// ============================================================================
//
// Backward pass: recompute attention scores from Q, K, V and logsumexp.
//
// Grid:  (batch_size * num_heads, ceil(seq_len_k / BLOCK_KV))
// Block: (BLOCK_KV) -- 1D block, one thread per K/V row
//
// Each thread owns one K/V row and accumulates grad_K[k_row] and grad_V[k_row].
// It loops over all Q rows (in BLOCK_Q-sized chunks), recomputing P and
// applying the correct softmax backward formula:
//   ds_ij = P_ij * (dP_ij - D_i)
// where dP_ij = dO_i . V_j  and  D_i = dO_i . O_i
//
// grad_Q is accumulated via atomicAdd (multiple K threads write to same Q row).

template<int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void flash_attention_backward_kernel(
    const float* __restrict__ grad_output,  // [B, N_q, H, D]
    const float* __restrict__ Q,            // [B, N_q, H, D]
    const float* __restrict__ K,            // [B, N_k, H, D]
    const float* __restrict__ V,            // [B, N_k, H, D]
    const float* __restrict__ O,            // [B, N_q, H, D]
    const float* __restrict__ L,            // [B, H, N_q]
    float* __restrict__ grad_Q,             // [B, N_q, H, D]
    float* __restrict__ grad_K,             // [B, N_k, H, D]
    float* __restrict__ grad_V,             // [B, N_k, H, D]
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const float softmax_scale,
    const bool is_causal
) {
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int kv_block_idx = blockIdx.y;

    const int tx = threadIdx.x;  // 0..BLOCK_KV-1

    const int k_row = kv_block_idx * BLOCK_KV + tx;
    const bool valid_k = (k_row < seq_len_k);

    const int q_batch_offset = batch_idx * seq_len_q * num_heads * HEAD_DIM;
    const int kv_batch_offset = batch_idx * seq_len_k * num_heads * HEAD_DIM;
    const int l_batch_offset = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q;

    // Load K and V vectors for this thread's row into registers
    float k_vec[HEAD_DIM];
    float v_vec[HEAD_DIM];
    if (valid_k) {
        int base = kv_batch_offset + k_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            k_vec[d] = K[base + d];
            v_vec[d] = V[base + d];
        }
    } else {
        for (int d = 0; d < HEAD_DIM; ++d) {
            k_vec[d] = 0.0f;
            v_vec[d] = 0.0f;
        }
    }

    // Gradient accumulators
    float grad_k_acc[HEAD_DIM];
    float grad_v_acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) {
        grad_k_acc[d] = 0.0f;
        grad_v_acc[d] = 0.0f;
    }

    const int num_q_blocks = (seq_len_q + BLOCK_Q - 1) / BLOCK_Q;

    for (int q_block = 0; q_block < num_q_blocks; ++q_block) {
        // Causal skip: if last Q in block < k_row, all positions are masked
        if (is_causal && (q_block + 1) * BLOCK_Q - 1 < k_row) {
            continue;
        }

        for (int qi = 0; qi < BLOCK_Q; ++qi) {
            int q_pos = q_block * BLOCK_Q + qi;
            if (q_pos >= seq_len_q) continue;
            if (is_causal && k_row > q_pos) continue;
            if (!valid_k) continue;

            int q_base = q_batch_offset + q_pos * num_heads * HEAD_DIM + head_idx * HEAD_DIM;

            // Recompute attention score: s = Q[q_pos] . K[k_row] * scale
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q[q_base + d] * k_vec[d];
            }
            score *= softmax_scale;

            // Recompute attention probability: p = exp(s - logsumexp[q_pos])
            float logsumexp_val = L[l_batch_offset + q_pos];
            float p = safe_exp(score - logsumexp_val);

            // dP = dO[q_pos] . V[k_row]
            float dp = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dp += grad_output[q_base + d] * v_vec[d];
            }

            // D_i = dO[q_pos] . O[q_pos]  (needed for softmax backward)
            float Di = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                Di += grad_output[q_base + d] * O[q_base + d];
            }

            // Correct softmax backward: ds = P * (dP - D_i)
            float ds = p * (dp - Di);

            // grad_V[k_row] += P * dO[q_pos]
            for (int d = 0; d < HEAD_DIM; ++d) {
                grad_v_acc[d] += p * grad_output[q_base + d];
            }

            // grad_K[k_row] += ds * scale * Q[q_pos]
            for (int d = 0; d < HEAD_DIM; ++d) {
                grad_k_acc[d] += ds * softmax_scale * Q[q_base + d];
            }

            // grad_Q[q_pos] += ds * scale * K[k_row]  (atomicAdd: multiple K threads write)
            for (int d = 0; d < HEAD_DIM; ++d) {
                atomicAdd(&grad_Q[q_base + d], ds * softmax_scale * k_vec[d]);
            }
        }
    }

    // Write accumulated grad_K and grad_V
    if (valid_k) {
        int base = kv_batch_offset + k_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            atomicAdd(&grad_K[base + d], grad_k_acc[d]);
            atomicAdd(&grad_V[base + d], grad_v_acc[d]);
        }
    }
}

// ============================================================================
// FlashAttention Backward Host Function
// ============================================================================

std::tuple<Tensor, Tensor, Tensor> flash_attention_backward(
    const Tensor& grad_output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& output,
    const Tensor& logsumexp,
    const FlashAttentionConfig& config
) {
    const int batch_size = query.size(0);
    const int seq_len_q = query.size(1);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);
    const int seq_len_k = key.size(1);

    float softmax_scale = config.softmax_scale;
    if (softmax_scale < 0) {
        softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    // Zero-initialized for atomicAdd
    Tensor grad_query = zeros({batch_size, seq_len_q, num_heads, head_dim},
                              TensorOptions().dtype(query.dtype()).device(query.device()));
    Tensor grad_key = zeros({batch_size, seq_len_k, num_heads, head_dim},
                            TensorOptions().dtype(key.dtype()).device(key.device()));
    Tensor grad_value = zeros({batch_size, seq_len_k, num_heads, head_dim},
                              TensorOptions().dtype(value.dtype()).device(value.device()));

    constexpr int BKV = 32;  // 1D block, 32 threads -- well under 1024

    if (head_dim == 64) {
        constexpr int BQ = 32;
        constexpr int HD = 64;

        dim3 grid(batch_size * num_heads, (seq_len_k + BKV - 1) / BKV);
        dim3 block(BKV);

        flash_attention_backward_kernel<BQ, BKV, HD><<<grid, block, 0>>>(
            grad_output.data_ptr<float>(),
            query.data_ptr<float>(),
            key.data_ptr<float>(),
            value.data_ptr<float>(),
            output.data_ptr<float>(),
            logsumexp.data_ptr<float>(),
            grad_query.mutable_data_ptr<float>(),
            grad_key.mutable_data_ptr<float>(),
            grad_value.mutable_data_ptr<float>(),
            batch_size, num_heads, seq_len_q, seq_len_k,
            softmax_scale, config.is_causal
        );
    } else if (head_dim == 128) {
        constexpr int BQ = 16;
        constexpr int HD = 128;

        dim3 grid(batch_size * num_heads, (seq_len_k + BKV - 1) / BKV);
        dim3 block(BKV);

        flash_attention_backward_kernel<BQ, BKV, HD><<<grid, block, 0>>>(
            grad_output.data_ptr<float>(),
            query.data_ptr<float>(),
            key.data_ptr<float>(),
            value.data_ptr<float>(),
            output.data_ptr<float>(),
            logsumexp.data_ptr<float>(),
            grad_query.mutable_data_ptr<float>(),
            grad_key.mutable_data_ptr<float>(),
            grad_value.mutable_data_ptr<float>(),
            batch_size, num_heads, seq_len_q, seq_len_k,
            softmax_scale, config.is_causal
        );
    } else {
        PT_ERROR("FlashAttention backward: head_dim must be 64 or 128, got ", head_dim);
    }

    return std::make_tuple(grad_query, grad_key, grad_value);
}

// ============================================================================
// Scaled Dot-Product Attention (high-level API)
// ============================================================================

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask,
    float dropout_p,
    bool is_causal,
    float scale
) {
    if (can_use_flash_attention(query, key, value)) {
        FlashAttentionConfig config;
        config.is_causal = is_causal;
        config.dropout_p = dropout_p;
        config.softmax_scale = scale;

        auto [output, logsumexp, _] = flash_attention_forward(query, key, value, config);
        return output;
    }

    PT_ERROR("Standard attention fallback not implemented");
}

// ============================================================================
// Utility Functions
// ============================================================================

bool can_use_flash_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
) {
    if (query.device().type() != c10::DeviceType::CUDA) {
        return false;
    }
    if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
        return false;
    }
    int head_dim = query.size(3);
    if (head_dim != 64 && head_dim != 128) {
        return false;
    }
    auto dtype = query.dtype();
    if (dtype != c10::ScalarType::Float && dtype != c10::ScalarType::Half) {
        return false;
    }
    return true;
}

std::pair<int, int> get_flash_attention_block_sizes(
    int head_dim,
    int seq_len,
    c10::ScalarType dtype
) {
    if (head_dim <= 64) {
        return {32, 32};
    } else if (head_dim <= 128) {
        return {16, 32};
    } else {
        return {16, 16};
    }
}

} // namespace cuda
} // namespace at

#endif // PT_USE_CUDA
