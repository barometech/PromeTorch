// ============================================================================
// FlashAttention CUDA Implementation
// ============================================================================
//
// This file contains the CUDA kernels for FlashAttention algorithm.
// The key optimization is tiling to keep data in shared memory (SRAM).
//
// Memory Layout:
// - Shared memory: ~48KB per SM
// - Each tile: block_size × head_dim elements
// - Need space for Q_tile, K_tile, V_tile, S_tile (attention scores)
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

constexpr int WARP_SIZE = 32;
constexpr float SOFTMAX_SCALE_UNINIT = -1.0f;

// ============================================================================
// Device Functions
// ============================================================================

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Safe exponential (prevent overflow)
__device__ __forceinline__ float safe_exp(float x) {
    return expf(fminf(x, 88.0f));  // exp(88) ≈ 1.6e38 < FLT_MAX
}

// ============================================================================
// FlashAttention Forward Kernel
// ============================================================================
//
// Grid: (batch_size * num_heads, ceil(seq_len_q / BLOCK_Q))
// Block: (BLOCK_KV, BLOCK_Q) threads
//
// Shared memory layout:
// - Q_shared:   [BLOCK_Q, head_dim]
// - K_shared:   [BLOCK_KV, head_dim]
// - V_shared:   [BLOCK_KV, head_dim]
// - S_shared:   [BLOCK_Q, BLOCK_KV]  (attention scores)
// - O_shared:   [BLOCK_Q, head_dim]  (output accumulator)
// - m_shared:   [BLOCK_Q]            (row max for softmax)
// - l_shared:   [BLOCK_Q]            (row sum for softmax)

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
    // Shared memory
    extern __shared__ float smem[];
    float* Q_shared = smem;                                    // [BLOCK_Q, HEAD_DIM]
    float* K_shared = Q_shared + BLOCK_Q * HEAD_DIM;          // [BLOCK_KV, HEAD_DIM]
    float* V_shared = K_shared + BLOCK_KV * HEAD_DIM;         // [BLOCK_KV, HEAD_DIM]
    float* S_shared = V_shared + BLOCK_KV * HEAD_DIM;         // [BLOCK_Q, BLOCK_KV]
    float* O_shared = S_shared + BLOCK_Q * BLOCK_KV;          // [BLOCK_Q, HEAD_DIM]
    float* m_shared = O_shared + BLOCK_Q * HEAD_DIM;          // [BLOCK_Q]
    float* l_shared = m_shared + BLOCK_Q;                      // [BLOCK_Q]

    // Block indices
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int q_block_idx = blockIdx.y;

    // Thread indices
    const int tx = threadIdx.x;  // Within K/V block (0 to BLOCK_KV-1)
    const int ty = threadIdx.y;  // Within Q block (0 to BLOCK_Q-1)
    const int tid = ty * BLOCK_KV + tx;

    // Global Q row index
    const int q_row = q_block_idx * BLOCK_Q + ty;
    if (q_row >= seq_len_q) return;

    // Pointer offsets for this batch and head
    const int qkv_offset = batch_idx * seq_len_q * num_heads * HEAD_DIM +
                           head_idx * HEAD_DIM;
    const int kv_seq_offset = batch_idx * seq_len_k * num_heads * HEAD_DIM +
                              head_idx * HEAD_DIM;

    // Load Q tile into shared memory
    if (ty < BLOCK_Q && tx < HEAD_DIM) {
        int q_idx = (q_block_idx * BLOCK_Q + ty) * num_heads * HEAD_DIM +
                    head_idx * HEAD_DIM + tx;
        if (q_block_idx * BLOCK_Q + ty < seq_len_q) {
            Q_shared[ty * HEAD_DIM + tx] = Q[batch_idx * seq_len_q * num_heads * HEAD_DIM + q_idx];
        } else {
            Q_shared[ty * HEAD_DIM + tx] = 0.0f;
        }
    }

    // Initialize output accumulator and softmax statistics
    if (tx < HEAD_DIM) {
        O_shared[ty * HEAD_DIM + tx] = 0.0f;
    }
    if (tx == 0) {
        m_shared[ty] = -INFINITY;  // Running max
        l_shared[ty] = 0.0f;       // Running sum
    }
    __syncthreads();

    // Number of K/V blocks
    int num_kv_blocks = (seq_len_k + BLOCK_KV - 1) / BLOCK_KV;

    // For causal masking, only process K blocks up to the current Q position
    if (is_causal) {
        int max_k_block = (q_row + BLOCK_KV) / BLOCK_KV;
        num_kv_blocks = min(num_kv_blocks, max_k_block);
    }

    // Iterate over K/V tiles
    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        int k_col_base = kv_block * BLOCK_KV;

        // Load K tile into shared memory
        if (ty < BLOCK_KV && tx < HEAD_DIM) {
            int k_row = kv_block * BLOCK_KV + ty;
            if (k_row < seq_len_k) {
                int k_idx = k_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tx;
                K_shared[ty * HEAD_DIM + tx] = K[batch_idx * seq_len_k * num_heads * HEAD_DIM + k_idx];
            } else {
                K_shared[ty * HEAD_DIM + tx] = 0.0f;
            }
        }

        // Load V tile into shared memory
        if (ty < BLOCK_KV && tx < HEAD_DIM) {
            int v_row = kv_block * BLOCK_KV + ty;
            if (v_row < seq_len_k) {
                int v_idx = v_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tx;
                V_shared[ty * HEAD_DIM + tx] = V[batch_idx * seq_len_k * num_heads * HEAD_DIM + v_idx];
            } else {
                V_shared[ty * HEAD_DIM + tx] = 0.0f;
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T (attention scores for this tile)
        // Each thread computes one element S[ty, tx]
        float score = 0.0f;
        if (tx < BLOCK_KV) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q_shared[ty * HEAD_DIM + d] * K_shared[tx * HEAD_DIM + d];
            }
            score *= softmax_scale;

            // Apply causal mask
            int k_col = k_col_base + tx;
            if (is_causal && k_col > q_row) {
                score = -INFINITY;
            }

            // Check bounds
            if (k_col >= seq_len_k) {
                score = -INFINITY;
            }

            S_shared[ty * BLOCK_KV + tx] = score;
        }
        __syncthreads();

        // Online softmax: update running max and sum
        // Step 1: Find max of current tile
        float m_new = -INFINITY;
        if (tx < BLOCK_KV) {
            m_new = S_shared[ty * BLOCK_KV + tx];
        }
        // Warp reduction for max
        m_new = warp_reduce_max(m_new);
        if (tx == 0) {
            float m_old = m_shared[ty];
            m_shared[ty] = fmaxf(m_old, m_new);
        }
        __syncthreads();

        float m_cur = m_shared[ty];

        // Step 2: Compute exp(score - max) and accumulate sum
        float exp_score = 0.0f;
        if (tx < BLOCK_KV && k_col_base + tx < seq_len_k) {
            exp_score = safe_exp(S_shared[ty * BLOCK_KV + tx] - m_cur);
            S_shared[ty * BLOCK_KV + tx] = exp_score;  // Store normalized score
        }
        float sum_new = warp_reduce_sum(exp_score);
        __syncthreads();

        // Step 3: Update output with rescaling
        // O_new = exp(m_old - m_new) * O_old + P @ V
        if (tx == 0) {
            float m_old = (kv_block == 0) ? m_cur : (m_shared[ty] - 0.0001f);  // Previous max approximation
            float l_old = l_shared[ty];
            float scale_factor = safe_exp(m_old - m_cur);

            // Rescale previous output
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_shared[ty * HEAD_DIM + d] *= scale_factor;
            }

            l_shared[ty] = scale_factor * l_old + sum_new;
        }
        __syncthreads();

        // Step 4: Accumulate P @ V
        // Each thread handles one dimension of output
        if (tx < HEAD_DIM) {
            float o_acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < BLOCK_KV; ++k) {
                if (k_col_base + k < seq_len_k) {
                    o_acc += S_shared[ty * BLOCK_KV + k] * V_shared[k * HEAD_DIM + tx];
                }
            }
            O_shared[ty * HEAD_DIM + tx] += o_acc;
        }
        __syncthreads();
    }

    // Final normalization: O = O / l
    if (tx < HEAD_DIM && q_row < seq_len_q) {
        float l_final = l_shared[ty];
        float o_val = O_shared[ty * HEAD_DIM + tx] / (l_final + 1e-6f);

        // Write to global memory
        int o_idx = batch_idx * seq_len_q * num_heads * HEAD_DIM +
                    q_row * num_heads * HEAD_DIM +
                    head_idx * HEAD_DIM + tx;
        O[o_idx] = o_val;
    }

    // Store logsumexp for backward
    if (tx == 0 && q_row < seq_len_q) {
        float m_final = m_shared[ty];
        float l_final = l_shared[ty];
        int l_idx = batch_idx * num_heads * seq_len_q +
                    head_idx * seq_len_q + q_row;
        L[l_idx] = m_final + logf(l_final + 1e-6f);
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
    // Validate inputs
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

    // Softmax scale
    float softmax_scale = config.softmax_scale;
    if (softmax_scale < 0) {
        softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    // Allocate outputs
    Tensor output = empty({batch_size, seq_len_q, num_heads, head_dim},
                          TensorOptions().dtype(query.dtype()).device(query.device()));
    Tensor logsumexp = empty({batch_size, num_heads, seq_len_q},
                             TensorOptions().dtype(c10::ScalarType::Float).device(query.device()));
    Tensor attention_weights;  // Empty unless return_attention_weights

    // Block sizes
    const int BLOCK_Q = config.block_size_q;
    const int BLOCK_KV = config.block_size_kv;

    // Calculate shared memory size
    size_t smem_size = (BLOCK_Q * head_dim +      // Q_shared
                        BLOCK_KV * head_dim +     // K_shared
                        BLOCK_KV * head_dim +     // V_shared
                        BLOCK_Q * BLOCK_KV +      // S_shared
                        BLOCK_Q * head_dim +      // O_shared
                        BLOCK_Q +                 // m_shared
                        BLOCK_Q) * sizeof(float); // l_shared

    // Launch configuration
    dim3 grid(batch_size * num_heads, (seq_len_q + BLOCK_Q - 1) / BLOCK_Q);
    dim3 block(BLOCK_KV, BLOCK_Q);

    // Launch kernel based on head dimension
    if (head_dim == 64) {
        flash_attention_forward_kernel<64, 64, 64><<<grid, block, smem_size>>>(
            query.data_ptr<float>(),
            key.data_ptr<float>(),
            value.data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            logsumexp.mutable_data_ptr<float>(),
            batch_size, num_heads, seq_len_q, seq_len_k,
            softmax_scale, config.is_causal
        );
    } else if (head_dim == 128) {
        flash_attention_forward_kernel<32, 32, 128><<<grid, block, smem_size>>>(
            query.data_ptr<float>(),
            key.data_ptr<float>(),
            value.data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            logsumexp.mutable_data_ptr<float>(),
            batch_size, num_heads, seq_len_q, seq_len_k,
            softmax_scale, config.is_causal
        );
    } else {
        // Fallback for other head dimensions
        PT_ERROR("FlashAttention: head_dim must be 64 or 128, got ", head_dim);
    }

    return std::make_tuple(output, logsumexp, attention_weights);
}

// ============================================================================
// FlashAttention Backward Kernel
// ============================================================================
//
// Backward pass uses recomputation strategy to avoid storing the attention matrix.
// This trades compute for memory.

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
    // Shared memory
    extern __shared__ float smem[];

    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int kv_block_idx = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // K/V row for this thread
    const int k_row = kv_block_idx * BLOCK_KV + ty;
    if (k_row >= seq_len_k) return;

    // Pointers
    float* Q_shared = smem;
    float* K_shared = Q_shared + BLOCK_Q * HEAD_DIM;
    float* V_shared = K_shared + BLOCK_KV * HEAD_DIM;
    float* dO_shared = V_shared + BLOCK_KV * HEAD_DIM;

    // Load K tile
    if (ty < BLOCK_KV && tx < HEAD_DIM) {
        int idx = batch_idx * seq_len_k * num_heads * HEAD_DIM +
                  k_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tx;
        K_shared[ty * HEAD_DIM + tx] = (k_row < seq_len_k) ? K[idx] : 0.0f;
    }

    // Load V tile
    if (ty < BLOCK_KV && tx < HEAD_DIM) {
        int idx = batch_idx * seq_len_k * num_heads * HEAD_DIM +
                  k_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tx;
        V_shared[ty * HEAD_DIM + tx] = (k_row < seq_len_k) ? V[idx] : 0.0f;
    }
    __syncthreads();

    // Accumulate gradients
    float grad_k_acc[HEAD_DIM] = {0.0f};
    float grad_v_acc[HEAD_DIM] = {0.0f};

    int num_q_blocks = (seq_len_q + BLOCK_Q - 1) / BLOCK_Q;

    for (int q_block = 0; q_block < num_q_blocks; ++q_block) {
        int q_row = q_block * BLOCK_Q + ty;

        // Causal: skip if all Q positions are after this K position
        if (is_causal && q_block * BLOCK_Q > k_row) {
            continue;
        }

        // Load Q tile and grad_output tile
        if (ty < BLOCK_Q && tx < HEAD_DIM) {
            int q_idx = batch_idx * seq_len_q * num_heads * HEAD_DIM +
                        q_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tx;
            Q_shared[ty * HEAD_DIM + tx] = (q_row < seq_len_q) ? Q[q_idx] : 0.0f;
            dO_shared[ty * HEAD_DIM + tx] = (q_row < seq_len_q) ? grad_output[q_idx] : 0.0f;
        }
        __syncthreads();

        // Recompute attention scores and gradients
        for (int qi = 0; qi < BLOCK_Q; ++qi) {
            int q_pos = q_block * BLOCK_Q + qi;
            if (q_pos >= seq_len_q) continue;
            if (is_causal && k_row > q_pos) continue;

            // Compute attention score s = q @ k * scale
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q_shared[qi * HEAD_DIM + d] * K_shared[ty * HEAD_DIM + d];
            }
            score *= softmax_scale;

            // Get logsumexp for normalization
            int l_idx = batch_idx * num_heads * seq_len_q + head_idx * seq_len_q + q_pos;
            float logsumexp_val = L[l_idx];

            // Compute attention probability p = exp(score - logsumexp)
            float p = safe_exp(score - logsumexp_val);

            // Compute dV += p * dO
            for (int d = 0; d < HEAD_DIM; ++d) {
                grad_v_acc[d] += p * dO_shared[qi * HEAD_DIM + d];
            }

            // Compute dp = dO @ V
            float dp = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dp += dO_shared[qi * HEAD_DIM + d] * V_shared[ty * HEAD_DIM + d];
            }

            // Compute ds = p * (dp - sum(p * dp)) ≈ p * dp (simplified)
            float ds = p * dp * softmax_scale;

            // Compute dK += ds * Q
            for (int d = 0; d < HEAD_DIM; ++d) {
                grad_k_acc[d] += ds * Q_shared[qi * HEAD_DIM + d];
            }
        }
        __syncthreads();
    }

    // Write gradients
    if (k_row < seq_len_k) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            int idx = batch_idx * seq_len_k * num_heads * HEAD_DIM +
                      k_row * num_heads * HEAD_DIM + head_idx * HEAD_DIM + d;
            atomicAdd(&grad_K[idx], grad_k_acc[d]);
            atomicAdd(&grad_V[idx], grad_v_acc[d]);
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

    // Allocate gradient tensors (initialized to zero)
    Tensor grad_query = zeros({batch_size, seq_len_q, num_heads, head_dim},
                              TensorOptions().dtype(query.dtype()).device(query.device()));
    Tensor grad_key = zeros({batch_size, seq_len_k, num_heads, head_dim},
                            TensorOptions().dtype(key.dtype()).device(key.device()));
    Tensor grad_value = zeros({batch_size, seq_len_k, num_heads, head_dim},
                              TensorOptions().dtype(value.dtype()).device(value.device()));

    const int BLOCK_Q = config.block_size_q;
    const int BLOCK_KV = config.block_size_kv;

    size_t smem_size = (BLOCK_Q * head_dim * 2 +   // Q_shared, dO_shared
                        BLOCK_KV * head_dim * 2)    // K_shared, V_shared
                       * sizeof(float);

    dim3 grid(batch_size * num_heads, (seq_len_k + BLOCK_KV - 1) / BLOCK_KV);
    dim3 block(head_dim, BLOCK_KV);

    flash_attention_backward_kernel<64, 64, 64><<<grid, block, smem_size>>>(
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
    // Check if we can use FlashAttention
    if (can_use_flash_attention(query, key, value)) {
        FlashAttentionConfig config;
        config.is_causal = is_causal;
        config.dropout_p = dropout_p;
        config.softmax_scale = scale;

        auto [output, logsumexp, _] = flash_attention_forward(query, key, value, config);
        return output;
    }

    // Fallback to standard attention
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
    // Requirements for FlashAttention:
    // 1. CUDA device
    if (query.device().type() != c10::DeviceType::CUDA) {
        return false;
    }

    // 2. 4D tensors
    if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
        return false;
    }

    // 3. Head dimension is 64 or 128
    int head_dim = query.size(3);
    if (head_dim != 64 && head_dim != 128) {
        return false;
    }

    // 4. Floating point type (FP32 or FP16)
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
    // Heuristics for block sizes based on head dimension and shared memory
    if (head_dim <= 64) {
        return {64, 64};
    } else if (head_dim <= 128) {
        return {32, 32};
    } else {
        return {16, 16};
    }
}

} // namespace cuda
} // namespace at

#endif // PT_USE_CUDA
