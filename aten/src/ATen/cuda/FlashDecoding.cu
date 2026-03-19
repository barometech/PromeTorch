// ============================================================================
// Flash-Decoding + Fused Decode Kernels for PromeTorch
// ============================================================================
// Flash-decoding: split KV cache across thread blocks for parallel attention
// Fused kernels: reduce kernel launch count from ~14 to ~4 per layer
//
// Reference: Dao et al. "Flash-Decoding" (2023)
// Key idea: for decode (seq_len_q=1), split KV across blocks, each computes
// partial softmax(Q*K^T)*V, then reduce using log-sum-exp trick.

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "aten/src/ATen/cuda/CUDAOps.h"

namespace at {
namespace cuda {

// ============================================================================
// Flash-Decode Kernel
// ============================================================================
// Grid: (num_splits, num_heads)
// Each block processes a chunk of KV cache for one head.
// Phase 1: compute partial_O and partial_lse per block
// Phase 2: reduce across splits (separate kernel or in-block if single split)
//
// KV_CHUNK_SIZE: number of KV entries each block processes
// Shared memory: Q vector (head_dim) + scores (KV_CHUNK_SIZE)

static constexpr int KV_CHUNK_SIZE = 256;

__global__ void flash_decode_partial_kernel(
    const float* __restrict__ Q,          // [n_heads * head_dim]
    const float* __restrict__ K_cache,    // [total_seq, n_kv_heads * head_dim]
    const float* __restrict__ V_cache,    // [total_seq, n_kv_heads * head_dim]
    float* __restrict__ partial_O,        // [num_splits, n_heads, head_dim]
    float* __restrict__ partial_lse,      // [num_splits, n_heads]
    float* __restrict__ partial_max,      // [num_splits, n_heads]
    int n_heads, int n_kv_heads, int head_dim,
    int total_seq, float scale)
{
    int split_idx = blockIdx.x;     // which KV chunk
    int head_idx = blockIdx.y;      // which query head
    int tid = threadIdx.x;
    int heads_per_group = n_heads / n_kv_heads;
    int kv_head = head_idx / heads_per_group;  // GQA mapping
    int kv_dim = n_kv_heads * head_dim;

    // KV range for this block
    int kv_start = split_idx * KV_CHUNK_SIZE;
    int kv_end = kv_start + KV_CHUNK_SIZE;
    if (kv_end > total_seq) kv_end = total_seq;
    int chunk_len = kv_end - kv_start;

    if (chunk_len <= 0) {
        // This split has no work — write sentinel values
        if (tid == 0) {
            int idx = split_idx * n_heads + head_idx;
            partial_lse[idx] = 0.0f;
            partial_max[idx] = -FLT_MAX;
        }
        if (tid < head_dim) {
            partial_O[(split_idx * n_heads + head_idx) * head_dim + tid] = 0.0f;
        }
        return;
    }

    // Shared memory layout:
    // [0 .. head_dim-1]: Q vector for this head
    // [head_dim .. head_dim + KV_CHUNK_SIZE - 1]: attention scores
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* scores = smem + head_dim;

    // Load Q into shared memory
    const float* q_head = Q + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = q_head[d];
    }
    __syncthreads();

    // Step 1: Compute Q*K^T scores for this chunk
    for (int t = tid; t < chunk_len; t += blockDim.x) {
        int kv_pos = kv_start + t;
        const float* k_vec = K_cache + kv_pos * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_shared[d] * k_vec[d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Step 2: Find max for numerical stability (parallel reduction)
    // Use thread 0 for simplicity — chunk_len <= KV_CHUNK_SIZE = 256
    __shared__ float s_max;
    __shared__ float s_sum;
    if (tid == 0) {
        float m = -FLT_MAX;
        for (int t = 0; t < chunk_len; t++) {
            if (scores[t] > m) m = scores[t];
        }
        s_max = m;
    }
    __syncthreads();

    // Step 3: exp(score - max) and sum
    for (int t = tid; t < chunk_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - s_max);
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int t = 0; t < chunk_len; t++) {
            sum += scores[t];
        }
        s_sum = sum;
    }
    __syncthreads();

    // Step 4: Compute weighted V sum (partial output)
    int out_base = (split_idx * n_heads + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < chunk_len; t++) {
            int kv_pos = kv_start + t;
            acc += scores[t] * V_cache[kv_pos * kv_dim + kv_head * head_dim + d];
        }
        partial_O[out_base + d] = acc;  // unnormalized: sum(softmax_unnorm * V)
    }

    // Store max and sum for cross-block reduction
    if (tid == 0) {
        int idx = split_idx * n_heads + head_idx;
        partial_max[idx] = s_max;
        partial_lse[idx] = s_sum;   // sum of exp(score - max)
    }
}

// ============================================================================
// Flash-Decode Reduce Kernel
// ============================================================================
// Combines partial results from all splits using log-sum-exp trick:
//   O = sum_i (exp(max_i - global_max) * partial_sum_i * partial_O_i) / total_sum
// Grid: (n_heads,)  Block: (head_dim,) or (256,)

__global__ void flash_decode_reduce_kernel(
    const float* __restrict__ partial_O,    // [num_splits, n_heads, head_dim]
    const float* __restrict__ partial_lse,  // [num_splits, n_heads]
    const float* __restrict__ partial_max,  // [num_splits, n_heads]
    float* __restrict__ O,                  // [n_heads * head_dim]
    int n_heads, int head_dim, int num_splits)
{
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Step 1: Find global max across all splits (thread 0)
    __shared__ float global_max;
    __shared__ float total_sum;
    // Scratch for per-split rescaled sums
    extern __shared__ float smem[];
    // smem[0..num_splits-1]: rescaled sums

    if (tid == 0) {
        float gm = -FLT_MAX;
        for (int s = 0; s < num_splits; s++) {
            float m = partial_max[s * n_heads + head_idx];
            if (m > gm) gm = m;
        }
        global_max = gm;

        // Step 2: Compute rescaled sums
        float ts = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float m = partial_max[s * n_heads + head_idx];
            float sum_s = partial_lse[s * n_heads + head_idx];
            float rescaled = sum_s * expf(m - gm);
            smem[s] = rescaled;
            ts += rescaled;
        }
        total_sum = ts;
    }
    __syncthreads();

    float inv_total = 1.0f / (total_sum + 1e-10f);

    // Step 3: Weighted sum of partial outputs
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float w = smem[s];
            acc += w * partial_O[(s * n_heads + head_idx) * head_dim + d];
        }
        O[head_idx * head_dim + d] = acc * inv_total;
    }
}

// ============================================================================
// Launch Flash-Decode
// ============================================================================

ATEN_CUDA_API void launch_flash_decode(
    const float* Q, const float* K_cache, const float* V_cache,
    float* O,
    float* partial_O, float* partial_lse, float* partial_max,
    int n_heads, int n_kv_heads, int head_dim,
    int total_seq, float scale,
    cudaStream_t stream)
{
    int num_splits = (total_seq + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;

    // Phase 1: Partial attention per split
    {
        dim3 grid(num_splits, n_heads);
        int block_size = 128;
        if (head_dim > 128) block_size = 256;
        // Shared: Q vector + scores
        int shared_mem = (head_dim + KV_CHUNK_SIZE) * sizeof(float);
        flash_decode_partial_kernel<<<grid, block_size, shared_mem, stream>>>(
            Q, K_cache, V_cache,
            partial_O, partial_lse, partial_max,
            n_heads, n_kv_heads, head_dim, total_seq, scale);
    }

    // Phase 2: Reduce across splits
    if (num_splits == 1) {
        // Single split — just normalize partial_O in place and copy to O
        // partial_O already has sum(exp*V), partial_lse has sum(exp)
        // O[d] = partial_O[d] / partial_lse
        // Use reduce kernel with num_splits=1 (it handles this)
    }
    {
        int block_size = 128;
        if (head_dim > 128) block_size = 256;
        int shared_mem = num_splits * sizeof(float);
        flash_decode_reduce_kernel<<<n_heads, block_size, shared_mem, stream>>>(
            partial_O, partial_lse, partial_max, O,
            n_heads, head_dim, num_splits);
    }
}

// ============================================================================
// Fused Attention Prep: RMSNorm + QKV GEMV + RoPE + KV Cache Write
// ============================================================================
// This is NOT a single monolithic kernel — QKV projection requires GEMV which
// needs the full weight matrices. Instead, we fuse the smaller ops:
//   fused_norm_rope_kvwrite: RMSNorm → (result feeds GEMV externally) →
//                            QK-norm + RoPE + KV-cache-write
// This eliminates 5 kernel launches (qk_norm_q, qk_norm_k, rope_q, rope_k,
// kv_write_k, kv_write_v) → 1 kernel launch.

__global__ void fused_qknorm_rope_kvwrite_kernel(
    float* __restrict__ Q,          // [n_heads * head_dim] — in/out (QK-norm + RoPE in-place)
    float* __restrict__ K,          // [n_kv_heads * head_dim] — in/out
    const float* __restrict__ V,    // [n_kv_heads * head_dim] — input (just write to cache)
    const float* __restrict__ q_norm_w,  // [head_dim] (nullptr if no QK-norm)
    const float* __restrict__ k_norm_w,  // [head_dim]
    float* __restrict__ K_cache,    // [max_seq, n_kv_heads * head_dim]
    float* __restrict__ V_cache,    // [max_seq, n_kv_heads * head_dim]
    int n_heads, int n_kv_heads, int head_dim,
    int position, float rope_freq_base, float eps, bool add_one,
    int64_t cache_offset_row)
{
    // Grid: (max(n_heads, n_kv_heads),)
    // Each block processes one head: QK-norm + RoPE
    // Then kv_heads blocks also write K,V to cache

    int head = blockIdx.x;
    int tid = threadIdx.x;

    // Part A: Q head processing (QK-norm + RoPE)
    if (head < n_heads) {
        float* q_head = Q + head * head_dim;

        // QK-norm on Q (if weight provided)
        if (q_norm_w != nullptr) {
            // Compute RMS
            extern __shared__ float smem[];
            float local_sum = 0.0f;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float val = q_head[d];
                local_sum += val * val;
            }
            smem[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) smem[tid] += smem[tid + s];
                __syncthreads();
            }
            float rms = rsqrtf(smem[0] / head_dim + eps);
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float w = add_one ? (1.0f + q_norm_w[d]) : q_norm_w[d];
                q_head[d] = q_head[d] * rms * w;
            }
            __syncthreads();
        }

        // RoPE on Q
        int half_dim = head_dim / 2;
        for (int d = tid; d < half_dim; d += blockDim.x) {
            float freq = 1.0f / powf(rope_freq_base, 2.0f * d / head_dim);
            float theta = position * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float x0 = q_head[2 * d];
            float x1 = q_head[2 * d + 1];
            q_head[2 * d]     = x0 * cos_t - x1 * sin_t;
            q_head[2 * d + 1] = x0 * sin_t + x1 * cos_t;
        }
    }

    // Part B: K head processing (QK-norm + RoPE + cache write)
    // Also V cache write
    int kv_head = head;  // blockIdx.x maps to kv_head
    if (kv_head < n_kv_heads) {
        float* k_head = K + kv_head * head_dim;
        int kv_dim = n_kv_heads * head_dim;

        // QK-norm on K
        if (k_norm_w != nullptr) {
            extern __shared__ float smem[];
            float local_sum = 0.0f;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float val = k_head[d];
                local_sum += val * val;
            }
            smem[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) smem[tid] += smem[tid + s];
                __syncthreads();
            }
            float rms = rsqrtf(smem[0] / head_dim + eps);
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float w = add_one ? (1.0f + k_norm_w[d]) : k_norm_w[d];
                k_head[d] = k_head[d] * rms * w;
            }
            __syncthreads();
        }

        // RoPE on K
        int half_dim = head_dim / 2;
        for (int d = tid; d < half_dim; d += blockDim.x) {
            float freq = 1.0f / powf(rope_freq_base, 2.0f * d / head_dim);
            float theta = position * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float x0 = k_head[2 * d];
            float x1 = k_head[2 * d + 1];
            k_head[2 * d]     = x0 * cos_t - x1 * sin_t;
            k_head[2 * d + 1] = x0 * sin_t + x1 * cos_t;
        }
        __syncthreads();

        // Write K to cache
        for (int d = tid; d < head_dim; d += blockDim.x) {
            K_cache[cache_offset_row * kv_dim + kv_head * head_dim + d] = k_head[d];
        }

        // Write V to cache
        const float* v_head = V + kv_head * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            V_cache[cache_offset_row * kv_dim + kv_head * head_dim + d] = v_head[d];
        }
    }
}

ATEN_CUDA_API void launch_fused_qknorm_rope_kvwrite(
    float* Q, float* K, const float* V,
    const float* q_norm_w, const float* k_norm_w,
    float* K_cache, float* V_cache,
    int n_heads, int n_kv_heads, int head_dim,
    int position, float rope_freq_base, float eps, bool add_one,
    int64_t cache_offset_row,
    cudaStream_t stream)
{
    int num_blocks = n_heads > n_kv_heads ? n_heads : n_kv_heads;
    int block_size = 128;
    if (head_dim > 128) block_size = 256;
    // Power of 2 for reduction
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;
    if (block_size > 256) block_size = 256;

    int shared_mem = block_size * sizeof(float);
    fused_qknorm_rope_kvwrite_kernel<<<num_blocks, block_size, shared_mem, stream>>>(
        Q, K, V, q_norm_w, k_norm_w, K_cache, V_cache,
        n_heads, n_kv_heads, head_dim,
        position, rope_freq_base, eps, add_one, cache_offset_row);
}

// ============================================================================
// Fused Output Proj + Residual Add
// ============================================================================
// out[i] = x[i] + proj[i]  (elementwise, trivial fusion)
// This just saves one kernel launch vs separate add.

__global__ void fused_residual_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = x[idx] + y[idx];
}

// ============================================================================
// Fused RMSNorm + SiLU-Mul + Residual
// ============================================================================
// FFN path: norm(h) → gate/up GEMV (external) → silu(gate)*up → down GEMV (external) → h + down
// We can fuse: RMSNorm into the start, and residual add at the end.
// Since GEMVs are separate kernels, we fuse:
//   fused_norm_residual: RMSNorm + write to output (saves 1 launch vs separate norm)
//   fused_silu_mul_residual_add: silu(gate)*up in-place + add residual at end
//     NOT possible because down projection sits between silu_mul and residual.
//
// Practical fusion: silu_mul + residual_add after down projection:
//   out[i] = residual[i] + down[i]
//   This is just an add — already fused above.
//
// Most impactful fusion: merge the TWO rope launches (Q and K) into ONE kernel,
// merge the TWO kv_cache_write launches into ONE kernel.
// That's what fused_qknorm_rope_kvwrite does above.
//
// Additional: fused_silu_mul_store avoids one intermediate buffer
// gate and up → out = silu(gate) * up, directly usable by down projection.
// Already have launch_silu_mul. No extra fusion needed.

// ============================================================================
// Fused RMSNorm + Bias Add (for QKV with bias models like Qwen3)
// ============================================================================
// norm(x) → GEMV (external) → add bias
// Fuse: skip — bias add is 1 launch for 3 vectors, trivial cost.

// ============================================================================
// Flash-Decode Buffer Size Query
// ============================================================================

ATEN_CUDA_API int flash_decode_num_splits(int total_seq) {
    return (total_seq + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;
}

ATEN_CUDA_API int flash_decode_kv_chunk_size() {
    return KV_CHUNK_SIZE;
}

} // namespace cuda
} // namespace at
