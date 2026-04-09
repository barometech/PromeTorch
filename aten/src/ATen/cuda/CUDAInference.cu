// ============================================================================
// CUDA Inference Kernels for PromeTorch
// ============================================================================
// Specialized kernels for LLM inference: RMSNorm, RoPE, Causal Attention

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "aten/src/ATen/cuda/CUDAOps.h"

namespace at {
namespace cuda {

// ============================================================================
// RMSNorm Kernel
// ============================================================================
// One block per row. Shared memory reduction for sum-of-squares.

__global__ void rms_norm_kernel(
    const float* __restrict__ input,    // [rows, hidden]
    const float* __restrict__ weight,   // [hidden]
    float* __restrict__ output,         // [rows, hidden]
    int hidden, float eps, bool add_one)
{
    int row = blockIdx.x;
    const float* x = input + row * hidden;
    float* y = output + row * hidden;

    extern __shared__ float shared[];

    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float val = x[i];
        local_sum += val * val;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / hidden + eps);

    // Apply normalization
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float w = add_one ? (1.0f + weight[i]) : weight[i];
        y[i] = x[i] * rms * w;
    }
}

ATEN_CUDA_API void launch_rms_norm(
    const float* input, const float* weight, float* output,
    int rows, int hidden, float eps, bool add_one,
    cudaStream_t stream)
{
    int block_size = 256;
    if (hidden > 512) block_size = 512;
    if (hidden > 1024) block_size = 1024;
    // Ensure block_size doesn't exceed hidden (for correct reduction)
    if (block_size > hidden) block_size = hidden;
    // Round up to next power of 2 for reduction
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;
    if (block_size > 1024) block_size = 1024;

    int shared_mem = block_size * sizeof(float);
    rms_norm_kernel<<<rows, block_size, shared_mem, stream>>>(
        input, weight, output, hidden, eps, add_one);
}

// ============================================================================
// Per-Head RMSNorm (QK-norm) Kernel
// ============================================================================
// One block per (row, head) pair.

__global__ void per_head_rms_norm_kernel(
    float* __restrict__ data,           // [rows, n_heads * head_dim] — in-place
    const float* __restrict__ weight,   // [head_dim]
    int n_heads, int head_dim, float eps, bool add_one)
{
    int row = blockIdx.x / n_heads;
    int head = blockIdx.x % n_heads;
    int total_dim = n_heads * head_dim;

    float* head_data = data + row * total_dim + head * head_dim;

    extern __shared__ float shared[];

    // Sum of squares
    float local_sum = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = head_data[d];
        local_sum += val * val;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / head_dim + eps);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float w = add_one ? (1.0f + weight[d]) : weight[d];
        head_data[d] = head_data[d] * rms * w;
    }
}

ATEN_CUDA_API void launch_per_head_rms_norm(
    float* data, const float* weight,
    int rows, int n_heads, int head_dim, float eps, bool add_one,
    cudaStream_t stream)
{
    int total_blocks = rows * n_heads;
    int block_size = 128;
    if (head_dim > 128) block_size = 256;
    // Power of 2
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;
    if (block_size > head_dim) {
        bs = 1;
        while (bs < head_dim) bs <<= 1;
        block_size = bs;
    }
    if (block_size > 1024) block_size = 1024;

    int shared_mem = block_size * sizeof(float);
    per_head_rms_norm_kernel<<<total_blocks, block_size, shared_mem, stream>>>(
        data, weight, n_heads, head_dim, eps, add_one);
}

// ============================================================================
// RoPE (Rotary Position Embeddings) Kernel
// ============================================================================
// Each thread handles one rotation pair (2 values).

__global__ void rope_kernel(
    float* __restrict__ data,   // [seq_len, n_heads * head_dim]
    int seq_len, int n_heads, int head_dim,
    int position_offset, float freq_base)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = seq_len * n_heads * (head_dim / 2);
    if (idx >= total_pairs) return;

    int d = idx % (head_dim / 2);
    int tmp = idx / (head_dim / 2);
    int h = tmp % n_heads;
    int s = tmp / n_heads;

    int pos = position_offset + s;
    float freq = 1.0f / powf(freq_base, 2.0f * d / head_dim);
    float theta = pos * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    int offset = s * (n_heads * head_dim) + h * head_dim + 2 * d;
    float x0 = data[offset];
    float x1 = data[offset + 1];

    data[offset]     = x0 * cos_t - x1 * sin_t;
    data[offset + 1] = x0 * sin_t + x1 * cos_t;
}

ATEN_CUDA_API void launch_rope(
    float* data, int seq_len, int n_heads, int head_dim,
    int position_offset, float freq_base,
    cudaStream_t stream)
{
    int total_pairs = seq_len * n_heads * (head_dim / 2);
    int block_size = 256;
    int num_blocks = (total_pairs + block_size - 1) / block_size;
    rope_kernel<<<num_blocks, block_size, 0, stream>>>(
        data, seq_len, n_heads, head_dim, position_offset, freq_base);
}

// ============================================================================
// Causal Attention Kernel (GQA support)
// ============================================================================
// One block per (query_position, head) pair.
// For decode (seq_len=1): n_heads blocks.
// For prefill: seq_len * n_heads blocks.
//
// Steps per block:
// 1. Compute scores: Q[s,h,:] @ K[t,kv_h,:] * scale for t=0..total_seq-1
// 2. Causal mask (scores[t] = -inf for t > past_len + s)
// 3. Online softmax
// 4. Weighted sum of V

__global__ void causal_attention_kernel(
    const float* __restrict__ Q,       // [seq_len, n_heads * head_dim]
    const float* __restrict__ K,       // [total_seq, n_kv_heads * head_dim]
    const float* __restrict__ V,       // [total_seq, n_kv_heads * head_dim]
    float* __restrict__ output,        // [seq_len, n_heads * head_dim]
    int seq_len, int total_seq,
    int n_heads, int n_kv_heads, int head_dim,
    int past_len, float scale)
{
    int block_id = blockIdx.x;
    int s = block_id / n_heads;          // query position
    int h = block_id % n_heads;          // head index
    int heads_per_group = n_heads / n_kv_heads;
    int kv_h = h / heads_per_group;      // GQA mapping
    int max_t = past_len + s;            // causal: can attend up to this

    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    const float* q_head = Q + s * q_dim + h * head_dim;
    float* out_head = output + s * q_dim + h * head_dim;

    // Use shared memory for scores + partial sums
    extern __shared__ float shared[];
    // shared[0..total_seq-1] = scores (reused for attn_weights)

    // Step 1: Compute dot product scores
    for (int t = threadIdx.x; t < total_seq; t += blockDim.x) {
        if (t <= max_t) {
            const float* k_head = K + t * kv_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_head[d] * k_head[d];
            }
            shared[t] = dot * scale;
        } else {
            shared[t] = -1e9f;  // causal mask
        }
    }
    __syncthreads();

    // Step 2: Softmax — find max
    // Use thread 0 for simplicity (total_seq is usually small for decode)
    __shared__ float s_max;
    __shared__ float s_sum;

    if (threadIdx.x == 0) {
        float max_val = -1e9f;
        int end = (max_t < total_seq - 1) ? max_t + 1 : total_seq;
        for (int t = 0; t < end; t++) {
            if (shared[t] > max_val) max_val = shared[t];
        }
        s_max = max_val;
    }
    __syncthreads();

    // Exp and sum
    for (int t = threadIdx.x; t < total_seq; t += blockDim.x) {
        if (t <= max_t) {
            shared[t] = expf(shared[t] - s_max);
        } else {
            shared[t] = 0.0f;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        int end = (max_t < total_seq - 1) ? max_t + 1 : total_seq;
        for (int t = 0; t < end; t++) {
            sum += shared[t];
        }
        s_sum = 1.0f / (sum + 1e-10f);
    }
    __syncthreads();

    // Normalize
    for (int t = threadIdx.x; t < total_seq; t += blockDim.x) {
        shared[t] *= s_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        int end = (max_t < total_seq - 1) ? max_t + 1 : total_seq;
        for (int t = 0; t < end; t++) {
            acc += shared[t] * V[t * kv_dim + kv_h * head_dim + d];
        }
        out_head[d] = acc;
    }
}

ATEN_CUDA_API void launch_causal_attention(
    const float* Q, const float* K, const float* V, float* output,
    int seq_len, int total_seq,
    int n_heads, int n_kv_heads, int head_dim,
    int past_len, float scale,
    cudaStream_t stream)
{
    int total_blocks = seq_len * n_heads;
    // Block size: at least head_dim threads, at most 256
    int block_size = 128;
    if (head_dim > 128) block_size = 256;
    // Need enough shared memory for total_seq floats
    int shared_mem = total_seq * sizeof(float);
    // CUDA shared memory limit is 48KB = 12288 floats
    // For very long sequences, fall back (but for inference this should be fine)

    causal_attention_kernel<<<total_blocks, block_size, shared_mem, stream>>>(
        Q, K, V, output,
        seq_len, total_seq, n_heads, n_kv_heads, head_dim,
        past_len, scale);
}

// ============================================================================
// GPU Tensor Concat (for KV cache)
// ============================================================================
// Concatenate two 2D tensors along dim 0: [M1, K] + [M2, K] → [M1+M2, K]

__global__ void concat_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output,
    int64_t a_rows, int64_t b_rows, int64_t cols)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (a_rows + b_rows) * cols;
    if (idx >= total) return;

    int64_t row = idx / cols;
    int64_t col = idx % cols;

    if (row < a_rows) {
        output[idx] = a[row * cols + col];
    } else {
        output[idx] = b[(row - a_rows) * cols + col];
    }
}

ATEN_CUDA_API void launch_concat(
    const float* a, const float* b, float* output,
    int64_t a_rows, int64_t b_rows, int64_t cols,
    cudaStream_t stream)
{
    int64_t total = (a_rows + b_rows) * cols;
    int block_size = 256;
    int num_blocks = (int)((total + block_size - 1) / block_size);
    if (num_blocks > 65535) num_blocks = 65535;
    concat_kernel<<<num_blocks, block_size, 0, stream>>>(
        a, b, output, a_rows, b_rows, cols);
}

// ============================================================================
// KV Cache Write Kernel
// ============================================================================
// Copy new_rows of [cols] into dst_cache starting at offset_row.
// dst_cache layout: [max_seq, cols], write at [offset_row..offset_row+num_new_rows-1, :]

__global__ void kv_cache_write_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t num_new_rows, int64_t cols, int64_t offset_row)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_new_rows * cols;
    if (idx >= total) return;
    dst[(offset_row * cols) + idx] = src[idx];
}

ATEN_CUDA_API void launch_kv_cache_write(
    const float* src, float* dst_cache,
    int64_t num_new_rows, int64_t cols, int64_t offset_row,
    cudaStream_t stream)
{
    int64_t total = num_new_rows * cols;
    int block_size = 256;
    int num_blocks = (int)((total + block_size - 1) / block_size);
    if (num_blocks > 65535) num_blocks = 65535;
    kv_cache_write_kernel<<<num_blocks, block_size, 0, stream>>>(
        src, dst_cache, num_new_rows, cols, offset_row);
}

// ============================================================================
// Fused SiLU-Mul Kernel
// ============================================================================
// out[i] = silu(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
// Replaces 2 kernel launches (silu + mul) + intermediate tensor

__global__ void silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = gate[idx];
    output[idx] = (g / (1.0f + expf(-g))) * up[idx];
}

ATEN_CUDA_API void launch_silu_mul(
    const float* gate, const float* up, float* output,
    int64_t n, cudaStream_t stream)
{
    int block_size = 256;
    int num_blocks = (int)((n + block_size - 1) / block_size);
    if (num_blocks > 65535) num_blocks = 65535;
    silu_mul_kernel<<<num_blocks, block_size, 0, stream>>>(gate, up, output, n);
}

// ============================================================================
// Inference GEMV Kernel (for decode: [1,K] @ [K,N] → [1,N])
// ============================================================================
// W is row-major [K, N]. Each thread computes one output y[n].
// Adjacent threads have adjacent n values → coalesced W reads.
// W[k*N + n] with adjacent n = adjacent memory addresses.
//
// For large K: load x into shared memory for reuse across threads.

__global__ void inference_gemv_kernel(
    const float* __restrict__ x,   // [K]
    const float* __restrict__ W,   // [K, N] row-major
    float* __restrict__ y,         // [N]
    int K, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // Load x into shared memory — ALL threads must participate before __syncthreads
    extern __shared__ float x_shared[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    // Now safe to exit for out-of-bounds threads
    if (n >= N) return;

    float sum = 0.0f;

    // Process 4 elements at a time
    int K4 = (K / 4) * 4;
    for (int k = 0; k < K4; k += 4) {
        float x0 = x_shared[k], x1 = x_shared[k+1], x2 = x_shared[k+2], x3 = x_shared[k+3];
        sum += x0 * W[(k  ) * N + n]
             + x1 * W[(k+1) * N + n]
             + x2 * W[(k+2) * N + n]
             + x3 * W[(k+3) * N + n];
    }
    for (int k = K4; k < K; ++k) {
        sum += x_shared[k] * W[k * N + n];
    }

    y[n] = sum;
}

ATEN_CUDA_API void launch_inference_gemv(
    const float* x, const float* W, float* y,
    int K, int N, cudaStream_t stream)
{
    // Each thread computes one output element
    // 256 threads per block, ceil(N/256) blocks
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    int shared_mem = K * sizeof(float);  // x vector in shared memory (max ~39 KB for K=9728)
    inference_gemv_kernel<<<grid_size, block_size, shared_mem, stream>>>(x, W, y, K, N);
}

// ============================================================================
// Embedding Lookup Kernel (CUDA Graph compatible)
// ============================================================================
// Reads token_id from device memory, copies one row from embedding table.
// This replaces cudaMemcpyAsync with a kernel that can be captured in a graph.

__global__ void embedding_lookup_kernel(
    const float* __restrict__ emb_table,  // [vocab_size, hidden_size]
    float* __restrict__ output,           // [hidden_size]
    const int* __restrict__ d_token_id,   // device pointer to token ID
    int hidden_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_size) return;
    int token_id = *d_token_id;
    output[tid] = emb_table[token_id * hidden_size + tid];
}

ATEN_CUDA_API void launch_embedding_lookup(
    const float* emb_table, float* output,
    const int* d_token_id, int hidden_size,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (hidden_size + block - 1) / block;
    embedding_lookup_kernel<<<grid, block, 0, stream>>>(
        emb_table, output, d_token_id, hidden_size);
}

} // namespace cuda
} // namespace at
