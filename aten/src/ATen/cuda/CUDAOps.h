#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "c10/macros/Macros.h"

// ============================================================================
// CUDA Operation Declarations for PromeTorch
// ============================================================================
// These functions launch CUDA kernels - include this header in C++ code
// to dispatch operations to GPU

// Export/Import macro for aten_cuda library
// ATEN_CUDA_EXPORTS is defined only when building aten_cuda.dll
#if defined(PT_PLATFORM_WINDOWS)
    #if defined(ATEN_CUDA_EXPORTS)
        #define ATEN_CUDA_API __declspec(dllexport)
    #else
        #define ATEN_CUDA_API __declspec(dllimport)
    #endif
#else
    #define ATEN_CUDA_API __attribute__((visibility("default")))
#endif

namespace at {
namespace cuda {

// ============================================================================
// Element-wise Unary Operations
// ============================================================================

ATEN_CUDA_API void launch_neg(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_abs(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_sqrt(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_rsqrt(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_square(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_exp(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_log(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_sin(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_cos(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_tanh(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_sigmoid(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_relu(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_leaky_relu(const float* input, float* output, float alpha, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_silu(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_gelu(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);

// Additional unary operations
ATEN_CUDA_API void launch_log2(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_log10(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_tan(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_ceil(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_floor(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_round(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_sign(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_reciprocal(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);

// ============================================================================
// Element-wise Binary Operations
// ============================================================================

ATEN_CUDA_API void launch_add(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_add_scalar(const float* a, float scalar, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_sub(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_mul(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_mul_scalar(const float* a, float scalar, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_mul_broadcast_row(const float* a, const float* b, float* out, int64_t outer_size, int64_t inner_size, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_mul_broadcast_col(const float* a, const float* b, float* out, int64_t outer_size, int64_t inner_size, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_add_broadcast_col(const float* a, const float* b, float* out, int64_t outer_size, int64_t inner_size, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_div(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_div_scalar(const float* a, float scalar, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_pow(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_pow_scalar(const float* a, float exp, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_maximum(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_minimum(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);

// ============================================================================
// Fill and Copy
// ============================================================================

ATEN_CUDA_API void launch_fill(float* data, float value, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_copy(const float* src, float* dst, int64_t n, cudaStream_t stream = nullptr);

// ============================================================================
// Comparison Operations (float-returning: 0.0f or 1.0f)
// ============================================================================

// Tensor vs tensor comparisons
ATEN_CUDA_API void launch_eq(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_ne(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_lt(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_le(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_gt(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_ge(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);

// Scalar comparisons
ATEN_CUDA_API void launch_eq_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_ne_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_lt_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_le_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_gt_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_ge_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream = nullptr);

// ============================================================================
// Fused Operations
// ============================================================================

ATEN_CUDA_API void launch_addcmul(const float* self, const float* t1, const float* t2, float value, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_addcdiv(const float* self, const float* t1, const float* t2, float value, float* out, int64_t n, cudaStream_t stream = nullptr);

// ============================================================================
// Conditional Operations
// ============================================================================

ATEN_CUDA_API void launch_clamp(const float* input, float* output, float min_val, float max_val, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_where(const bool* cond, const float* a, const float* b, float* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_masked_fill(float* data, const bool* mask, float value, int64_t n, cudaStream_t stream = nullptr);

// ============================================================================
// Softmax
// ============================================================================

ATEN_CUDA_API void launch_softmax(const float* input, float* output, int64_t outer_size, int64_t dim_size, int64_t inner_size, cudaStream_t stream = nullptr);

// ============================================================================
// Reduction Operations
// ============================================================================

ATEN_CUDA_API void launch_sum(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_sum_dim(const float* input, float* output, int64_t outer_size, int64_t reduce_size, int64_t inner_size, cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_mean(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_mean_dim(const float* input, float* output, int64_t outer_size, int64_t reduce_size, int64_t inner_size, cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_max(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_max_dim(const float* input, float* output, int64_t* indices, int64_t outer_size, int64_t reduce_size, int64_t inner_size, cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_min(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_min_dim(const float* input, float* output, int64_t* indices, int64_t outer_size, int64_t reduce_size, int64_t inner_size, cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_l1_norm(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_l2_norm(const float* input, float* output, int64_t n, cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_argmax(const float* input, int64_t* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_argmin(const float* input, int64_t* output, int64_t n, cudaStream_t stream = nullptr);

// Embedding lookup from GPU table using device token_id pointer (CUDA Graph compatible)
ATEN_CUDA_API void launch_embedding_lookup(
    const float* emb_table, float* output,
    const int* d_token_id, int hidden_size,
    cudaStream_t stream = nullptr);

// ============================================================================
// Linear Algebra / BLAS Operations
// ============================================================================

// General matrix multiplication: C = alpha * op(A) @ op(B) + beta * C
// trans_a: if true, A is transposed
// trans_b: if true, B is transposed
ATEN_CUDA_API void launch_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b,
    cudaStream_t stream = nullptr
);

// Native tiled GEMM — no cuBLAS, uses hand-written kernels from CUDABlas.cu.
// Slower than launch_gemm (cuBLAS) but works without cuBLAS at link time.
ATEN_CUDA_API void launch_gemm_native(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b,
    cudaStream_t stream = nullptr
);

// Batched GEMM: batch matrix multiplications
ATEN_CUDA_API void launch_batched_gemm(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream = nullptr
);

// Matrix-vector: y = A @ x
ATEN_CUDA_API void launch_gemv(
    const float* A, const float* x, float* y,
    int M, int N,
    cudaStream_t stream = nullptr
);

// Vector dot product
ATEN_CUDA_API void launch_dot(
    const float* a, const float* b, float* result,
    int64_t n,
    cudaStream_t stream = nullptr
);

// Outer product: C = a @ b^T
ATEN_CUDA_API void launch_outer(
    const float* a, const float* b, float* C,
    int M, int N,
    cudaStream_t stream = nullptr
);

// Matrix transpose
ATEN_CUDA_API void launch_transpose(
    const float* input, float* output,
    int rows, int cols,
    cudaStream_t stream = nullptr
);

// Addmm: C = beta * C + alpha * A @ B
ATEN_CUDA_API void launch_addmm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b,
    cudaStream_t stream = nullptr
);

// ============================================================================
// Convolution Operations
// ============================================================================

ATEN_CUDA_API void launch_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,  // can be nullptr
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    cudaStream_t stream = nullptr
);

// ============================================================================
// Pooling Operations
// ============================================================================

ATEN_CUDA_API void launch_max_pool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    cudaStream_t stream = nullptr
);

ATEN_CUDA_API void launch_avg_pool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    bool count_include_pad,
    cudaStream_t stream = nullptr
);

ATEN_CUDA_API void launch_adaptive_avg_pool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    cudaStream_t stream = nullptr
);

// ============================================================================
// Batch Normalization
// ============================================================================

ATEN_CUDA_API void launch_batch_norm2d_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps,
    cudaStream_t stream = nullptr
);

// ============================================================================
// Loss Functions
// ============================================================================

// Cross entropy loss: -log(softmax(logits)[target])
// logits: (batch_size, num_classes)
// targets: (batch_size,) - class indices as float
// output: scalar (Mean/Sum reduction) or (batch_size,) for None reduction
// reduction: 0=None, 1=Mean, 2=Sum
ATEN_CUDA_API void launch_cross_entropy_loss(
    const float* logits,
    const float* targets,
    float* output,
    int batch_size,
    int num_classes,
    int reduction,
    cudaStream_t stream = nullptr
);

// NLL loss: -log_probs[target]
// log_probs: (batch_size, num_classes) - already log softmax
// targets: (batch_size,) - class indices as float
ATEN_CUDA_API void launch_nll_loss(
    const float* log_probs,
    const float* targets,
    float* output,
    int batch_size,
    int num_classes,
    int reduction,
    cudaStream_t stream = nullptr
);

// ============================================================================
// Convenience aliases for common operations
// ============================================================================

// Matrix multiplication: C = A @ B (no transpose, alpha=1, beta=0)
inline void launch_mm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    launch_gemm(A, B, C, M, N, K, 1.0f, 0.0f, false, false, stream);
}

// Batched matrix multiplication
inline void launch_bmm(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    launch_batched_gemm(A, B, C, batch, M, N, K, 1.0f, 0.0f, stream);
}

// ============================================================================
// Inference Kernels (LLM: RMSNorm, RoPE, Causal Attention)
// ============================================================================

ATEN_CUDA_API void launch_rms_norm(
    const float* input, const float* weight, float* output,
    int rows, int hidden, float eps, bool add_one,
    cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_per_head_rms_norm(
    float* data, const float* weight,
    int rows, int n_heads, int head_dim, float eps, bool add_one,
    cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_rope(
    float* data, int seq_len, int n_heads, int head_dim,
    int position_offset, float freq_base,
    cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_causal_attention(
    const float* Q, const float* K, const float* V, float* output,
    int seq_len, int total_seq,
    int n_heads, int n_kv_heads, int head_dim,
    int past_len, float scale,
    cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_concat(
    const float* a, const float* b, float* output,
    int64_t a_rows, int64_t b_rows, int64_t cols,
    cudaStream_t stream = nullptr);

// Write new KV rows into pre-allocated cache at offset position
ATEN_CUDA_API void launch_kv_cache_write(
    const float* src, float* dst_cache,
    int64_t num_new_rows, int64_t cols, int64_t offset_row,
    cudaStream_t stream = nullptr);

// Fused SiLU(gate) * up: out[i] = silu(gate[i]) * up[i]
ATEN_CUDA_API void launch_silu_mul(
    const float* gate, const float* up, float* output,
    int64_t n, cudaStream_t stream = nullptr);

// Inference GEMV: y[n] = sum_k x[k] * W[k * N + n], for [1,K] @ [K,N] = [1,N]
// W is row-major [K, N] (pre-transposed weights)
ATEN_CUDA_API void launch_inference_gemv(
    const float* x, const float* W, float* y,
    int K, int N, cudaStream_t stream = nullptr);

// Fused Q4_K_M dequant-GEMV: y[n] = sum_k dequant(W_q4km[n,k]) * x[k]
ATEN_CUDA_API void launch_q4km_gemv(
    const void* weights, const float* x, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// Persistent Q4_K_M GEMV: one block per SM, grid-stride over rows.
// Loads x into shared memory ONCE, then processes all rows.
// Reduces kernel launch overhead by ~4x for large N.
ATEN_CUDA_API void launch_q4km_persistent_gemv(
    const void* weights, const float* x, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// Q4_K_M GEMV v2: llama.cpp-style hand-tuned variant.
// - Bulk 16-byte (uint4) header load via __ldg (d, dmin, 12 scale bytes)
// - NROWS=2: each warp handles 2 output rows, amortizing Q8_1 smem reads
// - Produces byte-identical results to launch_q4km_persistent_gemv at T=0
// Expected +30-50% on A100 vs v1 for large N/K GEMV shapes.
ATEN_CUDA_API void launch_q4km_persistent_gemv_v2(
    const void* weights, const float* x, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// Fused multi-GEMV: runs gate+up projections in a single kernel launch.
// Writes gate output to y_gate[0..N_gate-1] and up output to y_up[0..N_up-1].
// Both weight matrices must be Q4_K with the same K (input dim).
ATEN_CUDA_API void launch_q4km_fused_gate_up_gemv(
    const void* w_gate, const void* w_up,
    const float* x, float* y_gate, float* y_up,
    int K, int N_gate, int N_up,
    int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// Quantize float32 x vector to Q8_1 (int8) for dp4a GEMV
// y_q8 must point to (K/32) * 36 bytes of GPU memory
ATEN_CUDA_API void launch_quantize_q8_1(
    const float* x, void* y_q8, int K,
    cudaStream_t stream = nullptr);

// Q4_K × Q8_1 dp4a GEMV: x pre-quantized to Q8_1, 4x faster than float GEMV
ATEN_CUDA_API void launch_q4km_q8_gemv(
    const void* weights, const void* x_q8, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// Pre-initialize kernel shared memory attributes for CUDA Graph compatibility.
// Must be called ONCE before any graph capture. Pass model's max hidden and intermediate dims.
ATEN_CUDA_API void init_cuda_kernel_smem_attributes(int max_K, int max_inter);

// Dequantize Q4_K_M weights to FP16 (one-time at load)
ATEN_CUDA_API void launch_dequant_q4k_to_fp16(
    const void* weights, void* out_fp16,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// cuBLAS Hgemm GEMV: FP16 weights × FP32 vector → FP32 output
// Requires FP16 scratch buffers for x and y conversion
ATEN_CUDA_API void launch_cublas_hgemv(
    const void* W_fp16, const float* x, float* y,
    int K, int N,
    void* x_fp16_buf, void* y_fp16_buf,
    cudaStream_t stream = nullptr,
    bool row_major = false);

// FP16 dequant GEMV: y[n] = sum_k fp16_to_fp32(W[n,k]) * x[k]
// W is [N, K] row-major FP16 (GGML layout: ne[0]=K contiguous)
ATEN_CUDA_API void launch_fp16_gemv(
    const void* W_fp16, const float* x, float* y,
    int K, int N, cudaStream_t stream = nullptr);

// Fused Q6_K dequant-GEMV: y[n] = sum_k dequant(W_q6k[n,k]) * x[k]
ATEN_CUDA_API void launch_q6k_gemv(
    const void* weights, const float* x, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// Fused Q5_K dequant-GEMV: y[n] = sum_k dequant(W_q5k[n,k]) * x[k]
ATEN_CUDA_API void launch_q5k_gemv(
    const void* weights, const float* x, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// ============================================================================
// PIR Operations (Parallel Scan for Recurrent Networks)
// ============================================================================

// Parallel recurrent scan: h[t] = gate[t] * h[t-1] + x[t]
// Runs on GPU without CPU transfer
ATEN_CUDA_API void launch_parallel_scan(
    const float* x,           // [B, T, D] input
    const float* gate_logits, // [B, T, D] gate logits
    const float* base_decay,  // [D] base decay values
    float* output,            // [B, T, D] output
    float* gates,             // [B, T, D] computed gates (for backward)
    int64_t B, int64_t T, int64_t D,
    cudaStream_t stream = nullptr
);

// Rotary positional embedding application
ATEN_CUDA_API void launch_rotary_embedding(
    const float* x,           // [B, T, D] input
    const float* cos_cache,   // [max_seq, D] precomputed cos
    const float* sin_cache,   // [max_seq, D] precomputed sin
    float* output,            // [B, T, D] output
    int64_t B, int64_t T, int64_t D, int64_t cache_dim,
    cudaStream_t stream = nullptr
);

// ============================================================================
// Flash-Decoding (parallel decode attention across KV splits)
// ============================================================================

// Flash-decode: split KV cache across thread blocks for parallel attention.
// Requires scratch buffers: partial_O [num_splits * n_heads * head_dim],
//   partial_lse [num_splits * n_heads], partial_max [num_splits * n_heads].
// Use flash_decode_num_splits() to compute num_splits for buffer allocation.
ATEN_CUDA_API void launch_flash_decode(
    const float* Q, const float* K_cache, const float* V_cache,
    float* O,
    float* partial_O, float* partial_lse, float* partial_max,
    int n_heads, int n_kv_heads, int head_dim,
    int total_seq, float scale,
    cudaStream_t stream = nullptr);

// Query helpers for scratch buffer sizing
ATEN_CUDA_API int flash_decode_num_splits(int total_seq);
ATEN_CUDA_API int flash_decode_kv_chunk_size();

// ============================================================================
// Fused Decode Kernels (reduce kernel launch count)
// ============================================================================

// Fused QK-norm + RoPE + KV cache write (replaces 6 separate launches)
// Q and K are modified in-place (QK-norm + RoPE applied).
// K and V are written to their respective caches at cache_offset_row.
// q_norm_w/k_norm_w can be nullptr to skip QK-norm.
ATEN_CUDA_API void launch_fused_qknorm_rope_kvwrite(
    float* Q, float* K, const float* V,
    const float* q_norm_w, const float* k_norm_w,
    float* K_cache, float* V_cache,
    int n_heads, int n_kv_heads, int head_dim,
    int position, float rope_freq_base, float eps, bool add_one,
    int64_t cache_offset_row,
    cudaStream_t stream = nullptr);

// ============================================================================
// Fused QKV GEMV — Three projections (Q, K, V) in a single kernel launch
// ============================================================================
// Loads x into shared memory ONCE, then computes Q, K, V outputs.
// Saves 2 kernel launches + 2 shared memory loads of x per layer.
// All three weight matrices must be Q4_K with the same K and row_stride.
ATEN_CUDA_API void launch_q4km_fused_qkv_gemv(
    const void* w_q, const void* w_k, const void* w_v,
    const float* x, float* out_q, float* out_k, float* out_v,
    int K, int N_q, int N_k, int N_v,
    int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// ============================================================================
// Fused RMSNorm + GEMV — Normalize x in shared memory, then compute GEMV
// ============================================================================
// Eliminates separate RMSNorm kernel + global memory round-trip.
ATEN_CUDA_API void launch_q4km_fused_rmsnorm_gemv(
    const float* x, const float* norm_weight,
    const void* weights, float* y,
    int K, int N, int64_t row_stride_bytes,
    float eps, bool add_one,
    cudaStream_t stream = nullptr);

// ============================================================================
// Fused RMSNorm + QKV GEMV — Single kernel: normalize x → Q, K, V projections
// ============================================================================
// Combines RMSNorm + fused QKV: saves 4 kernel launches per layer.
ATEN_CUDA_API void launch_q4km_fused_rmsnorm_qkv_gemv(
    const float* x, const float* norm_weight,
    const void* w_q, const void* w_k, const void* w_v,
    float* out_q, float* out_k, float* out_v,
    int K, int N_q, int N_k, int N_v,
    int64_t row_stride_bytes,
    float eps, bool add_one,
    cudaStream_t stream = nullptr);

// ============================================================================
// GEMV with accumulate (beta=1): y[n] += W@x (fused residual add)
// ============================================================================
// Used for output_proj and down_proj to fold residual_add into GEMV.
ATEN_CUDA_API void launch_q4km_persistent_gemv_accumulate(
    const void* weights, const float* x, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream = nullptr);

// ============================================================================
// Fused RMSNorm + Gate+Up GEMV — Single kernel: normalize → gate, up
// ============================================================================
ATEN_CUDA_API void launch_q4km_fused_rmsnorm_gate_up_gemv(
    const float* x, const float* norm_weight,
    const void* w_gate, const void* w_up,
    float* y_gate, float* y_up,
    int K, int N_gate, int N_up,
    int64_t row_stride_bytes,
    float eps, bool add_one,
    cudaStream_t stream = nullptr);

// ============================================================================
// FP16 KV Cache Operations
// ============================================================================

// Write FP32 K/V to FP16 cache (conversion on the fly)
ATEN_CUDA_API void launch_fp16_kv_cache_write(
    const float* src, void* dst_cache_fp16,
    int64_t num_new_rows, int64_t cols, int64_t offset_row,
    cudaStream_t stream = nullptr);

// Flash-decode with FP16 KV cache — halves memory bandwidth for attention
ATEN_CUDA_API void launch_flash_decode_fp16(
    const float* Q, const void* K_cache_fp16, const void* V_cache_fp16,
    float* O,
    float* partial_O, float* partial_lse, float* partial_max,
    int n_heads, int n_kv_heads, int head_dim,
    int total_seq, float scale,
    cudaStream_t stream = nullptr);

// ============================================================================
// CUDA Graph Compatible Kernels (past_len via Device Pointer)
// ============================================================================

ATEN_CUDA_API void launch_fused_qknorm_rope_kvwrite_graph(
    float* Q, float* K, const float* V,
    const float* q_norm_w, const float* k_norm_w,
    float* K_cache, float* V_cache,
    int n_heads, int n_kv_heads, int head_dim,
    const int64_t* d_past_len, float rope_freq_base, float eps, bool add_one,
    cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_flash_decode_graph(
    const float* Q, const float* K_cache, const float* V_cache,
    float* O, float* partial_O, float* partial_lse, float* partial_max,
    int n_heads, int n_kv_heads, int head_dim,
    const int64_t* d_past_len, int max_seq, float scale,
    cudaStream_t stream = nullptr);

ATEN_CUDA_API void launch_flash_decode_fp16_graph(
    const float* Q, const void* K_cache_fp16, const void* V_cache_fp16,
    float* O, float* partial_O, float* partial_lse, float* partial_max,
    int n_heads, int n_kv_heads, int head_dim,
    const int64_t* d_past_len, int max_seq, float scale,
    cudaStream_t stream = nullptr);

// ============================================================================
// FP16 Kernels (AMP / Automatic Mixed Precision)
// ============================================================================
// These mirror the float32 elementwise/activation/norm launchers above but
// operate on __half storage with FP32 accumulation. Used when a tensor's
// dtype == c10::ScalarType::Half under AutocastGuard.

// Element-wise binary
ATEN_CUDA_API void launch_add_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_sub_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_mul_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_div_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_add_broadcast_fp16(
    const __half* a, const __half* b, __half* out,
    int64_t outer_size, int64_t inner_size,
    cudaStream_t stream = nullptr);

// Activations
ATEN_CUDA_API void launch_relu_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_sigmoid_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_tanh_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_gelu_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_silu_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream = nullptr);

// Fused softmax (max + exp + sum + normalize in one kernel, FP32 accumulate)
ATEN_CUDA_API void launch_softmax_fp16(
    const __half* input, __half* output,
    int64_t outer_size, int64_t dim_size, int64_t inner_size,
    cudaStream_t stream = nullptr);

// LayerNorm: input [rows, hidden]; gamma/beta may be nullptr for affine=false
ATEN_CUDA_API void launch_layernorm_fp16(
    const __half* input,
    const __half* gamma,
    const __half* beta,
    __half* output,
    int rows, int hidden, float eps,
    cudaStream_t stream = nullptr);

// RMSNorm: input [rows, hidden]; weight may be nullptr (use identity)
ATEN_CUDA_API void launch_rmsnorm_fp16(
    const __half* input,
    const __half* weight,
    __half* output,
    int rows, int hidden, float eps, bool add_one,
    cudaStream_t stream = nullptr);

// Device-side inf/nan check. `found_device` must point to an int on device;
// caller zeroes it before launch and copies back afterwards to host.
ATEN_CUDA_API void launch_check_inf_nan_fp32(
    const float* data, int64_t n, int* found_device,
    cudaStream_t stream = nullptr);
ATEN_CUDA_API void launch_check_inf_nan_fp16(
    const __half* data, int64_t n, int* found_device,
    cudaStream_t stream = nullptr);

} // namespace cuda
} // namespace at
