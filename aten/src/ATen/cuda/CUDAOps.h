#pragma once

#include <cuda_runtime.h>
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
// Comparison Operations
// ============================================================================

// Note: These output bool arrays (stored as uint8_t on GPU)

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

} // namespace cuda
} // namespace at
