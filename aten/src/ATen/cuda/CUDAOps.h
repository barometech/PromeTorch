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
    cudaStream_t stream = nullptr);

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

} // namespace cuda
} // namespace at
