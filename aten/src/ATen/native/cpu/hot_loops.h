#pragma once
// ============================================================================
// hot_loops.h — Non-inline inner loop declarations for LTO on Elbrus
// ============================================================================
// On E2K (Elbrus VLIW), inline functions in headers prevent LCC from
// performing cross-function VLIW scheduling. By moving hot inner loops
// into a .cpp file compiled once with -O3, LCC can:
//   1. Schedule VLIW bundles with full pipeline visibility
//   2. Share compiled code across all translation units (no duplication)
//   3. Apply LTO across the entire aten library
//
// On x86 (AVX2), these are still beneficial: reduces binary size and
// allows the linker to deduplicate previously-inline function copies.
// ============================================================================

#include <cstdint>

namespace at {
namespace native {
namespace hot {

// ============================================================================
// BLAS wrappers (delegate to TudaBLAS sgemm/sgemv/sdot)
// ============================================================================

// C = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
void sgemm(int64_t M, int64_t K, int64_t N,
           float alpha, const float* A, int64_t lda,
           const float* B, int64_t ldb,
           float beta, float* C, int64_t ldc);

// C = alpha * A[M,K] @ B^T[N,K] + beta * C[M,N]  (B stored row-major [N,K])
void sgemm_nt(int64_t M, int64_t K, int64_t N,
              float alpha, const float* A, int64_t lda,
              const float* B, int64_t ldb,
              float beta, float* C, int64_t ldc);

// C[K,N] = alpha * A^T[K,M] @ B[M,N] + beta * C[K,N]
// A is stored row-major [M,K], transposed logically.
// Used in backward: grad_weight = grad^T @ input
void sgemm_tn(int64_t M, int64_t K, int64_t N,
              float alpha, const float* A, int64_t lda,
              const float* B, int64_t ldb,
              float beta, float* C, int64_t ldc);

// y = alpha * A[M,N] @ x + beta * y
void sgemv(int64_t M, int64_t N,
           float alpha, const float* A, int64_t lda,
           const float* x,
           float beta, float* y);

// result = dot(a, b)
float sdot(int64_t n, const float* a, const float* b);

// ============================================================================
// Element-wise loops (float32 SIMD)
// ============================================================================

// out[i] = a[i] + alpha * b[i],  i in [0, n)
void add_loop(const float* a, const float* b, float* out, int64_t n, float alpha);

// out[i] = a[i] - b[i],  i in [0, n)
void sub_loop(const float* a, const float* b, float* out, int64_t n);

// out[i] = a[i] * b[i],  i in [0, n)
void mul_loop(const float* a, const float* b, float* out, int64_t n);

// out[i] = a[i] / b[i],  i in [0, n)
void div_loop(const float* a, const float* b, float* out, int64_t n);

// out[i] = a[i] + alpha * b[i],  broadcast b[j] over rows  (bias add pattern)
// a is [outer, inner], b is [inner], out is [outer, inner]
void add_broadcast_loop(const float* a, const float* b, float* out,
                        int64_t outer, int64_t inner, float alpha);

// ============================================================================
// Unary loops (float32 SIMD)
// ============================================================================

void neg_loop(const float* in, float* out, int64_t n);
void abs_loop(const float* in, float* out, int64_t n);
void sqrt_loop(const float* in, float* out, int64_t n);
void rsqrt_loop(const float* in, float* out, int64_t n);
void square_loop(const float* in, float* out, int64_t n);
void exp_loop(const float* in, float* out, int64_t n);
void log_loop(const float* in, float* out, int64_t n);
void log2_loop(const float* in, float* out, int64_t n);
void log10_loop(const float* in, float* out, int64_t n);
void sin_loop(const float* in, float* out, int64_t n);
void cos_loop(const float* in, float* out, int64_t n);
void tan_loop(const float* in, float* out, int64_t n);
void tanh_loop(const float* in, float* out, int64_t n);
void sigmoid_loop(const float* in, float* out, int64_t n);
void relu_loop(const float* in, float* out, int64_t n);
void reciprocal_loop(const float* in, float* out, int64_t n);
void ceil_loop(const float* in, float* out, int64_t n);
void floor_loop(const float* in, float* out, int64_t n);
void round_loop(const float* in, float* out, int64_t n);
void sign_loop(const float* in, float* out, int64_t n);

// ============================================================================
// Reduction loops (float32 SIMD)
// ============================================================================

// Sum all elements
float sum_loop(const float* data, int64_t n);

// Sum along a dimension: accumulate reduce_size rows of inner_size elements
void sum_dim_loop(const float* in, float* out,
                  int64_t outer_size, int64_t reduce_size, int64_t inner_size);

// ============================================================================
// In-place scalar loops (optimizer hot paths)
// ============================================================================

// data[i] *= scalar
void mul_scalar_inplace(float* data, float scalar, int64_t n);

// data[i] += scalar * other[i]
void axpy_inplace(float* data, float scalar, const float* other, int64_t n);

// Adam/AdamW inner loop: single pass over all parameter elements
// p -= lr * m_hat / (sqrt(v_hat) + eps)
// where m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
void adam_step_loop(float* param, const float* grad,
                    float* exp_avg, float* exp_avg_sq,
                    int64_t n, float lr, float beta1, float beta2,
                    float eps, float weight_decay,
                    float bias_correction1, float bias_correction2,
                    bool amsgrad, float* max_exp_avg_sq);

// SGD inner loop with momentum
void sgd_step_loop(float* param, const float* grad, float* momentum_buf,
                   int64_t n, float lr, float momentum, float dampening,
                   float weight_decay, bool nesterov);

// ============================================================================
// Fused multi-parameter optimizer steps (E2K cache-friendly)
// ============================================================================
// On Elbrus E2K, per-parameter function calls cause instruction cache misses
// because parameters are scattered in memory. These fused versions collect ALL
// parameters into a single array of packs and process them in ONE function call.
// Benefits:
//   1. Single function entry — no per-parameter call overhead
//   2. Bias correction computed once, not per-parameter
//   3. LCC can schedule the entire outer+inner loop as one VLIW region
//   4. Better branch predictor utilization (one loop structure, not N)

struct AdamParamPack {
    float* param;
    const float* grad;
    float* exp_avg;
    float* exp_avg_sq;
    int64_t numel;
};

// Fused multi-parameter Adam: process ALL parameters in one function call
// Reduces per-parameter function call overhead and improves instruction cache
void fused_adam_multi(AdamParamPack* params, int num_params,
                     float lr, float beta1, float beta2, float eps,
                     float weight_decay, int step,
                     bool amsgrad = false, float** max_exp_avg_sq = nullptr);

struct SGDParamPack {
    float* param;
    const float* grad;
    float* momentum_buf;  // nullptr if no momentum
    int64_t numel;
};

// Fused multi-parameter SGD: process ALL parameters in one function call
void fused_sgd_multi(SGDParamPack* params, int num_params,
                    float lr, float momentum, float dampening,
                    float weight_decay, bool nesterov);

// ============================================================================
// Fused kernels (E2K-optimized: large loop bodies, no branches)
// ============================================================================

// ============================================================================
// Fused backward helpers (zero-intermediate backward passes)
// ============================================================================

// Apply relu mask in-place: out[i] = (mask[i] > 0) ? grad[i] : 0
// mask is the post-relu output (positive = active)
void relu_mask_mul(const float* grad, const float* mask, float* out, int64_t n);

// Column sum: out[j] = sum_i(data[i*cols + j]) for j in [0, cols)
// Used for grad_bias = grad.sum(dim=0)
void col_sum(const float* data, float* out, int64_t rows, int64_t cols);

// In-place add: dst[i] += src[i]  (no allocation)
void add_inplace(float* dst, const float* src, int64_t n);

// Fused bias_add + relu: out[i*N+j] = max(0, out[i*N+j] + bias[j])
// Single pass avoids extra memory traffic (write-back + re-read)
void bias_relu_fused(float* out, const float* bias, int64_t M, int64_t N);

// Fused bias_add + gelu: out[i*N+j] = gelu(out[i*N+j] + bias[j])
void bias_gelu_fused(float* out, const float* bias, int64_t M, int64_t N);

// Fused cross-entropy: softmax + log + nll in one pass
// Avoids 3 separate passes over logits (max, exp+sum, log+gather)
// loss = -log(softmax(logits)[target]) averaged over batch
// grad[i,j] = softmax[i,j] - (j == target[i]) / batch
void cross_entropy_fused(const float* logits, const int64_t* targets,
                         float* loss, float* grad,
                         int64_t batch, int64_t classes);

// Fused softmax: max + exp + sum + normalize in one pass per row
// Avoids 3 separate passes over data
void softmax_fused(const float* in, float* out, int64_t rows, int64_t cols);

// Fused residual + layer_norm: out = LayerNorm(x + residual)
// Avoids materializing the intermediate x + residual tensor
void residual_layernorm_fused(const float* x, const float* residual,
                              const float* gamma, const float* beta_param,
                              float* out, int64_t rows, int64_t cols, float eps);

} // namespace hot
} // namespace native
} // namespace at
