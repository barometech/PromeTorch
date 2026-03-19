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

} // namespace hot
} // namespace native
} // namespace at
