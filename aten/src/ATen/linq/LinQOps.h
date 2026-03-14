#pragma once
// ============================================================================
// LinQOps.h — Operation launch wrappers for LinQ H1M
// ============================================================================
// Dispatches to emulator (or hardware when SDK available)
// ============================================================================

#include "aten/src/ATen/linq/LinQEmulator.h"

namespace at {
namespace linq_ops {

// ============================================================================
// Matrix operations
// ============================================================================

inline void launch_matmul(const float* A, const float* B, float* C,
                          int64_t M, int64_t K, int64_t N) {
    linq::LinQEmulator::get().matmul_fp32(A, B, C, M, K, N);
}

inline void launch_matvec(const float* A, const float* x, float* y,
                          int64_t M, int64_t N) {
    linq::LinQEmulator::get().matvec(A, x, y, M, N);
}

inline float launch_dot(const float* a, const float* b, int64_t n) {
    return linq::LinQEmulator::get().dot(a, b, n);
}

// ============================================================================
// Activation functions
// ============================================================================

inline void launch_relu(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().relu(in, out, n);
}
inline void launch_silu(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().silu(in, out, n);
}
inline void launch_gelu(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().gelu(in, out, n);
}
inline void launch_sigmoid(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().sigmoid(in, out, n);
}
inline void launch_tanh(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().tanh_op(in, out, n);
}
inline void launch_leaky_relu(const float* in, float* out, int64_t n, float alpha) {
    linq::LinQEmulator::get().leaky_relu(in, out, n, alpha);
}
inline void launch_softmax(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().softmax(in, out, n);
}

// ============================================================================
// Normalization
// ============================================================================

inline void launch_layernorm(const float* in, const float* w, const float* b,
                             float* out, int64_t batch, int64_t hidden, float eps) {
    linq::LinQEmulator::get().layernorm(in, w, b, out, batch, hidden, eps);
}
inline void launch_rmsnorm(const float* in, const float* w,
                           float* out, int64_t numel, float eps) {
    linq::LinQEmulator::get().rmsnorm(in, w, out, numel, eps);
}

// ============================================================================
// Element-wise binary
// ============================================================================

inline void launch_add(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().elem_add(a, b, out, n);
}
inline void launch_sub(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().elem_sub(a, b, out, n);
}
inline void launch_mul(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().elem_mul(a, b, out, n);
}
inline void launch_div(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().elem_div(a, b, out, n);
}
inline void launch_maximum(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().maximum_op(a, b, out, n);
}
inline void launch_minimum(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().minimum_op(a, b, out, n);
}

// ============================================================================
// Unary ops
// ============================================================================

inline void launch_neg(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().neg(in, out, n);
}
inline void launch_abs(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().abs_op(in, out, n);
}
inline void launch_sqrt(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().sqrt_op(in, out, n);
}
inline void launch_exp(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().exp_op(in, out, n);
}
inline void launch_log(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().log_op(in, out, n);
}
inline void launch_sin(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().sin_op(in, out, n);
}
inline void launch_cos(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().cos_op(in, out, n);
}
inline void launch_tan(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().tan_op(in, out, n);
}
inline void launch_log2(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().log2_op(in, out, n);
}
inline void launch_log10(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().log10_op(in, out, n);
}
inline void launch_rsqrt(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().rsqrt_op(in, out, n);
}
inline void launch_square(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().square_op(in, out, n);
}
inline void launch_reciprocal(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().reciprocal_op(in, out, n);
}
inline void launch_ceil(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().ceil_op(in, out, n);
}
inline void launch_floor(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().floor_op(in, out, n);
}
inline void launch_round(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().round_op(in, out, n);
}
inline void launch_sign(const float* in, float* out, int64_t n) {
    linq::LinQEmulator::get().sign_op(in, out, n);
}

// ============================================================================
// Clamp
// ============================================================================

inline void launch_clamp(const float* in, float* out, int64_t n, float lo, float hi) {
    linq::LinQEmulator::get().clamp_op(in, out, n, lo, hi);
}

// ============================================================================
// Scalar ops
// ============================================================================

inline void launch_add_scalar(const float* in, float* out, int64_t n, float s) {
    linq::LinQEmulator::get().add_scalar(in, out, n, s);
}
inline void launch_mul_scalar(const float* in, float* out, int64_t n, float s) {
    linq::LinQEmulator::get().mul_scalar(in, out, n, s);
}
inline void launch_pow_scalar(const float* in, float* out, int64_t n, float p) {
    linq::LinQEmulator::get().pow_scalar(in, out, n, p);
}

// ============================================================================
// Comparison ops
// ============================================================================

inline void launch_eq(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().eq_op(a, b, out, n);
}
inline void launch_ne(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().ne_op(a, b, out, n);
}
inline void launch_lt(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().lt_op(a, b, out, n);
}
inline void launch_le(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().le_op(a, b, out, n);
}
inline void launch_gt(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().gt_op(a, b, out, n);
}
inline void launch_ge(const float* a, const float* b, float* out, int64_t n) {
    linq::LinQEmulator::get().ge_op(a, b, out, n);
}
inline void launch_eq_scalar(const float* a, float* out, int64_t n, float s) {
    linq::LinQEmulator::get().eq_scalar(a, out, n, s);
}
inline void launch_ne_scalar(const float* a, float* out, int64_t n, float s) {
    linq::LinQEmulator::get().ne_scalar(a, out, n, s);
}
inline void launch_lt_scalar(const float* a, float* out, int64_t n, float s) {
    linq::LinQEmulator::get().lt_scalar(a, out, n, s);
}
inline void launch_le_scalar(const float* a, float* out, int64_t n, float s) {
    linq::LinQEmulator::get().le_scalar(a, out, n, s);
}
inline void launch_gt_scalar(const float* a, float* out, int64_t n, float s) {
    linq::LinQEmulator::get().gt_scalar(a, out, n, s);
}
inline void launch_ge_scalar(const float* a, float* out, int64_t n, float s) {
    linq::LinQEmulator::get().ge_scalar(a, out, n, s);
}

// ============================================================================
// Reductions
// ============================================================================

inline float launch_sum(const float* in, int64_t n) {
    return linq::LinQEmulator::get().sum(in, n);
}
inline float launch_max(const float* in, int64_t n) {
    return linq::LinQEmulator::get().max_val(in, n);
}
inline float launch_min(const float* in, int64_t n) {
    return linq::LinQEmulator::get().min_val(in, n);
}
inline int64_t launch_argmax(const float* in, int64_t n) {
    return linq::LinQEmulator::get().argmax(in, n);
}
inline int64_t launch_argmin(const float* in, int64_t n) {
    return linq::LinQEmulator::get().argmin(in, n);
}

// ============================================================================
// Memory ops
// ============================================================================

inline void launch_fill(float* data, int64_t n, float val) {
    linq::LinQEmulator::get().fill(data, n, val);
}
inline void launch_copy(const float* src, float* dst, int64_t n) {
    linq::LinQEmulator::get().copy(src, dst, n);
}

// ============================================================================
// Fused ops
// ============================================================================

inline void launch_addcmul(const float* self, const float* t1, const float* t2,
                           float* out, int64_t n, float value) {
    linq::LinQEmulator::get().addcmul(self, t1, t2, out, n, value);
}
inline void launch_addcdiv(const float* self, const float* t1, const float* t2,
                           float* out, int64_t n, float value) {
    linq::LinQEmulator::get().addcdiv(self, t1, t2, out, n, value);
}

} // namespace linq_ops
} // namespace at
