#pragma once
// ============================================================================
// TudaMath.h — Vectorized transcendental math for all TUDA architectures
// ============================================================================
// Provides exp, log, sin, cos, tanh, sigmoid, etc. using the VecF abstraction.
// AVX2: uses existing Cephes implementations from VectorizedOps.h
// NEON: Cephes polynomial with NEON intrinsics
// E2K/Scalar: std::math (LCC optimizes to packed FMA on Elbrus)
// ============================================================================

#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"

#if defined(TUDA_AVX2)
#include "aten/src/ATen/native/cpu/VectorizedOps.h"
#endif

#include <cmath>
#include <algorithm>

namespace at {
namespace native {
namespace tuda {

// ============================================================================
// exp
// ============================================================================

static inline VecF exp_vec(VecF x) {
#if defined(TUDA_AVX2)
    return VecF(vec::exp256_ps(x.val));
#elif defined(TUDA_NEON)
    // Cephes exp: same algorithm as AVX2, NEON intrinsics
    const float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
    const float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    const float32x4_t log2e  = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t c1     = vdupq_n_f32(0.693359375f);
    const float32x4_t c2     = vdupq_n_f32(-2.12194440e-4f);
    const float32x4_t p0 = vdupq_n_f32(1.9875691500E-4f);
    const float32x4_t p1 = vdupq_n_f32(1.3981999507E-3f);
    const float32x4_t p2 = vdupq_n_f32(8.3334519073E-3f);
    const float32x4_t p3 = vdupq_n_f32(4.1665795894E-2f);
    const float32x4_t p4 = vdupq_n_f32(1.6666665459E-1f);
    const float32x4_t p5 = vdupq_n_f32(5.0000001201E-1f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    float32x4_t v = x.val;
    v = vminq_f32(v, exp_hi);
    v = vmaxq_f32(v, exp_lo);

    float32x4_t fx = vmulq_f32(v, log2e);
    fx = vrndnq_f32(fx); // round to nearest

    v = vsubq_f32(v, vmulq_f32(fx, c1));
    v = vsubq_f32(v, vmulq_f32(fx, c2));

    float32x4_t y = p0;
    y = vfmaq_f32(p1, y, v);
    y = vfmaq_f32(p2, y, v);
    y = vfmaq_f32(p3, y, v);
    y = vfmaq_f32(p4, y, v);
    y = vfmaq_f32(p5, y, v);
    y = vfmaq_f32(vaddq_f32(v, one), y, vmulq_f32(v, v));

    // 2^n: shift integer into float exponent
    int32x4_t n = vcvtnq_s32_f32(fx);
    n = vaddq_s32(n, vdupq_n_s32(0x7f));
    n = vshlq_n_s32(n, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(n);

    return VecF(vmulq_f32(y, pow2n));
#else
    // E2K / Scalar: element-wise std::exp
    VecF result;
#if defined(TUDA_E2K)
    result.v[0] = std::exp(x.v[0]); result.v[1] = std::exp(x.v[1]);
    result.v[2] = std::exp(x.v[2]); result.v[3] = std::exp(x.v[3]);
#else
    result = VecF(std::exp(x.val));
#endif
    return result;
#endif
}

// ============================================================================
// log
// ============================================================================

static inline VecF log_vec(VecF x) {
#if defined(TUDA_AVX2)
    return VecF(vec::log256_ps(x.val));
#elif defined(TUDA_E2K)
    VecF r;
    r.v[0] = std::log(x.v[0]); r.v[1] = std::log(x.v[1]);
    r.v[2] = std::log(x.v[2]); r.v[3] = std::log(x.v[3]);
    return r;
#elif defined(TUDA_NEON)
    // Simplified: use element-wise for NEON (can optimize later)
    float tmp[4];
    vst1q_f32(tmp, x.val);
    tmp[0] = std::log(tmp[0]); tmp[1] = std::log(tmp[1]);
    tmp[2] = std::log(tmp[2]); tmp[3] = std::log(tmp[3]);
    return VecF(vld1q_f32(tmp));
#else
    return VecF(std::log(x.val));
#endif
}

// ============================================================================
// sin, cos
// ============================================================================

static inline VecF sin_vec(VecF x) {
#if defined(TUDA_AVX2)
    return VecF(vec::sin256_ps(x.val));
#elif defined(TUDA_E2K)
    VecF r;
    r.v[0] = std::sin(x.v[0]); r.v[1] = std::sin(x.v[1]);
    r.v[2] = std::sin(x.v[2]); r.v[3] = std::sin(x.v[3]);
    return r;
#elif defined(TUDA_NEON)
    float tmp[4];
    vst1q_f32(tmp, x.val);
    tmp[0] = std::sin(tmp[0]); tmp[1] = std::sin(tmp[1]);
    tmp[2] = std::sin(tmp[2]); tmp[3] = std::sin(tmp[3]);
    return VecF(vld1q_f32(tmp));
#else
    return VecF(std::sin(x.val));
#endif
}

static inline VecF cos_vec(VecF x) {
#if defined(TUDA_AVX2)
    return VecF(vec::cos256_ps(x.val));
#elif defined(TUDA_E2K)
    VecF r;
    r.v[0] = std::cos(x.v[0]); r.v[1] = std::cos(x.v[1]);
    r.v[2] = std::cos(x.v[2]); r.v[3] = std::cos(x.v[3]);
    return r;
#elif defined(TUDA_NEON)
    float tmp[4];
    vst1q_f32(tmp, x.val);
    tmp[0] = std::cos(tmp[0]); tmp[1] = std::cos(tmp[1]);
    tmp[2] = std::cos(tmp[2]); tmp[3] = std::cos(tmp[3]);
    return VecF(vld1q_f32(tmp));
#else
    return VecF(std::cos(x.val));
#endif
}

// ============================================================================
// tanh = 1 - 2/(1 + exp(2x))
// ============================================================================

static inline VecF tanh_vec(VecF x) {
#if defined(TUDA_AVX2)
    return VecF(vec::tanh256_ps(x.val));
#else
    VecF two = VecF::broadcast(2.0f);
    VecF one = VecF::broadcast(1.0f);
    VecF nine = VecF::broadcast(9.0f);
    VecF neg_nine = VecF::broadcast(-9.0f);
    x = x.max(neg_nine).min(nine);
    VecF e2x = exp_vec(x * two);
    return one - two / (one + e2x);
#endif
}

// ============================================================================
// sigmoid = 1 / (1 + exp(-x))
// ============================================================================

static inline VecF sigmoid_vec(VecF x) {
#if defined(TUDA_AVX2)
    return VecF(vec::sigmoid256_ps(x.val));
#else
    VecF one = VecF::broadcast(1.0f);
    VecF clamp_hi = VecF::broadcast(20.0f);
    VecF clamp_lo = VecF::broadcast(-20.0f);
    x = x.max(clamp_lo).min(clamp_hi);
    VecF negx = VecF::zero() - x;
    VecF ex = exp_vec(negx);
    return one / (one + ex);
#endif
}

// ============================================================================
// Simple unary ops via VecF
// ============================================================================

static inline VecF neg_vec(VecF v) { return v.neg(); }
static inline VecF abs_vec(VecF v) { return v.abs(); }
static inline VecF sqrt_vec(VecF v) { return v.sqrt(); }
static inline VecF rsqrt_vec(VecF v) { return v.rsqrt(); }
static inline VecF reciprocal_vec(VecF v) { return v.reciprocal(); }

static inline VecF relu_vec(VecF v) {
    return v.max(VecF::zero());
}

static inline VecF silu_vec(VecF v) {
    return v * sigmoid_vec(v);
}

static inline VecF gelu_vec(VecF v) {
    // GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    VecF half = VecF::broadcast(0.5f);
    VecF one = VecF::broadcast(1.0f);
    VecF c = VecF::broadcast(0.044715f);
    VecF sqrt2pi = VecF::broadcast(0.7978845608028654f); // sqrt(2/pi)
    VecF x3 = v * v * v;
    VecF inner = sqrt2pi * (v + c * x3);
    return half * v * (one + tanh_vec(inner));
}

static inline VecF leaky_relu_vec(VecF v, float alpha = 0.01f) {
    VecF pos = v.max(VecF::zero());
    VecF neg_part = v.min(VecF::zero()) * VecF::broadcast(alpha);
    return pos + neg_part;
}

static inline VecF clamp_vec(VecF v, float lo, float hi) {
    return v.max(VecF::broadcast(lo)).min(VecF::broadcast(hi));
}

static inline VecF square_vec(VecF v) {
    return v * v;
}

static inline VecF ceil_vec(VecF v) { return v.ceil(); }
static inline VecF floor_vec(VecF v) { return v.floor(); }
static inline VecF round_vec(VecF v) { return v.round(); }

static inline VecF tan_vec(VecF v) {
    return sin_vec(v) / cos_vec(v);
}

static inline VecF log2_vec(VecF v) {
    return log_vec(v) * VecF::broadcast(1.4426950408889634f); // 1/ln(2)
}

static inline VecF log10_vec(VecF v) {
    return log_vec(v) * VecF::broadcast(0.4342944819032518f); // 1/ln(10)
}

static inline VecF sign_vec(VecF v) {
#if defined(TUDA_AVX2)
    __m256 zero = _mm256_setzero_ps();
    __m256 pos = _mm256_and_ps(_mm256_cmp_ps(v.val, zero, _CMP_GT_OS), _mm256_set1_ps(1.0f));
    __m256 neg = _mm256_and_ps(_mm256_cmp_ps(v.val, zero, _CMP_LT_OS), _mm256_set1_ps(-1.0f));
    return VecF(_mm256_or_ps(pos, neg));
#elif defined(TUDA_E2K)
    VecF r;
    for (int i = 0; i < 4; ++i) r.v[i] = (v.v[i] > 0.0f) ? 1.0f : ((v.v[i] < 0.0f) ? -1.0f : 0.0f);
    return r;
#elif defined(TUDA_NEON)
    float tmp[4];
    vst1q_f32(tmp, v.val);
    for (int i = 0; i < 4; ++i) tmp[i] = (tmp[i] > 0.0f) ? 1.0f : ((tmp[i] < 0.0f) ? -1.0f : 0.0f);
    return VecF(vld1q_f32(tmp));
#else
    return VecF((v.val > 0.0f) ? 1.0f : ((v.val < 0.0f) ? -1.0f : 0.0f));
#endif
}

// ============================================================================
// Reduction helpers
// ============================================================================

static inline float vec_max(const float* data, int64_t n) {
    constexpr int W = VecF::width;
    if (n == 0) return -INFINITY;
    VecF vmax = VecF::broadcast(-INFINITY);
    int64_t i = 0;
    for (; i + W <= n; i += W) {
        vmax = vmax.max(VecF::load(data + i));
    }
    float result = vmax.hsum(); // not ideal for max, but handles scalar
    // For actual max, unroll manually
#if defined(TUDA_AVX2)
    __m128 hi = _mm256_extractf128_ps(vmax.val, 1);
    __m128 lo = _mm256_castps256_ps128(vmax.val);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
    result = _mm_cvtss_f32(lo);
#elif VecF::width == 4
    float tmp[4];
    vmax.store(tmp);
    result = std::max(std::max(tmp[0], tmp[1]), std::max(tmp[2], tmp[3]));
#elif VecF::width == 1
    result = vmax.val;
#endif
    for (; i < n; ++i) if (data[i] > result) result = data[i];
    return result;
}

static inline float vec_min(const float* data, int64_t n) {
    constexpr int W = VecF::width;
    if (n == 0) return INFINITY;
    VecF vmin = VecF::broadcast(INFINITY);
    int64_t i = 0;
    for (; i + W <= n; i += W) {
        vmin = vmin.min(VecF::load(data + i));
    }
    float result;
#if defined(TUDA_AVX2)
    __m128 hi = _mm256_extractf128_ps(vmin.val, 1);
    __m128 lo = _mm256_castps256_ps128(vmin.val);
    lo = _mm_min_ps(lo, hi);
    lo = _mm_min_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_min_ss(lo, _mm_movehdup_ps(lo));
    result = _mm_cvtss_f32(lo);
#elif VecF::width == 4
    float tmp[4];
    vmin.store(tmp);
    result = std::min(std::min(tmp[0], tmp[1]), std::min(tmp[2], tmp[3]));
#elif VecF::width == 1
    result = vmin.val;
#endif
    for (; i < n; ++i) if (data[i] < result) result = data[i];
    return result;
}

} // namespace tuda
} // namespace native
} // namespace at
