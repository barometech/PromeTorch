#pragma once
// ============================================================================
// TudaVec.h — Portable SIMD vector abstraction for all TUDA architectures
// ============================================================================
// Provides a zero-overhead VecF type that maps to the optimal SIMD register
// for the target architecture. All operations compile to direct intrinsics.
//
// VecF = Vec8 (AVX2) | Vec4 (NEON/E2K) | Vec1 (Scalar)
// ============================================================================

#include "aten/src/ATen/native/cpu/tuda/TudaConfig.h"

#if defined(TUDA_AVX2)
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <immintrin.h>
    #endif
#elif defined(TUDA_NEON)
    #include <arm_neon.h>
#endif

#include <cmath>

namespace at {
namespace native {
namespace tuda {

// ============================================================================
// AVX2: Vec8 — 8 × float32 in __m256
// ============================================================================

#if defined(TUDA_AVX2)

struct Vec8 {
    __m256 val;

    Vec8() : val(_mm256_setzero_ps()) {}
    explicit Vec8(__m256 v) : val(v) {}
    explicit Vec8(float s) : val(_mm256_set1_ps(s)) {}

    static Vec8 load(const float* p) { return Vec8(_mm256_loadu_ps(p)); }
    static Vec8 load_aligned(const float* p) { return Vec8(_mm256_load_ps(p)); }
    void store(float* p) const { _mm256_storeu_ps(p, val); }
    void store_aligned(float* p) const { _mm256_store_ps(p, val); }

    static Vec8 zero() { return Vec8(_mm256_setzero_ps()); }
    static Vec8 broadcast(float s) { return Vec8(_mm256_set1_ps(s)); }

    Vec8 operator+(Vec8 b) const { return Vec8(_mm256_add_ps(val, b.val)); }
    Vec8 operator-(Vec8 b) const { return Vec8(_mm256_sub_ps(val, b.val)); }
    Vec8 operator*(Vec8 b) const { return Vec8(_mm256_mul_ps(val, b.val)); }
    Vec8 operator/(Vec8 b) const { return Vec8(_mm256_div_ps(val, b.val)); }
    Vec8 operator-() const { return Vec8(_mm256_xor_ps(val, _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000))))); }

    static Vec8 fmadd(Vec8 a, Vec8 b, Vec8 c) {
        return Vec8(_mm256_fmadd_ps(a.val, b.val, c.val));
    }

    Vec8 abs() const { return Vec8(_mm256_and_ps(val, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)))); }
    Vec8 neg() const { return -(*this); }
    Vec8 max(Vec8 b) const { return Vec8(_mm256_max_ps(val, b.val)); }
    Vec8 min(Vec8 b) const { return Vec8(_mm256_min_ps(val, b.val)); }
    Vec8 sqrt() const { return Vec8(_mm256_sqrt_ps(val)); }
    Vec8 rsqrt() const { return Vec8(_mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(val))); }
    Vec8 reciprocal() const { return Vec8(_mm256_div_ps(_mm256_set1_ps(1.0f), val)); }
    Vec8 ceil() const { return Vec8(_mm256_ceil_ps(val)); }
    Vec8 floor() const { return Vec8(_mm256_floor_ps(val)); }
    Vec8 round() const { return Vec8(_mm256_round_ps(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); }

    float hsum() const {
        __m128 hi = _mm256_extractf128_ps(val, 1);
        __m128 lo = _mm256_castps256_ps128(val);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        lo = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, lo);
        lo = _mm_add_ss(lo, shuf);
        return _mm_cvtss_f32(lo);
    }

    static constexpr int width = 8;
};

using VecF = Vec8;

// ============================================================================
// ARM NEON: Vec4 — 4 × float32 in float32x4_t
// ============================================================================

#elif defined(TUDA_NEON)

struct Vec4 {
    float32x4_t val;

    Vec4() : val(vdupq_n_f32(0.0f)) {}
    explicit Vec4(float32x4_t v) : val(v) {}
    explicit Vec4(float s) : val(vdupq_n_f32(s)) {}

    static Vec4 load(const float* p) { return Vec4(vld1q_f32(p)); }
    void store(float* p) const { vst1q_f32(p, val); }

    static Vec4 zero() { return Vec4(vdupq_n_f32(0.0f)); }
    static Vec4 broadcast(float s) { return Vec4(vdupq_n_f32(s)); }

    Vec4 operator+(Vec4 b) const { return Vec4(vaddq_f32(val, b.val)); }
    Vec4 operator-(Vec4 b) const { return Vec4(vsubq_f32(val, b.val)); }
    Vec4 operator*(Vec4 b) const { return Vec4(vmulq_f32(val, b.val)); }
    Vec4 operator/(Vec4 b) const { return Vec4(vdivq_f32(val, b.val)); }
    Vec4 operator-() const { return Vec4(vnegq_f32(val)); }

    static Vec4 fmadd(Vec4 a, Vec4 b, Vec4 c) {
        return Vec4(vfmaq_f32(c.val, a.val, b.val)); // c + a*b
    }

    Vec4 abs() const { return Vec4(vabsq_f32(val)); }
    Vec4 neg() const { return -(*this); }
    Vec4 max(Vec4 b) const { return Vec4(vmaxq_f32(val, b.val)); }
    Vec4 min(Vec4 b) const { return Vec4(vminq_f32(val, b.val)); }
    Vec4 sqrt() const { return Vec4(vsqrtq_f32(val)); }
    Vec4 rsqrt() const {
        float32x4_t est = vrsqrteq_f32(val);
        est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(val, est), est));
        est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(val, est), est));
        return Vec4(est);
    }
    Vec4 reciprocal() const {
        float32x4_t rec = vrecpeq_f32(val);
        rec = vmulq_f32(rec, vrecpsq_f32(val, rec));
        rec = vmulq_f32(rec, vrecpsq_f32(val, rec));
        return Vec4(rec);
    }
    Vec4 ceil() const { return Vec4(vrndpq_f32(val)); }
    Vec4 floor() const { return Vec4(vrndmq_f32(val)); }
    Vec4 round() const { return Vec4(vrndnq_f32(val)); }

    float hsum() const {
        float32x2_t sum = vadd_f32(vget_low_f32(val), vget_high_f32(val));
        sum = vpadd_f32(sum, sum);
        return vget_lane_f32(sum, 0);
    }

    static constexpr int width = 4;
};

using VecF = Vec4;

// ============================================================================
// E2K Elbrus: Vec4 — plain C, LCC auto-vectorizes to packed FMA
// ============================================================================

#elif defined(TUDA_E2K)

struct Vec4 {
    float v[4];

    Vec4() : v{0,0,0,0} {}
    explicit Vec4(float s) : v{s,s,s,s} {}
    Vec4(float a, float b, float c, float d) : v{a,b,c,d} {}

    static Vec4 load(const float* p) {
        Vec4 r; r.v[0]=p[0]; r.v[1]=p[1]; r.v[2]=p[2]; r.v[3]=p[3]; return r;
    }
    void store(float* p) const { p[0]=v[0]; p[1]=v[1]; p[2]=v[2]; p[3]=v[3]; }

    static Vec4 zero() { return Vec4(0.0f); }
    static Vec4 broadcast(float s) { return Vec4(s); }

    Vec4 operator+(Vec4 b) const { return {v[0]+b.v[0], v[1]+b.v[1], v[2]+b.v[2], v[3]+b.v[3]}; }
    Vec4 operator-(Vec4 b) const { return {v[0]-b.v[0], v[1]-b.v[1], v[2]-b.v[2], v[3]-b.v[3]}; }
    Vec4 operator*(Vec4 b) const { return {v[0]*b.v[0], v[1]*b.v[1], v[2]*b.v[2], v[3]*b.v[3]}; }
    Vec4 operator/(Vec4 b) const { return {v[0]/b.v[0], v[1]/b.v[1], v[2]/b.v[2], v[3]/b.v[3]}; }
    Vec4 operator-() const { return {-v[0], -v[1], -v[2], -v[3]}; }

    static Vec4 fmadd(Vec4 a, Vec4 b, Vec4 c) {
        return {a.v[0]*b.v[0]+c.v[0], a.v[1]*b.v[1]+c.v[1],
                a.v[2]*b.v[2]+c.v[2], a.v[3]*b.v[3]+c.v[3]};
    }

    Vec4 abs() const { return {std::fabs(v[0]), std::fabs(v[1]), std::fabs(v[2]), std::fabs(v[3])}; }
    Vec4 neg() const { return -(*this); }
    Vec4 max(Vec4 b) const { return {std::fmax(v[0],b.v[0]), std::fmax(v[1],b.v[1]), std::fmax(v[2],b.v[2]), std::fmax(v[3],b.v[3])}; }
    Vec4 min(Vec4 b) const { return {std::fmin(v[0],b.v[0]), std::fmin(v[1],b.v[1]), std::fmin(v[2],b.v[2]), std::fmin(v[3],b.v[3])}; }
    Vec4 sqrt() const { return {std::sqrt(v[0]), std::sqrt(v[1]), std::sqrt(v[2]), std::sqrt(v[3])}; }
    Vec4 rsqrt() const { return {1.0f/std::sqrt(v[0]), 1.0f/std::sqrt(v[1]), 1.0f/std::sqrt(v[2]), 1.0f/std::sqrt(v[3])}; }
    Vec4 reciprocal() const { return {1.0f/v[0], 1.0f/v[1], 1.0f/v[2], 1.0f/v[3]}; }
    Vec4 ceil() const { return {std::ceil(v[0]), std::ceil(v[1]), std::ceil(v[2]), std::ceil(v[3])}; }
    Vec4 floor() const { return {std::floor(v[0]), std::floor(v[1]), std::floor(v[2]), std::floor(v[3])}; }
    Vec4 round() const { return {std::round(v[0]), std::round(v[1]), std::round(v[2]), std::round(v[3])}; }

    float hsum() const { return v[0]+v[1]+v[2]+v[3]; }

    static constexpr int width = 4;
};

using VecF = Vec4;

// ============================================================================
// Scalar fallback: Vec1 — single float
// ============================================================================

#else

struct Vec1 {
    float val;

    Vec1() : val(0) {}
    explicit Vec1(float s) : val(s) {}

    static Vec1 load(const float* p) { return Vec1(*p); }
    void store(float* p) const { *p = val; }

    static Vec1 zero() { return Vec1(0.0f); }
    static Vec1 broadcast(float s) { return Vec1(s); }

    Vec1 operator+(Vec1 b) const { return Vec1(val + b.val); }
    Vec1 operator-(Vec1 b) const { return Vec1(val - b.val); }
    Vec1 operator*(Vec1 b) const { return Vec1(val * b.val); }
    Vec1 operator/(Vec1 b) const { return Vec1(val / b.val); }
    Vec1 operator-() const { return Vec1(-val); }

    static Vec1 fmadd(Vec1 a, Vec1 b, Vec1 c) { return Vec1(a.val*b.val + c.val); }

    Vec1 abs() const { return Vec1(std::fabs(val)); }
    Vec1 neg() const { return Vec1(-val); }
    Vec1 max(Vec1 b) const { return Vec1(std::fmax(val, b.val)); }
    Vec1 min(Vec1 b) const { return Vec1(std::fmin(val, b.val)); }
    Vec1 sqrt() const { return Vec1(std::sqrt(val)); }
    Vec1 rsqrt() const { return Vec1(1.0f / std::sqrt(val)); }
    Vec1 reciprocal() const { return Vec1(1.0f / val); }
    Vec1 ceil() const { return Vec1(std::ceil(val)); }
    Vec1 floor() const { return Vec1(std::floor(val)); }
    Vec1 round() const { return Vec1(std::round(val)); }

    float hsum() const { return val; }

    static constexpr int width = 1;
};

using VecF = Vec1;

#endif

// ============================================================================
// Utility: vectorized apply over array using VecF
// ============================================================================

template<typename VecFunc, typename ScalarFunc>
static inline void vec_apply(const float* in, float* out, int64_t n,
                             VecFunc vec_fn, ScalarFunc scalar_fn) {
    constexpr int W = VecF::width;
    int64_t i = 0;
    for (; i + 4*W <= n; i += 4*W) {
        vec_fn(VecF::load(in + i)).store(out + i);
        vec_fn(VecF::load(in + i + W)).store(out + i + W);
        vec_fn(VecF::load(in + i + 2*W)).store(out + i + 2*W);
        vec_fn(VecF::load(in + i + 3*W)).store(out + i + 3*W);
    }
    for (; i + W <= n; i += W) {
        vec_fn(VecF::load(in + i)).store(out + i);
    }
    for (; i < n; ++i) {
        out[i] = scalar_fn(in[i]);
    }
}

template<typename VecFunc, typename ScalarFunc>
static inline void vec_apply_binary(const float* a, const float* b, float* out, int64_t n,
                                    VecFunc vec_fn, ScalarFunc scalar_fn) {
    constexpr int W = VecF::width;
    int64_t i = 0;
    for (; i + W <= n; i += W) {
        vec_fn(VecF::load(a + i), VecF::load(b + i)).store(out + i);
    }
    for (; i < n; ++i) {
        out[i] = scalar_fn(a[i], b[i]);
    }
}

static inline void vec_fill(float* out, float val, int64_t n) {
    constexpr int W = VecF::width;
    VecF v = VecF::broadcast(val);
    int64_t i = 0;
    for (; i + 4*W <= n; i += 4*W) {
        v.store(out + i);
        v.store(out + i + W);
        v.store(out + i + 2*W);
        v.store(out + i + 3*W);
    }
    for (; i + W <= n; i += W) {
        v.store(out + i);
    }
    for (; i < n; ++i) {
        out[i] = val;
    }
}

static inline float vec_sum(const float* data, int64_t n) {
    constexpr int W = VecF::width;
    VecF acc0 = VecF::zero(), acc1 = VecF::zero();
    VecF acc2 = VecF::zero(), acc3 = VecF::zero();
    int64_t i = 0;
    for (; i + 4*W <= n; i += 4*W) {
        acc0 = acc0 + VecF::load(data + i);
        acc1 = acc1 + VecF::load(data + i + W);
        acc2 = acc2 + VecF::load(data + i + 2*W);
        acc3 = acc3 + VecF::load(data + i + 3*W);
    }
    VecF total = (acc0 + acc1) + (acc2 + acc3);
    float result = total.hsum();
    for (; i < n; ++i) result += data[i];
    return result;
}

static inline float vec_dot(int64_t n, const float* x, const float* y) {
    constexpr int W = VecF::width;
    VecF acc0 = VecF::zero(), acc1 = VecF::zero();
    VecF acc2 = VecF::zero(), acc3 = VecF::zero();
    int64_t i = 0;
    for (; i + 4*W <= n; i += 4*W) {
        acc0 = VecF::fmadd(VecF::load(x + i),       VecF::load(y + i),       acc0);
        acc1 = VecF::fmadd(VecF::load(x + i + W),   VecF::load(y + i + W),   acc1);
        acc2 = VecF::fmadd(VecF::load(x + i + 2*W), VecF::load(y + i + 2*W), acc2);
        acc3 = VecF::fmadd(VecF::load(x + i + 3*W), VecF::load(y + i + 3*W), acc3);
    }
    VecF total = (acc0 + acc1) + (acc2 + acc3);
    float result = total.hsum();
    for (; i < n; ++i) result += x[i] * y[i];
    return result;
}

} // namespace tuda
} // namespace native
} // namespace at
