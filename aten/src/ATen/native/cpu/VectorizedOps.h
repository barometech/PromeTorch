#pragma once
// ============================================================================
// VectorizedOps.h — AVX2 SIMD implementations for transcendental functions
// Cephes-derived polynomial approximations, float32 accuracy ~1e-7
// ============================================================================

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include <cmath>
#include <cstdint>

namespace at {
namespace native {
namespace vec {

// ============================================================================
// AVX2 Constants
// ============================================================================

static inline __m256 _ps256_1()    { return _mm256_set1_ps(1.0f); }
static inline __m256 _ps256_0p5()  { return _mm256_set1_ps(0.5f); }
static inline __m256 _ps256_0()    { return _mm256_setzero_ps(); }
static inline __m256i _pi256_0x7f(){ return _mm256_set1_epi32(0x7f); }

static inline __m256 _ps256_sign_mask() {
    return _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000)));
}
static inline __m256 _ps256_abs_mask() {
    return _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
}

// ============================================================================
// Horizontal sum of 8 floats
// ============================================================================
static inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);          // 4 floats
    __m128 shuf = _mm_movehdup_ps(lo); // {1,1,3,3}
    lo = _mm_add_ps(lo, shuf);        // {0+1, _, 2+3, _}
    shuf = _mm_movehl_ps(shuf, lo);   // {2+3, ...}
    lo = _mm_add_ss(lo, shuf);        // {0+1+2+3, ...}
    return _mm_cvtss_f32(lo);
}

// ============================================================================
// AVX2 exp(x) — Cephes-derived, range [-88, 88], accuracy ~1e-7
// ============================================================================
static inline __m256 exp256_ps(__m256 x) {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2e  = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1     = _mm256_set1_ps(0.693359375f);
    const __m256 c2     = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 p0     = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 p1     = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 p2     = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 p3     = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 p4     = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 p5     = _mm256_set1_ps(5.0000001201E-1f);
    const __m256 one    = _ps256_1();

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    // fx = round(x * log2(e))
    __m256 fx = _mm256_mul_ps(x, log2e);
    fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // x = x - fx * ln(2)  (high+low for precision)
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c1));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c2));

    // Polynomial: 1 + x*(p4 + x*(p3 + x*(p2 + x*(p1 + x*p0))))
    __m256 y = p0;
    y = _mm256_fmadd_ps(y, x, p1);
    y = _mm256_fmadd_ps(y, x, p2);
    y = _mm256_fmadd_ps(y, x, p3);
    y = _mm256_fmadd_ps(y, x, p4);
    y = _mm256_fmadd_ps(y, x, p5);
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, one));

    // Build 2^n by shifting integer into float exponent
    __m256i n = _mm256_cvtps_epi32(fx);
    n = _mm256_add_epi32(n, _pi256_0x7f());
    n = _mm256_slli_epi32(n, 23);
    __m256 pow2n = _mm256_castsi256_ps(n);

    return _mm256_mul_ps(y, pow2n);
}

// ============================================================================
// AVX2 log(x) — Cephes-derived, accuracy ~1e-7
// ============================================================================
static inline __m256 log256_ps(__m256 x) {
    const __m256 min_norm = _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000));
    const __m256 inv_mant_mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x7f800000));
    const __m256 c_0p5    = _ps256_0p5();
    const __m256 c_one    = _ps256_1();
    const __m256 c_sqrthf = _mm256_set1_ps(0.707106781186547524f);

    const __m256 log_p0 = _mm256_set1_ps(7.0376836292E-2f);
    const __m256 log_p1 = _mm256_set1_ps(-1.1514610310E-1f);
    const __m256 log_p2 = _mm256_set1_ps(1.1676998740E-1f);
    const __m256 log_p3 = _mm256_set1_ps(-1.2420140846E-1f);
    const __m256 log_p4 = _mm256_set1_ps(1.4249322787E-1f);
    const __m256 log_p5 = _mm256_set1_ps(-1.6668057665E-1f);
    const __m256 log_p6 = _mm256_set1_ps(2.0000714765E-1f);
    const __m256 log_p7 = _mm256_set1_ps(-2.4999993993E-1f);
    const __m256 log_p8 = _mm256_set1_ps(3.3333331174E-1f);
    const __m256 log_q1 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 log_q2 = _mm256_set1_ps(0.693359375f);

    // Clamp to valid range
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
    x = _mm256_max_ps(x, min_norm);

    // Extract exponent
    __m256i emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    x = _mm256_and_ps(x, inv_mant_mask);
    x = _mm256_or_ps(x, c_0p5);

    emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
    __m256 e = _mm256_cvtepi32_ps(emm0);
    e = _mm256_add_ps(e, c_one);

    // Adjust if mantissa < sqrt(0.5)
    __m256 mask = _mm256_cmp_ps(x, c_sqrthf, _CMP_LT_OS);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, c_one);
    e = _mm256_sub_ps(e, _mm256_and_ps(c_one, mask));
    x = _mm256_add_ps(x, tmp);

    __m256 z = _mm256_mul_ps(x, x);

    // Polynomial
    __m256 y = log_p0;
    y = _mm256_fmadd_ps(y, x, log_p1);
    y = _mm256_fmadd_ps(y, x, log_p2);
    y = _mm256_fmadd_ps(y, x, log_p3);
    y = _mm256_fmadd_ps(y, x, log_p4);
    y = _mm256_fmadd_ps(y, x, log_p5);
    y = _mm256_fmadd_ps(y, x, log_p6);
    y = _mm256_fmadd_ps(y, x, log_p7);
    y = _mm256_fmadd_ps(y, x, log_p8);
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);

    y = _mm256_fmadd_ps(e, log_q1, y);
    y = _mm256_sub_ps(y, _mm256_mul_ps(c_0p5, z));

    x = _mm256_add_ps(x, y);
    x = _mm256_fmadd_ps(e, log_q2, x);

    // NaN for x <= 0
    x = _mm256_or_ps(x, invalid_mask);
    return x;
}

// ============================================================================
// AVX2 sin(x) & cos(x) — Cephes, accuracy ~1e-7
// ============================================================================
static inline __m256 sin256_ps(__m256 x) {
    const __m256 sign_mask = _ps256_sign_mask();
    const __m256 abs_mask = _ps256_abs_mask();
    const __m256 c_4overpi = _mm256_set1_ps(1.27323954473516f);
    const __m256 c_dp1 = _mm256_set1_ps(-0.78515625f);
    const __m256 c_dp2 = _mm256_set1_ps(-2.4187564849853515625e-4f);
    const __m256 c_dp3 = _mm256_set1_ps(-3.77489497744594108e-8f);
    const __m256 c_sin_p0 = _mm256_set1_ps(-1.9515295891E-4f);
    const __m256 c_sin_p1 = _mm256_set1_ps(8.3321608736E-3f);
    const __m256 c_sin_p2 = _mm256_set1_ps(-1.6666654611E-1f);
    const __m256 c_cos_p0 = _mm256_set1_ps(2.443315711809948E-005f);
    const __m256 c_cos_p1 = _mm256_set1_ps(-1.388731625493765E-003f);
    const __m256 c_cos_p2 = _mm256_set1_ps(4.166664568298827E-002f);

    __m256 sign = _mm256_and_ps(x, sign_mask);
    x = _mm256_and_ps(x, abs_mask);

    // Range reduction: j = (int)(x * 4/pi)
    __m256 y = _mm256_mul_ps(x, c_4overpi);
    __m256i j = _mm256_cvtps_epi32(y);
    // Make j odd
    j = _mm256_add_epi32(j, _mm256_and_si256(j, _mm256_set1_epi32(1)));
    y = _mm256_cvtepi32_ps(j);

    // Extended precision modular arithmetic
    x = _mm256_fmadd_ps(y, c_dp1, x);
    x = _mm256_fmadd_ps(y, c_dp2, x);
    x = _mm256_fmadd_ps(y, c_dp3, x);

    // Sign
    __m256i j_and_4 = _mm256_slli_epi32(_mm256_and_si256(j, _mm256_set1_epi32(4)), 29);
    sign = _mm256_xor_ps(sign, _mm256_castsi256_ps(j_and_4));

    // Polynomial selection: sin or cos polynomial based on j&2
    __m256 poly_mask = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(_mm256_and_si256(j, _mm256_set1_epi32(2)), _mm256_setzero_si256()));

    __m256 z = _mm256_mul_ps(x, x);

    // cos polynomial
    __m256 yc = c_cos_p0;
    yc = _mm256_fmadd_ps(yc, z, c_cos_p1);
    yc = _mm256_fmadd_ps(yc, z, c_cos_p2);
    yc = _mm256_mul_ps(yc, z);
    yc = _mm256_mul_ps(yc, z);
    yc = _mm256_sub_ps(yc, _mm256_mul_ps(_ps256_0p5(), z));
    yc = _mm256_add_ps(yc, _ps256_1());

    // sin polynomial
    __m256 ys = c_sin_p0;
    ys = _mm256_fmadd_ps(ys, z, c_sin_p1);
    ys = _mm256_fmadd_ps(ys, z, c_sin_p2);
    ys = _mm256_mul_ps(ys, z);
    ys = _mm256_fmadd_ps(ys, x, x);

    // Select
    y = _mm256_blendv_ps(yc, ys, poly_mask);
    y = _mm256_xor_ps(y, sign);
    return y;
}

static inline __m256 cos256_ps(__m256 x) {
    // cos(x) = sin(x + pi/2)
    return sin256_ps(_mm256_add_ps(x, _mm256_set1_ps(1.5707963267948966f)));
}

// ============================================================================
// AVX2 tanh(x) = 1 - 2/(1 + exp(2x))
// ============================================================================
static inline __m256 tanh256_ps(__m256 x) {
    const __m256 one = _ps256_1();
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 neg_two = _mm256_set1_ps(-2.0f);

    // Clamp to [-9, 9] for stability
    x = _mm256_max_ps(x, _mm256_set1_ps(-9.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(9.0f));

    __m256 e2x = exp256_ps(_mm256_mul_ps(two, x));
    __m256 denom = _mm256_add_ps(one, e2x);
    // tanh = 1 - 2/denom = (denom - 2) / denom
    return _mm256_sub_ps(one, _mm256_div_ps(two, denom));
}

// ============================================================================
// AVX2 sigmoid(x) = 1 / (1 + exp(-x))
// ============================================================================
static inline __m256 sigmoid256_ps(__m256 x) {
    const __m256 one = _ps256_1();

    // Clamp for stability
    x = _mm256_max_ps(x, _mm256_set1_ps(-20.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(20.0f));

    __m256 negx = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 ex = exp256_ps(negx);
    return _mm256_div_ps(one, _mm256_add_ps(one, ex));
}

// ============================================================================
// Vectorized apply: process array with AVX2 function + scalar tail
// ============================================================================
template<typename AVXFunc, typename ScalarFunc>
static inline void vectorized_apply(const float* in, float* out, int64_t n,
                                     AVXFunc avx_fn, ScalarFunc scalar_fn) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, avx_fn(v));
    }
    for (; i < n; ++i) {
        out[i] = scalar_fn(in[i]);
    }
}

// Binary variant
template<typename AVXFunc, typename ScalarFunc>
static inline void vectorized_binary(const float* a, const float* b, float* out, int64_t n,
                                      AVXFunc avx_fn, ScalarFunc scalar_fn) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, avx_fn(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = scalar_fn(a[i], b[i]);
    }
}

// Fill variant
static inline void vectorized_fill(float* out, float val, int64_t n) {
    __m256 v = _mm256_set1_ps(val);
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(out + i,      v);
        _mm256_storeu_ps(out + i + 8,  v);
        _mm256_storeu_ps(out + i + 16, v);
        _mm256_storeu_ps(out + i + 24, v);
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) {
        out[i] = val;
    }
}

// ============================================================================
// AVX2 Reduction helpers
// ============================================================================
static inline float vectorized_sum(const float* data, int64_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(data + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(data + i + 24));
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float total = hsum_avx2(acc0);
    for (; i < n; ++i) total += data[i];
    return total;
}

static inline float vectorized_max(const float* data, int64_t n) {
    if (n == 0) return -INFINITY;
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(data + i));
    }
    // Horizontal max
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
    float result = _mm_cvtss_f32(lo);
    for (; i < n; ++i) if (data[i] > result) result = data[i];
    return result;
}

static inline float vectorized_min(const float* data, int64_t n) {
    if (n == 0) return INFINITY;
    __m256 vmin = _mm256_set1_ps(INFINITY);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vmin = _mm256_min_ps(vmin, _mm256_loadu_ps(data + i));
    }
    __m128 hi = _mm256_extractf128_ps(vmin, 1);
    __m128 lo = _mm256_castps256_ps128(vmin);
    lo = _mm_min_ps(lo, hi);
    lo = _mm_min_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_min_ss(lo, _mm_movehdup_ps(lo));
    float result = _mm_cvtss_f32(lo);
    for (; i < n; ++i) if (data[i] < result) result = data[i];
    return result;
}

} // namespace vec
} // namespace native
} // namespace at
