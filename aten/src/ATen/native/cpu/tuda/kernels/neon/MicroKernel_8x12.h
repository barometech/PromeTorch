#pragma once
// ============================================================================
// MicroKernel_8x12.h — NEON GEMM micro-kernel for Baikal-S (Cortex-A75)
// ============================================================================
// C[8×12] += A[8×K] × B[K×12]
// 24 accumulator registers (8 rows × 3 columns-of-4), fits 32 V registers
// ============================================================================

#ifdef TUDA_NEON
#include <arm_neon.h>
#include <cstdint>

namespace at {
namespace native {
namespace tuda {
namespace kernels {

static inline void microkernel_8x12_neon(
    int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
    // 24 accumulators: 8 rows × 3 groups of 4
    float32x4_t c00={}, c01={}, c02={};
    float32x4_t c10={}, c11={}, c12={};
    float32x4_t c20={}, c21={}, c22={};
    float32x4_t c30={}, c31={}, c32={};
    float32x4_t c40={}, c41={}, c42={};
    float32x4_t c50={}, c51={}, c52={};
    float32x4_t c60={}, c61={}, c62={};
    float32x4_t c70={}, c71={}, c72={};

    c00=vdupq_n_f32(0); c01=vdupq_n_f32(0); c02=vdupq_n_f32(0);
    c10=vdupq_n_f32(0); c11=vdupq_n_f32(0); c12=vdupq_n_f32(0);
    c20=vdupq_n_f32(0); c21=vdupq_n_f32(0); c22=vdupq_n_f32(0);
    c30=vdupq_n_f32(0); c31=vdupq_n_f32(0); c32=vdupq_n_f32(0);
    c40=vdupq_n_f32(0); c41=vdupq_n_f32(0); c42=vdupq_n_f32(0);
    c50=vdupq_n_f32(0); c51=vdupq_n_f32(0); c52=vdupq_n_f32(0);
    c60=vdupq_n_f32(0); c61=vdupq_n_f32(0); c62=vdupq_n_f32(0);
    c70=vdupq_n_f32(0); c71=vdupq_n_f32(0); c72=vdupq_n_f32(0);

    for (int64_t p = 0; p < K; ++p) {
        float32x4_t b0 = vld1q_f32(B + p * 12);
        float32x4_t b1 = vld1q_f32(B + p * 12 + 4);
        float32x4_t b2 = vld1q_f32(B + p * 12 + 8);

        float32x4_t a;
        #define FMA_ROW(row) \
            a = vdupq_n_f32(A[p * 8 + row]); \
            c##row##0 = vfmaq_f32(c##row##0, a, b0); \
            c##row##1 = vfmaq_f32(c##row##1, a, b1); \
            c##row##2 = vfmaq_f32(c##row##2, a, b2);

        FMA_ROW(0); FMA_ROW(1); FMA_ROW(2); FMA_ROW(3);
        FMA_ROW(4); FMA_ROW(5); FMA_ROW(6); FMA_ROW(7);
        #undef FMA_ROW
    }

    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta = vdupq_n_f32(beta);

    #define STORE_ROW3(row, r0, r1, r2) \
    { \
        float* crow = C + (row) * ldc; \
        if (beta == 0.0f) { \
            vst1q_f32(crow,     vmulq_f32(valpha, r0)); \
            vst1q_f32(crow + 4, vmulq_f32(valpha, r1)); \
            vst1q_f32(crow + 8, vmulq_f32(valpha, r2)); \
        } else { \
            vst1q_f32(crow,     vfmaq_f32(vmulq_f32(vbeta, vld1q_f32(crow)),     valpha, r0)); \
            vst1q_f32(crow + 4, vfmaq_f32(vmulq_f32(vbeta, vld1q_f32(crow + 4)), valpha, r1)); \
            vst1q_f32(crow + 8, vfmaq_f32(vmulq_f32(vbeta, vld1q_f32(crow + 8)), valpha, r2)); \
        } \
    }

    STORE_ROW3(0, c00, c01, c02);
    STORE_ROW3(1, c10, c11, c12);
    STORE_ROW3(2, c20, c21, c22);
    STORE_ROW3(3, c30, c31, c32);
    STORE_ROW3(4, c40, c41, c42);
    STORE_ROW3(5, c50, c51, c52);
    STORE_ROW3(6, c60, c61, c62);
    STORE_ROW3(7, c70, c71, c72);

    #undef STORE_ROW3
}

} // namespace kernels
} // namespace tuda
} // namespace native
} // namespace at

#endif // TUDA_NEON
