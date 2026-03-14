#pragma once
// ============================================================================
// MicroKernel_4x8.h — NEON GEMM micro-kernel for Baikal-M (Cortex-A57)
// ============================================================================
// C[4×8] += A[4×K] × B[K×8]
// 8 accumulator registers (4 rows × 2 halves of 8), fits 32 V registers
// ============================================================================

#ifdef TUDA_NEON
#include <arm_neon.h>
#include <cstdint>

namespace at {
namespace native {
namespace tuda {
namespace kernels {

static inline void microkernel_4x8_neon(
    int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);

    for (int64_t p = 0; p < K; ++p) {
        float32x4_t b0 = vld1q_f32(B + p * 8);
        float32x4_t b1 = vld1q_f32(B + p * 8 + 4);

        float32x4_t a;
        a = vdupq_n_f32(A[p * 4 + 0]);
        c00 = vfmaq_f32(c00, a, b0); c01 = vfmaq_f32(c01, a, b1);
        a = vdupq_n_f32(A[p * 4 + 1]);
        c10 = vfmaq_f32(c10, a, b0); c11 = vfmaq_f32(c11, a, b1);
        a = vdupq_n_f32(A[p * 4 + 2]);
        c20 = vfmaq_f32(c20, a, b0); c21 = vfmaq_f32(c21, a, b1);
        a = vdupq_n_f32(A[p * 4 + 3]);
        c30 = vfmaq_f32(c30, a, b0); c31 = vfmaq_f32(c31, a, b1);
    }

    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta = vdupq_n_f32(beta);

    #define STORE_ROW_NEON(row, r0, r1) \
    { \
        float* crow = C + (row) * ldc; \
        if (beta == 0.0f) { \
            vst1q_f32(crow,     vmulq_f32(valpha, r0)); \
            vst1q_f32(crow + 4, vmulq_f32(valpha, r1)); \
        } else { \
            vst1q_f32(crow,     vfmaq_f32(vmulq_f32(vbeta, vld1q_f32(crow)),     valpha, r0)); \
            vst1q_f32(crow + 4, vfmaq_f32(vmulq_f32(vbeta, vld1q_f32(crow + 4)), valpha, r1)); \
        } \
    }

    STORE_ROW_NEON(0, c00, c01);
    STORE_ROW_NEON(1, c10, c11);
    STORE_ROW_NEON(2, c20, c21);
    STORE_ROW_NEON(3, c30, c31);

    #undef STORE_ROW_NEON
}

} // namespace kernels
} // namespace tuda
} // namespace native
} // namespace at

#endif // TUDA_NEON
