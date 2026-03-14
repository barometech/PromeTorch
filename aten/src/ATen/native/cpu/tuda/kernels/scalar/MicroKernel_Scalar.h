#pragma once
// ============================================================================
// MicroKernel_Scalar.h — Generic scalar GEMM micro-kernel (fallback)
// ============================================================================

#include <cstdint>

namespace at {
namespace native {
namespace tuda {
namespace kernels {

static void microkernel_scalar(
    int64_t mr, int64_t nr, int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta,
    int64_t MR_pack,
    int64_t NR_pack
) {
    float tmp[16 * 16] = {0}; // max MR×NR = 16×16

    for (int64_t p = 0; p < K; ++p) {
        for (int64_t i = 0; i < mr; ++i) {
            float aval = A[p * MR_pack + i];
            for (int64_t j = 0; j < nr; ++j) {
                tmp[i * NR_pack + j] += aval * B[p * NR_pack + j];
            }
        }
    }

    for (int64_t i = 0; i < mr; ++i) {
        for (int64_t j = 0; j < nr; ++j) {
            if (beta == 0.0f) {
                C[i * ldc + j] = alpha * tmp[i * NR_pack + j];
            } else {
                C[i * ldc + j] = alpha * tmp[i * NR_pack + j] + beta * C[i * ldc + j];
            }
        }
    }
}

} // namespace kernels
} // namespace tuda
} // namespace native
} // namespace at
