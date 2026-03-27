#pragma once
// ============================================================================
// TudaBLAS.h — Goto BLAS GEMM with architecture-specific micro-kernels
// ============================================================================
// Provides sgemm, sgemm_nt, sgemv, sdot, saxpy using TUDA dispatch.
// AVX2 micro-kernel from PromeBLAS; NEON/E2K/Scalar alternatives.
// ============================================================================

#include "aten/src/ATen/native/cpu/tuda/TudaConfig.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"

// EML (Elbrus Math Library) BLAS integration
// DISABLED: EML cblas_sgemm has SIGILL bug with OMP_NUM_THREADS=32 + large matrices
// Using TUDA 6x6 micro-kernel GEMM instead (NUMA-safe, no SIGILL)
// To re-enable: uncomment the #define below
// #if defined(TUDA_E2K) && __has_include(<eml/cblas.h>)
// #define PT_USE_EML_BLAS 1
// #include <eml/cblas.h>
// #endif

// System BLAS (MKL, OpenBLAS, etc.) for x86 — if available and not on Elbrus
#if !defined(TUDA_E2K) && !defined(PT_USE_EML_BLAS) && defined(PT_USE_SYSTEM_BLAS)
#if __has_include(<mkl_cblas.h>)
#include <mkl_cblas.h>
#elif __has_include(<cblas.h>)
#include <cblas.h>
#else
#undef PT_USE_SYSTEM_BLAS
#endif
#endif

// Architecture-specific micro-kernels (only compiled on matching platform)
#if defined(TUDA_NEON_A75)
#include "aten/src/ATen/native/cpu/tuda/kernels/neon/MicroKernel_8x12.h"
#elif defined(TUDA_NEON)
#include "aten/src/ATen/native/cpu/tuda/kernels/neon/MicroKernel_4x8.h"
#endif
#if defined(TUDA_E2K)
#include "aten/src/ATen/native/cpu/tuda/kernels/e2k/MicroKernel_6x6.h"
#endif
#if defined(TUDA_NMC4)
#include "aten/src/ATen/native/cpu/tuda/kernels/nmc4/MicroKernel_4x4.h"
#endif
#include "aten/src/ATen/native/cpu/tuda/kernels/scalar/MicroKernel_Scalar.h"

#if defined(TUDA_AVX2)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#endif

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace at {
namespace native {
namespace tuda {
namespace blas {

// Use tuning from TudaConfig
static constexpr int64_t MR = kTuning.MR;
static constexpr int64_t NR = kTuning.NR;
static constexpr int64_t MC = kTuning.MC;
static constexpr int64_t KC = kTuning.KC;
static constexpr int64_t NC = kTuning.NC;

// ============================================================================
// Aligned allocation helpers
// ============================================================================

static inline float* aligned_alloc_f32(int64_t n) {
#ifdef _MSC_VER
    return static_cast<float*>(_aligned_malloc(n * sizeof(float), kTuning.ALIGN));
#else
    void* p = nullptr;
    posix_memalign(&p, kTuning.ALIGN, n * sizeof(float));
    return static_cast<float*>(p);
#endif
}

static inline void aligned_free(void* p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
}

// ============================================================================
// AVX2 micro-kernel 6×16 (inlined from original PromeBLAS)
// ============================================================================

#if defined(TUDA_AVX2)

static inline float hsum_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

static inline void microkernel_6x16_avx2(
    int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int64_t p = 0; p < K; ++p) {
        __m256 b0 = _mm256_loadu_ps(B + p * NR);
        __m256 b1 = _mm256_loadu_ps(B + p * NR + 8);
        __m256 a;
        a = _mm256_set1_ps(A[p * MR + 0]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_set1_ps(A[p * MR + 1]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_set1_ps(A[p * MR + 2]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_set1_ps(A[p * MR + 3]); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_set1_ps(A[p * MR + 4]); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_set1_ps(A[p * MR + 5]); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbeta  = _mm256_set1_ps(beta);

    #define STORE_ROW_AVX(row, r0, r1) \
    { \
        float* crow = C + (row) * ldc; \
        if (beta == 0.0f) { \
            _mm256_storeu_ps(crow,     _mm256_mul_ps(valpha, r0)); \
            _mm256_storeu_ps(crow + 8, _mm256_mul_ps(valpha, r1)); \
        } else { \
            _mm256_storeu_ps(crow,     _mm256_fmadd_ps(valpha, r0, _mm256_mul_ps(vbeta, _mm256_loadu_ps(crow)))); \
            _mm256_storeu_ps(crow + 8, _mm256_fmadd_ps(valpha, r1, _mm256_mul_ps(vbeta, _mm256_loadu_ps(crow + 8)))); \
        } \
    }

    STORE_ROW_AVX(0, c00, c01); STORE_ROW_AVX(1, c10, c11);
    STORE_ROW_AVX(2, c20, c21); STORE_ROW_AVX(3, c30, c31);
    STORE_ROW_AVX(4, c40, c41); STORE_ROW_AVX(5, c50, c51);
    #undef STORE_ROW_AVX
}

#endif // TUDA_AVX2

// ============================================================================
// Dispatch: call the right micro-kernel for current architecture
// ============================================================================

static inline void dispatch_microkernel(
    int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
#if defined(TUDA_AVX2)
    microkernel_6x16_avx2(K, A, B, C, ldc, alpha, beta);
#elif defined(TUDA_NEON_A75)
    kernels::microkernel_8x12_neon(K, A, B, C, ldc, alpha, beta);
#elif defined(TUDA_NEON)
    kernels::microkernel_4x8_neon(K, A, B, C, ldc, alpha, beta);
#elif defined(TUDA_E2K)
    kernels::microkernel_6x6_e2k(K, A, B, C, ldc, alpha, beta);
#elif defined(TUDA_NMC4)
    kernels::microkernel_4x4_nmc4(K, A, B, C, ldc, alpha, beta);
#else
    kernels::microkernel_scalar(MR, NR, K, A, B, C, ldc, alpha, beta, MR, NR);
#endif
}

static inline void dispatch_microkernel_edge(
    int64_t mr, int64_t nr, int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
    kernels::microkernel_scalar(mr, nr, K, A, B, C, ldc, alpha, beta, MR, NR);
}

// ============================================================================
// Pack A: column-major tiles [K × MR]
// ============================================================================

static inline void pack_a(
    int64_t mc, int64_t kc,
    const float* __restrict A, int64_t lda,
    float* __restrict packed
) {
    for (int64_t i = 0; i < mc; i += MR) {
        int64_t mr = std::min(MR, mc - i);
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t ii = 0; ii < mr; ++ii)
                packed[p * MR + ii] = A[(i + ii) * lda + p];
            for (int64_t ii = mr; ii < MR; ++ii)
                packed[p * MR + ii] = 0.0f;
        }
        packed += kc * MR;
    }
}

// ============================================================================
// Pack B: row-major tiles [K × NR]
// ============================================================================

static inline void pack_b(
    int64_t kc, int64_t nc,
    const float* __restrict B, int64_t ldb,
    float* __restrict packed
) {
    for (int64_t j = 0; j < nc; j += NR) {
        int64_t nr = std::min(NR, nc - j);
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t jj = 0; jj < nr; ++jj)
                packed[p * NR + jj] = B[p * ldb + (j + jj)];
            for (int64_t jj = nr; jj < NR; ++jj)
                packed[p * NR + jj] = 0.0f;
        }
        packed += kc * NR;
    }
}

// ============================================================================
// Pack B transposed: B_orig[N,K] row-major → B^T[K,N] tiles
// ============================================================================

static inline void pack_b_trans(
    int64_t kc, int64_t nc,
    const float* __restrict B, int64_t ldb,
    int64_t jc_offset, int64_t pc_offset,
    float* __restrict packed
) {
    for (int64_t j = 0; j < nc; j += NR) {
        int64_t nr = std::min(NR, nc - j);
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t jj = 0; jj < nr; ++jj)
                packed[p * NR + jj] = B[(jc_offset + j + jj) * ldb + (pc_offset + p)];
            for (int64_t jj = nr; jj < NR; ++jj)
                packed[p * NR + jj] = 0.0f;
        }
        packed += kc * NR;
    }
}

// ============================================================================
// Thread-local packing buffers
// ============================================================================

struct PackBuffers {
    float* a;
    float* b;
    PackBuffers() {
        a = aligned_alloc_f32(MC * KC);
        b = aligned_alloc_f32(KC * NC);
    }
    ~PackBuffers() {
        aligned_free(a);
        aligned_free(b);
    }
};

static inline PackBuffers& get_pack_buffers() {
    static thread_local PackBuffers bufs;
    return bufs;
}

// ============================================================================
// Macro-kernel
// ============================================================================

static inline void macro_kernel(
    int64_t mc, int64_t nc, int64_t kc,
    const float* __restrict packed_a,
    const float* __restrict packed_b,
    float* __restrict C, int64_t ldc,
    int64_t ic, int64_t jc,
    float alpha, float beta_k
) {
    const float* pb = packed_b;
    for (int64_t jr = 0; jr < nc; jr += NR) {
        int64_t nr = std::min(NR, nc - jr);
        const float* pa = packed_a;
        for (int64_t ir = 0; ir < mc; ir += MR) {
            int64_t mr = std::min(MR, mc - ir);
            float* c_ptr = C + (ic + ir) * ldc + (jc + jr);
            if (mr == MR && nr == NR) {
                dispatch_microkernel(kc, pa, pb, c_ptr, ldc, alpha, beta_k);
            } else {
                dispatch_microkernel_edge(mr, nr, kc, pa, pb, c_ptr, ldc, alpha, beta_k);
            }
            pa += kc * MR;
        }
        pb += kc * NR;
    }
}

// ============================================================================
// sgemm: C = alpha * A @ B + beta * C
// ============================================================================

static void sgemm(
    int64_t M, int64_t K, int64_t N,
    float alpha,
    const float* __restrict A, int64_t lda,
    const float* __restrict B, int64_t ldb,
    float beta,
    float* __restrict C, int64_t ldc
) {
#if defined(PT_USE_EML_BLAS)
    // Use EML (Elbrus Math Library) optimized BLAS — VLIW-tuned, multi-threaded
    // EML cblas_sgemm on E8C2: 230-269 GFLOPS (vs ~10 GFLOPS our scalar)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
    return;
#elif defined(PT_USE_SYSTEM_BLAS)
    // Use system BLAS (MKL, OpenBLAS) — highly optimized for x86 cache hierarchy
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
    return;
#endif
    // Small matrix fast path
    if (K < 64 || (M * K < MC * KC && M < MR * 2)) {
        // Simple accumulation with VecF
        constexpr int W = VecF::width;
        if (beta == 0.0f) {
            for (int64_t i = 0; i < M; ++i) {
                int64_t j = 0;
                for (; j + W <= N; j += W) VecF::zero().store(C + i * ldc + j);
                for (; j < N; ++j) C[i * ldc + j] = 0.0f;
            }
        } else if (beta != 1.0f) {
            VecF vbeta = VecF::broadcast(beta);
            for (int64_t i = 0; i < M; ++i) {
                int64_t j = 0;
                for (; j + W <= N; j += W)
                    (VecF::load(C + i*ldc + j) * vbeta).store(C + i*ldc + j);
                for (; j < N; ++j) C[i*ldc + j] *= beta;
            }
        }
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t p = 0; p < K; ++p) {
                float aval = alpha * A[i * lda + p];
                VecF va = VecF::broadcast(aval);
                const float* b_row = B + p * ldb;
                float* c_row = C + i * ldc;
                int64_t j = 0;
                for (; j + W <= N; j += W)
                    VecF::fmadd(va, VecF::load(b_row + j), VecF::load(c_row + j)).store(c_row + j);
                for (; j < N; ++j) c_row[j] += aval * b_row[j];
            }
        }
        return;
    }

    auto& bufs = get_pack_buffers();
    for (int64_t jc = 0; jc < N; jc += NC) {
        int64_t nc = std::min(NC, N - jc);
        for (int64_t pc = 0; pc < K; pc += KC) {
            int64_t kc = std::min(KC, K - pc);
            float beta_k = (pc == 0) ? beta : 1.0f;
            pack_b(kc, nc, B + pc * ldb + jc, ldb, bufs.b);
            for (int64_t ic = 0; ic < M; ic += MC) {
                int64_t mc = std::min(MC, M - ic);
                pack_a(mc, kc, A + ic * lda + pc, lda, bufs.a);
                macro_kernel(mc, nc, kc, bufs.a, bufs.b, C, ldc, ic, jc, alpha, beta_k);
            }
        }
    }
}

// ============================================================================
// sgemm_nt: C = alpha * A @ B^T + beta * C  (B stored as [N,K])
// ============================================================================

static void sgemm_nt(
    int64_t M, int64_t K, int64_t N,
    float alpha,
    const float* __restrict A, int64_t lda,
    const float* __restrict B, int64_t ldb,
    float beta,
    float* __restrict C, int64_t ldc
) {
#if defined(PT_USE_EML_BLAS)
    // B is transposed: C = alpha * A @ B^T + beta * C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
    return;
#elif defined(PT_USE_SYSTEM_BLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
    return;
#endif
    // Small matrix fast path
    if (M * N * K < 128 * 128 * 128 || K < 64 || M < MR) {
        constexpr int W = VecF::width;
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                VecF acc = VecF::zero();
                int64_t p = 0;
                for (; p + W <= K; p += W)
                    acc = VecF::fmadd(VecF::load(A + i*lda + p), VecF::load(B + j*ldb + p), acc);
                float sum = acc.hsum();
                for (; p < K; ++p) sum += A[i*lda + p] * B[j*ldb + p];
                C[i*ldc + j] = (beta == 0.0f) ? alpha*sum : alpha*sum + beta*C[i*ldc + j];
            }
        }
        return;
    }

    auto& bufs = get_pack_buffers();
    for (int64_t jc = 0; jc < N; jc += NC) {
        int64_t nc = std::min(NC, N - jc);
        for (int64_t pc = 0; pc < K; pc += KC) {
            int64_t kc = std::min(KC, K - pc);
            float beta_k = (pc == 0) ? beta : 1.0f;
            pack_b_trans(kc, nc, B, ldb, jc, pc, bufs.b);
            for (int64_t ic = 0; ic < M; ic += MC) {
                int64_t mc = std::min(MC, M - ic);
                pack_a(mc, kc, A + ic * lda + pc, lda, bufs.a);
                macro_kernel(mc, nc, kc, bufs.a, bufs.b, C, ldc, ic, jc, alpha, beta_k);
            }
        }
    }
}

// ============================================================================
// sgemv: y = alpha * A @ x + beta * y
// ============================================================================

static void sgemv(
    int64_t M, int64_t N,
    float alpha,
    const float* __restrict A, int64_t lda,
    const float* __restrict x,
    float beta,
    float* __restrict y
) {
    constexpr int W = VecF::width;
    int64_t i = 0;

    // 4-row unrolled
    for (; i + 4 <= M; i += 4) {
        VecF s0 = VecF::zero(), s1 = VecF::zero();
        VecF s2 = VecF::zero(), s3 = VecF::zero();
        const float* a0 = A + (i+0)*lda;
        const float* a1 = A + (i+1)*lda;
        const float* a2 = A + (i+2)*lda;
        const float* a3 = A + (i+3)*lda;
        int64_t j = 0;
        for (; j + W <= N; j += W) {
            VecF xv = VecF::load(x + j);
            s0 = VecF::fmadd(VecF::load(a0+j), xv, s0);
            s1 = VecF::fmadd(VecF::load(a1+j), xv, s1);
            s2 = VecF::fmadd(VecF::load(a2+j), xv, s2);
            s3 = VecF::fmadd(VecF::load(a3+j), xv, s3);
        }
        float r0=s0.hsum(), r1=s1.hsum(), r2=s2.hsum(), r3=s3.hsum();
        for (; j < N; ++j) {
            float xv = x[j];
            r0 += a0[j]*xv; r1 += a1[j]*xv;
            r2 += a2[j]*xv; r3 += a3[j]*xv;
        }
        if (beta == 0.0f) {
            y[i]=alpha*r0; y[i+1]=alpha*r1; y[i+2]=alpha*r2; y[i+3]=alpha*r3;
        } else {
            y[i]=alpha*r0+beta*y[i]; y[i+1]=alpha*r1+beta*y[i+1];
            y[i+2]=alpha*r2+beta*y[i+2]; y[i+3]=alpha*r3+beta*y[i+3];
        }
    }

    for (; i < M; ++i) {
        VecF s = VecF::zero();
        const float* ai = A + i*lda;
        int64_t j = 0;
        for (; j + W <= N; j += W)
            s = VecF::fmadd(VecF::load(ai+j), VecF::load(x+j), s);
        float r = s.hsum();
        for (; j < N; ++j) r += ai[j] * x[j];
        y[i] = (beta == 0.0f) ? alpha*r : alpha*r + beta*y[i];
    }
}

// ============================================================================
// sdot: dot product
// ============================================================================

static float sdot(int64_t N, const float* __restrict x, const float* __restrict y) {
    return vec_dot(N, x, y);
}

// ============================================================================
// saxpy: y = alpha * x + y
// ============================================================================

static void saxpy(int64_t N, float alpha, const float* __restrict x, float* __restrict y) {
    constexpr int W = VecF::width;
    VecF va = VecF::broadcast(alpha);
    int64_t i = 0;
    for (; i + W <= N; i += W)
        VecF::fmadd(va, VecF::load(x+i), VecF::load(y+i)).store(y+i);
    for (; i < N; ++i)
        y[i] += alpha * x[i];
}

} // namespace blas
} // namespace tuda
} // namespace native
} // namespace at
