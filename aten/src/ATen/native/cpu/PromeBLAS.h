#pragma once
// ============================================================================
// PromeBLAS — High-performance BLAS for PromeTorch
// Cache-tiled GEMM with AVX2 FMA micro-kernels
// ============================================================================

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace at {
namespace native {
namespace blas {

// ============================================================================
// Tuning constants (for Intel Haswell+/AMD Zen+)
// MC×KC panel of A fits in L2 (256KB), KC×NC panel of B streams from L3
// ============================================================================
static constexpr int64_t MR = 6;    // micro-kernel M tile
static constexpr int64_t NR = 16;   // micro-kernel N tile (2 × AVX2 register width)
static constexpr int64_t MC = 72;   // macro-kernel M block (MR * 12)
static constexpr int64_t KC = 256;  // K block (fits in L2)
static constexpr int64_t NC = 4096; // N block (streams from L3)

// ============================================================================
// Aligned allocation helpers
// ============================================================================
static inline float* aligned_alloc_f32(int64_t n) {
#ifdef _MSC_VER
    return static_cast<float*>(_aligned_malloc(n * sizeof(float), 64));
#else
    void* p = nullptr;
    posix_memalign(&p, 64, n * sizeof(float));
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
// Horizontal sum for AVX2
// ============================================================================
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

// ============================================================================
// Micro-kernel 6×16: C[6×16] += A[6×K] × B[K×16]
// A is packed: column-major within tile (K×MR, stride MR)
// B is packed: row-major within tile (K×NR, stride NR)
// ============================================================================
static inline void microkernel_6x16(
    int64_t K,
    const float* __restrict A, // packed A: K rows × MR=6 cols
    const float* __restrict B, // packed B: K rows × NR=16 cols
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
    // 12 accumulators = 6 rows × 2 halves of 16
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int64_t p = 0; p < K; ++p) {
        // Load B row: 16 floats
        __m256 b0 = _mm256_loadu_ps(B + p * NR);
        __m256 b1 = _mm256_loadu_ps(B + p * NR + 8);

        // Broadcast each A element and FMA
        __m256 a;
        a = _mm256_set1_ps(A[p * MR + 0]);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);

        a = _mm256_set1_ps(A[p * MR + 1]);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);

        a = _mm256_set1_ps(A[p * MR + 2]);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);

        a = _mm256_set1_ps(A[p * MR + 3]);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);

        a = _mm256_set1_ps(A[p * MR + 4]);
        c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);

        a = _mm256_set1_ps(A[p * MR + 5]);
        c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    // Store: C = alpha * AB + beta * C
    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbeta  = _mm256_set1_ps(beta);

    #define STORE_ROW(row, r0, r1) \
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

    STORE_ROW(0, c00, c01);
    STORE_ROW(1, c10, c11);
    STORE_ROW(2, c20, c21);
    STORE_ROW(3, c30, c31);
    STORE_ROW(4, c40, c41);
    STORE_ROW(5, c50, c51);

    #undef STORE_ROW
}

// ============================================================================
// Edge micro-kernel: handles M < MR or N < NR
// Uses a temporary buffer then copies back
// ============================================================================
static inline void microkernel_edge(
    int64_t mr, int64_t nr, int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
    // Use a temp buffer with full MR×NR
    float tmp[MR * NR] = {0};

    // Run the micro-kernel into tmp
    for (int64_t p = 0; p < K; ++p) {
        for (int64_t i = 0; i < mr; ++i) {
            float aval = A[p * MR + i];
            for (int64_t j = 0; j < nr; ++j) {
                tmp[i * NR + j] += aval * B[p * NR + j];
            }
        }
    }

    // Copy back
    for (int64_t i = 0; i < mr; ++i) {
        for (int64_t j = 0; j < nr; ++j) {
            if (beta == 0.0f) {
                C[i * ldc + j] = alpha * tmp[i * NR + j];
            } else {
                C[i * ldc + j] = alpha * tmp[i * NR + j] + beta * C[i * ldc + j];
            }
        }
    }
}

// ============================================================================
// Pack A: copy panel A[ic:ic+mc, pc:pc+kc] into column-major tiles
// Layout: for each MR×KC tile, store K groups of MR consecutive elements
// ============================================================================
static inline void pack_a(
    int64_t mc, int64_t kc,
    const float* __restrict A, int64_t lda,
    float* __restrict packed
) {
    for (int64_t i = 0; i < mc; i += MR) {
        int64_t mr = std::min(MR, mc - i);
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t ii = 0; ii < mr; ++ii) {
                packed[p * MR + ii] = A[(i + ii) * lda + p];
            }
            // Zero-pad remainder
            for (int64_t ii = mr; ii < MR; ++ii) {
                packed[p * MR + ii] = 0.0f;
            }
        }
        packed += kc * MR;
    }
}

// ============================================================================
// Pack B: copy panel B[pc:pc+kc, jc:jc+nc] into row-major tiles
// Layout: for each KC×NR tile, store K groups of NR consecutive elements
// ============================================================================
static inline void pack_b(
    int64_t kc, int64_t nc,
    const float* __restrict B, int64_t ldb,
    float* __restrict packed
) {
    for (int64_t j = 0; j < nc; j += NR) {
        int64_t nr = std::min(NR, nc - j);
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t jj = 0; jj < nr; ++jj) {
                packed[p * NR + jj] = B[p * ldb + (j + jj)];
            }
            // Zero-pad remainder
            for (int64_t jj = nr; jj < NR; ++jj) {
                packed[p * NR + jj] = 0.0f;
            }
        }
        packed += kc * NR;
    }
}

// ============================================================================
// Pack B transposed: B_orig is [N, K] row-major, we want B^T[K, N]
// B^T[p, j] = B_orig[j, p] = B_orig[j * ldb + p]
// ============================================================================
static inline void pack_b_trans(
    int64_t kc, int64_t nc,
    const float* __restrict B, int64_t ldb, // B_orig is [N, K], ldb = K
    int64_t jc_offset,                       // column offset into B^T = row offset into B_orig
    int64_t pc_offset,                       // row offset into B^T = column offset into B_orig
    float* __restrict packed
) {
    for (int64_t j = 0; j < nc; j += NR) {
        int64_t nr = std::min(NR, nc - j);
        for (int64_t p = 0; p < kc; ++p) {
            for (int64_t jj = 0; jj < nr; ++jj) {
                // B^T[pc_offset+p, jc_offset+j+jj] = B_orig[jc_offset+j+jj, pc_offset+p]
                packed[p * NR + jj] = B[(jc_offset + j + jj) * ldb + (pc_offset + p)];
            }
            for (int64_t jj = nr; jj < NR; ++jj) {
                packed[p * NR + jj] = 0.0f;
            }
        }
        packed += kc * NR;
    }
}

// ============================================================================
// Thread-local static packing buffers to avoid repeated allocation
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
    // Thread-local avoids allocation per call
    static thread_local PackBuffers bufs;
    return bufs;
}

// ============================================================================
// Macro-kernel: factored out to share between sgemm and sgemm_nt
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
                microkernel_6x16(kc, pa, pb, c_ptr, ldc, alpha, beta_k);
            } else {
                microkernel_edge(mr, nr, kc, pa, pb, c_ptr, ldc, alpha, beta_k);
            }

            pa += kc * MR;
        }
        pb += kc * NR;
    }
}

// ============================================================================
// sgemm: C[M×N] = alpha * A[M×K] @ B[K×N] + beta * C[M×N]
//
// Goto BLAS algorithm:
//   for jc (N blocks)
//     for pc (K blocks)
//       pack B panel
//       for ic (M blocks)
//         pack A panel
//         macro-kernel (micro-kernel calls)
// ============================================================================
static void sgemm(
    int64_t M, int64_t K, int64_t N,
    float alpha,
    const float* __restrict A, int64_t lda,
    const float* __restrict B, int64_t ldb,
    float beta,
    float* __restrict C, int64_t ldc
) {
    // Small/medium matrix fast path: avoid packing overhead
    // Use when K is too small for packing to be worthwhile, or total work is small
    if (K < 64 || (M * K < MC * KC && M < MR * 2)) {
        // AVX2 accumulate: broadcast each A element, FMA across B row
        if (beta == 0.0f) {
            for (int64_t i = 0; i < M; ++i) {
                int64_t j = 0;
                __m256 vzero = _mm256_setzero_ps();
                for (; j + 8 <= N; j += 8) _mm256_storeu_ps(C + i * ldc + j, vzero);
                for (; j < N; ++j) C[i * ldc + j] = 0.0f;
            }
        } else if (beta != 1.0f) {
            __m256 vbeta = _mm256_set1_ps(beta);
            for (int64_t i = 0; i < M; ++i) {
                int64_t j = 0;
                for (; j + 8 <= N; j += 8)
                    _mm256_storeu_ps(C + i * ldc + j, _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + i * ldc + j)));
                for (; j < N; ++j) C[i * ldc + j] *= beta;
            }
        }
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t p = 0; p < K; ++p) {
                float aval = alpha * A[i * lda + p];
                __m256 va = _mm256_set1_ps(aval);
                const float* b_row = B + p * ldb;
                float* c_row = C + i * ldc;
                int64_t j = 0;
                for (; j + 32 <= N; j += 32) {
                    _mm256_storeu_ps(c_row+j,    _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row+j),    _mm256_loadu_ps(c_row+j)));
                    _mm256_storeu_ps(c_row+j+8,  _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row+j+8),  _mm256_loadu_ps(c_row+j+8)));
                    _mm256_storeu_ps(c_row+j+16, _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row+j+16), _mm256_loadu_ps(c_row+j+16)));
                    _mm256_storeu_ps(c_row+j+24, _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row+j+24), _mm256_loadu_ps(c_row+j+24)));
                }
                for (; j + 8 <= N; j += 8) {
                    _mm256_storeu_ps(c_row+j, _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row+j), _mm256_loadu_ps(c_row+j)));
                }
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
// sgemm_nt: C[M×N] = alpha * A[M×K] @ B^T[K×N] + beta * C[M×N]
//
// B_orig is stored as [N, K] row-major (ldb = K).
// This avoids copying transposed weight matrices.
// ============================================================================
static void sgemm_nt(
    int64_t M, int64_t K, int64_t N,
    float alpha,
    const float* __restrict A, int64_t lda,
    const float* __restrict B, int64_t ldb, // B_orig is [N, K], ldb = K
    float beta,
    float* __restrict C, int64_t ldc
) {
    // Small/medium matrix fast path: avoid packing overhead
    if (M * N * K < 128 * 128 * 128 || K < 64 || M < MR) {
        // Row-wise: for each output row, accumulate dot products
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                // Dot product of A row i and B_orig row j (= B^T column j)
                __m256 acc = _mm256_setzero_ps();
                int64_t p = 0;
                for (; p + 8 <= K; p += 8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(A + i * lda + p),
                                          _mm256_loadu_ps(B + j * ldb + p), acc);
                }
                float sum = hsum_ps(acc);
                for (; p < K; ++p) sum += A[i * lda + p] * B[j * ldb + p];
                if (beta == 0.0f)
                    C[i * ldc + j] = alpha * sum;
                else
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
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

            // Pack B^T panel: rows pc..pc+kc, cols jc..jc+nc of B^T
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
// sgemv: y[M] = alpha * A[M×N] @ x[N] + beta * y[M]
// AVX2 vectorized, 4 rows at a time
// ============================================================================
static void sgemv(
    int64_t M, int64_t N,
    float alpha,
    const float* __restrict A, int64_t lda,
    const float* __restrict x,
    float beta,
    float* __restrict y
) {
    int64_t i = 0;

    // 4-row unrolled
    for (; i + 4 <= M; i += 4) {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();

        const float* a0 = A + (i + 0) * lda;
        const float* a1 = A + (i + 1) * lda;
        const float* a2 = A + (i + 2) * lda;
        const float* a3 = A + (i + 3) * lda;

        int64_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + j), xv, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + j), xv, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + j), xv, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + j), xv, s3);
        }

        float r0 = hsum_ps(s0), r1 = hsum_ps(s1), r2 = hsum_ps(s2), r3 = hsum_ps(s3);
        // Scalar tail
        for (; j < N; ++j) {
            float xv = x[j];
            r0 += a0[j] * xv;
            r1 += a1[j] * xv;
            r2 += a2[j] * xv;
            r3 += a3[j] * xv;
        }

        if (beta == 0.0f) {
            y[i+0] = alpha * r0;
            y[i+1] = alpha * r1;
            y[i+2] = alpha * r2;
            y[i+3] = alpha * r3;
        } else {
            y[i+0] = alpha * r0 + beta * y[i+0];
            y[i+1] = alpha * r1 + beta * y[i+1];
            y[i+2] = alpha * r2 + beta * y[i+2];
            y[i+3] = alpha * r3 + beta * y[i+3];
        }
    }

    // Remaining rows
    for (; i < M; ++i) {
        __m256 s = _mm256_setzero_ps();
        const float* ai = A + i * lda;
        int64_t j = 0;
        for (; j + 8 <= N; j += 8) {
            s = _mm256_fmadd_ps(_mm256_loadu_ps(ai + j), _mm256_loadu_ps(x + j), s);
        }
        float r = hsum_ps(s);
        for (; j < N; ++j) r += ai[j] * x[j];
        y[i] = (beta == 0.0f) ? alpha * r : alpha * r + beta * y[i];
    }
}

// ============================================================================
// sdot: dot product of two vectors, AVX2 with 4× unrolling
// ============================================================================
static float sdot(int64_t N, const float* __restrict x, const float* __restrict y) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int64_t i = 0;
    for (; i + 32 <= N; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),      _mm256_loadu_ps(y + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8),  _mm256_loadu_ps(y + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24), acc3);
    }

    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    for (; i + 8 <= N; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);
    }

    float result = hsum_ps(acc0);
    for (; i < N; ++i) result += x[i] * y[i];
    return result;
}

// ============================================================================
// saxpy: y = alpha * x + y
// ============================================================================
static void saxpy(int64_t N, float alpha, const float* __restrict x, float* __restrict y) {
    __m256 valpha = _mm256_set1_ps(alpha);
    int64_t i = 0;
    for (; i + 8 <= N; i += 8) {
        __m256 yv = _mm256_loadu_ps(y + i);
        __m256 xv = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(valpha, xv, yv));
    }
    for (; i < N; ++i) {
        y[i] += alpha * x[i];
    }
}

} // namespace blas
} // namespace native
} // namespace at
