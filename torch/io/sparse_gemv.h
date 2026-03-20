#pragma once

// ============================================================================
// Sparse Q4_K GEMV — Skip near-zero weight blocks
//
// Key insight: Q4_K weights have natural sparsity. Many super-blocks (256
// values) have very small scale factors, meaning the entire block contributes
// near-zero to the dot product. By pre-scanning weights during model load
// and building a "significant block" mask, we can skip these blocks during
// GEMV, reducing compute by 20-40%.
//
// Two levels of sparsity:
//   1. Block-level: skip entire 256-value super-blocks with tiny d * max_scale
//   2. Sub-block-level: skip 64-value groups with tiny d * sc
//
// We use block-level for simplicity and cache-friendliness.
//
// Structure:
//   - SparseQ4KWeight: holds per-row block masks + statistics
//   - sparse_q4k_gemv: GEMV that skips masked blocks
//
// Memory overhead: ~N * blocks_per_row / 8 bytes (bitmap) ≈ negligible
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include "c10/util/ThreadPool.h"
#include "torch/io/gguf_dequant.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace torch {
namespace io {

// ============================================================================
// Sparse weight structure for Q4_K
// ============================================================================
struct SparseQ4KWeight {
    // Per row: bitmap of significant blocks (1 = compute, 0 = skip)
    // Each uint64_t covers 64 blocks. For K=2560, blocks_per_row=10, so 1 uint64_t suffices.
    std::vector<std::vector<uint8_t>> block_mask;  // [N][blocks_per_row_bytes]

    // Per row: number of significant blocks (for statistics)
    std::vector<int32_t> nz_block_count;

    // Threshold for considering a block significant
    float threshold = 0.0f;

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks_per_row = 0;
    bool valid = false;

    // ====================================================================
    // Analyze quantized weight and build sparsity mask
    //
    // threshold_pct: blocks with max possible contribution below this
    //                fraction of the row's max contribution are skipped.
    //                Typical: 0.01 (skip blocks contributing < 1%)
    //
    // A Q4_K block's maximum contribution to the dot product is bounded by:
    //   max_contrib = |d| * max_scale * 15 * 256 * max(|x|)
    // But since x varies, we use a simpler heuristic:
    //   block_magnitude = |d| * max_scale
    // And skip blocks where block_magnitude < threshold_pct * max_block_magnitude_in_row
    // ====================================================================
    void analyze(const void* weight_data, int64_t N, int64_t K,
                 int64_t row_stride_bytes, float threshold_pct = 0.01f) {
        rows = N;
        cols = K;
        blocks_per_row = K / 256;

        int64_t mask_bytes = (blocks_per_row + 7) / 8;
        block_mask.resize(N);
        nz_block_count.resize(N);

        const uint8_t* raw = static_cast<const uint8_t*>(weight_data);

        int64_t total_blocks = 0;
        int64_t total_nz = 0;

        auto fp16_decode = [](uint16_t h) -> float {
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            if (exp == 0) {
                if (mant == 0) return sign ? -0.0f : 0.0f;
                float val = std::ldexp(static_cast<float>(mant), -24);
                return sign ? -val : val;
            }
            if (exp == 31) return sign ? -INFINITY : INFINITY;
            float val = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
            return sign ? -val : val;
        };

        auto get_scale_min = [](int is, const uint8_t* sc, uint8_t* s_out, uint8_t* m_out) {
            if (is < 4) {
                *s_out = sc[is] & 63;
                *m_out = sc[is + 4] & 63;
            } else {
                *s_out = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4);
                *m_out = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4);
            }
        };

        for (int64_t n = 0; n < N; ++n) {
            block_mask[n].resize(mask_bytes, 0);
            const uint8_t* row_data = raw + n * row_stride_bytes;

            // First pass: compute block magnitudes
            std::vector<float> block_mag(blocks_per_row);
            float max_mag = 0.0f;

            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* block = row_data + bi * 144;
                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits, block, 2);
                std::memcpy(&dmin_bits, block + 2, 2);
                float d = std::abs(fp16_decode(d_bits));
                float dmin = std::abs(fp16_decode(dmin_bits));
                const uint8_t* scales = block + 4;

                // Find max scale in this block
                float max_scale = 0.0f;
                float max_min = 0.0f;
                for (int is = 0; is < 8; ++is) {
                    uint8_t sc, mv;
                    get_scale_min(is, scales, &sc, &mv);
                    if (sc > max_scale) max_scale = sc;
                    if (mv > max_min) max_min = mv;
                }
                // Block magnitude: max possible |weight| in this block
                // w = d * sc * q - dmin * m, where q in [0,15], so max |w| ≈ d * max_scale * 15 + dmin * max_min
                block_mag[bi] = d * max_scale * 15.0f + dmin * max_min;
                if (block_mag[bi] > max_mag) max_mag = block_mag[bi];
            }

            // Second pass: mark significant blocks
            float thr = max_mag * threshold_pct;
            int32_t nz = 0;
            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                if (block_mag[bi] >= thr) {
                    block_mask[n][bi / 8] |= (1 << (bi % 8));
                    nz++;
                }
            }
            nz_block_count[n] = nz;
            total_blocks += blocks_per_row;
            total_nz += nz;
        }

        valid = true;

        float sparsity = 1.0f - static_cast<float>(total_nz) / static_cast<float>(total_blocks);
        float speedup = static_cast<float>(total_blocks) / static_cast<float>(total_nz);
        std::cout << "[SparseGEMV] Analyzed " << N << "x" << K
                  << ": " << (sparsity * 100.0f) << "% sparse"
                  << " (threshold=" << threshold_pct << ")"
                  << ", expected speedup: " << speedup << "x" << std::endl;
    }
};

// ============================================================================
// Sparse Q4_K GEMV — uses block mask to skip near-zero blocks
// ============================================================================
#ifdef __AVX2__

namespace sparse_detail {
inline float hsum_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}
} // namespace sparse_detail

inline void sparse_q4k_gemv_avx2(
        const void* weight_data, const float* x,
        float* y, int64_t K, int64_t N,
        int64_t row_stride_bytes,
        const SparseQ4KWeight& sparse) {

    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
        for (int64_t n = start; n < end; ++n) {
            const uint8_t* row_data = raw + n * row_stride_bytes;
            const auto& mask = sparse.block_mask[n];

            const __m256i mask_lo = _mm256_set1_epi32(0xF);
            __m256 acc = _mm256_setzero_ps();

            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                // Check block mask
                if (!(mask[bi / 8] & (1 << (bi % 8)))) continue;

                const uint8_t* block = row_data + bi * 144;
                const int64_t base_k = bi * 256;

                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits, block, 2);
                std::memcpy(&dmin_bits, block + 2, 2);
                const float d = gguf::fp16_to_fp32(d_bits);
                const float dmin = gguf::fp16_to_fp32(dmin_bits);
                const uint8_t* scales = block + 4;
                const uint8_t* qs = block + 16;

                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m_val;
                    gguf::get_scale_min_k4(is, scales, &sc, &m_val);
                    const float d1 = d * sc;
                    const float m1 = dmin * m_val;
                    gguf::get_scale_min_k4(is + 1, scales, &sc, &m_val);
                    const float d2 = d * sc;
                    const float m2 = dmin * m_val;

                    __m256 sum_qx_lo = _mm256_setzero_ps();
                    __m256 sum_x_lo  = _mm256_setzero_ps();
                    __m256 sum_qx_hi = _mm256_setzero_ps();
                    __m256 sum_x_hi  = _mm256_setzero_ps();

                    for (int l = 0; l < 32; l += 8) {
                        __m128i qs8 = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(qs + l));
                        __m256i qi = _mm256_cvtepu8_epi32(qs8);
                        __m256i q_lo_i = _mm256_and_si256(qi, mask_lo);
                        __m256i q_hi_i = _mm256_srli_epi32(qi, 4);
                        __m256 q_lo_f = _mm256_cvtepi32_ps(q_lo_i);
                        __m256 q_hi_f = _mm256_cvtepi32_ps(q_hi_i);
                        __m256 vx_lo = _mm256_loadu_ps(x + base_k + j + l);
                        __m256 vx_hi = _mm256_loadu_ps(x + base_k + j + 32 + l);
                        sum_qx_lo = _mm256_fmadd_ps(q_lo_f, vx_lo, sum_qx_lo);
                        sum_x_lo  = _mm256_add_ps(sum_x_lo, vx_lo);
                        sum_qx_hi = _mm256_fmadd_ps(q_hi_f, vx_hi, sum_qx_hi);
                        sum_x_hi  = _mm256_add_ps(sum_x_hi, vx_hi);
                    }

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d1), sum_qx_lo, acc);
                    acc = _mm256_fnmadd_ps(_mm256_set1_ps(m1), sum_x_lo, acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d2), sum_qx_hi, acc);
                    acc = _mm256_fnmadd_ps(_mm256_set1_ps(m2), sum_x_hi, acc);

                    qs += 32;
                    is += 2;
                }
            }
            y[n] = sparse_detail::hsum_avx(acc);
        }
    }, 1);
}
#endif

// Scalar fallback
inline void sparse_q4k_gemv_scalar(
        const void* weight_data, const float* x,
        float* y, int64_t K, int64_t N,
        int64_t row_stride_bytes,
        const SparseQ4KWeight& sparse) {

    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    auto fp16_decode = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            float val = std::ldexp(static_cast<float>(mant), -24);
            return sign ? -val : val;
        }
        if (exp == 31) return sign ? -INFINITY : INFINITY;
        float val = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
        return sign ? -val : val;
    };

    auto get_scale_min = [](int is, const uint8_t* sc, uint8_t* s_out, uint8_t* m_out) {
        if (is < 4) {
            *s_out = sc[is] & 63;
            *m_out = sc[is + 4] & 63;
        } else {
            *s_out = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4);
            *m_out = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4);
        }
    };

    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        const auto& mask = sparse.block_mask[n];
        float dot = 0.0f;

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            if (!(mask[bi / 8] & (1 << (bi % 8)))) continue;

            const uint8_t* block = row_data + bi * 144;
            const int64_t base_k = bi * 256;

            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits, block, 2);
            std::memcpy(&dmin_bits, block + 2, 2);
            float d = fp16_decode(d_bits);
            float dmin = fp16_decode(dmin_bits);
            const uint8_t* scales = block + 4;
            const uint8_t* qs = block + 16;

            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m_val;
                get_scale_min(is, scales, &sc, &m_val);
                float d1 = d * sc, m1 = dmin * m_val;
                get_scale_min(is + 1, scales, &sc, &m_val);
                float d2 = d * sc, m2 = dmin * m_val;
                for (int l = 0; l < 32; ++l) {
                    float q_lo = (float)(qs[l] & 0xF);
                    float q_hi = (float)(qs[l] >> 4);
                    dot += (d1 * q_lo - m1) * x[base_k + j + l];
                    dot += (d2 * q_hi - m2) * x[base_k + j + 32 + l];
                }
                qs += 32;
                is += 2;
            }
        }
        y[n] = dot;
    }
}

// Dispatch
inline void sparse_q4k_gemv(
        const void* weight_data, const float* x,
        float* y, int64_t K, int64_t N,
        int64_t row_stride_bytes,
        const SparseQ4KWeight& sparse) {
#ifdef __AVX2__
    sparse_q4k_gemv_avx2(weight_data, x, y, K, N, row_stride_bytes, sparse);
#else
    sparse_q4k_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes, sparse);
#endif
}

} // namespace io
} // namespace torch
