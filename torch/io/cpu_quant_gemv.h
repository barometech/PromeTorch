#pragma once

// ============================================================================
// CPU AVX2 Quantized GEMV Kernels
// Fused dequant + dot product — no intermediate FP32 buffer needed
//
// Supported formats:
//   - Q4_K (Q4_K_M): 256 values per super-block, 144 bytes
//   - Q8_0: 32 values per block, 34 bytes
//   - Q6_K: 256 values per super-block, 210 bytes
//   - Q5_K: 256 values per super-block, 176 bytes
//
// Each function computes y[N] = A_quant[N,K] @ x[K]
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cmath>
#include "torch/io/gguf_dequant.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace torch {
namespace io {
namespace cpu_quant {

// ============================================================================
// Horizontal sum helper
// ============================================================================

#ifdef __AVX2__
inline float hsum_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}
#endif

// ============================================================================
// Q4_K AVX2 GEMV — processes 32 values per inner iteration
//
// Block layout (144 bytes = 256 values):
//   d(fp16, 2B) + dmin(fp16, 2B) + scales[12] + qs[128]
//
// For each group of 64 values (4 groups per super-block):
//   - 2 sub-groups of 32: low nibbles (qs & 0xF) and high nibbles (qs >> 4)
//   - Each sub-group has its own (scale, min) pair
//   - val = d * scale * nibble - dmin * min
//   - Rewritten: d*sc * sum(q*x) - dmin*m * sum(x) to avoid per-element dequant
//
// Key AVX2 trick: load 16 bytes of qs via _mm_loadu_si128, then use
// _mm256_cvtepu8_epi32 to expand 8 bytes to 8 int32s, then mask nibbles.
// This avoids the slow _mm256_set_epi32 scalar path.
// ============================================================================

#ifdef __AVX2__
inline void q4k_gemv_avx2(const void* weight_data, const float* x,
                           float* y, int64_t K, int64_t N,
                           int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    #pragma omp parallel for schedule(static) if(N > 64)
    #pragma omp parallel for schedule(static) if(N > 64)
    for (int64_t n = 0; n < N; ++n) {
        const __m256i mask_lo = _mm256_set1_epi32(0xF);
        const uint8_t* row_data = raw + n * row_stride_bytes;
        __m256 acc = _mm256_setzero_ps();

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
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

                // Accumulate q*x and x sums for low (0..31) and high (32..63) halves
                __m256 sum_qx_lo = _mm256_setzero_ps();
                __m256 sum_x_lo  = _mm256_setzero_ps();
                __m256 sum_qx_hi = _mm256_setzero_ps();
                __m256 sum_x_hi  = _mm256_setzero_ps();

                // Process 32 bytes of qs (= 32 low nibbles + 32 high nibbles = 64 values)
                // Load 8 bytes at a time, expand to int32, extract nibbles
                for (int l = 0; l < 32; l += 8) {
                    // Load 8 bytes of qs and zero-extend to 8x int32
                    __m128i qs8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(qs + l));
                    __m256i qi = _mm256_cvtepu8_epi32(qs8);

                    // Extract low and high nibbles
                    __m256i q_lo_i = _mm256_and_si256(qi, mask_lo);
                    __m256i q_hi_i = _mm256_srli_epi32(qi, 4);

                    __m256 q_lo_f = _mm256_cvtepi32_ps(q_lo_i);
                    __m256 q_hi_f = _mm256_cvtepi32_ps(q_hi_i);

                    // Load corresponding x values
                    __m256 vx_lo = _mm256_loadu_ps(x + base_k + j + l);
                    __m256 vx_hi = _mm256_loadu_ps(x + base_k + j + 32 + l);

                    // Accumulate: q * x and x
                    sum_qx_lo = _mm256_fmadd_ps(q_lo_f, vx_lo, sum_qx_lo);
                    sum_x_lo  = _mm256_add_ps(sum_x_lo, vx_lo);
                    sum_qx_hi = _mm256_fmadd_ps(q_hi_f, vx_hi, sum_qx_hi);
                    sum_x_hi  = _mm256_add_ps(sum_x_hi, vx_hi);
                }

                // acc += d1 * sum(q_lo * x) - m1 * sum(x_lo)
                //      + d2 * sum(q_hi * x) - m2 * sum(x_hi)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d1), sum_qx_lo, acc);
                acc = _mm256_fnmadd_ps(_mm256_set1_ps(m1), sum_x_lo, acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d2), sum_qx_hi, acc);
                acc = _mm256_fnmadd_ps(_mm256_set1_ps(m2), sum_x_hi, acc);

                qs += 32;
                is += 2;
            }
        }
        y[n] = hsum_avx(acc);
    }
}
#endif

// Scalar fallback for Q4_K
inline void q4k_gemv_scalar(const void* weight_data, const float* x,
                             float* y, int64_t K, int64_t N,
                             int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    #pragma omp parallel for schedule(static) if(N > 64)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        float dot = 0.0f;

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
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
                float d1 = d * sc;
                float m1 = dmin * m_val;
                gguf::get_scale_min_k4(is + 1, scales, &sc, &m_val);
                float d2 = d * sc;
                float m2 = dmin * m_val;
                for (int l = 0; l < 32; ++l) {
                    dot += (d1 * (qs[l] & 0xF) - m1) * x[base_k + j + l];
                    dot += (d2 * (qs[l] >> 4) - m2) * x[base_k + j + 32 + l];
                }
                qs += 32;
                is += 2;
            }
        }
        y[n] = dot;
    }
}

// ============================================================================
// Q8_0 AVX2 GEMV
//
// Block layout (34 bytes = 32 values):
//   d(fp16, 2B) + qs[32] (int8)
//
// val = d * qs[i]
// dot += d * sum(qs[i] * x[i]) = d * dot(qs_vec, x_vec)
//
// AVX2 trick: load 32 int8 values via _mm256_loadu_si256, split into
// low/high 16, extend to int32 via _mm256_cvtepi8_epi32, convert to float,
// FMA with x.
// ============================================================================

#ifdef __AVX2__
inline void q8_0_gemv_avx2(const void* weight_data, const float* x,
                            float* y, int64_t K, int64_t N,
                            int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 32;  // QK8_0 = 32

    #pragma omp parallel for schedule(static) if(N > 64)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        __m256 acc = _mm256_setzero_ps();

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 34;  // 2 + 32 = 34 bytes
            const int64_t base_k = bi * 32;

            uint16_t d_bits;
            std::memcpy(&d_bits, block, 2);
            const float d = gguf::fp16_to_fp32(d_bits);
            const __m256 vd = _mm256_set1_ps(d);

            const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

            // Process 32 int8 values in 4 groups of 8
            // Each group: load 8 int8 -> extend to int32 -> convert to float -> FMA
            for (int l = 0; l < 32; l += 8) {
                // Load 8 int8 values and sign-extend to 8x int32
                __m128i qs8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(qs + l));
                __m256i qi = _mm256_cvtepi8_epi32(qs8);
                __m256 qf = _mm256_cvtepi32_ps(qi);

                // Load 8 float x values
                __m256 vx = _mm256_loadu_ps(x + base_k + l);

                // acc += d * qs * x
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, qf), vx, acc);
            }
        }
        y[n] = hsum_avx(acc);
    }
}
#endif

// Scalar fallback for Q8_0
inline void q8_0_gemv_scalar(const void* weight_data, const float* x,
                              float* y, int64_t K, int64_t N,
                              int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 32;

    #pragma omp parallel for schedule(static) if(N > 64)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        float dot = 0.0f;

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 34;
            const int64_t base_k = bi * 32;

            uint16_t d_bits;
            std::memcpy(&d_bits, block, 2);
            const float d = gguf::fp16_to_fp32(d_bits);
            const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

            for (int l = 0; l < 32; ++l) {
                dot += d * static_cast<float>(qs[l]) * x[base_k + l];
            }
        }
        y[n] = dot;
    }
}

// ============================================================================
// Q6_K AVX2 GEMV
//
// Block layout (210 bytes = 256 values):
//   ql[128] + qh[64] + scales[16](int8) + d(fp16, 2B)
//
// Each value is 6-bit: base from ql (4-bit low/high nibbles) + 2 high bits from qh
// val = d * scales[sub_block] * (q6 - 32)
// ============================================================================

#ifdef __AVX2__
inline void q6k_gemv_avx2(const void* weight_data, const float* x,
                           float* y, int64_t K, int64_t N,
                           int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    #pragma omp parallel for schedule(static) if(N > 64)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        __m256 acc = _mm256_setzero_ps();

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 210;
            const int64_t base_k = bi * 256;

            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);

            uint16_t d_bits;
            std::memcpy(&d_bits, block + 208, 2);
            const float d = gguf::fp16_to_fp32(d_bits);

            // Process 256 values in 2 halves of 128
            for (int n_half = 0; n_half < 256; n_half += 128) {
                for (int l = 0; l < 32; l += 8) {
                    int is = n_half / 16 + l / 16;

                    // Reconstruct 6-bit values for 4 sub-groups of 8
                    // Group 1: ql[l..l+7] low nibble + qh bits 0-1
                    // Group 2: ql[l+32..l+39] low nibble + qh bits 2-3
                    // Group 3: ql[l..l+7] high nibble + qh bits 4-5
                    // Group 4: ql[l+32..l+39] high nibble + qh bits 6-7

                    // Process group 1: offset l+0, scale is+0
                    {
                        float sc = d * scales[is + 0];
                        __m256 vsc = _mm256_set1_ps(sc);
                        __m256i offset32 = _mm256_set1_epi32(32);

                        __m128i ql8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l));
                        __m128i qh8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
                        __m256i ql_i = _mm256_cvtepu8_epi32(ql8);
                        __m256i qh_i = _mm256_cvtepu8_epi32(qh8);

                        __m256i q_lo = _mm256_and_si256(ql_i, _mm256_set1_epi32(0xF));
                        __m256i q_hi_bits = _mm256_and_si256(qh_i, _mm256_set1_epi32(0x3));
                        __m256i q6 = _mm256_or_si256(q_lo, _mm256_slli_epi32(q_hi_bits, 4));
                        q6 = _mm256_sub_epi32(q6, offset32);

                        __m256 qf = _mm256_cvtepi32_ps(q6);
                        __m256 vx = _mm256_loadu_ps(x + base_k + n_half + l);
                        acc = _mm256_fmadd_ps(_mm256_mul_ps(vsc, qf), vx, acc);
                    }

                    // Process group 2: offset l+32, scale is+2
                    {
                        float sc = d * scales[is + 2];
                        __m256 vsc = _mm256_set1_ps(sc);
                        __m256i offset32 = _mm256_set1_epi32(32);

                        __m128i ql8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l + 32));
                        __m128i qh8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
                        __m256i ql_i = _mm256_cvtepu8_epi32(ql8);
                        __m256i qh_i = _mm256_cvtepu8_epi32(qh8);

                        __m256i q_lo = _mm256_and_si256(ql_i, _mm256_set1_epi32(0xF));
                        __m256i q_hi_bits = _mm256_srli_epi32(_mm256_and_si256(qh_i, _mm256_set1_epi32(0xC)), 2);
                        __m256i q6 = _mm256_or_si256(q_lo, _mm256_slli_epi32(q_hi_bits, 4));
                        q6 = _mm256_sub_epi32(q6, offset32);

                        __m256 qf = _mm256_cvtepi32_ps(q6);
                        __m256 vx = _mm256_loadu_ps(x + base_k + n_half + l + 32);
                        acc = _mm256_fmadd_ps(_mm256_mul_ps(vsc, qf), vx, acc);
                    }

                    // Process group 3: offset l+64, scale is+4
                    {
                        float sc = d * scales[is + 4];
                        __m256 vsc = _mm256_set1_ps(sc);
                        __m256i offset32 = _mm256_set1_epi32(32);

                        __m128i ql8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l));
                        __m128i qh8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
                        __m256i ql_i = _mm256_cvtepu8_epi32(ql8);
                        __m256i qh_i = _mm256_cvtepu8_epi32(qh8);

                        __m256i q_hi_nib = _mm256_srli_epi32(ql_i, 4);
                        __m256i q_hi_bits = _mm256_srli_epi32(_mm256_and_si256(qh_i, _mm256_set1_epi32(0x30)), 4);
                        __m256i q6 = _mm256_or_si256(q_hi_nib, _mm256_slli_epi32(q_hi_bits, 4));
                        q6 = _mm256_sub_epi32(q6, offset32);

                        __m256 qf = _mm256_cvtepi32_ps(q6);
                        __m256 vx = _mm256_loadu_ps(x + base_k + n_half + l + 64);
                        acc = _mm256_fmadd_ps(_mm256_mul_ps(vsc, qf), vx, acc);
                    }

                    // Process group 4: offset l+96, scale is+6
                    {
                        float sc = d * scales[is + 6];
                        __m256 vsc = _mm256_set1_ps(sc);
                        __m256i offset32 = _mm256_set1_epi32(32);

                        __m128i ql8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l + 32));
                        __m128i qh8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
                        __m256i ql_i = _mm256_cvtepu8_epi32(ql8);
                        __m256i qh_i = _mm256_cvtepu8_epi32(qh8);

                        __m256i q_hi_nib = _mm256_srli_epi32(ql_i, 4);
                        __m256i q_hi_bits = _mm256_srli_epi32(_mm256_and_si256(qh_i, _mm256_set1_epi32(0xC0)), 6);
                        __m256i q6 = _mm256_or_si256(q_hi_nib, _mm256_slli_epi32(q_hi_bits, 4));
                        q6 = _mm256_sub_epi32(q6, offset32);

                        __m256 qf = _mm256_cvtepi32_ps(q6);
                        __m256 vx = _mm256_loadu_ps(x + base_k + n_half + l + 96);
                        acc = _mm256_fmadd_ps(_mm256_mul_ps(vsc, qf), vx, acc);
                    }
                }
                ql += 64;
                qh += 32;
            }
        }
        y[n] = hsum_avx(acc);
    }
}
#endif

// Scalar fallback for Q6_K
inline void q6k_gemv_scalar(const void* weight_data, const float* x,
                             float* y, int64_t K, int64_t N,
                             int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    #pragma omp parallel for schedule(static) if(N > 64)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        float dot = 0.0f;

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 210;
            const int64_t base_k = bi * 256;

            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);

            uint16_t d_bits;
            std::memcpy(&d_bits, block + 208, 2);
            const float d = gguf::fp16_to_fp32(d_bits);

            for (int n_half = 0; n_half < 256; n_half += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = n_half / 16 + l / 16;
                    int8_t q1 = static_cast<int8_t>(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4))) - 32;
                    int8_t q2 = static_cast<int8_t>(((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) - 32;
                    int8_t q3 = static_cast<int8_t>(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32;
                    int8_t q4 = static_cast<int8_t>(((ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32;

                    dot += d * scales[is+0] * q1 * x[base_k + n_half + l];
                    dot += d * scales[is+2] * q2 * x[base_k + n_half + l + 32];
                    dot += d * scales[is+4] * q3 * x[base_k + n_half + l + 64];
                    dot += d * scales[is+6] * q4 * x[base_k + n_half + l + 96];
                }
                ql += 64;
                qh += 32;
            }
        }
        y[n] = dot;
    }
}

// ============================================================================
// Q5_K AVX2 GEMV
//
// Block layout (176 bytes = 256 values):
//   d(fp16, 2B) + dmin(fp16, 2B) + scales[12] + qh[32] + qs[128]
//
// Like Q4_K but with an extra high bit per value from qh[32]
// val = d * scale * (nibble + 16*highbit) - dmin * min
// ============================================================================

#ifdef __AVX2__
inline void q5k_gemv_avx2(const void* weight_data, const float* x,
                           float* y, int64_t K, int64_t N,
                           int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;
    const __m256i mask_lo = _mm256_set1_epi32(0xF);

    #pragma omp parallel for schedule(static) if(N > 64)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        __m256 acc = _mm256_setzero_ps();

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 176;
            const int64_t base_k = bi * 256;

            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits, block, 2);
            std::memcpy(&dmin_bits, block + 2, 2);
            const float d = gguf::fp16_to_fp32(d_bits);
            const float dmin = gguf::fp16_to_fp32(dmin_bits);
            const uint8_t* scales = block + 4;
            const uint8_t* qh = block + 16;   // 32 bytes of high bits
            const uint8_t* qs = block + 48;   // 128 bytes of nibbles

            int is = 0;
            uint8_t u1 = 1, u2 = 2;
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
                    __m128i qs8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + l));
                    __m256i qi = _mm256_cvtepu8_epi32(qs8);

                    // Load high bits
                    __m128i qh8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
                    __m256i qhi = _mm256_cvtepu8_epi32(qh8);

                    // Low nibble + high bit for low half
                    __m256i q_lo_i = _mm256_and_si256(qi, mask_lo);
                    __m256i hbit_lo = _mm256_and_si256(qhi, _mm256_set1_epi32(u1));
                    // If bit is set, add 16
                    __m256i hbit_lo_shifted = _mm256_slli_epi32(
                        _mm256_min_epi32(hbit_lo, _mm256_set1_epi32(1)), 4);
                    // Normalize: hbit_lo could be u1 (1,2,4,...), need to check != 0
                    // Simpler: compare with zero
                    __m256i hbit_lo_mask = _mm256_cmpgt_epi32(hbit_lo, _mm256_setzero_si256());
                    __m256i hbit_lo_16 = _mm256_and_si256(hbit_lo_mask, _mm256_set1_epi32(16));
                    q_lo_i = _mm256_add_epi32(q_lo_i, hbit_lo_16);

                    // High nibble + high bit for high half
                    __m256i q_hi_i = _mm256_srli_epi32(qi, 4);
                    __m256i hbit_hi = _mm256_and_si256(qhi, _mm256_set1_epi32(u2));
                    __m256i hbit_hi_mask = _mm256_cmpgt_epi32(hbit_hi, _mm256_setzero_si256());
                    __m256i hbit_hi_16 = _mm256_and_si256(hbit_hi_mask, _mm256_set1_epi32(16));
                    q_hi_i = _mm256_add_epi32(q_hi_i, hbit_hi_16);

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
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        y[n] = hsum_avx(acc);
    }
}
#endif

// Scalar fallback for Q5_K
inline void q5k_gemv_scalar(const void* weight_data, const float* x,
                             float* y, int64_t K, int64_t N,
                             int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    #pragma omp parallel for schedule(static) if(N > 64)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        float dot = 0.0f;

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 176;
            const int64_t base_k = bi * 256;

            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits, block, 2);
            std::memcpy(&dmin_bits, block + 2, 2);
            const float d = gguf::fp16_to_fp32(d_bits);
            const float dmin = gguf::fp16_to_fp32(dmin_bits);
            const uint8_t* scales = block + 4;
            const uint8_t* qh = block + 16;
            const uint8_t* qs = block + 48;

            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m_val;
                gguf::get_scale_min_k4(is, scales, &sc, &m_val);
                float d1 = d * sc;
                float m1 = dmin * m_val;
                gguf::get_scale_min_k4(is + 1, scales, &sc, &m_val);
                float d2 = d * sc;
                float m2 = dmin * m_val;
                for (int l = 0; l < 32; ++l) {
                    float q_lo = (float)(qs[l] & 0xF) + ((qh[l] & u1) ? 16.0f : 0.0f);
                    float q_hi = (float)(qs[l] >> 4)  + ((qh[l] & u2) ? 16.0f : 0.0f);
                    dot += (d1 * q_lo - m1) * x[base_k + j + l];
                    dot += (d2 * q_hi - m2) * x[base_k + j + 32 + l];
                }
                qs += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        y[n] = dot;
    }
}

// ============================================================================
// Dispatch function: auto-selects AVX2 or scalar based on quant type
// ============================================================================

inline void cpu_quant_gemv(uint32_t quant_type,
                           const void* weight_data, const float* x,
                           float* y, int64_t K, int64_t N,
                           int64_t row_stride_bytes) {
    switch (quant_type) {
        case 12: // GGML_TYPE_Q4_K
#ifdef __AVX2__
            q4k_gemv_avx2(weight_data, x, y, K, N, row_stride_bytes);
#else
            q4k_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes);
#endif
            break;
        case 8: // GGML_TYPE_Q8_0
#ifdef __AVX2__
            q8_0_gemv_avx2(weight_data, x, y, K, N, row_stride_bytes);
#else
            q8_0_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes);
#endif
            break;
        case 14: // GGML_TYPE_Q6_K
#ifdef __AVX2__
            q6k_gemv_avx2(weight_data, x, y, K, N, row_stride_bytes);
#else
            q6k_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes);
#endif
            break;
        case 13: // GGML_TYPE_Q5_K
#ifdef __AVX2__
            q5k_gemv_avx2(weight_data, x, y, K, N, row_stride_bytes);
#else
            q5k_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes);
#endif
            break;
        default:
            // Unsupported quant type — caller should fall back to dequant+sgemv
            break;
    }
}

// Check if a quant type is supported by cpu_quant_gemv
inline bool cpu_quant_gemv_supported(uint32_t quant_type) {
    return quant_type == 12 || quant_type == 8 || quant_type == 14 || quant_type == 13;
}

} // namespace cpu_quant
} // namespace io
} // namespace torch
