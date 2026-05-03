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
//
// Attribution: super-block layouts, scale packing and the GGML_FP16 helper
// conventions are defined by GGML (https://github.com/ggerganov/ggml, MIT)
// and llama.cpp (https://github.com/ggerganov/llama.cpp, MIT). The SIMD
// dequant+dot kernels below are independent implementations that target the
// same bit layout for binary compatibility. See THIRD_PARTY_NOTICES.md.
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include "torch/io/gguf_dequant.h"
#include "torch/io/numa_weight_replica.h"
#include "c10/util/ThreadPool.h"

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
// Q4_K AVX2 GEMV — ORIGINAL (float32 path, kept as fallback)
//
// Block layout (144 bytes = 256 values):
//   d(fp16, 2B) + dmin(fp16, 2B) + scales[12] + qs[128]
// ============================================================================

#ifdef __AVX2__
inline void q4k_gemv_avx2_float(const void* weight_data, const float* x,
                                 float* y, int64_t K, int64_t N,
                                 int64_t row_stride_bytes,
                                 const torch::io::ReplicatedWeight* numa = nullptr) {
    const int64_t blocks_per_row = K / 256;

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    // NUMA: each parallel_for chunk runs on ONE worker thread, so select
    // that thread's node-local weight replica once per chunk.
    const uint8_t* raw = (numa && numa->num_replicas > 1)
        ? static_cast<const uint8_t*>(numa->get(c10::current_numa_node()))
        : static_cast<const uint8_t*>(weight_data);
    for (int64_t n = start; n < end; ++n) {
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
            _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
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

                _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t l = 0; l < 32; l += 8) {
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
        y[n] = hsum_avx(acc);
    }
    }, 1);
}

// ============================================================================
// Q8 block for pre-quantized x vector
// ============================================================================
struct alignas(32) Q8Block {
    float d;          // scale
    float sum;        // sum of qs[i] (for min correction)
    int8_t qs[32];    // quantized values
};

// Pre-quantize x[K] to Q8 format with per-32-element blocks
// This is done ONCE per GEMV call and amortized across all N rows
inline void quantize_x_q8(const float* x, Q8Block* x_q8, int64_t K) {
    const int64_t nb = K / 32;
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * 32;
        // Find max absolute value using AVX2
        __m256 vmax = _mm256_setzero_ps();
        _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 32; j += 8) {
            __m256 vx = _mm256_loadu_ps(xb + j);
            __m256 vabs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vx);
            vmax = _mm256_max_ps(vmax, vabs);
        }
        // Horizontal max of 8 floats
        __m128 hi128 = _mm256_extractf128_ps(vmax, 1);
        __m128 lo128 = _mm256_castps256_ps128(vmax);
        __m128 mx128 = _mm_max_ps(lo128, hi128);
        mx128 = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, _MM_SHUFFLE(1,0,3,2)));
        mx128 = _mm_max_ps(mx128, _mm_shuffle_ps(mx128, mx128, _MM_SHUFFLE(2,3,0,1)));
        float amax = _mm_cvtss_f32(mx128);

        const float d = amax / 127.0f;
        x_q8[i].d = d;
        const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        const __m256 vid = _mm256_set1_ps(id);

        // Quantize + compute sum
        __m256i sum_i32 = _mm256_setzero_si256();
        _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 32; j += 8) {
            __m256 vx = _mm256_loadu_ps(xb + j);
            __m256 scaled = _mm256_mul_ps(vx, vid);
            // Round to nearest int
            __m256i vi = _mm256_cvtps_epi32(scaled);  // rounds to nearest
            // Clamp to [-127, 127]
            vi = _mm256_max_epi32(vi, _mm256_set1_epi32(-127));
            vi = _mm256_min_epi32(vi, _mm256_set1_epi32(127));
            sum_i32 = _mm256_add_epi32(sum_i32, vi);
            // Store as int8 (pack 8 int32 -> 8 int8)
            // Extract to scalar - this is the simplest correct path
            alignas(32) int32_t tmp[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vi);
            for (int k = 0; k < 8; ++k) {
                x_q8[i].qs[j + k] = static_cast<int8_t>(tmp[k]);
            }
        }
        // Horizontal sum of sum_i32
        __m128i lo_s = _mm256_castsi256_si128(sum_i32);
        __m128i hi_s = _mm256_extracti128_si256(sum_i32, 1);
        __m128i s4 = _mm_add_epi32(lo_s, hi_s);
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1,0,3,2)));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(2,3,0,1)));
        x_q8[i].sum = d * static_cast<float>(_mm_cvtsi128_si32(s4));
    }
}

// ============================================================================
// Q4_K x Q8 integer dot product for one super-block (256 values = 8 Q8 blocks)
//
// For Q4_K, each 64-value group has two sub-groups of 32:
//   lo = qs[i] & 0xF (with scale d1, min m1)
//   hi = qs[i] >> 4  (with scale d2, min m2)
//
// dot = d_w * sc * sum(q4 * q8) * d_x  -  dmin * m * sum(q8) * d_x
//     = d_w * sc * d_x * integer_dot   -  dmin * m * x_q8.sum
//
// The integer dot uses _mm256_maddubs_epi16:
//   Takes unsigned bytes (a) and signed bytes (b)
//   Computes: a[2i]*b[2i] + a[2i+1]*b[2i+1] -> int16
//   Then _mm256_madd_epi16 does horizontal pair sum -> int32
//
// This gives us 32 multiplications in 2 instructions (vs 32 float muls + adds)
// ============================================================================

// Process one Q4_K block (256 values) against 8 Q8 blocks
// Returns the dot product contribution
inline float q4k_q8_dot_avx2(const uint8_t* block, const Q8Block* x_q8) {
    uint16_t d_bits, dmin_bits;
    std::memcpy(&d_bits, block, 2);
    std::memcpy(&dmin_bits, block + 2, 2);
    const float d = gguf::fp16_to_fp32(d_bits);
    const float dmin = gguf::fp16_to_fp32(dmin_bits);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);

    float sumf = 0.0f;
    int64_t q8_idx = 0;  // index into Q8 blocks (0..7 for 256 values)

    _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
        // Get scales and mins for this 64-value group
        uint8_t sc1, m1, sc2, m2;
        int64_t is = j / 32;
        gguf::get_scale_min_k4(is, scales, &sc1, &m1);
        gguf::get_scale_min_k4(is + 1, scales, &sc2, &m2);

        // === Low sub-group: 32 values = qs[0..31] & 0xF ===
        // Load 32 bytes of qs
        __m256i raw_qs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
        // Extract low nibbles as unsigned bytes
        __m256i q4_lo = _mm256_and_si256(raw_qs, mask_lo4);
        // Extract high nibbles as unsigned bytes
        __m256i q4_hi = _mm256_and_si256(_mm256_srli_epi16(raw_qs, 4), mask_lo4);

        // Load Q8 values for low sub-group (32 int8 values)
        __m256i q8_lo = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(x_q8[q8_idx].qs));
        // Load Q8 values for high sub-group (32 int8 values)
        __m256i q8_hi = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(x_q8[q8_idx + 1].qs));

        // Integer dot product: q4_lo (unsigned) * q8_lo (signed)
        // _mm256_maddubs_epi16: pairs of (u8*i8) -> i16
        __m256i prod_lo16 = _mm256_maddubs_epi16(q4_lo, q8_lo);
        // _mm256_madd_epi16: pairs of i16 -> i32 (horizontal add)
        __m256i prod_lo32 = _mm256_madd_epi16(prod_lo16, _mm256_set1_epi16(1));

        // Same for high nibbles
        __m256i prod_hi16 = _mm256_maddubs_epi16(q4_hi, q8_hi);
        __m256i prod_hi32 = _mm256_madd_epi16(prod_hi16, _mm256_set1_epi16(1));

        // Horizontal sum of int32 lanes -> scalar int32
        // For prod_lo32:
        __m128i lo_lo = _mm256_castsi256_si128(prod_lo32);
        __m128i hi_lo = _mm256_extracti128_si256(prod_lo32, 1);
        __m128i s_lo = _mm_add_epi32(lo_lo, hi_lo);
        s_lo = _mm_add_epi32(s_lo, _mm_shuffle_epi32(s_lo, _MM_SHUFFLE(1,0,3,2)));
        s_lo = _mm_add_epi32(s_lo, _mm_shuffle_epi32(s_lo, _MM_SHUFFLE(2,3,0,1)));
        int32_t isum_lo = _mm_cvtsi128_si32(s_lo);

        // For prod_hi32:
        __m128i lo_hi = _mm256_castsi256_si128(prod_hi32);
        __m128i hi_hi = _mm256_extracti128_si256(prod_hi32, 1);
        __m128i s_hi = _mm_add_epi32(lo_hi, hi_hi);
        s_hi = _mm_add_epi32(s_hi, _mm_shuffle_epi32(s_hi, _MM_SHUFFLE(1,0,3,2)));
        s_hi = _mm_add_epi32(s_hi, _mm_shuffle_epi32(s_hi, _MM_SHUFFLE(2,3,0,1)));
        int32_t isum_hi = _mm_cvtsi128_si32(s_hi);

        // Combine: d * sc * d_x * isum  -  dmin * m * sum_x
        float d_x_lo = x_q8[q8_idx].d;
        float d_x_hi = x_q8[q8_idx + 1].d;
        float sum_x_lo = x_q8[q8_idx].sum;
        float sum_x_hi = x_q8[q8_idx + 1].sum;

        sumf += d * sc1 * d_x_lo * static_cast<float>(isum_lo) - dmin * m1 * sum_x_lo;
        sumf += d * sc2 * d_x_hi * static_cast<float>(isum_hi) - dmin * m2 * sum_x_hi;

        qs += 32;
        q8_idx += 2;
    }
    return sumf;
}

// ============================================================================
// Q4_K AVX2 GEMV — OPTIMIZED (integer dot product + 2-row processing)
//
// 3 key optimizations over the float32 path:
//   1. Pre-quantize x to Q8 (once, amortized across N rows)
//   2. Integer dot product via _mm256_maddubs_epi16 (2x throughput vs float)
//   3. Process 2 rows simultaneously (share x_q8 cache loads)
//
// Performance model:
//   Old: N * blocks * (32 float loads + 32 float muls + 32 float adds) per subgroup
//   New: 1 x quantize + N * blocks * (1 int load + maddubs + madd) per subgroup
//   Expected speedup: ~2x from int8 path + cache savings
// ============================================================================

// ============================================================================
// Phase 6: SSE4.1 / 128-bit vertical-accumulate Q4_K GEMV kernel
// ============================================================================
// Rewrite of q4k_gemv_avx2 per Gemini 3.1 Pro's design in
// vliw_mission/gemini_sse41_design.md.
//
// Key difference vs the AVX2 path:
//   * ALL ops 128-bit (__m128i / __m128). Matches E2K's native QR register
//     width — LCC 1.29 translates SSE4.1/SSSE3 → native QP 1:1; AVX2 256-bit
//     is split into 2x128 with scheduling overhead.
//   * NO horizontal reduce inside the j-loop. The 4 int32 lanes from
//     maddubs→madd are converted to fp32 and accumulated VERTICALLY into a
//     per-row __m128 acc. The super-block scale d_r is baked into the
//     per-sub-block FP scale so acc aggregates across super-blocks too.
//     A single _mm_hadd_ps + shuffle finalizes per row at the very end.
//   * dmin correction is a pure scalar running sum (one fmadd per sub-block).
//
// Opt-in via PT_Q4K_V2=1 for A/B testing. Will become default after bisect
// confirms speedup on Elbrus 8C2.
// ============================================================================

#ifdef __AVX2__
inline float hsum_m128(__m128 v) {
    __m128 sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
    v = _mm_add_ps(v, sh);
    sh = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm_add_ps(v, sh);
    return _mm_cvtss_f32(v);
}

inline void q4k_gemv_sse41_v(const void* weight_data, const float* x,
                              float* y, int64_t K, int64_t N,
                              int64_t row_stride_bytes,
                              const torch::io::ReplicatedWeight* numa = nullptr) {
    const int64_t blocks_per_row = K / 256;
    const int64_t nb_q8 = K / 32;

    Q8Block* x_q8;
    Q8Block x_q8_stack[512];
    std::unique_ptr<Q8Block[]> x_q8_heap;
    if (nb_q8 <= 512) {
        x_q8 = x_q8_stack;
    } else {
        x_q8_heap.reset(new Q8Block[nb_q8]);
        x_q8 = x_q8_heap.get();
    }
    quantize_x_q8(x, x_q8, K);

    const __m128i mask_lo4 = _mm_set1_epi8(0x0F);
    const __m128i ones_16 = _mm_set1_epi16(1);

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
        const uint8_t* __restrict raw = (numa && numa->num_replicas > 1)
            ? static_cast<const uint8_t*>(numa->get(c10::current_numa_node()))
            : static_cast<const uint8_t*>(weight_data);
        int64_t n = start;

        for (; n + 1 < end; n += 2) {
            const uint8_t* __restrict row0 = raw + n * row_stride_bytes;
            const uint8_t* __restrict row1 = raw + (n + 1) * row_stride_bytes;

            // Per-row vertical 4-lane FP accumulator. Cross super-block safe
            // because d_r (super-block scale) is baked into the scale we
            // multiply with at the sub-block level.
            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            // dmin correction — pure scalar running sum.
            float dmin_sum0 = 0.0f;
            float dmin_sum1 = 0.0f;

            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* __restrict blk0 = row0 + bi * 144;
                const uint8_t* __restrict blk1 = row1 + bi * 144;
                const Q8Block* __restrict xq = x_q8 + bi * 8;

                if (bi + 1 < blocks_per_row) {
                    __builtin_prefetch(row0 + (bi + 1) * 144,       0, 3);
                    __builtin_prefetch(row0 + (bi + 1) * 144 + 64,  0, 3);
                    __builtin_prefetch(row0 + (bi + 1) * 144 + 128, 0, 3);
                    __builtin_prefetch(row1 + (bi + 1) * 144,       0, 3);
                    __builtin_prefetch(row1 + (bi + 1) * 144 + 64,  0, 3);
                    __builtin_prefetch(row1 + (bi + 1) * 144 + 128, 0, 3);
                    // Round2 agent_3 item 6: multi-level. bi+1 covers only
                    // 15 cycles of 200-cycle DRAM latency. Add bi+16 into
                    // L2 (locality 2) to cover the rest.
                    if (bi + 16 < blocks_per_row) {
                        __builtin_prefetch(row0 + (bi + 16) * 144, 0, 2);
                        __builtin_prefetch(row1 + (bi + 16) * 144, 0, 2);
                    }
                }

                uint16_t d0_bits, dmin0_bits, d1_bits, dmin1_bits;
                std::memcpy(&d0_bits,    blk0,     2);
                std::memcpy(&dmin0_bits, blk0 + 2, 2);
                std::memcpy(&d1_bits,    blk1,     2);
                std::memcpy(&dmin1_bits, blk1 + 2, 2);
                const float d_r0    = gguf::fp16_to_fp32(d0_bits);
                const float dmin_r0 = gguf::fp16_to_fp32(dmin0_bits);
                const float d_r1    = gguf::fp16_to_fp32(d1_bits);
                const float dmin_r1 = gguf::fp16_to_fp32(dmin1_bits);
                const uint8_t* sc0 = blk0 + 4;
                const uint8_t* sc1 = blk1 + 4;
                const uint8_t* qs0 = blk0 + 16;
                const uint8_t* qs1 = blk1 + 16;

                int64_t q8_idx = 0;
                _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
                    int64_t is = j / 32;
                    uint8_t sc_a0, m_a0, sc_b0, m_b0;
                    uint8_t sc_a1, m_a1, sc_b1, m_b1;
                    gguf::get_scale_min_k4(is,     sc0, &sc_a0, &m_a0);
                    gguf::get_scale_min_k4(is + 1, sc0, &sc_b0, &m_b0);
                    gguf::get_scale_min_k4(is,     sc1, &sc_a1, &m_a1);
                    gguf::get_scale_min_k4(is + 1, sc1, &sc_b1, &m_b1);

                    // Load 4 × 16 bytes (64 int8 input values) from two Q8Blocks.
                    const int8_t* x_lo = reinterpret_cast<const int8_t*>(xq[q8_idx].qs);
                    const int8_t* x_hi = reinterpret_cast<const int8_t*>(xq[q8_idx + 1].qs);
                    __m128i x0 = _mm_loadu_si128((const __m128i*)(x_lo));
                    __m128i x1 = _mm_loadu_si128((const __m128i*)(x_lo + 16));
                    __m128i x2 = _mm_loadu_si128((const __m128i*)(x_hi));
                    __m128i x3 = _mm_loadu_si128((const __m128i*)(x_hi + 16));
                    const float dx_lo = xq[q8_idx].d;
                    const float dx_hi = xq[q8_idx + 1].d;
                    const float sx_lo = xq[q8_idx].sum;
                    const float sx_hi = xq[q8_idx + 1].sum;

                    // --- Row 0 ---
                    __m128i w0_a = _mm_loadu_si128((const __m128i*)(qs0));
                    __m128i w0_b = _mm_loadu_si128((const __m128i*)(qs0 + 16));
                    __m128i w0a_lo = _mm_and_si128(w0_a, mask_lo4);
                    __m128i w0a_hi = _mm_and_si128(_mm_srli_epi16(w0_a, 4), mask_lo4);
                    __m128i w0b_lo = _mm_and_si128(w0_b, mask_lo4);
                    __m128i w0b_hi = _mm_and_si128(_mm_srli_epi16(w0_b, 4), mask_lo4);

                    __m128i s0a = _mm_add_epi32(
                        _mm_madd_epi16(_mm_maddubs_epi16(w0a_lo, x0), ones_16),
                        _mm_madd_epi16(_mm_maddubs_epi16(w0a_hi, x1), ones_16));
                    __m128i s0b = _mm_add_epi32(
                        _mm_madd_epi16(_mm_maddubs_epi16(w0b_lo, x2), ones_16),
                        _mm_madd_epi16(_mm_maddubs_epi16(w0b_hi, x3), ones_16));

                    // Bake d_r0 into the scale so acc0 can accumulate across super-blocks.
                    __m128 scale0_a = _mm_set1_ps(d_r0 * sc_a0 * dx_lo);
                    __m128 scale0_b = _mm_set1_ps(d_r0 * sc_b0 * dx_hi);
                    acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(s0a), scale0_a));
                    acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_cvtepi32_ps(s0b), scale0_b));
                    dmin_sum0 += dmin_r0 * (m_a0 * sx_lo + m_b0 * sx_hi);

                    // --- Row 1 --- (reuses x0..x3)
                    __m128i w1_a = _mm_loadu_si128((const __m128i*)(qs1));
                    __m128i w1_b = _mm_loadu_si128((const __m128i*)(qs1 + 16));
                    __m128i w1a_lo = _mm_and_si128(w1_a, mask_lo4);
                    __m128i w1a_hi = _mm_and_si128(_mm_srli_epi16(w1_a, 4), mask_lo4);
                    __m128i w1b_lo = _mm_and_si128(w1_b, mask_lo4);
                    __m128i w1b_hi = _mm_and_si128(_mm_srli_epi16(w1_b, 4), mask_lo4);

                    __m128i s1a = _mm_add_epi32(
                        _mm_madd_epi16(_mm_maddubs_epi16(w1a_lo, x0), ones_16),
                        _mm_madd_epi16(_mm_maddubs_epi16(w1a_hi, x1), ones_16));
                    __m128i s1b = _mm_add_epi32(
                        _mm_madd_epi16(_mm_maddubs_epi16(w1b_lo, x2), ones_16),
                        _mm_madd_epi16(_mm_maddubs_epi16(w1b_hi, x3), ones_16));

                    __m128 scale1_a = _mm_set1_ps(d_r1 * sc_a1 * dx_lo);
                    __m128 scale1_b = _mm_set1_ps(d_r1 * sc_b1 * dx_hi);
                    acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(s1a), scale1_a));
                    acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_cvtepi32_ps(s1b), scale1_b));
                    dmin_sum1 += dmin_r1 * (m_a1 * sx_lo + m_b1 * sx_hi);

                    qs0 += 32;
                    qs1 += 32;
                    q8_idx += 2;
                }
            }
            // Horizontal reduce once per row at the very end.
            y[n]     = hsum_m128(acc0) - dmin_sum0;
            y[n + 1] = hsum_m128(acc1) - dmin_sum1;
        }

        // Odd remaining row — fall back to scalar dot (rare, not on hot path).
        if (n < end) {
            const uint8_t* row_data = raw + n * row_stride_bytes;
            float sum = 0.0f;
            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                sum += q4k_q8_dot_avx2(row_data + bi * 144, x_q8 + bi * 8);
            }
            y[n] = sum;
        }
    }, 1);
}
#endif

inline void q4k_gemv_avx2(const void* __restrict weight_data,
                           const float* __restrict x,
                           float* __restrict y,
                           int64_t K, int64_t N,
                           int64_t row_stride_bytes,
                           const torch::io::ReplicatedWeight* numa = nullptr) {
    const int64_t blocks_per_row = K / 256;
    const int64_t nb_q8 = K / 32;  // number of Q8 blocks

    // Step 1: Pre-quantize x to Q8 format (amortized across all N rows)
    // Stack alloc for typical sizes, heap for large K
    Q8Block* x_q8;
    Q8Block x_q8_stack[512];  // covers K up to 16384 (typical LLM hidden dim)
    std::unique_ptr<Q8Block[]> x_q8_heap;
    if (nb_q8 <= 512) {
        x_q8 = x_q8_stack;
    } else {
        x_q8_heap.reset(new Q8Block[nb_q8]);
        x_q8 = x_q8_heap.get();
    }
    quantize_x_q8(x, x_q8, K);

    // Step 2: GEMV with integer dot products, 2 rows at a time
    // The 2-row processing shares x_q8 loads between both rows,
    // effectively halving memory bandwidth for x data.
    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i ones_16 = _mm256_set1_epi16(1);

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
        // NUMA: pick this chunk's node-local replica once.
        const uint8_t* raw = (numa && numa->num_replicas > 1)
            ? static_cast<const uint8_t*>(numa->get(c10::current_numa_node()))
            : static_cast<const uint8_t*>(weight_data);
        int64_t n = start;

        // Process 2 rows at a time — share x_q8 loads
        for (; n + 1 < end; n += 2) {
            const uint8_t* row0 = raw + n * row_stride_bytes;
            const uint8_t* row1 = raw + (n + 1) * row_stride_bytes;
            // Split each row's accumulator into two independent chains
            // (sum*_a gets is_a terms, sum*_b gets is_b terms). Final sum
            // = a + b. This halves the FP dependency chain length per
            // super-block from 8 sequential +='s to 4, giving the LCC
            // scheduler 4 independent FP chains per 2-row iteration. On
            // E8C2 this moves us from ~2 to ~4 ALC channels utilized.
            // Float associativity change is safe for decode (no bit-exact
            // requirement against prior builds). agent_1_e2k_architecture.md P3.
            float sum0_a = 0.0f, sum0_b = 0.0f;
            float sum1_a = 0.0f, sum1_b = 0.0f;

            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* blk0 = row0 + bi * 144;
                const uint8_t* blk1 = row1 + bi * 144;
                const Q8Block* xq = x_q8 + bi * 8;

                // Prefetch NEXT block (both rows). Q4_K block = 144 bytes,
                // spans 3 × 64-byte cache lines; prefetch 3 lines each row.
                // Measured on Elbrus 8C2 single-thread: +12% throughput
                // (2832 ns/row vs 3199 ns/row baseline).
                if (bi + 1 < blocks_per_row) {
                    __builtin_prefetch(row0 + (bi + 1) * 144,       0, 3);
                    __builtin_prefetch(row0 + (bi + 1) * 144 + 64,  0, 3);
                    __builtin_prefetch(row0 + (bi + 1) * 144 + 128, 0, 3);
                    __builtin_prefetch(row1 + (bi + 1) * 144,       0, 3);
                    __builtin_prefetch(row1 + (bi + 1) * 144 + 64,  0, 3);
                    __builtin_prefetch(row1 + (bi + 1) * 144 + 128, 0, 3);
                    // Round2 agent_3 item 6: multi-level. bi+1 covers only
                    // 15 cycles of 200-cycle DRAM latency. Add bi+16 into
                    // L2 (locality 2) to cover the rest.
                    if (bi + 16 < blocks_per_row) {
                        __builtin_prefetch(row0 + (bi + 16) * 144, 0, 2);
                        __builtin_prefetch(row1 + (bi + 16) * 144, 0, 2);
                    }
                }

                // Read block headers for both rows
                uint16_t d0_bits, dmin0_bits, d1_bits, dmin1_bits;
                std::memcpy(&d0_bits, blk0, 2);
                std::memcpy(&dmin0_bits, blk0 + 2, 2);
                std::memcpy(&d1_bits, blk1, 2);
                std::memcpy(&dmin1_bits, blk1 + 2, 2);
                const float d_r0 = gguf::fp16_to_fp32(d0_bits);
                const float dmin_r0 = gguf::fp16_to_fp32(dmin0_bits);
                const float d_r1 = gguf::fp16_to_fp32(d1_bits);
                const float dmin_r1 = gguf::fp16_to_fp32(dmin1_bits);
                const uint8_t* sc0 = blk0 + 4;
                const uint8_t* sc1 = blk1 + 4;
                const uint8_t* qs0 = blk0 + 16;
                const uint8_t* qs1 = blk1 + 16;

                int64_t q8_idx = 0;
                _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
                    int64_t is = j / 32;
                    uint8_t sc_a0, m_a0, sc_b0, m_b0;
                    uint8_t sc_a1, m_a1, sc_b1, m_b1;
                    gguf::get_scale_min_k4(is, sc0, &sc_a0, &m_a0);
                    gguf::get_scale_min_k4(is + 1, sc0, &sc_b0, &m_b0);
                    gguf::get_scale_min_k4(is, sc1, &sc_a1, &m_a1);
                    gguf::get_scale_min_k4(is + 1, sc1, &sc_b1, &m_b1);

                    // Load x_q8 data ONCE for both rows
                    __m256i q8_lo = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xq[q8_idx].qs));
                    __m256i q8_hi = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xq[q8_idx + 1].qs));
                    float dx_lo = xq[q8_idx].d;
                    float dx_hi = xq[q8_idx + 1].d;
                    float sx_lo = xq[q8_idx].sum;
                    float sx_hi = xq[q8_idx + 1].sum;

                    // Row 0: extract nibbles + integer dot
                    __m256i raw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs0));
                    __m256i q4_lo0 = _mm256_and_si256(raw0, mask_lo4);
                    __m256i q4_hi0 = _mm256_and_si256(_mm256_srli_epi16(raw0, 4), mask_lo4);

                    __m256i p0_lo16 = _mm256_maddubs_epi16(q4_lo0, q8_lo);
                    __m256i p0_lo32 = _mm256_madd_epi16(p0_lo16, ones_16);
                    __m256i p0_hi16 = _mm256_maddubs_epi16(q4_hi0, q8_hi);
                    __m256i p0_hi32 = _mm256_madd_epi16(p0_hi16, ones_16);

                    // Horizontal sum for row 0
                    __m128i t0 = _mm_add_epi32(_mm256_castsi256_si128(p0_lo32),
                                               _mm256_extracti128_si256(p0_lo32, 1));
                    t0 = _mm_add_epi32(t0, _mm_shuffle_epi32(t0, _MM_SHUFFLE(1,0,3,2)));
                    t0 = _mm_add_epi32(t0, _mm_shuffle_epi32(t0, _MM_SHUFFLE(2,3,0,1)));
                    int32_t is0_lo = _mm_cvtsi128_si32(t0);

                    __m128i t1 = _mm_add_epi32(_mm256_castsi256_si128(p0_hi32),
                                               _mm256_extracti128_si256(p0_hi32, 1));
                    t1 = _mm_add_epi32(t1, _mm_shuffle_epi32(t1, _MM_SHUFFLE(1,0,3,2)));
                    t1 = _mm_add_epi32(t1, _mm_shuffle_epi32(t1, _MM_SHUFFLE(2,3,0,1)));
                    int32_t is0_hi = _mm_cvtsi128_si32(t1);

                    sum0_a += d_r0 * sc_a0 * dx_lo * (float)is0_lo - dmin_r0 * m_a0 * sx_lo;
                    sum0_b += d_r0 * sc_b0 * dx_hi * (float)is0_hi - dmin_r0 * m_b0 * sx_hi;

                    // Row 1: extract nibbles + integer dot (reuses q8_lo, q8_hi)
                    __m256i raw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs1));
                    __m256i q4_lo1 = _mm256_and_si256(raw1, mask_lo4);
                    __m256i q4_hi1 = _mm256_and_si256(_mm256_srli_epi16(raw1, 4), mask_lo4);

                    __m256i p1_lo16 = _mm256_maddubs_epi16(q4_lo1, q8_lo);
                    __m256i p1_lo32 = _mm256_madd_epi16(p1_lo16, ones_16);
                    __m256i p1_hi16 = _mm256_maddubs_epi16(q4_hi1, q8_hi);
                    __m256i p1_hi32 = _mm256_madd_epi16(p1_hi16, ones_16);

                    __m128i t2 = _mm_add_epi32(_mm256_castsi256_si128(p1_lo32),
                                               _mm256_extracti128_si256(p1_lo32, 1));
                    t2 = _mm_add_epi32(t2, _mm_shuffle_epi32(t2, _MM_SHUFFLE(1,0,3,2)));
                    t2 = _mm_add_epi32(t2, _mm_shuffle_epi32(t2, _MM_SHUFFLE(2,3,0,1)));
                    int32_t is1_lo = _mm_cvtsi128_si32(t2);

                    __m128i t3 = _mm_add_epi32(_mm256_castsi256_si128(p1_hi32),
                                               _mm256_extracti128_si256(p1_hi32, 1));
                    t3 = _mm_add_epi32(t3, _mm_shuffle_epi32(t3, _MM_SHUFFLE(1,0,3,2)));
                    t3 = _mm_add_epi32(t3, _mm_shuffle_epi32(t3, _MM_SHUFFLE(2,3,0,1)));
                    int32_t is1_hi = _mm_cvtsi128_si32(t3);

                    sum1_a += d_r1 * sc_a1 * dx_lo * (float)is1_lo - dmin_r1 * m_a1 * sx_lo;
                    sum1_b += d_r1 * sc_b1 * dx_hi * (float)is1_hi - dmin_r1 * m_b1 * sx_hi;

                    qs0 += 32;
                    qs1 += 32;
                    q8_idx += 2;
                }
            }
            y[n] = sum0_a + sum0_b;
            y[n + 1] = sum1_a + sum1_b;
        }

        // Handle odd remaining row
        if (n < end) {
            const uint8_t* row_data = raw + n * row_stride_bytes;
            float sum = 0.0f;
            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                sum += q4k_q8_dot_avx2(row_data + bi * 144, x_q8 + bi * 8);
            }
            y[n] = sum;
        }
    }, 1);  // grain_size=1: each row is independent
}
#endif

// Scalar fallback for Q4_K
inline void q4k_gemv_scalar(const void* weight_data, const float* x,
                             float* y, int64_t K, int64_t N,
                             int64_t row_stride_bytes,
                             const torch::io::ReplicatedWeight* numa = nullptr) {
    (void)numa;  // scalar path: replica unused, parallelism from thread pool only
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    // Per-row stack buffer for pre-converted fp32 (d, dmin) pairs. 8C2 has
    // no native fp16->fp32 instruction; LCC inlines ~16 ops of bit-twiddle
    // for each gguf::fp16_to_fp32, which fragments the SWP pattern of the
    // hot loop. Pre-converting moves those ops into a tight pre-loop the
    // compiler can pipeline cheaply, leaving the hot loop on clean fp32 +
    // 4-bit unpack + scalar FMA — what LCC's SWP scheduler is best at.
    // Max K = 256 * 80 = 20480 covers any GGUF tensor we'd hit.
    float dpair_buf[80 * 2];
    for (int64_t n = start; n < end; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        float dot = 0.0f;

        // Pre-loop: fp16 -> fp32 once per super-block in this row.
        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 144;
            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits, block, 2);
            std::memcpy(&dmin_bits, block + 2, 2);
            dpair_buf[bi*2 + 0] = gguf::fp16_to_fp32(d_bits);
            dpair_buf[bi*2 + 1] = gguf::fp16_to_fp32(dmin_bits);
        }

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 144;
            const int64_t base_k = bi * 256;

            // Software prefetch: warm next-block header into L1, far-block
            // payload into L2/L3. Critical on E2K — there's no AVX2 path here
            // and LCC's APB only sees current row; without these hints every
            // super-block's 144B header is a cold fetch (the dominant stall
            // per agent_5 audit). Lead distance 4 blocks ≈ 576B at 12 GB/s
            // ≈ 50 ns ahead — covers DDR latency.
            if (bi + 4 < blocks_per_row) {
                const uint8_t* nxt = row_data + (bi + 4) * 144;
                __builtin_prefetch(nxt,       0, 1);  // header → L2
                __builtin_prefetch(nxt + 64,  0, 1);
                __builtin_prefetch(nxt + 128, 0, 1);  // covers all 3 lines of 144B
            }
            if (n + 1 < end && bi == 0) {
                // Warm first cacheline of next row into L3 (long lead).
                const uint8_t* row_next = raw + (n + 1) * row_stride_bytes;
                __builtin_prefetch(row_next, 0, 0);
            }

            const float d    = dpair_buf[bi*2 + 0];
            const float dmin = dpair_buf[bi*2 + 1];
            const uint8_t* scales = block + 4;
            const uint8_t* qs = block + 16;

            int is = 0;
            _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
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
    }, 1);  // min_grain=1: always parallelize GEMV rows
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

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
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
            _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t l = 0; l < 32; l += 8) {
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
    }, 1);  // min_grain=1: always parallelize GEMV rows
}
#endif

// Scalar fallback for Q8_0
inline void q8_0_gemv_scalar(const void* weight_data, const float* x,
                              float* y, int64_t K, int64_t N,
                              int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 32;

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        float dot = 0.0f;

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 34;
            const int64_t base_k = bi * 32;

            // Software prefetch (no AVX2 here; APB covers current row only).
            // Q8_0 blocks are 34B — small, so 8-block lead = 272B = ~22 ns @
            // 12 GB/s, plenty for L2 fill. Less aggressive than Q4_K's 144B
            // blocks which need 4-block lead.
            if (bi + 8 < blocks_per_row) {
                const uint8_t* nxt = row_data + (bi + 8) * 34;
                __builtin_prefetch(nxt,      0, 1);
                __builtin_prefetch(nxt + 64, 0, 1);
            }
            if (n + 1 < end && bi == 0) {
                const uint8_t* row_next = raw + (n + 1) * row_stride_bytes;
                __builtin_prefetch(row_next, 0, 0);
            }

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
    }, 1);  // min_grain=1: always parallelize GEMV rows
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
                           int64_t row_stride_bytes,
                           const torch::io::ReplicatedWeight* numa = nullptr) {
    const int64_t blocks_per_row = K / 256;

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    const uint8_t* raw = (numa && numa->num_replicas > 1)
        ? static_cast<const uint8_t*>(numa->get(c10::current_numa_node()))
        : static_cast<const uint8_t*>(weight_data);
    for (int64_t n = start; n < end; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        __m256 acc = _mm256_setzero_ps();

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 210;
            const int64_t base_k = bi * 256;

            // Prefetch next Q6_K block (4 × 64B cache lines spanning 210B).
            if (bi + 1 < blocks_per_row) {
                const uint8_t* next = row_data + (bi + 1) * 210;
                __builtin_prefetch(next,       0, 3);
                __builtin_prefetch(next + 64,  0, 3);
                __builtin_prefetch(next + 128, 0, 3);
                __builtin_prefetch(next + 192, 0, 3);
            }

            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);

            uint16_t d_bits;
            std::memcpy(&d_bits, block + 208, 2);
            const float d = gguf::fp16_to_fp32(d_bits);

            // Process 256 values in 2 halves of 128
            for (int n_half = 0; n_half < 256; n_half += 128) {
                _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t l = 0; l < 32; l += 8) {
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
    }, 1);  // min_grain=1: always parallelize GEMV rows
}
#endif

// Scalar fallback for Q6_K
inline void q6k_gemv_scalar(const void* weight_data, const float* x,
                             float* y, int64_t K, int64_t N,
                             int64_t row_stride_bytes,
                             const torch::io::ReplicatedWeight* numa = nullptr) {
    (void)numa;
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
        const uint8_t* row_data = raw + n * row_stride_bytes;
        float dot = 0.0f;

        for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
            const uint8_t* block = row_data + bi * 210;
            const int64_t base_k = bi * 256;

            // Prefetch next Q6_K block (4 × 64B cache lines spanning 210B).
            if (bi + 1 < blocks_per_row) {
                const uint8_t* next = row_data + (bi + 1) * 210;
                __builtin_prefetch(next,       0, 3);
                __builtin_prefetch(next + 64,  0, 3);
                __builtin_prefetch(next + 128, 0, 3);
                __builtin_prefetch(next + 192, 0, 3);
            }

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
    }, 1);  // min_grain=1: always parallelize GEMV rows
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

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
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
            _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
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

                _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t l = 0; l < 32; l += 8) {
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
    }, 1);  // min_grain=1: always parallelize GEMV rows
}
#endif

// Scalar fallback for Q5_K
inline void q5k_gemv_scalar(const void* weight_data, const float* x,
                             float* y, int64_t K, int64_t N,
                             int64_t row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = K / 256;

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
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
            _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
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
    }, 1);  // min_grain=1: always parallelize GEMV rows
}

// ============================================================================
// K-sliced GEMV (for row-parallel / K-dim Tensor Parallel)
//
// Operates on a **compacted** weight buffer: the caller copies only a K-range
// [k_start_block..k_end_block) of super-blocks from each row into a rank-local
// buffer whose row_stride_bytes = local_blocks * bytes_per_block.
//
// The input x is the **local slice** of length k_local = (k_end-k_start)*256
// (already trimmed by caller; matches this rank's k-range globally).
//
// Output y has length N (full N rows — each rank produces full N). The result
// is a **partial sum** across K: y_partial = W[n, k_start*256:k_end*256] @ x
// All ranks must then AllReduce-sum their y to recover the full N-vector.
//
// NOTE: the caller is responsible for the NUMA-local copy of the sliced weight;
// we just read from the provided buffer assuming [N, k_local_blocks*bytes] layout.
// ============================================================================

#ifdef __AVX2__
// Q4_K K-slice — same code path as q4k_gemv_avx2 but with local_blocks
// super-blocks per row (no outer K, only blocks [0..local_blocks) read).
inline void q4k_gemv_k_slice_avx2(const void* sliced_weight, const float* x_local,
                                   float* y, int64_t local_blocks, int64_t N,
                                   int64_t local_row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(sliced_weight);
    const int64_t K_local = local_blocks * 256;
    const int64_t nb_q8 = K_local / 32;

    Q8Block* x_q8;
    Q8Block x_q8_stack[512];  // covers K_local up to 16384
    std::unique_ptr<Q8Block[]> x_q8_heap;
    if (nb_q8 <= 512) {
        x_q8 = x_q8_stack;
    } else {
        x_q8_heap.reset(new Q8Block[nb_q8]);
        x_q8 = x_q8_heap.get();
    }
    quantize_x_q8(x_local, x_q8, K_local);

    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i ones_16 = _mm256_set1_epi16(1);

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
        int64_t n = start;
        for (; n + 1 < end; n += 2) {
            const uint8_t* row0 = raw + n * local_row_stride_bytes;
            const uint8_t* row1 = raw + (n + 1) * local_row_stride_bytes;
            // Split accumulators (agent_1_e2k_architecture.md P3): 4 chains
            // instead of 2 → better VLIW issue packing on E8C2.
            float sum0_a = 0.0f, sum0_b = 0.0f;
            float sum1_a = 0.0f, sum1_b = 0.0f;

            for (int64_t bi = 0; bi < local_blocks; ++bi) {
                const uint8_t* blk0 = row0 + bi * 144;
                const uint8_t* blk1 = row1 + bi * 144;
                const Q8Block* xq = x_q8 + bi * 8;

                uint16_t d0_bits, dmin0_bits, d1_bits, dmin1_bits;
                std::memcpy(&d0_bits, blk0, 2);
                std::memcpy(&dmin0_bits, blk0 + 2, 2);
                std::memcpy(&d1_bits, blk1, 2);
                std::memcpy(&dmin1_bits, blk1 + 2, 2);
                const float d_r0 = gguf::fp16_to_fp32(d0_bits);
                const float dmin_r0 = gguf::fp16_to_fp32(dmin0_bits);
                const float d_r1 = gguf::fp16_to_fp32(d1_bits);
                const float dmin_r1 = gguf::fp16_to_fp32(dmin1_bits);
                const uint8_t* sc0 = blk0 + 4;
                const uint8_t* sc1 = blk1 + 4;
                const uint8_t* qs0 = blk0 + 16;
                const uint8_t* qs1 = blk1 + 16;

                int64_t q8_idx = 0;
                _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
                    int64_t is = j / 32;
                    uint8_t sc_a0, m_a0, sc_b0, m_b0;
                    uint8_t sc_a1, m_a1, sc_b1, m_b1;
                    gguf::get_scale_min_k4(is, sc0, &sc_a0, &m_a0);
                    gguf::get_scale_min_k4(is + 1, sc0, &sc_b0, &m_b0);
                    gguf::get_scale_min_k4(is, sc1, &sc_a1, &m_a1);
                    gguf::get_scale_min_k4(is + 1, sc1, &sc_b1, &m_b1);

                    __m256i q8_lo = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xq[q8_idx].qs));
                    __m256i q8_hi = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(xq[q8_idx + 1].qs));
                    float dx_lo = xq[q8_idx].d;
                    float dx_hi = xq[q8_idx + 1].d;
                    float sx_lo = xq[q8_idx].sum;
                    float sx_hi = xq[q8_idx + 1].sum;

                    // Row 0
                    __m256i raw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs0));
                    __m256i q4_lo0 = _mm256_and_si256(raw0, mask_lo4);
                    __m256i q4_hi0 = _mm256_and_si256(_mm256_srli_epi16(raw0, 4), mask_lo4);
                    __m256i p0_lo16 = _mm256_maddubs_epi16(q4_lo0, q8_lo);
                    __m256i p0_lo32 = _mm256_madd_epi16(p0_lo16, ones_16);
                    __m256i p0_hi16 = _mm256_maddubs_epi16(q4_hi0, q8_hi);
                    __m256i p0_hi32 = _mm256_madd_epi16(p0_hi16, ones_16);
                    __m128i t0 = _mm_add_epi32(_mm256_castsi256_si128(p0_lo32),
                                                _mm256_extracti128_si256(p0_lo32, 1));
                    t0 = _mm_add_epi32(t0, _mm_shuffle_epi32(t0, _MM_SHUFFLE(1,0,3,2)));
                    t0 = _mm_add_epi32(t0, _mm_shuffle_epi32(t0, _MM_SHUFFLE(2,3,0,1)));
                    int32_t is0_lo = _mm_cvtsi128_si32(t0);
                    __m128i t1 = _mm_add_epi32(_mm256_castsi256_si128(p0_hi32),
                                                _mm256_extracti128_si256(p0_hi32, 1));
                    t1 = _mm_add_epi32(t1, _mm_shuffle_epi32(t1, _MM_SHUFFLE(1,0,3,2)));
                    t1 = _mm_add_epi32(t1, _mm_shuffle_epi32(t1, _MM_SHUFFLE(2,3,0,1)));
                    int32_t is0_hi = _mm_cvtsi128_si32(t1);
                    sum0_a += d_r0 * sc_a0 * dx_lo * (float)is0_lo - dmin_r0 * m_a0 * sx_lo;
                    sum0_b += d_r0 * sc_b0 * dx_hi * (float)is0_hi - dmin_r0 * m_b0 * sx_hi;

                    // Row 1
                    __m256i raw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs1));
                    __m256i q4_lo1 = _mm256_and_si256(raw1, mask_lo4);
                    __m256i q4_hi1 = _mm256_and_si256(_mm256_srli_epi16(raw1, 4), mask_lo4);
                    __m256i p1_lo16 = _mm256_maddubs_epi16(q4_lo1, q8_lo);
                    __m256i p1_lo32 = _mm256_madd_epi16(p1_lo16, ones_16);
                    __m256i p1_hi16 = _mm256_maddubs_epi16(q4_hi1, q8_hi);
                    __m256i p1_hi32 = _mm256_madd_epi16(p1_hi16, ones_16);
                    __m128i t2 = _mm_add_epi32(_mm256_castsi256_si128(p1_lo32),
                                                _mm256_extracti128_si256(p1_lo32, 1));
                    t2 = _mm_add_epi32(t2, _mm_shuffle_epi32(t2, _MM_SHUFFLE(1,0,3,2)));
                    t2 = _mm_add_epi32(t2, _mm_shuffle_epi32(t2, _MM_SHUFFLE(2,3,0,1)));
                    int32_t is1_lo = _mm_cvtsi128_si32(t2);
                    __m128i t3 = _mm_add_epi32(_mm256_castsi256_si128(p1_hi32),
                                                _mm256_extracti128_si256(p1_hi32, 1));
                    t3 = _mm_add_epi32(t3, _mm_shuffle_epi32(t3, _MM_SHUFFLE(1,0,3,2)));
                    t3 = _mm_add_epi32(t3, _mm_shuffle_epi32(t3, _MM_SHUFFLE(2,3,0,1)));
                    int32_t is1_hi = _mm_cvtsi128_si32(t3);
                    sum1_a += d_r1 * sc_a1 * dx_lo * (float)is1_lo - dmin_r1 * m_a1 * sx_lo;
                    sum1_b += d_r1 * sc_b1 * dx_hi * (float)is1_hi - dmin_r1 * m_b1 * sx_hi;

                    qs0 += 32;
                    qs1 += 32;
                    q8_idx += 2;
                }
            }
            y[n] = sum0_a + sum0_b;
            y[n + 1] = sum1_a + sum1_b;
        }
        if (n < end) {
            const uint8_t* row_data = raw + n * local_row_stride_bytes;
            float sum = 0.0f;
            for (int64_t bi = 0; bi < local_blocks; ++bi) {
                sum += q4k_q8_dot_avx2(row_data + bi * 144, x_q8 + bi * 8);
            }
            y[n] = sum;
        }
    }, 1);
}
#endif

// Scalar fallback for Q4_K K-slice
inline void q4k_gemv_k_slice_scalar(const void* sliced_weight, const float* x_local,
                                      float* y, int64_t local_blocks, int64_t N,
                                      int64_t local_row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(sliced_weight);

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    // Per-row pre-converted (d, dmin) fp32 buffer — see q4k_gemv_scalar
    // comment for rationale. local_blocks max ≤ inter_max/256 ≈ 80 for qwen3.
    float dpair_buf[80 * 2];
    for (int64_t n = start; n < end; ++n) {
        const uint8_t* row_data = raw + n * local_row_stride_bytes;
        float dot = 0.0f;

        for (int64_t bi = 0; bi < local_blocks; ++bi) {
            const uint8_t* block = row_data + bi * 144;
            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits, block, 2);
            std::memcpy(&dmin_bits, block + 2, 2);
            dpair_buf[bi*2 + 0] = gguf::fp16_to_fp32(d_bits);
            dpair_buf[bi*2 + 1] = gguf::fp16_to_fp32(dmin_bits);
        }

        for (int64_t bi = 0; bi < local_blocks; ++bi) {
            const uint8_t* block = row_data + bi * 144;
            const int64_t base_k = bi * 256;  // index into x_local

            // Software prefetch (no AVX2 here; APB covers only current row).
            if (bi + 4 < local_blocks) {
                const uint8_t* nxt = row_data + (bi + 4) * 144;
                __builtin_prefetch(nxt,       0, 1);
                __builtin_prefetch(nxt + 64,  0, 1);
                __builtin_prefetch(nxt + 128, 0, 1);
            }
            if (n + 1 < end && bi == 0) {
                const uint8_t* row_next = raw + (n + 1) * local_row_stride_bytes;
                __builtin_prefetch(row_next, 0, 0);
            }

            const float d = dpair_buf[bi*2 + 0];
            const float dmin = dpair_buf[bi*2 + 1];
            const uint8_t* scales = block + 4;
            const uint8_t* qs = block + 16;

            int is = 0;
            _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
                uint8_t sc, m_val;
                gguf::get_scale_min_k4(is, scales, &sc, &m_val);
                float d1 = d * sc;
                float m1 = dmin * m_val;
                gguf::get_scale_min_k4(is + 1, scales, &sc, &m_val);
                float d2 = d * sc;
                float m2 = dmin * m_val;
                for (int l = 0; l < 32; ++l) {
                    dot += (d1 * (qs[l] & 0xF) - m1) * x_local[base_k + j + l];
                    dot += (d2 * (qs[l] >> 4) - m2) * x_local[base_k + j + 32 + l];
                }
                qs += 32;
                is += 2;
            }
        }
        y[n] = dot;
    }
    }, 1);
}

// Q6_K K-slice scalar — ffn_down in qwen3:4b / gemma3:4b uses Q6_K for
// higher precision on the down-projection. Super-block layout same 256
// elements, stride 210 bytes.
inline void q6k_gemv_k_slice_scalar(const void* sliced_weight, const float* x_local,
                                      float* y, int64_t local_blocks, int64_t N,
                                      int64_t local_row_stride_bytes) {
    const uint8_t* raw = static_cast<const uint8_t*>(sliced_weight);

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
        const uint8_t* row_data = raw + n * local_row_stride_bytes;
        float dot = 0.0f;

        for (int64_t bi = 0; bi < local_blocks; ++bi) {
            const uint8_t* block = row_data + bi * 210;
            const int64_t base_k = bi * 256;  // index into x_local

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

                    dot += d * scales[is+0] * q1 * x_local[base_k + n_half + l];
                    dot += d * scales[is+2] * q2 * x_local[base_k + n_half + l + 32];
                    dot += d * scales[is+4] * q3 * x_local[base_k + n_half + l + 64];
                    dot += d * scales[is+6] * q4 * x_local[base_k + n_half + l + 96];
                }
                ql += 64;
                qh += 32;
            }
        }
        y[n] = dot;
    }
    }, 1);
}

// Dispatch: K-sliced GEMV. Supports Q4_K and Q6_K. Caller must prepare
// compacted rank-local buffer [N, local_blocks * bytes_per_block] — Q4_K
// uses 144 bytes/block, Q6_K uses 210. Partial output → AllReduce-sum.
inline void cpu_quant_gemv_k_slice(
    uint32_t quant_type,
    const void* sliced_weight,  // compacted rank-local buffer [N, local_blocks*144]
    const float* x_local,        // length = local_blocks * 256
    float* y,                    // length = N (partial, needs AllReduce after)
    int64_t local_blocks,
    int64_t N,
    int64_t local_row_stride_bytes) {
    switch (quant_type) {
        case 12: // Q4_K
#ifdef __AVX2__
            q4k_gemv_k_slice_avx2(sliced_weight, x_local, y,
                local_blocks, N, local_row_stride_bytes);
#else
            q4k_gemv_k_slice_scalar(sliced_weight, x_local, y,
                local_blocks, N, local_row_stride_bytes);
#endif
            break;
        case 14: // Q6_K
            // Existing q6k_gemv_avx2 works on any (data, K, N, row_stride)
            // tuple. With sliced_weight compacted and row_stride = local_blocks*210
            // bytes, blocks_per_row = K/256 = local_blocks — identical layout.
#ifdef __AVX2__
            q6k_gemv_avx2(sliced_weight, x_local, y,
                local_blocks * 256, N, local_row_stride_bytes);
#else
            q6k_gemv_scalar(sliced_weight, x_local, y,
                local_blocks * 256, N, local_row_stride_bytes);
#endif
            break;
        case 8: // Q8_0
            // Q8_0 has 32-element blocks (vs 256 for Q4_K/Q6_K). Sliced
            // layout = compact subset of blocks per row, identical layout,
            // just smaller K. Reuse the standard Q8_0 GEMV with K = local_blocks*32.
            //
            // local_blocks here is in 32-element units (caller's slice_k_blocks
            // accounts for the smaller QK_BLOCK = 32 for Q8_0).
#ifdef __AVX2__
            q8_0_gemv_avx2(sliced_weight, x_local, y,
                local_blocks * 32, N, local_row_stride_bytes);
#else
            q8_0_gemv_scalar(sliced_weight, x_local, y,
                local_blocks * 32, N, local_row_stride_bytes);
#endif
            break;
        default:
            // Q5_K — unsupported k-slice fast path. Caller detects valid=false
            // and falls back to replicated GEMV + AllReduce.
            for (int64_t i = 0; i < N; ++i) y[i] = std::numeric_limits<float>::quiet_NaN();
            break;
    }
}

inline bool cpu_quant_gemv_k_slice_supported(uint32_t quant_type) {
    return quant_type == 12 || quant_type == 14 || quant_type == 8; // Q4_K, Q6_K, Q8_0
}

// ============================================================================
// Q5_0 scalar GEMV — 32 elements per block, 22 bytes:
//   2 bytes fp16 d (scale)
//   4 bytes qh (32 high bits, 1 bit per element)
//   16 bytes qs (32 low 4-bit nibbles, 2 elements per byte)
// Element value: ((qs[i/2] >> (4*(i%2))) & 0x0F) | (((qh >> i) & 1) << 4) - 16, then * d
// ============================================================================
inline void q5_0_gemv_scalar(const void* __restrict weight_data,
                              const float* __restrict x,
                              float* __restrict y,
                              int64_t K, int64_t N,
                              int64_t row_stride_bytes) {
    constexpr int QK = 32;
    const int64_t blocks_per_row = K / QK;  // K must be multiple of 32
    const uint8_t* base = static_cast<const uint8_t*>(weight_data);
    auto fp16_to_fp32 = [](uint16_t h) -> float {
        // Standard IEEE 754 fp16 -> fp32 conversion.
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h & 0x7C00) >> 10;
        uint32_t mant = (h & 0x03FF);
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) f = sign;
            else {
                exp = 1;
                while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
                mant &= 0x03FF;
                f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f = sign | 0x7F800000 | (mant << 13);
        } else {
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        float r; std::memcpy(&r, &f, 4); return r;
    };
    for (int64_t row = 0; row < N; ++row) {
        const uint8_t* rp = base + row * row_stride_bytes;
        float acc = 0.0f;
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            uint16_t d_h;
            std::memcpy(&d_h, rp + b * 22, 2);
            float d = fp16_to_fp32(d_h);
            uint32_t qh;
            std::memcpy(&qh, rp + b * 22 + 2, 4);
            const uint8_t* qs = rp + b * 22 + 6;
            const float* xp = x + b * QK;
            for (int i = 0; i < QK; ++i) {
                uint8_t low = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
                uint8_t high = (qh >> i) & 1;
                int q = (int)(low | (high << 4)) - 16;  // signed -16..15
                acc += d * static_cast<float>(q) * xp[i];
            }
        }
        y[row] = acc;
    }
}

// ============================================================================
// Dispatch function: auto-selects AVX2 or scalar based on quant type
// ============================================================================

inline void cpu_quant_gemv(uint32_t quant_type,
                           const void* weight_data, const float* x,
                           float* y, int64_t K, int64_t N,
                           int64_t row_stride_bytes,
                           const torch::io::ReplicatedWeight* numa = nullptr) {
    switch (quant_type) {
        case 12: // GGML_TYPE_Q4_K
#ifdef __AVX2__
        {
            // PT_Q4K_V2=1 selects the Phase 6 vertical-accumulate kernel
            // (gemini_sse41_design.md). Checked once and cached.
            static const bool use_v2 = [] {
                const char* e = std::getenv("PT_Q4K_V2");
                return e && e[0] == '1';
            }();
            if (use_v2) {
                q4k_gemv_sse41_v(weight_data, x, y, K, N, row_stride_bytes, numa);
            } else {
                q4k_gemv_avx2(weight_data, x, y, K, N, row_stride_bytes, numa);
            }
        }
#else
            q4k_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes, numa);
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
            q6k_gemv_avx2(weight_data, x, y, K, N, row_stride_bytes, numa);
#else
            q6k_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes, numa);
#endif
            break;
        case 13: // GGML_TYPE_Q5_K
#ifdef __AVX2__
            q5k_gemv_avx2(weight_data, x, y, K, N, row_stride_bytes);
#else
            q5k_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes);
#endif
            break;
        case 6: // GGML_TYPE_Q5_0
            q5_0_gemv_scalar(weight_data, x, y, K, N, row_stride_bytes);
            break;
        default:
            // Unsupported quant type — caller should fall back to dequant+sgemv
            break;
    }
}

// Check if a quant type is supported by cpu_quant_gemv
inline bool cpu_quant_gemv_supported(uint32_t quant_type) {
    return quant_type == 12 || quant_type == 8 || quant_type == 14 ||
           quant_type == 13 || quant_type == 6;
}

// ============================================================================
// 3D-indexed quantized GEMV — for MoE experts and DeepSeek-V2 MLA up-projs
// ============================================================================
//
// GigaChat3 / DeepSeek-V2 use 3D quantized tensors:
//   ffn_gate_exps   [hidden_dim, expert_intermediate, num_experts]
//   ffn_up_exps     [hidden_dim, expert_intermediate, num_experts]
//   ffn_down_exps   [expert_intermediate, hidden_dim, num_experts]
//   attn_k_b        [no_rope_per_head, kv_lora_rank, n_heads]
//   attn_v_b        [kv_lora_rank, value_per_head, n_heads]
//
// Each "slice" along dim 2 is a self-contained 2D Q4_K/Q5_K/etc matrix.
// Slice `i` stride = N * row_stride_bytes  (where N = output dim, K = input dim).
//
// This wrapper picks the slice and calls the existing 2D GEMV. No new kernel
// math required — Q4_K/Q6_K/Q5_K/Q8_0 super-block layouts are bit-identical
// between 2D and 3D weights; only the byte offset to the chosen slice differs.
inline void cpu_quant_gemv_3d_indexed(
    uint32_t quant_type,
    const void* __restrict weight_data_3d,
    const float* __restrict x,
    float* __restrict y,
    int64_t K, int64_t N, int64_t /*num_experts*/,
    int64_t expert_idx,
    int64_t row_stride_bytes,
    const torch::io::ReplicatedWeight* numa = nullptr) {
    const uint8_t* base = static_cast<const uint8_t*>(weight_data_3d);
    // 3D GGUF layout (matches llama.cpp ggml_tensor->nb[2]):
    //   slice_stride = N * row_stride_bytes  (rows of one expert × bytes-per-row)
    const int64_t slice_stride = N * row_stride_bytes;
    const void* slice = base + expert_idx * slice_stride;
    cpu_quant_gemv(quant_type, slice, x, y, K, N, row_stride_bytes, numa);
}

// ============================================================================
// Batched Q4_K GEMV over N rows × 2 x vectors.
// Core primitive for speculative decoding: when we have 2 query tokens
// sharing the same weight matrix, read W once and dot against both x's.
//
// Weight bandwidth per row is amortized 2× — compute doubles but memory
// traffic stays the same. Measured gain depends on how memory-bound the
// single-row version is on Elbrus (typically L2/L3 resident weights, some
// gain measurable).
//
// Caller must pre-quantize BOTH x0 and x1 to Q8 (same helper as scalar
// path). Outputs y0[N], y1[N] are written independently.
// ============================================================================
#ifdef __AVX2__
inline void q4k_gemv_avx2_batch2(
    const void* weight_data,
    const float* x0, const float* x1,   // two x vectors, same K
    float* y0, float* y1,                // two output rows, size N each
    int64_t K, int64_t N,
    int64_t row_stride_bytes,
    const torch::io::ReplicatedWeight* numa = nullptr) {

    const int64_t blocks_per_row = K / 256;
    const int64_t nb_q8 = K / 32;

    Q8Block xq0_stack[512], xq1_stack[512];
    std::unique_ptr<Q8Block[]> xq0_heap, xq1_heap;
    Q8Block *xq0, *xq1;
    if (nb_q8 <= 512) {
        xq0 = xq0_stack; xq1 = xq1_stack;
    } else {
        xq0_heap.reset(new Q8Block[nb_q8]);
        xq1_heap.reset(new Q8Block[nb_q8]);
        xq0 = xq0_heap.get(); xq1 = xq1_heap.get();
    }
    quantize_x_q8(x0, xq0, K);
    quantize_x_q8(x1, xq1, K);

    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i ones_16  = _mm256_set1_epi16(1);

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
        const uint8_t* raw = (numa && numa->num_replicas > 1)
            ? static_cast<const uint8_t*>(numa->get(c10::current_numa_node()))
            : static_cast<const uint8_t*>(weight_data);

        for (int64_t n = start; n < end; ++n) {
            const uint8_t* row = raw + n * row_stride_bytes;
            float s0 = 0.0f, s1 = 0.0f;

            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* blk = row + bi * 144;

                // Prefetch next block to L1 (same win as single-row kernel).
                if (bi + 1 < blocks_per_row) {
                    const uint8_t* nxt = row + (bi + 1) * 144;
                    __builtin_prefetch(nxt,       0, 3);
                    __builtin_prefetch(nxt + 64,  0, 3);
                    __builtin_prefetch(nxt + 128, 0, 3);
                }

                uint16_t d_bits, dm_bits;
                std::memcpy(&d_bits,  blk,     2);
                std::memcpy(&dm_bits, blk + 2, 2);
                const float d    = gguf::fp16_to_fp32(d_bits);
                const float dmin = gguf::fp16_to_fp32(dm_bits);
                const uint8_t* sc = blk + 4;
                const uint8_t* qs = blk + 16;

                // Q8 blocks for both x's at this position.
                const Q8Block* xq0b = xq0 + bi * 8;
                const Q8Block* xq1b = xq1 + bi * 8;

                int64_t q8_idx = 0;
                _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
                    int64_t is = j / 32;
                    uint8_t sca, ma, scb, mb;
                    gguf::get_scale_min_k4(is,     sc, &sca, &ma);
                    gguf::get_scale_min_k4(is + 1, sc, &scb, &mb);

                    // Load Q4 nibbles from weight ONCE per 64-chunk.
                    __m256i raw_q4 = _mm256_loadu_si256((const __m256i*)qs);
                    __m256i q4_lo = _mm256_and_si256(raw_q4, mask_lo4);
                    __m256i q4_hi = _mm256_and_si256(_mm256_srli_epi16(raw_q4, 4), mask_lo4);

                    // Dot with BOTH x0 and x1 Q8 blocks for lo+hi sub-blocks.
                    auto dot_against = [&](const Q8Block* xqb, float& sum) {
                        __m256i q8_lo = _mm256_loadu_si256((const __m256i*)xqb[q8_idx].qs);
                        __m256i q8_hi = _mm256_loadu_si256((const __m256i*)xqb[q8_idx + 1].qs);
                        float dx_lo = xqb[q8_idx].d,  dx_hi = xqb[q8_idx + 1].d;
                        float sx_lo = xqb[q8_idx].sum, sx_hi = xqb[q8_idx + 1].sum;

                        __m256i p_lo32 = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(q4_lo, q8_lo), ones_16);
                        __m256i p_hi32 = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(q4_hi, q8_hi), ones_16);

                        __m128i tlo = _mm_add_epi32(_mm256_castsi256_si128(p_lo32),
                                                     _mm256_extracti128_si256(p_lo32, 1));
                        tlo = _mm_add_epi32(tlo, _mm_shuffle_epi32(tlo, _MM_SHUFFLE(1,0,3,2)));
                        tlo = _mm_add_epi32(tlo, _mm_shuffle_epi32(tlo, _MM_SHUFFLE(2,3,0,1)));
                        int32_t is_lo = _mm_cvtsi128_si32(tlo);

                        __m128i thi = _mm_add_epi32(_mm256_castsi256_si128(p_hi32),
                                                     _mm256_extracti128_si256(p_hi32, 1));
                        thi = _mm_add_epi32(thi, _mm_shuffle_epi32(thi, _MM_SHUFFLE(1,0,3,2)));
                        thi = _mm_add_epi32(thi, _mm_shuffle_epi32(thi, _MM_SHUFFLE(2,3,0,1)));
                        int32_t is_hi = _mm_cvtsi128_si32(thi);

                        sum += d * sca * dx_lo * (float)is_lo - dmin * ma * sx_lo;
                        sum += d * scb * dx_hi * (float)is_hi - dmin * mb * sx_hi;
                    };

                    dot_against(xq0b, s0);
                    dot_against(xq1b, s1);

                    qs += 32;
                    q8_idx += 2;
                }
            }
            y0[n] = s0;
            y1[n] = s1;
        }
    }, 1);
}
#endif

// ============================================================================
// Batched QKV GEMV -- shared x vector across 3 weight matrices
//
// Instead of 3 separate GEMVs where each re-reads x from memory,
// we split total rows (N_q + N_k + N_v) across threads. Each thread
// reads from the correct weight matrix but x stays hot in L1/L2 cache
// because all threads reference the same x.
//
// This reduces x memory traffic from 3*K to ~K (shared across threads).
// ============================================================================

// ============================================================================
// Phase 7.1 — batched Q4_K GEMV for speculative verify.
//
// Standard q4k_gemv reads 2.5 GB of weights per forward. For spec decode with
// K draft tokens we want K forwards for the cost of ~1 weight read. That's
// what this function does: weight read once, K partial dots computed per
// weight super-block, K outputs written.
//
// Contract:
//   x         — row-major [K_batch × H]. x[k*H + j] is element j of query k.
//   y         — row-major [K_batch × N]. y[k*N + n] is the dot of query k
//               with weight row n.
//   K_batch   — number of queries to batch (1..kMaxSpecBatch=4). When K=1
//               this is equivalent to q4k_gemv_avx2.
//
// Implementation: outer parallel_for over N (weight rows). Inside, per super-
// block, the weight is loaded once and K independent sum accumulators are
// updated. At row end, each sum is stored to y[k*N + n].
// ============================================================================

static constexpr int kMaxSpecBatch = 6;  // Gemini Deep Research finds K=5 optimal for p≈0.8; headroom for K=6

#ifdef __AVX2__
inline void q4k_gemv_batched_avx2(const void* weight_data,
                                   const float* x,
                                   float* y,
                                   int K_batch,
                                   int64_t H, int64_t N,
                                   int64_t row_stride_bytes,
                                   const torch::io::ReplicatedWeight* numa = nullptr) {
    if (K_batch <= 0 || K_batch > kMaxSpecBatch) return;
    const int64_t blocks_per_row = H / 256;
    const int64_t nb_q8 = H / 32;

    // Pre-quantize each of K_batch rows of x to its own Q8Block array.
    // Stack: up to 6 × 512 = 3072 Q8Blocks × 40 B = 120 KB — safe with
    // /STACK:67108864 on MSVC and default 8 MB pthread stack on Linux.
    Q8Block x_q8_stack[kMaxSpecBatch * 512];
    std::unique_ptr<Q8Block[]> x_q8_heap;
    Q8Block* x_q8_all;
    if (nb_q8 * K_batch <= (int64_t)(kMaxSpecBatch * 512)) {
        x_q8_all = x_q8_stack;
    } else {
        x_q8_heap.reset(new Q8Block[nb_q8 * K_batch]);
        x_q8_all = x_q8_heap.get();
    }
    for (int k = 0; k < K_batch; ++k) {
        quantize_x_q8(x + k * H, x_q8_all + k * nb_q8, H);
    }

    const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
    const __m256i ones_16  = _mm256_set1_epi16(1);

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
        const uint8_t* __restrict raw = (numa && numa->num_replicas > 1)
            ? static_cast<const uint8_t*>(numa->get(c10::current_numa_node()))
            : static_cast<const uint8_t*>(weight_data);

        for (int64_t n = start; n < end; ++n) {
            const uint8_t* __restrict row = raw + n * row_stride_bytes;

            // Split accumulator trick (Phase 3 / agent_1 P3): one pair per
            // query. 2*K_batch independent FP chains so LCC can schedule
            // plenty of packed adds in each VLIW bundle.
            float sum_a[kMaxSpecBatch] = {0};
            float sum_b[kMaxSpecBatch] = {0};

            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* __restrict blk = row + bi * 144;

                if (bi + 1 < blocks_per_row) {
                    __builtin_prefetch(row + (bi + 1) * 144,       0, 3);
                    __builtin_prefetch(row + (bi + 1) * 144 + 64,  0, 3);
                    __builtin_prefetch(row + (bi + 1) * 144 + 128, 0, 3);
                }

                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits,    blk,     2);
                std::memcpy(&dmin_bits, blk + 2, 2);
                const float d_r    = gguf::fp16_to_fp32(d_bits);
                const float dmin_r = gguf::fp16_to_fp32(dmin_bits);
                const uint8_t* sc = blk + 4;
                const uint8_t* qs = blk + 16;

                int64_t q8_idx = 0;
                _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
                    int64_t is = j / 32;
                    uint8_t sc_a, m_a, sc_b, m_b;
                    gguf::get_scale_min_k4(is,     sc, &sc_a, &m_a);
                    gguf::get_scale_min_k4(is + 1, sc, &sc_b, &m_b);

                    // Weight block is loaded ONCE per super-block iteration,
                    // reused across all K_batch queries.
                    __m256i raw_w = _mm256_loadu_si256((const __m256i*)qs);
                    __m256i q4_lo = _mm256_and_si256(raw_w, mask_lo4);
                    __m256i q4_hi = _mm256_and_si256(_mm256_srli_epi16(raw_w, 4), mask_lo4);

                    for (int k = 0; k < K_batch; ++k) {
                        const Q8Block* xqk = x_q8_all + k * nb_q8 + bi * 8;
                        __m256i q8_lo = _mm256_loadu_si256((const __m256i*)xqk[q8_idx].qs);
                        __m256i q8_hi = _mm256_loadu_si256((const __m256i*)xqk[q8_idx + 1].qs);
                        const float dx_lo = xqk[q8_idx].d;
                        const float dx_hi = xqk[q8_idx + 1].d;
                        const float sx_lo = xqk[q8_idx].sum;
                        const float sx_hi = xqk[q8_idx + 1].sum;

                        __m256i p_lo = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(q4_lo, q8_lo), ones_16);
                        __m256i p_hi = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(q4_hi, q8_hi), ones_16);

                        __m128i t_lo = _mm_add_epi32(_mm256_castsi256_si128(p_lo),
                                                     _mm256_extracti128_si256(p_lo, 1));
                        t_lo = _mm_add_epi32(t_lo, _mm_shuffle_epi32(t_lo, _MM_SHUFFLE(1,0,3,2)));
                        t_lo = _mm_add_epi32(t_lo, _mm_shuffle_epi32(t_lo, _MM_SHUFFLE(2,3,0,1)));
                        int32_t is_lo = _mm_cvtsi128_si32(t_lo);

                        __m128i t_hi = _mm_add_epi32(_mm256_castsi256_si128(p_hi),
                                                     _mm256_extracti128_si256(p_hi, 1));
                        t_hi = _mm_add_epi32(t_hi, _mm_shuffle_epi32(t_hi, _MM_SHUFFLE(1,0,3,2)));
                        t_hi = _mm_add_epi32(t_hi, _mm_shuffle_epi32(t_hi, _MM_SHUFFLE(2,3,0,1)));
                        int32_t is_hi = _mm_cvtsi128_si32(t_hi);

                        sum_a[k] += d_r * sc_a * dx_lo * (float)is_lo - dmin_r * m_a * sx_lo;
                        sum_b[k] += d_r * sc_b * dx_hi * (float)is_hi - dmin_r * m_b * sx_hi;
                    }

                    qs += 32;
                    q8_idx += 2;
                }
            }
            for (int k = 0; k < K_batch; ++k) {
                y[k * N + n] = sum_a[k] + sum_b[k];
            }
        }
    }, 1);
}
#endif  // __AVX2__

// Scalar fallback for batched Q4_K GEMV (when AVX2 not available).
inline void q4k_gemv_batched_scalar(const void* weight_data,
                                     const float* x,
                                     float* y,
                                     int K_batch,
                                     int64_t H, int64_t N,
                                     int64_t row_stride_bytes,
                                     const torch::io::ReplicatedWeight* numa = nullptr) {
    (void)numa;
    const uint8_t* raw = static_cast<const uint8_t*>(weight_data);
    const int64_t blocks_per_row = H / 256;

    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
        // Per-row stack buffer for pre-converted fp32 scales (d, dmin pairs).
        // 8C2 has no native fp16->fp32 instruction; LCC emits ~16 ops of bit-
        // twiddle for each gguf::fp16_to_fp32, fragmenting the inner-loop
        // SWP pattern. Pre-converting in a tight pre-loop gives the hot loop
        // a clean (fp32 read + 4-bit unpack + scalar FMA) shape that LCC's
        // pipeline scheduler can pack tightly into VLIW bundles.
        float dpair_buf[80 * 2];
        for (int64_t n = start; n < end; ++n) {
            const uint8_t* row = raw + n * row_stride_bytes;
            // Pre-loop: fp16 -> fp32 once for all super-blocks of this row.
            for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                const uint8_t* block = row + bi * 144;
                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits,    block,     2);
                std::memcpy(&dmin_bits, block + 2, 2);
                dpair_buf[bi*2 + 0] = gguf::fp16_to_fp32(d_bits);
                dpair_buf[bi*2 + 1] = gguf::fp16_to_fp32(dmin_bits);
            }
            for (int k = 0; k < K_batch; ++k) {
                float dot = 0.0f;
                for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                    const uint8_t* block = row + bi * 144;
                    const int64_t base_k = bi * 256;
                    // Software prefetch (E2K has no AVX2 SIMD prefetch hints
                    // here; APB covers only current row). 4-block lead.
                    if (bi + 4 < blocks_per_row) {
                        const uint8_t* nxt = row + (bi + 4) * 144;
                        __builtin_prefetch(nxt,       0, 1);
                        __builtin_prefetch(nxt + 64,  0, 1);
                        __builtin_prefetch(nxt + 128, 0, 1);
                    }
                    if (n + 1 < end && bi == 0 && k == 0) {
                        const uint8_t* row_next = raw + (n + 1) * row_stride_bytes;
                        __builtin_prefetch(row_next, 0, 0);
                    }
                    const float d    = dpair_buf[bi*2 + 0];
                    const float dmin = dpair_buf[bi*2 + 1];
                    const uint8_t* sc = block + 4;
                    const uint8_t* qs = block + 16;
                    _Pragma("loop count(8)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 32) {
                        int64_t is = j / 32;
                        uint8_t sc_j, m_j;
                        gguf::get_scale_min_k4(is, sc, &sc_j, &m_j);
                        for (int l = 0; l < 32; ++l) {
                            uint8_t raw_w = qs[(j / 2 + l / 2)];
                            uint8_t q4 = (l & 1) ? (raw_w >> 4) : (raw_w & 0x0F);
                            dot += (d * sc_j * (float)q4 - dmin * m_j)
                                 * x[k * H + base_k + j + l];
                        }
                    }
                }
                y[k * N + n] = dot;
            }
        }
    }, 1);
}

// Public dispatcher for batched Q4_K GEMV.
inline void cpu_quant_gemv_batched(uint32_t quant_type,
                                    const void* weight_data,
                                    const float* x,
                                    float* y,
                                    int K_batch,
                                    int64_t H, int64_t N,
                                    int64_t row_stride_bytes,
                                    const torch::io::ReplicatedWeight* numa = nullptr) {
    switch (quant_type) {
        case 12: // Q4_K
#ifdef __AVX2__
            q4k_gemv_batched_avx2(weight_data, x, y, K_batch, H, N, row_stride_bytes, numa);
#else
            q4k_gemv_batched_scalar(weight_data, x, y, K_batch, H, N, row_stride_bytes, numa);
#endif
            break;
        default:
            // Fallback: call non-batched kernel K_batch times. Correct but
            // slow; not a speedup. Phase 7.2 will add batched Q6_K/Q8_0.
            for (int k = 0; k < K_batch; ++k) {
                cpu_quant_gemv(quant_type, weight_data, x + k * H, y + k * N,
                               H, N, row_stride_bytes, numa);
            }
    }
}

inline void cpu_quant_gemv_batched_qkv(
    uint32_t quant_type,
    const void* w_q, const void* w_k, const void* w_v,
    const float* x,
    float* y_q, float* y_k, float* y_v,
    int64_t K, int64_t N_q, int64_t N_k, int64_t N_v,
    int64_t row_stride_bytes) {

    int64_t N_total = N_q + N_k + N_v;

    auto get_weight_row = [&](int64_t global_row) -> const uint8_t* {
        if (global_row < N_q) {
            return static_cast<const uint8_t*>(w_q) + global_row * row_stride_bytes;
        } else if (global_row < N_q + N_k) {
            return static_cast<const uint8_t*>(w_k) + (global_row - N_q) * row_stride_bytes;
        } else {
            return static_cast<const uint8_t*>(w_v) + (global_row - N_q - N_k) * row_stride_bytes;
        }
    };

    auto get_output_ptr = [&](int64_t global_row) -> float* {
        if (global_row < N_q) {
            return y_q + global_row;
        } else if (global_row < N_q + N_k) {
            return y_k + (global_row - N_q);
        } else {
            return y_v + (global_row - N_q - N_k);
        }
    };

#ifdef __AVX2__
    if (quant_type == 12) {  // Q4_K — use Q8 pre-quantized integer dot products
        const int64_t blocks_per_row = K / 256;
        const int64_t nb_q8 = K / 32;

        // Pre-quantize x to Q8 ONCE (amortized across all Q+K+V rows)
        Q8Block* x_q8;
        Q8Block x_q8_stack[512];  // covers K up to 16384
        std::unique_ptr<Q8Block[]> x_q8_heap;
        if (nb_q8 <= 512) {
            x_q8 = x_q8_stack;
        } else {
            x_q8_heap.reset(new Q8Block[nb_q8]);
            x_q8 = x_q8_heap.get();
        }
        quantize_x_q8(x, x_q8, K);

        const __m256i mask_lo4 = _mm256_set1_epi8(0x0F);
        const __m256i ones_16 = _mm256_set1_epi16(1);

        c10::get_thread_pool().parallel_for(0, N_total, [&](int64_t start, int64_t end) {
            int64_t n = start;
            // Process 2 rows at a time for better x_q8 reuse
            for (; n + 1 < end; n += 2) {
                const uint8_t* row0 = get_weight_row(n);
                const uint8_t* row1 = get_weight_row(n + 1);
                // Split accumulators (agent_1 P3) — 4 chains per iteration.
                float sum0_a = 0.0f, sum0_b = 0.0f;
                float sum1_a = 0.0f, sum1_b = 0.0f;

                for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                    const uint8_t* blk0 = row0 + bi * 144;
                    const uint8_t* blk1 = row1 + bi * 144;
                    const Q8Block* xq = x_q8 + bi * 8;

                    // Prefetch next Q4_K block for both rows (3 cache lines each).
                    if (bi + 1 < blocks_per_row) {
                        __builtin_prefetch(row0 + (bi + 1) * 144,       0, 3);
                        __builtin_prefetch(row0 + (bi + 1) * 144 + 64,  0, 3);
                        __builtin_prefetch(row0 + (bi + 1) * 144 + 128, 0, 3);
                        __builtin_prefetch(row1 + (bi + 1) * 144,       0, 3);
                        __builtin_prefetch(row1 + (bi + 1) * 144 + 64,  0, 3);
                        __builtin_prefetch(row1 + (bi + 1) * 144 + 128, 0, 3);
                    }

                    uint16_t d0_bits, dmin0_bits, d1_bits, dmin1_bits;
                    std::memcpy(&d0_bits, blk0, 2);
                    std::memcpy(&dmin0_bits, blk0 + 2, 2);
                    std::memcpy(&d1_bits, blk1, 2);
                    std::memcpy(&dmin1_bits, blk1 + 2, 2);
                    const float d_r0 = gguf::fp16_to_fp32(d0_bits);
                    const float dmin_r0 = gguf::fp16_to_fp32(dmin0_bits);
                    const float d_r1 = gguf::fp16_to_fp32(d1_bits);
                    const float dmin_r1 = gguf::fp16_to_fp32(dmin1_bits);
                    const uint8_t* sc0 = blk0 + 4;
                    const uint8_t* sc1 = blk1 + 4;
                    const uint8_t* qs0 = blk0 + 16;
                    const uint8_t* qs1 = blk1 + 16;

                    int64_t q8_idx = 0;
                    _Pragma("loop count(4)") _Pragma("ivdep") for (int64_t j = 0; j < 256; j += 64) {
                        int64_t is = j / 32;
                        uint8_t sc_a0, m_a0, sc_b0, m_b0;
                        uint8_t sc_a1, m_a1, sc_b1, m_b1;
                        gguf::get_scale_min_k4(is, sc0, &sc_a0, &m_a0);
                        gguf::get_scale_min_k4(is + 1, sc0, &sc_b0, &m_b0);
                        gguf::get_scale_min_k4(is, sc1, &sc_a1, &m_a1);
                        gguf::get_scale_min_k4(is + 1, sc1, &sc_b1, &m_b1);

                        __m256i q8_lo = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(xq[q8_idx].qs));
                        __m256i q8_hi = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(xq[q8_idx + 1].qs));
                        float dx_lo = xq[q8_idx].d;
                        float dx_hi = xq[q8_idx + 1].d;
                        float sx_lo = xq[q8_idx].sum;
                        float sx_hi = xq[q8_idx + 1].sum;

                        // Row 0
                        __m256i raw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs0));
                        __m256i q4_lo0 = _mm256_and_si256(raw0, mask_lo4);
                        __m256i q4_hi0 = _mm256_and_si256(_mm256_srli_epi16(raw0, 4), mask_lo4);
                        __m256i p0_lo16 = _mm256_maddubs_epi16(q4_lo0, q8_lo);
                        __m256i p0_lo32 = _mm256_madd_epi16(p0_lo16, ones_16);
                        __m256i p0_hi16 = _mm256_maddubs_epi16(q4_hi0, q8_hi);
                        __m256i p0_hi32 = _mm256_madd_epi16(p0_hi16, ones_16);
                        __m128i t0 = _mm_add_epi32(_mm256_castsi256_si128(p0_lo32),
                                                     _mm256_extracti128_si256(p0_lo32, 1));
                        t0 = _mm_add_epi32(t0, _mm_shuffle_epi32(t0, _MM_SHUFFLE(1,0,3,2)));
                        t0 = _mm_add_epi32(t0, _mm_shuffle_epi32(t0, _MM_SHUFFLE(2,3,0,1)));
                        int32_t is0_lo = _mm_cvtsi128_si32(t0);
                        __m128i t1 = _mm_add_epi32(_mm256_castsi256_si128(p0_hi32),
                                                     _mm256_extracti128_si256(p0_hi32, 1));
                        t1 = _mm_add_epi32(t1, _mm_shuffle_epi32(t1, _MM_SHUFFLE(1,0,3,2)));
                        t1 = _mm_add_epi32(t1, _mm_shuffle_epi32(t1, _MM_SHUFFLE(2,3,0,1)));
                        int32_t is0_hi = _mm_cvtsi128_si32(t1);
                        sum0_a += d_r0 * sc_a0 * dx_lo * (float)is0_lo - dmin_r0 * m_a0 * sx_lo;
                        sum0_b += d_r0 * sc_b0 * dx_hi * (float)is0_hi - dmin_r0 * m_b0 * sx_hi;

                        // Row 1
                        __m256i raw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs1));
                        __m256i q4_lo1 = _mm256_and_si256(raw1, mask_lo4);
                        __m256i q4_hi1 = _mm256_and_si256(_mm256_srli_epi16(raw1, 4), mask_lo4);
                        __m256i p1_lo16 = _mm256_maddubs_epi16(q4_lo1, q8_lo);
                        __m256i p1_lo32 = _mm256_madd_epi16(p1_lo16, ones_16);
                        __m256i p1_hi16 = _mm256_maddubs_epi16(q4_hi1, q8_hi);
                        __m256i p1_hi32 = _mm256_madd_epi16(p1_hi16, ones_16);
                        __m128i t2 = _mm_add_epi32(_mm256_castsi256_si128(p1_lo32),
                                                     _mm256_extracti128_si256(p1_lo32, 1));
                        t2 = _mm_add_epi32(t2, _mm_shuffle_epi32(t2, _MM_SHUFFLE(1,0,3,2)));
                        t2 = _mm_add_epi32(t2, _mm_shuffle_epi32(t2, _MM_SHUFFLE(2,3,0,1)));
                        int32_t is1_lo = _mm_cvtsi128_si32(t2);
                        __m128i t3 = _mm_add_epi32(_mm256_castsi256_si128(p1_hi32),
                                                     _mm256_extracti128_si256(p1_hi32, 1));
                        t3 = _mm_add_epi32(t3, _mm_shuffle_epi32(t3, _MM_SHUFFLE(1,0,3,2)));
                        t3 = _mm_add_epi32(t3, _mm_shuffle_epi32(t3, _MM_SHUFFLE(2,3,0,1)));
                        int32_t is1_hi = _mm_cvtsi128_si32(t3);
                        sum1_a += d_r1 * sc_a1 * dx_lo * (float)is1_lo - dmin_r1 * m_a1 * sx_lo;
                        sum1_b += d_r1 * sc_b1 * dx_hi * (float)is1_hi - dmin_r1 * m_b1 * sx_hi;

                        qs0 += 32;
                        qs1 += 32;
                        q8_idx += 2;
                    }
                }
                *get_output_ptr(n) = sum0_a + sum0_b;
                *get_output_ptr(n + 1) = sum1_a + sum1_b;
            }
            // Handle odd remaining row
            if (n < end) {
                const uint8_t* row_data = get_weight_row(n);
                float sumf = 0.0f;
                for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
                    sumf += q4k_q8_dot_avx2(row_data + bi * 144, x_q8 + bi * 8);
                }
                *get_output_ptr(n) = sumf;
            }
        }, 1);
        return;
    }
#endif

    // Fallback: 3 separate GEMVs (handles Q6_K, Q5_K, Q8_0, etc.)
    cpu_quant_gemv(quant_type, w_q, x, y_q, K, N_q, row_stride_bytes);
    cpu_quant_gemv(quant_type, w_k, x, y_k, K, N_k, row_stride_bytes);
    cpu_quant_gemv(quant_type, w_v, x, y_v, K, N_v, row_stride_bytes);
}

// ============================================================================
// Fused RMSNorm + GEMV on CPU
//
// Instead of: RMSNorm -> write to temp buffer -> read temp buffer for GEMV
// We do:      RMSNorm -> stack buffer (L1 hot) -> GEMV reads from L1
//
// Saves one full hidden_size write+read from/to L2/L3 per call.
// For hidden=3584, that's 14 KB saved per GEMV call.
// ============================================================================

inline void cpu_fused_rmsnorm_gemv(
    const float* x, const float* gamma, float eps, bool add_one,
    uint32_t quant_type, const void* weight_data,
    float* y, int64_t hidden, int64_t N, int64_t row_stride_bytes) {

    // Step 1: RMSNorm into stack-allocated buffer (stays in L1)
    constexpr int64_t MAX_STACK_HIDDEN = 8192;
    float stack_buf[MAX_STACK_HIDDEN];
    float* x_norm = (hidden <= MAX_STACK_HIDDEN) ? stack_buf
                   : static_cast<float*>(std::malloc(hidden * sizeof(float)));

#ifdef __AVX2__
    __m256 sum_sq_vec = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 7 < hidden; j += 8) {
        __m256 vx = _mm256_loadu_ps(x + j);
        sum_sq_vec = _mm256_fmadd_ps(vx, vx, sum_sq_vec);
    }
    float sum_sq = hsum_avx(sum_sq_vec);
    for (; j < hidden; ++j) sum_sq += x[j] * x[j];

    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    __m256 vrms = _mm256_set1_ps(rms);
    j = 0;
    if (add_one) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(x_norm + j,
                _mm256_mul_ps(_mm256_mul_ps(vx, vrms), _mm256_add_ps(vg, one)));
        }
    } else {
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(x_norm + j, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vg));
        }
    }
    for (; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        x_norm[j] = x[j] * rms * w;
    }
#else
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < hidden; ++j) sum_sq += x[j] * x[j];
    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    for (int64_t j = 0; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        x_norm[j] = x[j] * rms * w;
    }
#endif

    // Step 2: GEMV with normalized x (still hot in L1 cache)
    cpu_quant_gemv(quant_type, weight_data, x_norm, y, hidden, N, row_stride_bytes);

    if (hidden > MAX_STACK_HIDDEN) std::free(x_norm);
}

// ============================================================================
// Fused RMSNorm + batched QKV GEMV
// Combines both fusions: RMSNorm in L1 + shared x across Q/K/V
// ============================================================================

inline void cpu_fused_rmsnorm_qkv_gemv(
    const float* x, const float* gamma, float eps, bool add_one,
    uint32_t quant_type,
    const void* w_q, const void* w_k, const void* w_v,
    float* y_q, float* y_k, float* y_v,
    int64_t hidden, int64_t N_q, int64_t N_k, int64_t N_v,
    int64_t row_stride_bytes) {

    constexpr int64_t MAX_STACK_HIDDEN = 8192;
    float stack_buf[MAX_STACK_HIDDEN];
    float* x_norm = (hidden <= MAX_STACK_HIDDEN) ? stack_buf
                   : static_cast<float*>(std::malloc(hidden * sizeof(float)));

#ifdef __AVX2__
    __m256 sum_sq_vec = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 7 < hidden; j += 8) {
        __m256 vx = _mm256_loadu_ps(x + j);
        sum_sq_vec = _mm256_fmadd_ps(vx, vx, sum_sq_vec);
    }
    float sum_sq = hsum_avx(sum_sq_vec);
    for (; j < hidden; ++j) sum_sq += x[j] * x[j];

    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    __m256 vrms = _mm256_set1_ps(rms);
    j = 0;
    if (add_one) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(x_norm + j,
                _mm256_mul_ps(_mm256_mul_ps(vx, vrms), _mm256_add_ps(vg, one)));
        }
    } else {
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(x_norm + j, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vg));
        }
    }
    for (; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        x_norm[j] = x[j] * rms * w;
    }
#else
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < hidden; ++j) sum_sq += x[j] * x[j];
    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    for (int64_t j = 0; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        x_norm[j] = x[j] * rms * w;
    }
#endif

    cpu_quant_gemv_batched_qkv(quant_type, w_q, w_k, w_v,
        x_norm, y_q, y_k, y_v,
        hidden, N_q, N_k, N_v, row_stride_bytes);

    if (hidden > MAX_STACK_HIDDEN) std::free(x_norm);
}

// ============================================================================
// Fused RMSNorm + batched gate+up GEMV
// Same fusion for FFN: normalize x once, then gate and up share the L1-hot x
// ============================================================================

inline void cpu_fused_rmsnorm_gate_up_gemv(
    const float* x, const float* gamma, float eps, bool add_one,
    uint32_t quant_type_gate, const void* w_gate,
    uint32_t quant_type_up, const void* w_up,
    float* y_gate, float* y_up,
    int64_t hidden, int64_t N_gate, int64_t N_up,
    int64_t row_stride_gate, int64_t row_stride_up,
    const torch::io::ReplicatedWeight* numa_gate = nullptr,
    const torch::io::ReplicatedWeight* numa_up = nullptr) {

    constexpr int64_t MAX_STACK_HIDDEN = 8192;
    float stack_buf[MAX_STACK_HIDDEN];
    float* x_norm = (hidden <= MAX_STACK_HIDDEN) ? stack_buf
                   : static_cast<float*>(std::malloc(hidden * sizeof(float)));

#ifdef __AVX2__
    __m256 sum_sq_vec = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 7 < hidden; j += 8) {
        __m256 vx = _mm256_loadu_ps(x + j);
        sum_sq_vec = _mm256_fmadd_ps(vx, vx, sum_sq_vec);
    }
    float sum_sq = hsum_avx(sum_sq_vec);
    for (; j < hidden; ++j) sum_sq += x[j] * x[j];

    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    __m256 vrms = _mm256_set1_ps(rms);
    j = 0;
    if (add_one) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(x_norm + j,
                _mm256_mul_ps(_mm256_mul_ps(vx, vrms), _mm256_add_ps(vg, one)));
        }
    } else {
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(x_norm + j, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vg));
        }
    }
    for (; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        x_norm[j] = x[j] * rms * w;
    }
#else
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < hidden; ++j) sum_sq += x[j] * x[j];
    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    for (int64_t j = 0; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        x_norm[j] = x[j] * rms * w;
    }
#endif

    // Fuse gate and up into ONE parallel_for via batched_qkv (with N_v=0).
    // Saves 1 parallel_for sync (~120 us) per layer × 36 layers = ~4 ms/token.
    //
    // Round2 agent_5 item 3a: previously we bailed to the 2-call path when
    // either weight had a NUMA replica, losing fusion exactly in the TP
    // scenario that motivated replication. Now we resolve master-thread-
    // local replica pointers here and still take the batched path. In TP
    // mode every rank's worker pool is already `--cpunodebind`'d to one
    // node, so the resolved pointer IS node-local for every worker — no
    // cross-chip read regression.
    const void* w_gate_local = w_gate;
    const void* w_up_local   = w_up;
    if (numa_gate && numa_gate->num_replicas > 1) {
        const void* p = numa_gate->get(c10::current_numa_node());
        if (p) w_gate_local = p;
    }
    if (numa_up && numa_up->num_replicas > 1) {
        const void* p = numa_up->get(c10::current_numa_node());
        if (p) w_up_local = p;
    }
    if (quant_type_gate == quant_type_up &&
        row_stride_gate == row_stride_up) {
        cpu_quant_gemv_batched_qkv(
            quant_type_gate, w_gate_local, w_up_local, nullptr,
            x_norm, y_gate, y_up, nullptr,
            hidden, N_gate, N_up, 0, row_stride_gate);
    } else {
        cpu_quant_gemv(quant_type_gate, w_gate_local, x_norm, y_gate, hidden, N_gate, row_stride_gate, nullptr);
        cpu_quant_gemv(quant_type_up,   w_up_local,   x_norm, y_up,   hidden, N_up,   row_stride_up,   nullptr);
    }

    if (hidden > MAX_STACK_HIDDEN) std::free(x_norm);
}

// ============================================================================
// In-place RMSNorm on a buffer (for use in zero-alloc decode path)
// ============================================================================

// Out-of-place RMSNorm: reads src, writes dst — saves the pointless
// memcpy+inplace pattern (copy-and-then-overwrite) in batched paths.
// Round2 agent_5 item 7.
inline void cpu_rmsnorm_out(const float* src, float* dst,
                             const float* gamma, float eps,
                             bool add_one, int64_t hidden) {
#ifdef __AVX2__
    __m256 sum_sq_vec = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 7 < hidden; j += 8) {
        __m256 vx = _mm256_loadu_ps(src + j);
        sum_sq_vec = _mm256_fmadd_ps(vx, vx, sum_sq_vec);
    }
    float sum_sq = hsum_avx(sum_sq_vec);
    for (; j < hidden; ++j) sum_sq += src[j] * src[j];

    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    __m256 vrms = _mm256_set1_ps(rms);
    j = 0;
    if (add_one) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(src + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(dst + j,
                _mm256_mul_ps(_mm256_mul_ps(vx, vrms), _mm256_add_ps(vg, one)));
        }
    } else {
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(src + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(dst + j, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vg));
        }
    }
    for (; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        dst[j] = src[j] * rms * w;
    }
#else
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < hidden; ++j) sum_sq += src[j] * src[j];
    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    for (int64_t j = 0; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        dst[j] = src[j] * rms * w;
    }
#endif
}

inline void cpu_rmsnorm_inplace(float* x, const float* gamma, float eps,
                                 bool add_one, int64_t hidden) {
#ifdef __AVX2__
    __m256 sum_sq_vec = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 7 < hidden; j += 8) {
        __m256 vx = _mm256_loadu_ps(x + j);
        sum_sq_vec = _mm256_fmadd_ps(vx, vx, sum_sq_vec);
    }
    float sum_sq = hsum_avx(sum_sq_vec);
    for (; j < hidden; ++j) sum_sq += x[j] * x[j];

    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    __m256 vrms = _mm256_set1_ps(rms);
    j = 0;
    if (add_one) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(x + j,
                _mm256_mul_ps(_mm256_mul_ps(vx, vrms), _mm256_add_ps(vg, one)));
        }
    } else {
        for (; j + 7 < hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(gamma + j);
            _mm256_storeu_ps(x + j, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vg));
        }
    }
    for (; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        x[j] = x[j] * rms * w;
    }
#else
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < hidden; ++j) sum_sq += x[j] * x[j];
    float rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
    for (int64_t j = 0; j < hidden; ++j) {
        float w = add_one ? (1.0f + gamma[j]) : gamma[j];
        x[j] = x[j] * rms * w;
    }
#endif
}

} // namespace cpu_quant
} // namespace io
} // namespace torch
