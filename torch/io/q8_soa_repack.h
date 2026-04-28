// Q8_SoA4 — 4-row interleaved INT8 weight layout for E2K v5 (8C2) qpmaddubsh.
//
// Round 3 Agent 5 design. Microbench (vliw_mission/e2k_vnni/q8_soa4_microbench.c)
// validated 1.21 ms / GEMV K=2560 N=2432 single-core = 0.85x EML cblas_sgemv on
// 8C2. Closes 4× of the gap that capped Round 2's single-row VNNI kernel.
//
// Layout per super-row (4 N-rows × 32 K-elements = 176 bytes):
//   bytes  0..15  : 4× fp32 d_w     (per-row block scale = d × sub_scale_q4k)
//   bytes 16..31  : 4× fp32 dmin_m  (per-row min      = dmin × sub_min_q4k)
//   bytes 32..47  : 4× int32 sum_q  (per-row sum of int8 weights in this block)
//   bytes 48..175 : 128 bytes byte-interleaved weights, 8 K-groups × 16B each:
//                    bytes [r0[0..3], r1[0..3], r2[0..3], r3[0..3]] per K-group
//
// qpmaddubsh(W_v, A_v) → 8 i16 lanes. With activation broadcast (4 bytes
// repeated 4×) the pair-mul-add gives lane[i] = a[K]*w[K]+a[K+1]*w[K+1]
// where lanes 0,1 are row 0; 2,3 row 1; 4,5 row 2; 6,7 row 3. Then
// qpmaddh(p16, ONES16) collapses adjacent pairs: 4 i32 lanes, ONE per row.
// No horizontal reduce inside K-loop. After K-loop: per-block fp32 fold.
//
// Used via env PT_Q8_SOA=1. Falls back to default Q4_K kernel otherwise.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "torch/io/gguf_dequant.h"   // gguf::fp16_to_fp32, gguf::get_scale_min_k4
#include "c10/util/ThreadPool.h"

namespace torch {
namespace io {
namespace cpu_quant {

#ifdef __e2k__
typedef long long v2di __attribute__((vector_size(16)));
static const v2di SOA4_ONES16 = {0x0001000100010001LL, 0x0001000100010001LL};
static const v2di SOA4_SHIFT128 = {0x0000008000000080LL, 0x0000008000000080LL};
#endif

#define SOA4_GROUP_BYTES 176
#define SOA4_HEADER_BYTES 48
#define SOA4_WEIGHT_BYTES 128

// Q8 4-row interleaved weight tensor.
// Sized [N/4 groups × bpr=K/32 blocks × 176 bytes/block].
struct Q8SoA4 {
    uint8_t* mem = nullptr;
    int64_t  N = 0;
    int64_t  K = 0;
    int64_t  group_stride = 0;   // bytes per super-row group = bpr * 176
    bool     valid = false;

    Q8SoA4() = default;
    Q8SoA4(const Q8SoA4&) = delete;
    Q8SoA4& operator=(const Q8SoA4&) = delete;
    Q8SoA4(Q8SoA4&& o) noexcept { *this = std::move(o); }
    Q8SoA4& operator=(Q8SoA4&& o) noexcept {
        if (this != &o) {
            if (mem) std::free(mem);
            mem = o.mem; N = o.N; K = o.K;
            group_stride = o.group_stride; valid = o.valid;
            o.mem = nullptr; o.valid = false;
        }
        return *this;
    }
    ~Q8SoA4() { if (mem) std::free(mem); }
};

// Allocate Q8SoA4 storage. N must be divisible by 4, K by 32.
inline bool q8_soa4_alloc(Q8SoA4* w, int64_t N, int64_t K) {
    if (N % 4 != 0 || K % 32 != 0) return false;
    int64_t bpr = K / 32;
    int64_t gpr = N / 4;
    w->N = N;
    w->K = K;
    w->group_stride = bpr * SOA4_GROUP_BYTES;
    void* p = nullptr;
    if (posix_memalign(&p, 64, gpr * w->group_stride) != 0) return false;
    w->mem = static_cast<uint8_t*>(p);
    w->valid = true;
    return true;
}

// Repack Q4_K source matrix [N rows × K cols] into Q8_SoA4.
// Q4_K block layout (144B / 256 elements):
//   2B fp16 d, 2B fp16 dmin, 12B packed 6-bit sub-scales (8 sc + 8 m),
//   128B packed 4-bit qs (256 unsigned nibbles).
// Per super-block of 256 elements there are 8 sub-blocks of 32. Each
// sub-block becomes one Q8SoA4 32-elem block. Hence input bpr (Q4_K) =
// K/256 super-blocks; output bpr (Q8_SoA4) = K/32 = 8 × input bpr.
inline bool repack_q4k_to_q8soa4(const void* q4k_data, int64_t N, int64_t K,
                                  int64_t q4k_row_stride_bytes,
                                  Q8SoA4* out) {
    if (!q8_soa4_alloc(out, N, K)) return false;
    int64_t soa_bpr = K / 32;
    int64_t q4k_super_blocks_per_row = K / 256;
    if (q4k_super_blocks_per_row * 256 != K) return false;

    const uint8_t* src = static_cast<const uint8_t*>(q4k_data);

    for (int64_t g = 0; g < N / 4; g++) {
        uint8_t* gp = out->mem + g * out->group_stride;
        // Pre-convert all 4 rows' fp32 (d_w, dmin_m) per sub-block, plus
        // extract int8 weights and sum_q.
        // Per row r: q4k_row pointer = src + (g*4+r) * q4k_row_stride_bytes.
        // Per super-block sb (0..K/256-1): 8 sub-blocks (32 elements each).
        // The Q8_SoA4 block index b in [0, soa_bpr) corresponds to:
        //   sb = b / 8, j = b % 8 (Q4_K sub-block index).
        for (int64_t b = 0; b < soa_bpr; b++) {
            int64_t sb = b / 8;
            int j = (int)(b % 8);
            uint8_t* dst = gp + b * SOA4_GROUP_BYTES;
            float* d_w_field   = reinterpret_cast<float*>(dst + 0);
            float* dmin_field  = reinterpret_cast<float*>(dst + 16);
            int32_t* sum_q_fld = reinterpret_cast<int32_t*>(dst + 32);
            int8_t* W_field    = reinterpret_cast<int8_t*>(dst + 48);

            for (int r = 0; r < 4; r++) {
                int64_t row = g * 4 + r;
                const uint8_t* q4k_row = src + row * q4k_row_stride_bytes;
                const uint8_t* block = q4k_row + sb * 144;
                uint16_t d_bits, dmin_bits;
                std::memcpy(&d_bits,    block,     2);
                std::memcpy(&dmin_bits, block + 2, 2);
                float d = gguf::fp16_to_fp32(d_bits);
                float dmin = gguf::fp16_to_fp32(dmin_bits);
                const uint8_t* scales12 = block + 4;
                uint8_t sc, m;
                gguf::get_scale_min_k4(j, scales12, &sc, &m);
                d_w_field[r] = d * static_cast<float>(sc);
                dmin_field[r] = dmin * static_cast<float>(m);

                // Extract 32 4-bit values for sub-block j of super-block sb.
                // qs layout: 128 bytes for 256 elements, packed two per byte.
                // For sub-block j (32 elements), first 16 elements come from
                // the LOW nibbles of qs[j*16..(j+1)*16] (when j is even? need
                // to double-check Q4_K ordering).
                //
                // Standard llama.cpp Q4_K iteration: outer loop j in 0..7
                // pairs sub-blocks (0,1), (2,3), (4,5), (6,7) sharing one
                // 32-byte qs window. For pair index p = j/2:
                //   qs window = block + 16 + p * 32
                //   sub-block 2p:    LOW nibbles of these 32 bytes (32 elems
                //                    from byte[0..31] LOW bits)... actually
                //                    from byte[0..15] LOW + byte[0..15] HIGH
                //                    -- need to reread.
                // Looking at gguf_dequant.h pattern (line 698-720) used
                // elsewhere: for each pair (j, j+1) iterating j=0..6 step 2:
                //   d1 = d * sc_j;  m1 = dmin * m_j;
                //   d2 = d * sc_{j+1}; m2 = dmin * m_{j+1};
                //   qs += 32 (per pair); per pair iterate l in 0..31:
                //     dot += (d1*(qs[l] & 0xF) - m1) * x[base_k + l]
                //     dot += (d2*(qs[l] >> 4)  - m2) * x[base_k + 32 + l]
                // So pair p covers sub-blocks 2p (low nibbles of qs) and
                // 2p+1 (high nibbles of qs), each 32 elements.
                // The qs window for pair p is qs[(p)*32 .. (p+1)*32 - 1].
                int p = j / 2;
                bool is_high = (j & 1) != 0;
                const uint8_t* qs_window = block + 16 + p * 32;  // 32 bytes per pair
                int sum = 0;
                for (int l = 0; l < 32; l++) {
                    uint8_t byte = qs_window[l];
                    uint8_t q4 = is_high ? (byte >> 4) : (byte & 0x0F);
                    // Store as int8. q4 is 0..15 (unsigned), fits int8.
                    // sum_q tracked for activation-shift correction.
                    sum += (int)q4;
                    // Will be placed into W_field at correct interleaved
                    // position below; for now temporarily store in row-major
                    // scratch.
                    // We'll build interleaved layout directly in second pass.
                    // (Simpler: store row scratch in sum_q-overlapping area? No;
                    // use a stack array.)
                    // --- Postponed: use static thread_local row_q8[32] below.
                    (void)l;
                }
                sum_q_fld[r] = sum;
            }

            // Second pass: extract int8 weights for each row and write into
            // interleaved layout. Layout (16-byte K-group of 4 K elements ×
            // 4 rows): bytes [r0[0..3], r1[0..3], r2[0..3], r3[0..3]].
            // 8 K-groups per 32-elem block.
            int8_t row_q8[4][32];
            for (int r = 0; r < 4; r++) {
                int64_t row = g * 4 + r;
                const uint8_t* q4k_row = src + row * q4k_row_stride_bytes;
                const uint8_t* block = q4k_row + sb * 144;
                int p = j / 2;
                bool is_high = (j & 1) != 0;
                const uint8_t* qs_window = block + 16 + p * 32;
                for (int l = 0; l < 32; l++) {
                    uint8_t byte = qs_window[l];
                    uint8_t q4 = is_high ? (byte >> 4) : (byte & 0x0F);
                    row_q8[r][l] = static_cast<int8_t>(q4);  // 0..15 fits int8
                }
            }
            for (int kg = 0; kg < 8; kg++) {
                for (int r = 0; r < 4; r++) {
                    for (int k = 0; k < 4; k++) {
                        W_field[kg*16 + r*4 + k] = row_q8[r][kg*4 + k];
                    }
                }
            }
        }
    }
    return true;
}

// Quantize fp32 activation into uint8 (a_s8 + 128) + per-block sum + scale.
// Output:
//   a_b16: K*4 bytes = K/4 K-groups × 16 bytes (a[k..k+3] repeated 4×)
//   sum_a_per_block: K/32 int32
//   *out_scale_a
inline void q8_soa4_quant_activation(const float* x, int64_t K,
                                      uint8_t* a_b16,
                                      int32_t* sum_a_per_block,
                                      float* out_scale_a) {
    float max_a = 0;
    for (int64_t i = 0; i < K; i++) {
        float v = std::fabs(x[i]);
        if (v > max_a) max_a = v;
    }
    float scale_a = max_a > 0 ? max_a / 127.0f : 1.0f;
    *out_scale_a = scale_a;
    float inv_a = 1.0f / scale_a;

    // First pass: produce a_u8[K] in scratch buffer.
    // Reuse a_b16 first K bytes as scratch (will overwrite during broadcast).
    std::vector<uint8_t> a_u8(K);
    for (int64_t i = 0; i < K; i++) {
        int v = (int)std::lrint(x[i] * inv_a);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        a_u8[i] = static_cast<uint8_t>(v + 128);
    }
    // Broadcast: 16 bytes per K-group of 4. Layout: [a[k..k+3]] repeated 4 times.
    int64_t n_groups = K / 4;
    for (int64_t g = 0; g < n_groups; g++) {
        const uint8_t* p = a_u8.data() + g * 4;
        uint8_t* dst = a_b16 + g * 16;
        for (int r = 0; r < 4; r++) {
            dst[r*4 + 0] = p[0];
            dst[r*4 + 1] = p[1];
            dst[r*4 + 2] = p[2];
            dst[r*4 + 3] = p[3];
        }
    }
    // Sum per 32-elem block (signed, original a_s8 = a_u8 - 128).
    int64_t bpr = K / 32;
    for (int64_t b = 0; b < bpr; b++) {
        int s = 0;
        for (int k = 0; k < 32; k++) s += (int)a_u8[b*32 + k] - 128;
        sum_a_per_block[b] = s;
    }
}

// ==============================================================================
// q8_soa4_gemv: production multi-threaded GEMV.
// ==============================================================================
inline void q8_soa4_gemv(const Q8SoA4* w,
                          const uint8_t* a_b16,
                          const int32_t* sum_a_per_block,
                          float scale_a,
                          float* y) {
    int64_t bpr = w->K / 32;
    int64_t gpr = w->N / 4;

#ifdef __e2k__
    v2di scale_a_v;
    {
        float arr[4] = {scale_a, scale_a, scale_a, scale_a};
        std::memcpy(&scale_a_v, arr, 16);
    }

    c10::get_thread_pool().parallel_for(0, gpr, [&](int64_t g_start, int64_t g_end) {
        for (int64_t g = g_start; g < g_end; g++) {
            const uint8_t* gp = w->mem + g * w->group_stride;
            v2di fp_acc = {0, 0};  // 4 fp32 lanes (one per row in this group)
            for (int64_t b = 0; b < bpr; b++) {
                const uint8_t* sb = gp + b * SOA4_GROUP_BYTES;
                v2di scales_v = *(const v2di*)(sb + 0);
                v2di dmins_v  = *(const v2di*)(sb + 16);
                v2di sum_q_v  = *(const v2di*)(sb + 32);
                const v2di* W_v = (const v2di*)(sb + 48);
                const v2di* A_v = (const v2di*)(a_b16 + b*128);

                v2di acc_i32 = {0, 0};
                // Unrolled 8-iter inner K loop (8 K-groups × 4 elements = 32)
                _Pragma("loop count(8)") _Pragma("ivdep")
                for (int kg = 0; kg < 8; kg++) {
                    v2di p16 = __builtin_e2k_qpmaddubsh(W_v[kg], A_v[kg]);
                    v2di p32 = __builtin_e2k_qpmaddh(p16, SOA4_ONES16);
                    acc_i32  = __builtin_e2k_qpaddw(acc_i32, p32);
                }

                // Per-block fold: dot_signed = acc - 128*sum_q;
                //   fp_acc += scale_a * (d_w * dot_signed - dmin_m * sum_a_block)
                v2di shift_v = __builtin_e2k_qpmullw(sum_q_v, SOA4_SHIFT128);
                v2di dot_signed = __builtin_e2k_qpsubw(acc_i32, shift_v);
                v2di acc_f = __builtin_e2k_qpistofs(dot_signed);
                float sa_b_val = static_cast<float>(sum_a_per_block[b]);
                v2di sa_v;
                {
                    float arr[4] = {sa_b_val, sa_b_val, sa_b_val, sa_b_val};
                    std::memcpy(&sa_v, arr, 16);
                }
                v2di term_w = __builtin_e2k_qpfmuls(scales_v, acc_f);
                v2di term_d = __builtin_e2k_qpfmuls(dmins_v, sa_v);
                v2di delta  = __builtin_e2k_qpfmuls(scale_a_v,
                              __builtin_e2k_qpfsubs(term_w, term_d));
                fp_acc = __builtin_e2k_qpfadds(fp_acc, delta);
            }
            float lanes[4]; std::memcpy(lanes, &fp_acc, 16);
            y[g*4 + 0] = lanes[0];
            y[g*4 + 1] = lanes[1];
            y[g*4 + 2] = lanes[2];
            y[g*4 + 3] = lanes[3];
        }
    }, 1);
#else
    // Non-E2K fallback: scalar dequant (correctness baseline for x86 dev builds).
    c10::get_thread_pool().parallel_for(0, gpr, [&](int64_t g_start, int64_t g_end) {
        for (int64_t g = g_start; g < g_end; g++) {
            const uint8_t* gp = w->mem + g * w->group_stride;
            float fp_acc[4] = {0,0,0,0};
            for (int64_t b = 0; b < bpr; b++) {
                const uint8_t* sb = gp + b * SOA4_GROUP_BYTES;
                const float* scales = (const float*)(sb + 0);
                const float* dmins  = (const float*)(sb + 16);
                const int32_t* sum_q = (const int32_t*)(sb + 32);
                const int8_t* W = (const int8_t*)(sb + 48);
                const uint8_t* A = a_b16 + b*128;
                for (int r = 0; r < 4; r++) {
                    int dot_us = 0;
                    for (int kg = 0; kg < 8; kg++) {
                        const int8_t* w_k = W + kg*16 + r*4;
                        const uint8_t* a_k = A + kg*16 + r*4;
                        for (int k = 0; k < 4; k++) dot_us += (int)w_k[k] * (int)a_k[k];
                    }
                    int dot_signed = dot_us - 128 * sum_q[r];
                    fp_acc[r] += scale_a * (scales[r] * (float)dot_signed
                                            - dmins[r] * (float)sum_a_per_block[b]);
                }
            }
            y[g*4 + 0] = fp_acc[0];
            y[g*4 + 1] = fp_acc[1];
            y[g*4 + 2] = fp_acc[2];
            y[g*4 + 3] = fp_acc[3];
        }
    }, 1);
#endif
}

}  // namespace cpu_quant
}  // namespace io
}  // namespace torch
