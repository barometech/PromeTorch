// Q4_SOA4 microbench — Round 4 Step 4 gate.
// Сравнить cycles vs Q8 SoA4 baseline (1.21 ms / GEMV K=2560 N=2432).
// Цель: ≤ 0.7 ms (50%+ faster) или хотя бы ≤ 0.9 ms (still net win по BW).
//
// Layout:
//   Per super-block of 4 rows × 32 K-elements = 128 weights:
//     bytes  0..7:   4× fp16 d_w
//     bytes  8..15:  4× fp16 dmin_m
//     bytes 16..23:  4× int16 sum_q
//     bytes 24..87:  64 bytes packed nibbles, 8 K-groups × 8 bytes each.
//                    Per K-group of 4 K-elements × 4 rows:
//                      byte 0: row0 [K0=lo, K1=hi]
//                      byte 1: row0 [K2=lo, K3=hi]
//                      byte 2: row1 [K0=lo, K1=hi]
//                      byte 3: row1 [K2=lo, K3=hi]
//                      byte 4: row2 [K0=lo, K1=hi]
//                      byte 5: row2 [K2=lo, K3=hi]
//                      byte 6: row3 [K0=lo, K1=hi]
//                      byte 7: row3 [K2=lo, K3=hi]
//
// Inner kernel per K-group:
//   1. Load 8 bytes packed (only first half of __m128i)
//   2. lo = packed & 0x0F   (8 i8 low nibbles)
//   3. hi = (packed >> 4) & 0x0F   (8 i8 high nibbles)
//   4. expanded = unpacklo_epi8(lo, hi)
//        = [lo[0],hi[0],lo[1],hi[1],...,lo[7],hi[7]]
//        = [r0_K0, r0_K1, r0_K2, r0_K3, r1_K0, r1_K1, r1_K2, r1_K3,
//           r2_K0, r2_K1, r2_K2, r2_K3, r3_K0, r3_K1, r3_K2, r3_K3]
//   5. qpmaddubsh(activation_16, expanded) → 8 i16 lanes (per-row partial)
//   6. qpmaddh(p16, ONES16) → 4 i32 lanes (one per row)
//   7. Accumulate.
//
// Bandwidth: 88 B / 128 weights = 0.6875 B/param (vs Q8 SoA4 1.375).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __e2k__
#include <emmintrin.h>  // __m128i, _mm_unpacklo_epi8, etc.

typedef long long v2di __attribute__((vector_size(16)));
typedef short v8hi __attribute__((vector_size(16)));
typedef unsigned char v16qu __attribute__((vector_size(16)));

static const v2di ONES16   = {0x0001000100010001LL, 0x0001000100010001LL};
static const v2di MASK_0F  = {0x0F0F0F0F0F0F0F0FLL, 0x0F0F0F0F0F0F0F0FLL};
static const v2di SHIFT128 = {0x0000008000000080LL, 0x0000008000000080LL};
#endif

#define K 2560
#define N 2432
#define ITERS 50

#define Q4SOA4_GROUP_BYTES 88

// fp16 helpers (только для init данных)
static inline unsigned short fp32_to_fp16_half(float f) {
    unsigned int x; memcpy(&x, &f, 4);
    int s = (x >> 16) & 0x8000;
    int e = ((x >> 23) & 0xFF) - 112;
    int m = (x >> 13) & 0x3FF;
    if (e <= 0) return (unsigned short)s;
    if (e >= 31) return (unsigned short)(s | 0x7C00);
    return (unsigned short)(s | (e << 10) | m);
}

static inline float fp16_to_fp32_half(unsigned short h) {
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1F;
    int m = h & 0x3FF;
    int xs = s << 31;
    int xe, xm;
    if (e == 0) { xe = m ? 113 : 0; xm = m << 13; while (m && !(m & 0x400)) { m <<= 1; xe--; } xm = (m & 0x3FF) << 13; }
    else if (e == 31) { xe = 255 << 23; xm = m << 13; }
    else { xe = (e + 112) << 23; xm = m << 13; }
    int xb = xs | xe | xm;
    float f; memcpy(&f, &xb, 4);
    return f;
}

#ifdef __e2k__
// Q4_SOA4 GEMV — single rank's portion (no parallel_for, just inner kernel).
static void q4_soa4_gemv_one(
    const unsigned char* w_mem,        // [N/4, K/32, 88]
    int64_t group_stride,
    const unsigned char* a_b16,        // K*4 bytes (existing Q8 SoA4 activation layout)
    const int* sum_a_per_block,        // K/32 ints
    float scale_a,
    float* y                           // [N]
) {
    const int64_t bpr = K / 32;
    const int64_t gpr = N / 4;

    v2di scale_a_v;
    {
        float arr[4] = {scale_a, scale_a, scale_a, scale_a};
        memcpy(&scale_a_v, arr, 16);
    }

    for (int64_t g = 0; g < gpr; g++) {
        const unsigned char* gp = w_mem + g * group_stride;
        v2di fp_acc = {0, 0};

        for (int64_t b = 0; b < bpr; b++) {
            const unsigned char* sb = gp + b * Q4SOA4_GROUP_BYTES;

            // Header (24 bytes): fp16 d_w[4], fp16 dmin_m[4], int16 sum_q[4]
            const unsigned short* d_w_h    = (const unsigned short*)(sb + 0);
            const unsigned short* dmin_m_h = (const unsigned short*)(sb + 8);
            const short*          sum_q_h  = (const short*)(sb + 16);

            v2di scales_v, dmins_v, sum_q_v;
            {
                float d_w_arr[4], dmin_m_arr[4];
                int sum_q_arr[4];
                for (int r = 0; r < 4; r++) {
                    d_w_arr[r]    = fp16_to_fp32_half(d_w_h[r]);
                    dmin_m_arr[r] = fp16_to_fp32_half(dmin_m_h[r]);
                    sum_q_arr[r]  = (int)sum_q_h[r];
                }
                memcpy(&scales_v, d_w_arr, 16);
                memcpy(&dmins_v,  dmin_m_arr, 16);
                memcpy(&sum_q_v,  sum_q_arr,  16);
            }

            v2di acc_i32 = {0, 0};

            // 8 K-groups × 8 bytes packed
            const unsigned char* qs = sb + 24;
            const v2di* A_v = (const v2di*)(a_b16 + b*128);  // 8 K-groups × 16 B activation

            _Pragma("loop count(8)") _Pragma("ivdep")
            for (int kg = 0; kg < 8; kg++) {
                // Load 8 packed bytes via memcpy (alignment-safe)
                __m128i packed_8;
                {
                    long long lo64;
                    memcpy(&lo64, qs + kg*8, 8);
                    long long zero = 0;
                    long long arr2[2] = {lo64, zero};
                    memcpy(&packed_8, arr2, 16);
                }
                __m128i lo  = _mm_and_si128(packed_8, (__m128i)MASK_0F);
                __m128i shr = _mm_srli_epi16(packed_8, 4);
                __m128i hi  = _mm_and_si128(shr, (__m128i)MASK_0F);
                __m128i expanded = _mm_unpacklo_epi8(lo, hi);
                // expanded layout = Q8 SoA4 K-group:
                //   [r0_K0, r0_K1, r0_K2, r0_K3, r1_K0..K3, r2_K0..K3, r3_K0..K3]

                v2di W_v = (v2di)expanded;
                v2di p16 = __builtin_e2k_qpmaddubsh(W_v, A_v[kg]);
                v2di p32 = __builtin_e2k_qpmaddh(p16, ONES16);
                acc_i32  = __builtin_e2k_qpaddw(acc_i32, p32);
            }

            // Per-block fold (same as Q8 SoA4)
            v2di shift_v = __builtin_e2k_qpmullw(sum_q_v, SHIFT128);
            v2di dot_signed = __builtin_e2k_qpsubw(acc_i32, shift_v);
            v2di acc_f = __builtin_e2k_qpistofs(dot_signed);
            float sa_b_val = (float)sum_a_per_block[b];
            v2di sa_v;
            {
                float arr[4] = {sa_b_val, sa_b_val, sa_b_val, sa_b_val};
                memcpy(&sa_v, arr, 16);
            }
            v2di term_w = __builtin_e2k_qpfmuls(scales_v, acc_f);
            v2di term_d = __builtin_e2k_qpfmuls(dmins_v, sa_v);
            v2di delta  = __builtin_e2k_qpfmuls(scale_a_v,
                          __builtin_e2k_qpfsubs(term_w, term_d));
            fp_acc = __builtin_e2k_qpfadds(fp_acc, delta);
        }

        float lanes[4]; memcpy(lanes, &fp_acc, 16);
        y[g*4 + 0] = lanes[0];
        y[g*4 + 1] = lanes[1];
        y[g*4 + 2] = lanes[2];
        y[g*4 + 3] = lanes[3];
    }
}

// Reference: scalar dequant + dot (sanity check)
static void q4_soa4_gemv_scalar(
    const unsigned char* w_mem, int64_t group_stride,
    const unsigned char* a_b16, const int* sum_a_per_block,
    float scale_a, float* y)
{
    int64_t bpr = K / 32;
    int64_t gpr = N / 4;

    for (int64_t g = 0; g < gpr; g++) {
        const unsigned char* gp = w_mem + g * group_stride;
        float fp_acc[4] = {0,0,0,0};

        for (int64_t b = 0; b < bpr; b++) {
            const unsigned char* sb = gp + b * Q4SOA4_GROUP_BYTES;
            const unsigned short* d_w_h    = (const unsigned short*)(sb + 0);
            const unsigned short* dmin_m_h = (const unsigned short*)(sb + 8);
            const short*          sum_q_h  = (const short*)(sb + 16);
            const unsigned char*  qs = sb + 24;
            const unsigned char*  A = a_b16 + b*128;

            for (int r = 0; r < 4; r++) {
                int dot_us = 0;
                for (int kg = 0; kg < 8; kg++) {
                    // K-group's 8 bytes: r=0 in bytes 0,1; r=1 in 2,3; ...
                    int byte0_idx = kg*8 + r*2;
                    int byte1_idx = byte0_idx + 1;
                    unsigned char b0 = qs[byte0_idx];
                    unsigned char b1 = qs[byte1_idx];
                    // K=4kg+0..3
                    unsigned char w0 = b0 & 0x0F;       // K=4kg
                    unsigned char w1 = (b0 >> 4) & 0x0F; // K=4kg+1
                    unsigned char w2 = b1 & 0x0F;       // K=4kg+2
                    unsigned char w3 = (b1 >> 4) & 0x0F; // K=4kg+3
                    // a_b16 layout: per K-group of 4 K-elems, 16 bytes,
                    //   [a_K0..K3] × 4 rows broadcast
                    const unsigned char* A_kg = A + kg*16 + r*4;
                    dot_us += (int)w0 * (int)A_kg[0]
                            + (int)w1 * (int)A_kg[1]
                            + (int)w2 * (int)A_kg[2]
                            + (int)w3 * (int)A_kg[3];
                }
                int dot_signed = dot_us - 128 * (int)sum_q_h[r];
                float d_w = fp16_to_fp32_half(d_w_h[r]);
                float dm  = fp16_to_fp32_half(dmin_m_h[r]);
                float sa_b = (float)sum_a_per_block[b];
                fp_acc[r] += scale_a * (d_w * (float)dot_signed - dm * sa_b);
            }
        }
        y[g*4 + 0] = fp_acc[0];
        y[g*4 + 1] = fp_acc[1];
        y[g*4 + 2] = fp_acc[2];
        y[g*4 + 3] = fp_acc[3];
    }
}

int main(void) {
    int64_t bpr = K / 32;
    int64_t gpr = N / 4;
    size_t w_total = gpr * bpr * Q4SOA4_GROUP_BYTES;
    unsigned char* w_mem = (unsigned char*)aligned_alloc(64, w_total + 64);
    unsigned char* a_b16 = (unsigned char*)aligned_alloc(64, K * 4);
    int* sum_a = (int*)aligned_alloc(64, bpr * sizeof(int));
    float* y_simd = (float*)aligned_alloc(64, N * sizeof(float));
    float* y_scal = (float*)aligned_alloc(64, N * sizeof(float));

    // Random init (deterministic)
    srand(42);
    for (size_t i = 0; i < w_total; i++) w_mem[i] = (unsigned char)(rand() & 0xFF);
    for (int i = 0; i < K; i++) {
        int kg = i / 4;
        int kg_off = i % 4;
        for (int r = 0; r < 4; r++) {
            a_b16[kg*16 + r*4 + kg_off] = (unsigned char)((128 + (i % 50)) & 0xFF);
        }
    }
    for (int b = 0; b < bpr; b++) sum_a[b] = (b * 7) % 200 - 100;
    // Initialize valid fp16 headers
    for (int64_t g = 0; g < gpr; g++) {
        for (int64_t b = 0; b < bpr; b++) {
            unsigned char* sb = w_mem + (g * bpr + b) * Q4SOA4_GROUP_BYTES;
            unsigned short* dw  = (unsigned short*)(sb + 0);
            unsigned short* dm  = (unsigned short*)(sb + 8);
            short* sq           = (short*)(sb + 16);
            for (int r = 0; r < 4; r++) {
                dw[r] = fp32_to_fp16_half(0.01f * ((g + r) % 20 + 1));
                dm[r] = fp32_to_fp16_half(0.005f * ((g + b + r) % 10));
                int s = 0;
                for (int kg = 0; kg < 8; kg++) {
                    int by0 = kg*8 + r*2;
                    int by1 = by0 + 1;
                    unsigned char qb0 = sb[24 + by0];
                    unsigned char qb1 = sb[24 + by1];
                    s += (qb0 & 0xF) + ((qb0 >> 4) & 0xF) + (qb1 & 0xF) + ((qb1 >> 4) & 0xF);
                }
                sq[r] = (short)s;
            }
        }
    }

    // Sanity check: SIMD vs scalar
    q4_soa4_gemv_one(w_mem, bpr * Q4SOA4_GROUP_BYTES, a_b16, sum_a, 1.0f, y_simd);
    q4_soa4_gemv_scalar(w_mem, bpr * Q4SOA4_GROUP_BYTES, a_b16, sum_a, 1.0f, y_scal);
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = fabsf(y_simd[i] - y_scal[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("SIMD vs scalar max abs diff: %.6e\n", max_diff);
    if (max_diff > 0.01f) {
        printf("FAIL: SIMD path numerics off — check layout!\n");
        return 1;
    }
    printf("PASS: numerics match\n");

    // Warmup
    for (int i = 0; i < 5; i++)
        q4_soa4_gemv_one(w_mem, bpr * Q4SOA4_GROUP_BYTES, a_b16, sum_a, 1.0f, y_simd);

    // Time SIMD
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        q4_soa4_gemv_one(w_mem, bpr * Q4SOA4_GROUP_BYTES, a_b16, sum_a, 1.0f, y_simd);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_simd_total_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
    printf("Q4 SoA4 SIMD: %.2f ms / GEMV (avg over %d iters)\n",
           t_simd_total_us / ITERS / 1000.0, ITERS);

    // Time scalar (slow but reference)
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < 5; it++) {
        q4_soa4_gemv_scalar(w_mem, bpr * Q4SOA4_GROUP_BYTES, a_b16, sum_a, 1.0f, y_scal);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_scal_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
    printf("Q4 SoA4 scalar: %.2f ms / GEMV (avg over 5 iters)\n",
           t_scal_us / 5 / 1000.0);

    free(w_mem); free(a_b16); free(sum_a); free(y_simd); free(y_scal);
    return 0;
}
#else
int main(void) { puts("E2K only"); return 1; }
#endif
