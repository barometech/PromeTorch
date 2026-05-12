/* ============================================================================
 * nmc_qwen_step1.c — Qwen forward STEP 1: RMSNorm + Q-projection (Q4_K GEMV).
 *
 * Composes 2 building blocks в один pass:
 *   y[K] = RMSNorm(x[K], gamma[K])
 *   q[M] = Q4_K_GEMV(y[K], W[M, K])
 *
 * Qwen3-4B: K=2560 (hidden), M=32 rows (subset of Q-proj output rows).
 *           Real Q-proj would be M=2560 rows; this tests subset для correctness.
 *
 * Built с -mnmc4-float -O3 -funroll-loops.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define K 2560
#define M 32
#define BLOCKS_PER_ROW (K / 256)
#define ROW_WORDS (BLOCKS_PER_ROW * 144)
#define EPS 1.0e-6f

float        x[K];                /* input (residual stream) */
float        gamma_v[K];          /* RMSNorm scale */
unsigned int W[M * ROW_WORDS];    /* Q-proj Q4_K weights, M=32 rows */
float        y[K];                /* normalized */
float        q[M];                /* Q-projection output (subset) */

static float fp16_to_fp32(unsigned int h) {
    unsigned int sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
               mant &= 0x3FF; exp++;
               bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 0x1F) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

static float q4k_block_fused_mac(int row_off, const float *xb) {
    unsigned int d_bits    = (W[row_off+0] & 0xff) | ((W[row_off+1] & 0xff) << 8);
    unsigned int dmin_bits = (W[row_off+2] & 0xff) | ((W[row_off+3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    int scales_off = row_off + 4;
    int qs_off     = row_off + 16;
    float acc = 0.0f;
    int is = 0, j;
    for (j = 0; j < 256; j += 64) {
        unsigned int sc, m;
        if (is < 4) { sc = W[scales_off + is] & 63; m  = W[scales_off + is + 4] & 63; }
        else { sc = (W[scales_off + is + 4] & 0xF) | ((W[scales_off + is - 4] >> 6) << 4);
               m  = (W[scales_off + is + 4] >> 4) | ((W[scales_off + is    ] >> 6) << 4); }
        float d1 = d * (float)sc;  float m1 = dmin * (float)m;
        int is2 = is + 1;
        if (is2 < 4) { sc = W[scales_off + is2] & 63; m  = W[scales_off + is2 + 4] & 63; }
        else { sc = (W[scales_off + is2 + 4] & 0xF) | ((W[scales_off + is2 - 4] >> 6) << 4);
               m  = (W[scales_off + is2 + 4] >> 4) | ((W[scales_off + is2    ] >> 6) << 4); }
        float d2 = d * (float)sc;  float m2 = dmin * (float)m;

        int l;
        for (l = 0; l < 32; ++l) {
            unsigned int qb = W[qs_off + l] & 0xff;
            float v_lo = d1 * (float)(qb & 0xF)      - m1;
            float v_hi = d2 * (float)((qb >> 4) & 0xF) - m2;
            acc += v_lo * xb[j + l +  0];
            acc += v_hi * xb[j + l + 32];
        }
        qs_off += 32; is += 2;
    }
    return acc;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle */
    { volatile int w; for (w = 0; w < 200000; ++w) ; }

    /* STEP 1: RMSNorm */
    float sum = 0.0f;
    int i;
    for (i = 0; i < K; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum / (float)K + EPS);
    for (i = 0; i < K; ++i) y[i] = x[i] * inv_rms * gamma_v[i];

    /* STEP 2: Q-projection Q4_K GEMV (M rows × K=2560) */
    int r, blk;
    for (r = 0; r < M; ++r) {
        float row_acc = 0.0f;
        int row_base = r * ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk) {
            int byte_off = row_base + blk * 144;
            row_acc += q4k_block_fused_mac(byte_off, y + blk * 256);
        }
        q[r] = row_acc;
    }

    printf("NMC%d:%d step1 done: inv_rms=%f q[0..3]=%f %f %f %f\n",
        cluster, core, inv_rms, q[0], q[1], q[2], q[3]);
    return 0;
}
