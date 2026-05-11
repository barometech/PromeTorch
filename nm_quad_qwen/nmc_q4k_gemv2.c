/* ============================================================================
 * nmc_q4k_gemv2.c — optimized Q4_K GEMV on NMC4.
 *
 * Strategy: two-pass per row
 *   1. Dequantize entire block to fp32 scratch (256 floats)
 *   2. Pure FP dot product (256 MACs) — fast inner loop, easier for VLIW
 *
 * vs original (fused dequant+mac), this should give 2-4× speedup because
 * the unroll-able pure-FP inner loop maps well to NMC4 FMA units.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define M 32
#define K 256

unsigned int  gemv_W[M * 144];    /* M blocks of 144 bytes each */
float         gemv_x[K];
float         gemv_y[M];

static float fp16_to_fp32(unsigned int h) {
    unsigned int sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else { while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
               mant &= 0x3FF; exp++;
               bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4);
    return f;
}

/* Pre-dequantize a single Q4_K block to 256-fp32 scratch */
static void dequant_one_block(int row_off, float *out) {
    unsigned int d_bits = ((unsigned int)gemv_W[row_off+0] & 0xff) | (((unsigned int)gemv_W[row_off+1] & 0xff) << 8);
    unsigned int dmin_bits = ((unsigned int)gemv_W[row_off+2] & 0xff) | (((unsigned int)gemv_W[row_off+3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    int scales_off = row_off + 4;
    int qs_off     = row_off + 16;
    int is = 0, j;
    for (j = 0; j < 256; j += 64) {
        unsigned char sc, m;
        if (is < 4) { sc = gemv_W[scales_off + is] & 63; m  = gemv_W[scales_off + is + 4] & 63; }
        else { sc = (gemv_W[scales_off + is + 4] & 0xF) | ((gemv_W[scales_off + is - 4] >> 6) << 4);
               m  = (gemv_W[scales_off + is + 4] >> 4) | ((gemv_W[scales_off + is    ] >> 6) << 4); }
        float d1 = d * (float)sc;  float m1 = dmin * (float)m;
        int is2 = is + 1;
        if (is2 < 4) { sc = gemv_W[scales_off + is2] & 63; m  = gemv_W[scales_off + is2 + 4] & 63; }
        else { sc = (gemv_W[scales_off + is2 + 4] & 0xF) | ((gemv_W[scales_off + is2 - 4] >> 6) << 4);
               m  = (gemv_W[scales_off + is2 + 4] >> 4) | ((gemv_W[scales_off + is2    ] >> 6) << 4); }
        float d2 = d * (float)sc;  float m2 = dmin * (float)m;

        int l;
        for (l = 0; l < 32; ++l) {
            unsigned int qb = (unsigned int)gemv_W[qs_off + l] & 0xff;
            out[j + l +  0] = d1 * (float)(qb & 0xF)      - m1;
            out[j + l + 32] = d2 * (float)((qb >> 4) & 0xF) - m2;
        }
        qs_off += 32; is += 2;
    }
}

/* Unrolled FP dot product of 256 elements (32 × 8-unrolled) */
static float fp_dot_256(const float *a, const float *b) {
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    int i;
    for (i = 0; i < 256; i += 8) {
        acc0 += a[i+0] * b[i+0];
        acc1 += a[i+1] * b[i+1];
        acc2 += a[i+2] * b[i+2];
        acc3 += a[i+3] * b[i+3];
        acc4 += a[i+4] * b[i+4];
        acc5 += a[i+5] * b[i+5];
        acc6 += a[i+6] * b[i+6];
        acc7 += a[i+7] * b[i+7];
    }
    return (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    float scratch[256];   /* dequant buffer for one block */

    /* Cache warmup: touch gemv_W[0..15] and gemv_x[0..7] before real GEMV.
     * Without this, NMC4 returns 0 for row 0 (suspected D-cache miss on
     * first read from start of EMI-mapped array after IO_ServiceStart). */
    {
        volatile unsigned int warm = 0;
        int i;
        for (i = 0; i < 16; ++i) warm += gemv_W[i];
        volatile float warmf = 0;
        for (i = 0; i < 8; ++i) warmf += gemv_x[i];
        (void)warm; (void)warmf;
    }

    int r;
    for (r = 0; r < M; ++r) {
        dequant_one_block(r * 144, scratch);
        gemv_y[r] = fp_dot_256(scratch, gemv_x);
    }

    printf("NMC%d:%d Q4_K GEMV v2: M=%d K=%d\n", cluster, core, M, K);
    printf("  y[0..3] = %f %f %f %f\n", gemv_y[0], gemv_y[1], gemv_y[2], gemv_y[3]);
    return (int)(fabsf(gemv_y[0]) * 10000.0f);
}
