/* ============================================================================
 * nmc_q4k_gemv_fast.c — Q4_K GEMV with fused dequant+MAC (no fp32 scratch).
 *
 * Optimization: dequant constants (d1, m1, d2, m2) computed once per 64-block
 * of a Q4_K superblock. Then fused dequant+MAC inner loop — no fp32 scratch.
 * Removes 256 stores+loads to/from scratch (12 ms baseline).
 *
 * Inner loop:
 *   for l in 0..32:
 *     v_lo = d1 * (qb & 0xF) - m1
 *     v_hi = d2 * ((qb>>4) & 0xF) - m2
 *     acc += v_lo * x[j+l]
 *     acc += v_hi * x[j+l+32]
 *
 * Built with -mnmc4-float -O3 -funroll-loops.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define M 32
#define K 2560
#define BLOCKS_PER_ROW (K / 256)
#define ROW_WORDS (BLOCKS_PER_ROW * 144)

unsigned int  gemv_W[M * ROW_WORDS];
float         gemv_x[K];
float         gemv_y[M];

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

/* Fused dequant + MAC for one Q4_K block. Returns dot(dequant(blk), x). */
static float q4k_block_fused_mac(int row_off, const float *xb) {
    unsigned int d_bits    = (gemv_W[row_off+0] & 0xff) | ((gemv_W[row_off+1] & 0xff) << 8);
    unsigned int dmin_bits = (gemv_W[row_off+2] & 0xff) | ((gemv_W[row_off+3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    int scales_off = row_off + 4;
    int qs_off     = row_off + 16;
    float acc = 0.0f;
    int is = 0, j;
    for (j = 0; j < 256; j += 64) {
        unsigned int sc, m;
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
        /* Fused dequant + MAC, 8-way accumulator unroll */
        float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
        for (l = 0; l < 32; l += 8) {
            unsigned int qb0 = gemv_W[qs_off + l + 0] & 0xff;
            unsigned int qb1 = gemv_W[qs_off + l + 1] & 0xff;
            unsigned int qb2 = gemv_W[qs_off + l + 2] & 0xff;
            unsigned int qb3 = gemv_W[qs_off + l + 3] & 0xff;
            unsigned int qb4 = gemv_W[qs_off + l + 4] & 0xff;
            unsigned int qb5 = gemv_W[qs_off + l + 5] & 0xff;
            unsigned int qb6 = gemv_W[qs_off + l + 6] & 0xff;
            unsigned int qb7 = gemv_W[qs_off + l + 7] & 0xff;

            a0 += (d1 * (float)(qb0 & 0xF) - m1) * xb[j + l + 0]
                + (d2 * (float)(qb0 >> 4) - m2) * xb[j + l + 32];
            a1 += (d1 * (float)(qb1 & 0xF) - m1) * xb[j + l + 1]
                + (d2 * (float)(qb1 >> 4) - m2) * xb[j + l + 33];
            a2 += (d1 * (float)(qb2 & 0xF) - m1) * xb[j + l + 2]
                + (d2 * (float)(qb2 >> 4) - m2) * xb[j + l + 34];
            a3 += (d1 * (float)(qb3 & 0xF) - m1) * xb[j + l + 3]
                + (d2 * (float)(qb3 >> 4) - m2) * xb[j + l + 35];
            a4 += (d1 * (float)(qb4 & 0xF) - m1) * xb[j + l + 4]
                + (d2 * (float)(qb4 >> 4) - m2) * xb[j + l + 36];
            a5 += (d1 * (float)(qb5 & 0xF) - m1) * xb[j + l + 5]
                + (d2 * (float)(qb5 >> 4) - m2) * xb[j + l + 37];
            a6 += (d1 * (float)(qb6 & 0xF) - m1) * xb[j + l + 6]
                + (d2 * (float)(qb6 >> 4) - m2) * xb[j + l + 38];
            a7 += (d1 * (float)(qb7 & 0xF) - m1) * xb[j + l + 7]
                + (d2 * (float)(qb7 >> 4) - m2) * xb[j + l + 39];
        }
        acc += (a0+a1)+(a2+a3)+(a4+a5)+(a6+a7);
        qs_off += 32; is += 2;
    }
    return acc;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    int r, blk;
    for (r = 0; r < M; ++r) {
        float row_acc = 0.0f;
        int row_base = r * ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk) {
            int byte_off = row_base + blk * 144;
            row_acc += q4k_block_fused_mac(byte_off, gemv_x + blk * 256);
        }
        gemv_y[r] = row_acc;
    }

    printf("NMC%d:%d Q4_K GEMV FAST: M=%d K=%d\n", cluster, core, M, K);
    printf("  y[0..3] = %f %f %f %f\n", gemv_y[0], gemv_y[1], gemv_y[2], gemv_y[3]);
    return 0;
}
