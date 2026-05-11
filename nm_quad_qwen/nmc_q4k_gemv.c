/* ============================================================================
 * nmc_q4k_gemv.c — Q4_K matrix-vector multiply on NMC4.
 *
 * Computes y[N] = W[N, K] @ x[K] where W is Q4_K quantized (1 superblock per
 * 256 columns). Each row of W is `K/256` Q4_K blocks (144 bytes each).
 *
 * For first test: small case M=32, N=32, K=256 (one block per row).
 * Host fills W (32 blocks × 144 bytes = 4608 bytes), x (256 floats), reads y.
 *
 * After verification: scale to real Qwen layer (K=2560, N=2560 for attn_q
 * = 10 blocks per row × 2560 rows = 25600 blocks, 3.7 MB weights → fits EMI).
 *
 * Performance target: GEMV of full Qwen layer (~5M MACs) in <50ms on 1 NMC core
 * → ~100 MMACs/sec, which leaves room for 16-core parallel ≈ 1.6 GMACs/sec.
 * At ~5M MACs/token through full model (32 layers × 4 GEMVs/layer × 2560²):
 *   ~5M MACs/layer/token × 32 layers = 160M MACs/token
 *   16 cores × 100 MMACs/sec = 1.6 GMACs/sec
 *   = 10 tok/s theoretical with full parallelism
 *
 * Real numbers TBD by this microbench.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

/* Test params (small case verifiable against host) */
#define M    32      /* output rows */
#define K    256     /* reduction (1 Q4_K block per row) */

unsigned char gemv_W[M * 144];    /* M blocks of 144 bytes each */
float         gemv_x[K];          /* input vector */
float         gemv_y[M];          /* output vector */

static float fp16_to_fp32(unsigned int h) {
    unsigned int sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF; exp++;
            bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4);
    return f;
}

static void get_scale_min_k4(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* GEMV: y[row] = sum over K cols of dequant(W[row, k]) * x[k]
 * For each row's Q4_K block, dequant + multiply-accumulate in one pass. */
static void q4k_gemv_row(const unsigned char *block, const float *x, float *y_out) {
    unsigned int d_bits, dmin_bits;
    d_bits    = ((unsigned int)block[0] & 0xff) | (((unsigned int)block[1] & 0xff) << 8);
    dmin_bits = ((unsigned int)block[2] & 0xff) | (((unsigned int)block[3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    const unsigned char *scales = block + 4;
    const unsigned char *qs     = block + 16;

    float acc = 0.0f;
    int is = 0;
    int j;
    for (j = 0; j < 256; j += 64) {
        unsigned char sc, m;
        get_scale_min_k4(is, scales, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc;
        float m2 = dmin * (float)m;

        int l;
        for (l = 0; l < 32; ++l) {
            float v_lo = d1 * (float)(qs[l] & 0xF) - m1;
            float v_hi = d2 * (float)(qs[l] >> 4) - m2;
            acc += v_lo * x[j + l +  0];
            acc += v_hi * x[j + l + 32];
        }
        qs += 32;
        is += 2;
    }
    *y_out = acc;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* Run GEMV for each row */
    int r;
    for (r = 0; r < M; ++r) {
        q4k_gemv_row(gemv_W + r * 144, gemv_x, &gemv_y[r]);
    }

    /* Print first 4 outputs + stats */
    float mn = gemv_y[0], mx = gemv_y[0], sum = 0;
    int i;
    for (i = 0; i < M; ++i) {
        if (gemv_y[i] < mn) mn = gemv_y[i];
        if (gemv_y[i] > mx) mx = gemv_y[i];
        sum += gemv_y[i];
    }
    float mean = sum / M;

    printf("NMC%d:%d Q4_K GEMV: M=%d K=%d\n", cluster, core, M, K);
    printf("  y[0..3] = %f %f %f %f\n", gemv_y[0], gemv_y[1], gemv_y[2], gemv_y[3]);
    printf("  stats: min=%f max=%f mean=%f\n", mn, mx, mean);

    return (int)(fabsf(mean) * 10000.0f);
}
