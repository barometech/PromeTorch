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

unsigned int  gemv_W[M * 144];    /* M blocks of 144 bytes each (1 word per byte) */
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
 * Uses index-based access (avoids NMC compiler bug with pointer arith on row 0). */
static float q4k_gemv_row_at(int row_off) {
    int b0 = row_off;
    unsigned int d_bits    = ((unsigned int)gemv_W[b0+0] & 0xff) | (((unsigned int)gemv_W[b0+1] & 0xff) << 8);
    unsigned int dmin_bits = ((unsigned int)gemv_W[b0+2] & 0xff) | (((unsigned int)gemv_W[b0+3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    int scales_off = b0 + 4;
    int qs_off     = b0 + 16;

    float acc = 0.0f;
    int is = 0;
    int j;
    for (j = 0; j < 256; j += 64) {
        unsigned char sc, m;
        /* Inline get_scale_min_k4 with explicit indexing */
        if (is < 4) {
            sc = gemv_W[scales_off + is]     & 63;
            m  = gemv_W[scales_off + is + 4] & 63;
        } else {
            sc = (gemv_W[scales_off + is + 4] & 0xF) | ((gemv_W[scales_off + is - 4] >> 6) << 4);
            m  = (gemv_W[scales_off + is + 4] >> 4) | ((gemv_W[scales_off + is    ] >> 6) << 4);
        }
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        int is2 = is + 1;
        if (is2 < 4) {
            sc = gemv_W[scales_off + is2]     & 63;
            m  = gemv_W[scales_off + is2 + 4] & 63;
        } else {
            sc = (gemv_W[scales_off + is2 + 4] & 0xF) | ((gemv_W[scales_off + is2 - 4] >> 6) << 4);
            m  = (gemv_W[scales_off + is2 + 4] >> 4) | ((gemv_W[scales_off + is2    ] >> 6) << 4);
        }
        float d2 = d * (float)sc;
        float m2 = dmin * (float)m;

        int l;
        for (l = 0; l < 32; ++l) {
            unsigned int qbyte = (unsigned int)gemv_W[qs_off + l] & 0xff;
            float v_lo = d1 * (float)(qbyte & 0xF)     - m1;
            float v_hi = d2 * (float)((qbyte >> 4) & 0xF) - m2;
            acc += v_lo * gemv_x[j + l +  0];
            acc += v_hi * gemv_x[j + l + 32];
        }
        qs_off += 32;
        is += 2;
    }
    return acc;
}

/* Silence "unused function" warning */
static void __unused_dummy(void) {
    (void)get_scale_min_k4;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* Diag: print row 0 inputs */
    printf("NMC%d:%d row0 W[0..7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
        cluster, core,
        (unsigned int)gemv_W[0] & 0xff, (unsigned int)gemv_W[1] & 0xff,
        (unsigned int)gemv_W[2] & 0xff, (unsigned int)gemv_W[3] & 0xff,
        (unsigned int)gemv_W[4] & 0xff, (unsigned int)gemv_W[5] & 0xff,
        (unsigned int)gemv_W[6] & 0xff, (unsigned int)gemv_W[7] & 0xff);
    printf("NMC%d:%d row1 W[144..147]: %02x %02x %02x %02x\n",
        cluster, core,
        (unsigned int)gemv_W[144] & 0xff, (unsigned int)gemv_W[145] & 0xff,
        (unsigned int)gemv_W[146] & 0xff, (unsigned int)gemv_W[147] & 0xff);
    printf("NMC%d:%d x[0..3]: %f %f %f %f\n",
        cluster, core, gemv_x[0], gemv_x[1], gemv_x[2], gemv_x[3]);

    /* Run GEMV for each row */
    int r;
    for (r = 0; r < M; ++r) {
        gemv_y[r] = q4k_gemv_row_at(r * 144);
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
