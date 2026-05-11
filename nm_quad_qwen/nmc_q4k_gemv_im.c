/* ============================================================================
 * nmc_q4k_gemv_im.c — Q4_K GEMV with IM-resident caching.
 * Cache strategy:
 *   - gemv_x (2560 floats = 10240 bytes): copy whole to IM stack buffer
 *   - gemv_W per row (1440 words): copy row to IM stack buffer before processing
 * Eliminates EMI access in inner loop.
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

/* Dequant one block from IM buffer */
static void dequant_block_im(const unsigned int *blk, float *out) {
    unsigned int d_bits    = (blk[0] & 0xff) | ((blk[1] & 0xff) << 8);
    unsigned int dmin_bits = (blk[2] & 0xff) | ((blk[3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    const unsigned int *scales = blk + 4;
    const unsigned int *qs     = blk + 16;
    int is = 0, j;
    for (j = 0; j < 256; j += 64) {
        unsigned int sc, m;
        if (is < 4) { sc = scales[is] & 63; m  = scales[is + 4] & 63; }
        else { sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
               m  = (scales[is + 4] >> 4) | ((scales[is    ] >> 6) << 4); }
        float d1 = d * (float)sc;  float m1 = dmin * (float)m;
        int is2 = is + 1;
        if (is2 < 4) { sc = scales[is2] & 63; m  = scales[is2 + 4] & 63; }
        else { sc = (scales[is2 + 4] & 0xF) | ((scales[is2 - 4] >> 6) << 4);
               m  = (scales[is2 + 4] >> 4) | ((scales[is2    ] >> 6) << 4); }
        float d2 = d * (float)sc;  float m2 = dmin * (float)m;

        int l;
        for (l = 0; l < 32; ++l) {
            unsigned int qb = qs[l] & 0xff;
            out[j + l +  0] = d1 * (float)(qb & 0xF)      - m1;
            out[j + l + 32] = d2 * (float)((qb >> 4) & 0xF) - m2;
        }
        qs += 32; is += 2;
    }
}

static float fp_dot_256(const float *a, const float *b) {
    float acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;
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
    return (acc0+acc1)+(acc2+acc3)+(acc4+acc5)+(acc6+acc7);
}

/* IM-resident caches (stack-allocated → IMB) */
int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* Copy entire x vector to IM */
    float x_im[K];
    int i;
    for (i = 0; i < K; ++i) x_im[i] = gemv_x[i];

    unsigned int row_buf[ROW_WORDS];   /* 1440 words = 5760 bytes */
    float scratch[256];

    int r, blk;
    for (r = 0; r < M; ++r) {
        /* Copy row from EMI to IM */
        const unsigned int *src = &gemv_W[r * ROW_WORDS];
        for (i = 0; i < ROW_WORDS; ++i) row_buf[i] = src[i];

        float row_acc = 0.0f;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk) {
            dequant_block_im(&row_buf[blk * 144], scratch);
            row_acc += fp_dot_256(scratch, &x_im[blk * 256]);
        }
        gemv_y[r] = row_acc;
    }

    printf("NMC%d:%d Q4_K GEMV IM-cached: M=%d K=%d\n", cluster, core, M, K);
    printf("  y[0..3] = %f %f %f %f\n", gemv_y[0], gemv_y[1], gemv_y[2], gemv_y[3]);
    return 0;
}
