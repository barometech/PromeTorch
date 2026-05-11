/* ============================================================================
 * nmc_q4k_gemv_full.c — Q4_K GEMV at real Qwen layer dimensions.
 * K=2560 (10 Q4_K blocks per row), M=32 rows test.
 * Per-row: 10 blocks × dequant(256 fp32) + dot_256 + accumulate
 * Total MACs per row: 2560
 * Total weights: 32 rows × 10 blocks × 144 bytes = 46080 bytes
 *
 * Output: gemv_y[M=32]
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define M 32
#define K 2560
#define BLOCKS_PER_ROW (K / 256)   /* = 10 */
#define ROW_BYTES (BLOCKS_PER_ROW * 144)   /* = 1440 */

unsigned int  gemv_W[M * ROW_BYTES];   /* M × ROW_BYTES words */
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

/* Dequantize one Q4_K block at byte_offset → 256 fp32 in out[] */
static void dequant_one_block(int byte_off, float *out) {
    unsigned int d_bits    = ((unsigned int)gemv_W[byte_off+0] & 0xff) | (((unsigned int)gemv_W[byte_off+1] & 0xff) << 8);
    unsigned int dmin_bits = ((unsigned int)gemv_W[byte_off+2] & 0xff) | (((unsigned int)gemv_W[byte_off+3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    int scales_off = byte_off + 4;
    int qs_off     = byte_off + 16;
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

/* Unrolled FP dot product of 256-element block */
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

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    float scratch[256];
    int r, blk;
    for (r = 0; r < M; ++r) {
        float row_acc = 0.0f;
        int row_base = r * ROW_BYTES;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk) {
            int byte_off = row_base + blk * 144;
            dequant_one_block(byte_off, scratch);
            row_acc += fp_dot_256(scratch, gemv_x + blk * 256);
        }
        gemv_y[r] = row_acc;
    }

    printf("NMC%d:%d Q4_K GEMV full: M=%d K=%d (%d blocks/row)\n",
        cluster, core, M, K, BLOCKS_PER_ROW);
    printf("  y[0..3] = %f %f %f %f\n", gemv_y[0], gemv_y[1], gemv_y[2], gemv_y[3]);
    return (int)(fabsf(gemv_y[1]) * 1000.0f);
}
