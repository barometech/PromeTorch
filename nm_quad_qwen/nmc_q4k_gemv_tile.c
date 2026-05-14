/* nmc_q4k_gemv_tile.c — tiled Q4_K GEMV for full Wq via slicing.
 * Each invocation processes M=128 rows (4 cores × 32 rows) at offset
 * gemv_slice * 128. Host invokes 32x to cover M=4096.
 *
 * Per invocation: ~30 ms wall, ~11 MMACs/sec on 4 cores.
 * Full Wq @ y (M=4096): 32 × 30 = ~960 ms.
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define K 2560
#define BLOCKS_PER_ROW (K / 256)
#define ROW_WORDS (BLOCKS_PER_ROW * 144)
#define M_TILE 128            /* per-tile, 4 cores × 32 rows */
#define M_PER_CORE 32
#define M_FULL 4096

/* Shared cluster EMI */
unsigned int  gemv_W[M_FULL * ROW_WORDS];   /* full Wq (~5.9MW = 23.6MB) */
float         gemv_x[K];
float         gemv_y[M_FULL];               /* full output */
volatile int  gemv_slice;                   /* tile index 0..31, set by host */

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
        float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
        for (l = 0; l < 32; l += 8) {
            unsigned int qb0=gemv_W[qs_off+l+0]&0xff, qb1=gemv_W[qs_off+l+1]&0xff,
                         qb2=gemv_W[qs_off+l+2]&0xff, qb3=gemv_W[qs_off+l+3]&0xff,
                         qb4=gemv_W[qs_off+l+4]&0xff, qb5=gemv_W[qs_off+l+5]&0xff,
                         qb6=gemv_W[qs_off+l+6]&0xff, qb7=gemv_W[qs_off+l+7]&0xff;
            a0 += (d1*(float)(qb0&0xF)-m1)*xb[j+l+0] + (d2*(float)(qb0>>4)-m2)*xb[j+l+32];
            a1 += (d1*(float)(qb1&0xF)-m1)*xb[j+l+1] + (d2*(float)(qb1>>4)-m2)*xb[j+l+33];
            a2 += (d1*(float)(qb2&0xF)-m1)*xb[j+l+2] + (d2*(float)(qb2>>4)-m2)*xb[j+l+34];
            a3 += (d1*(float)(qb3&0xF)-m1)*xb[j+l+3] + (d2*(float)(qb3>>4)-m2)*xb[j+l+35];
            a4 += (d1*(float)(qb4&0xF)-m1)*xb[j+l+4] + (d2*(float)(qb4>>4)-m2)*xb[j+l+36];
            a5 += (d1*(float)(qb5&0xF)-m1)*xb[j+l+5] + (d2*(float)(qb5>>4)-m2)*xb[j+l+37];
            a6 += (d1*(float)(qb6&0xF)-m1)*xb[j+l+6] + (d2*(float)(qb6>>4)-m2)*xb[j+l+38];
            a7 += (d1*(float)(qb7&0xF)-m1)*xb[j+l+7] + (d2*(float)(qb7>>4)-m2)*xb[j+l+39];
        }
        acc += (a0+a1)+(a2+a3)+(a4+a5)+(a6+a7);
        qs_off += 32; is += 2;
    }
    return acc;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    for (volatile int w = 0; w < 100000; ++w);  /* DMA race fix */
    int rank = ncl_getCoreID();
    int slice = gemv_slice;  /* tile index 0..31 */
    int r_start = slice * M_TILE + rank * M_PER_CORE;
    int r_end   = r_start + M_PER_CORE;
    int r, blk;
    for (r = r_start; r < r_end; ++r) {
        float row_acc = 0.0f;
        int row_base = r * ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk) {
            int byte_off = row_base + blk * 144;
            row_acc += q4k_block_fused_mac(byte_off, gemv_x + blk * 256);
        }
        gemv_y[r] = row_acc;
    }
    return 0;
}
