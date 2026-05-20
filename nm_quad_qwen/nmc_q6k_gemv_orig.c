/* ============================================================================
 * nmc_q6k_gemv.c — Q6_K GEMV для V-projection в Qwen3-4B.
 *
 *   v[M] = Q6_K_GEMV(x[K], W[M, K])
 *
 * Test: M=32 K=2560 (subset для bit-exact verification).
 * Real Qwen V-proj: M=1024, K=2560.
 *
 * Q6_K block = 210 bytes per 256 values. K=2560 → 10 blocks per row.
 * Row size = 2100 bytes = 525 words (NMC4 char=word).
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define M 32
#define K 2560
#define BLOCKS_PER_ROW (K / 256)        /* 10 */
#define ROW_WORDS (BLOCKS_PER_ROW * 210)  /* 2100 */

unsigned int q6k_W[M * ROW_WORDS];     /* 32 * 2100 = 67200 words */
float        q6k_x[K];
float        q6k_y[M];

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

/* Q6_K dot for one 256-block at byte_off, against xb[256] */
static float q6k_block_dot(int byte_off, const float *xb) {
    unsigned int d_bits = (q6k_W[byte_off + 208] & 0xff) | ((q6k_W[byte_off + 209] & 0xff) << 8);
    float d = fp16_to_fp32(d_bits);

    int ql_base = byte_off;
    int qh_base = byte_off + 128;
    int sc_base = byte_off + 192;

    float acc = 0.0f;
    int i;
    for (i = 0; i < 256; ++i) {
        int is = i / 16;
        int ql_idx = (i % 64) + 64 * (i / 128);
        int ql_shift = 4 * ((i / 32) & 1);
        unsigned int ql_b = q6k_W[ql_base + ql_idx] & 0xff;
        int q_lo = (ql_b >> ql_shift) & 0xF;

        int qh_idx = (i % 32) + 32 * (i / 128);
        int qh_shift = 2 * ((i / 16) & 3);
        unsigned int qh_b = q6k_W[qh_base + qh_idx] & 0xff;
        int q_hi = (qh_b >> qh_shift) & 0x3;

        int q6 = q_lo | (q_hi << 4);

        unsigned int sc_b = q6k_W[sc_base + is] & 0xff;
        int sc = (sc_b & 0x80) ? (int)sc_b - 256 : (int)sc_b;
        float v = d * (float)sc * (float)(q6 - 32);

        acc += v * xb[i];
    }
    return acc;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle */
    { volatile int w; for (w = 0; w < 1000000; ++w) ; }

    int r, blk;
    for (r = 0; r < M; ++r) {
        float row_acc = 0.0f;
        int row_base = r * ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk) {
            row_acc += q6k_block_dot(row_base + blk * 210, q6k_x + blk * 256);
        }
        q6k_y[r] = row_acc;
    }

    printf("NMC%d:%d Q6_K GEMV: M=%d K=%d y[0..3]=%f %f %f %f\n",
        cluster, core, M, K, q6k_y[0], q6k_y[1], q6k_y[2], q6k_y[3]);
    return 0;
}
