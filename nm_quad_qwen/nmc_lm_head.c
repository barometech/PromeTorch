/* nmc_lm_head.c — Q6_K GEMV for lm_head subset (128 rows, K=2560) */
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define K_DIM 2560
#define M_OUT 128
#define BLOCKS_PER_ROW (K_DIM / 256)
#define Q6K_ROW_WORDS (BLOCKS_PER_ROW * 210)

unsigned int W_out[M_OUT * Q6K_ROW_WORDS];
float x[K_DIM];
float logits[M_OUT];

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
    float r; memcpy(&r, &bits, 4); return r;
}

static float q6k_block_dot(const unsigned int *Wp, int byte_off, const float *xb) {
    unsigned int d_bits = (Wp[byte_off + 208] & 0xff) | ((Wp[byte_off + 209] & 0xff) << 8);
    float d = fp16_to_fp32(d_bits);
    int ql_base = byte_off, qh_base = byte_off + 128, sc_base = byte_off + 192;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0; int i;
    for (i = 0; i < 256; i += 8) {
        int b;
        for (b = 0; b < 8; ++b) {
            int idx = i + b;
            int is = idx / 16;
            unsigned int ql_b = Wp[ql_base + (idx % 64) + 64 * (idx / 128)] & 0xff;
            int q_lo = (ql_b >> (4 * ((idx / 64) & 1))) & 0xF;
            unsigned int qh_b = Wp[qh_base + (idx % 32) + 32 * (idx / 128)] & 0xff;
            int q_hi = (qh_b >> (2 * ((idx / 32) & 3))) & 0x3;
            unsigned int sc_b = Wp[sc_base + is] & 0xff;
            int sc = (sc_b & 0x80) ? (int)sc_b - 256 : (int)sc_b;
            float v = d * (float)sc * (float)((q_lo | (q_hi << 4)) - 32) * xb[idx];
            switch (b) { case 0:a0+=v;break; case 1:a1+=v;break; case 2:a2+=v;break; case 3:a3+=v;break;
                         case 4:a4+=v;break; case 5:a5+=v;break; case 6:a6+=v;break; case 7:a7+=v;break; }
        }
    }
    return (a0+a1)+(a2+a3)+(a4+a5)+(a6+a7);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    { volatile int w; for (w = 0; w < 5000000  /* fast DMA settle */; ++w) ; }

    int r, blk;
    for (r = 0; r < M_OUT; ++r) {
        float a = 0;
        int row_base = r * Q6K_ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk)
            a += q6k_block_dot(W_out, row_base + blk * 210, x + blk * 256);
        logits[r] = a;
    }
    printf("[lm_head] logits[0..3]: %f %f %f %f\n", logits[0], logits[1], logits[2], logits[3]);
    return 0;
}
