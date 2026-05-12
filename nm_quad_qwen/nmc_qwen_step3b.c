/* ============================================================================
 * nmc_qwen_step3b.c — Qwen STEP 3b: RMSNorm + Q-proj (Q4_K) + V-proj (Q6_K).
 *
 * Mixed quant composition. Subset M=32 для bit-exact verification.
 *
 *   y[2560] = RMSNorm(x, gamma)
 *   q[32]   = Q4_K_GEMV(y, W_q[2560, K=2560])    (subset of attn_q)
 *   v[32]   = Q6_K_GEMV(y, W_v[2560, K=2560])    (subset of attn_v)
 *
 * Validates что Q4_K + Q6_K kernels coexist в одном NMC4 pass на real весах.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define K_DIM 2560
#define M 32
#define BLOCKS_PER_ROW (K_DIM / 256)
#define Q4K_ROW_WORDS (BLOCKS_PER_ROW * 144)
#define Q6K_ROW_WORDS (BLOCKS_PER_ROW * 210)
#define EPS 1.0e-6f

float        x[K_DIM];
float        gamma_v[K_DIM];
unsigned int W_q4k[M * Q4K_ROW_WORDS];     /* Q-proj weights, Q4_K */
unsigned int W_q6k[M * Q6K_ROW_WORDS];     /* V-proj weights, Q6_K */
float        y[K_DIM];
float        q[M];
float        v[M];

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

static float q4k_dot_one_block(const unsigned int *Wp, int byte_off, const float *xb) {
    unsigned int d_bits    = (Wp[byte_off+0] & 0xff) | ((Wp[byte_off+1] & 0xff) << 8);
    unsigned int dmin_bits = (Wp[byte_off+2] & 0xff) | ((Wp[byte_off+3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);
    int scales_off = byte_off + 4;
    int qs_off     = byte_off + 16;
    float acc = 0.0f; int is = 0, j;
    for (j = 0; j < 256; j += 64) {
        unsigned int sc, m;
        if (is < 4) { sc = Wp[scales_off + is] & 63; m  = Wp[scales_off + is + 4] & 63; }
        else { sc = (Wp[scales_off + is + 4] & 0xF) | ((Wp[scales_off + is - 4] >> 6) << 4);
               m  = (Wp[scales_off + is + 4] >> 4) | ((Wp[scales_off + is    ] >> 6) << 4); }
        float d1 = d * (float)sc; float m1 = dmin * (float)m;
        int is2 = is + 1;
        if (is2 < 4) { sc = Wp[scales_off + is2] & 63; m  = Wp[scales_off + is2 + 4] & 63; }
        else { sc = (Wp[scales_off + is2 + 4] & 0xF) | ((Wp[scales_off + is2 - 4] >> 6) << 4);
               m  = (Wp[scales_off + is2 + 4] >> 4) | ((Wp[scales_off + is2    ] >> 6) << 4); }
        float d2 = d * (float)sc; float m2 = dmin * (float)m;
        int l;
        for (l = 0; l < 32; ++l) {
            unsigned int qb = Wp[qs_off + l] & 0xff;
            float v_lo = d1 * (float)(qb & 0xF)      - m1;
            float v_hi = d2 * (float)((qb >> 4) & 0xF) - m2;
            acc += v_lo * xb[j + l +  0];
            acc += v_hi * xb[j + l + 32];
        }
        qs_off += 32; is += 2;
    }
    return acc;
}

static float q6k_dot_one_block(const unsigned int *Wp, int byte_off, const float *xb) {
    unsigned int d_bits = (Wp[byte_off + 208] & 0xff) | ((Wp[byte_off + 209] & 0xff) << 8);
    float d = fp16_to_fp32(d_bits);
    int ql_base = byte_off, qh_base = byte_off + 128, sc_base = byte_off + 192;
    float acc = 0.0f; int i;
    for (i = 0; i < 256; ++i) {
        int is = i / 16;
        int ql_idx = (i % 64) + 64 * (i / 128);
        unsigned int ql_b = Wp[ql_base + ql_idx] & 0xff;
        int q_lo = (ql_b >> (4 * ((i / 32) & 1))) & 0xF;
        int qh_idx = (i % 32) + 32 * (i / 128);
        unsigned int qh_b = Wp[qh_base + qh_idx] & 0xff;
        int q_hi = (qh_b >> (2 * ((i / 16) & 3))) & 0x3;
        unsigned int sc_b = Wp[sc_base + is] & 0xff;
        int sc = (sc_b & 0x80) ? (int)sc_b - 256 : (int)sc_b;
        float val = d * (float)sc * (float)((q_lo | (q_hi << 4)) - 32);
        acc += val * xb[i];
    }
    return acc;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle (W_q4k 46KB + W_q6k 67KB + x + gamma) */
    { volatile int w; for (w = 0; w < 1500000; ++w) ; }

    /* RMSNorm */
    float sum = 0.0f; int i;
    for (i = 0; i < K_DIM; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum / (float)K_DIM + EPS);
    for (i = 0; i < K_DIM; ++i) y[i] = x[i] * inv_rms * gamma_v[i];

    /* Q-projection Q4_K */
    int r, blk;
    for (r = 0; r < M; ++r) {
        float a = 0.0f;
        int row_base = r * Q4K_ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk)
            a += q4k_dot_one_block(W_q4k, row_base + blk * 144, y + blk * 256);
        q[r] = a;
    }
    /* V-projection Q6_K */
    for (r = 0; r < M; ++r) {
        float a = 0.0f;
        int row_base = r * Q6K_ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk)
            a += q6k_dot_one_block(W_q6k, row_base + blk * 210, y + blk * 256);
        v[r] = a;
    }

    printf("NMC%d:%d step3b: inv_rms=%f q[0..3]=%f %f %f %f v[0..3]=%f %f %f %f\n",
        cluster, core, inv_rms,
        q[0], q[1], q[2], q[3], v[0], v[1], v[2], v[3]);
    return 0;
}
