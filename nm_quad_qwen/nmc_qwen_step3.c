/* ============================================================================
 * nmc_qwen_step3.c — Qwen forward STEP 3: full QKV computation.
 *
 *   y[2560] = RMSNorm(x, gamma)
 *   q[2560] = Q4_K_GEMV(y, W_q[2560×2560])
 *   k[512]  = Q4_K_GEMV(y, W_k[512×2560])   (GQA: 4 KV heads × 128)
 *   v[512]  = Q4_K_GEMV(y, W_v[512×2560])
 *
 * Real Qwen3-4B attention input projection. All three weight matrices in one
 * NMC4 EMI region (~5.2 MB total). Validates что 3 sequential Q4_K GEMVs
 * композируются bit-exact.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define K_DIM 2560        /* hidden_size */
#define M_Q   2560
#define M_K   512         /* num_kv_heads × head_dim = 4 × 128 */
#define M_V   512
#define BLOCKS_PER_ROW (K_DIM / 256)
#define ROW_WORDS (BLOCKS_PER_ROW * 144)
#define EPS 1.0e-6f

float        x[K_DIM];
float        gamma_v[K_DIM];
unsigned int W_q[M_Q * ROW_WORDS];     /* 3.7 MB */
unsigned int W_k[M_K * ROW_WORDS];     /* 0.74 MB */
unsigned int W_v[M_V * ROW_WORDS];     /* 0.74 MB */
float        y[K_DIM];
float        q[M_Q];
float        k[M_K];
float        v[M_V];

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

static float q4k_dot_row(const unsigned int *W_ptr, int row_off, const float *xb) {
    int b0 = row_off;
    unsigned int d_bits    = (W_ptr[b0+0] & 0xff) | ((W_ptr[b0+1] & 0xff) << 8);
    unsigned int dmin_bits = (W_ptr[b0+2] & 0xff) | ((W_ptr[b0+3] & 0xff) << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    int scales_off = b0 + 4;
    int qs_off     = b0 + 16;
    float acc = 0.0f;
    int is = 0, j;
    for (j = 0; j < 256; j += 64) {
        unsigned int sc, m;
        if (is < 4) { sc = W_ptr[scales_off + is] & 63; m  = W_ptr[scales_off + is + 4] & 63; }
        else { sc = (W_ptr[scales_off + is + 4] & 0xF) | ((W_ptr[scales_off + is - 4] >> 6) << 4);
               m  = (W_ptr[scales_off + is + 4] >> 4) | ((W_ptr[scales_off + is    ] >> 6) << 4); }
        float d1 = d * (float)sc;  float m1 = dmin * (float)m;
        int is2 = is + 1;
        if (is2 < 4) { sc = W_ptr[scales_off + is2] & 63; m  = W_ptr[scales_off + is2 + 4] & 63; }
        else { sc = (W_ptr[scales_off + is2 + 4] & 0xF) | ((W_ptr[scales_off + is2 - 4] >> 6) << 4);
               m  = (W_ptr[scales_off + is2 + 4] >> 4) | ((W_ptr[scales_off + is2    ] >> 6) << 4); }
        float d2 = d * (float)sc;  float m2 = dmin * (float)m;

        int l;
        for (l = 0; l < 32; ++l) {
            unsigned int qb = W_ptr[qs_off + l] & 0xff;
            float v_lo = d1 * (float)(qb & 0xF)      - m1;
            float v_hi = d2 * (float)((qb >> 4) & 0xF) - m2;
            acc += v_lo * xb[j + l +  0];
            acc += v_hi * xb[j + l + 32];
        }
        qs_off += 32; is += 2;
    }
    return acc;
}

static void gemv_full(const unsigned int *W_ptr, int M, float *out, const float *xv) {
    int r, blk;
    for (r = 0; r < M; ++r) {
        float a = 0.0f;
        int row_base = r * ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk) {
            a += q4k_dot_row(W_ptr, row_base + blk * 144, xv + blk * 256);
        }
        out[r] = a;
    }
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle for large weights (~5.2 MB total) */
    { volatile int w; for (w = 0; w < 10000000; ++w) ; }

    /* RMSNorm */
    float sum = 0.0f;
    int i;
    for (i = 0; i < K_DIM; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum / (float)K_DIM + EPS);
    for (i = 0; i < K_DIM; ++i) y[i] = x[i] * inv_rms * gamma_v[i];

    /* Q-proj */
    gemv_full(W_q, M_Q, q, y);
    /* K-proj */
    gemv_full(W_k, M_K, k, y);
    /* V-proj */
    gemv_full(W_v, M_V, v, y);

    printf("NMC%d:%d step3 done: q[0..2]=%f %f %f  k[0..2]=%f %f %f  v[0..2]=%f %f %f\n",
        cluster, core,
        q[0], q[1], q[2],
        k[0], k[1], k[2],
        v[0], v[1], v[2]);
    return 0;
}
