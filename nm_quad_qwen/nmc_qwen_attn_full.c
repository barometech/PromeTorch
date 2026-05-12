/* ============================================================================
 * nmc_qwen_attn_full.c — Qwen3-4B FULL attention block with output projection.
 *
 * Extends 1-head attention (step4) с finishing pieces:
 *   - 2 heads (для attn_output K=256, Q4_K block compatible)
 *   - attn_output projection [M=2560, K=256] Q4_K → attn_out[2560]
 *   - residual: x_post[2560] = x + attn_out
 *
 * Compose: RMSNorm + Q-proj×2 heads + K-proj×2 + V-proj×2 + per-head norms +
 *          RoPE + attention + attn_output + residual.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define K_DIM 2560
#define HEAD_DIM 128
#define N_HEADS_SUB 2
#define ATTN_OUT_K (N_HEADS_SUB * HEAD_DIM)   /* = 256, Q4_K compat */
#define M_OUT 2560
#define BLOCKS_PER_ROW (K_DIM / 256)            /* 10 */
#define Q4K_ROW_WORDS (BLOCKS_PER_ROW * 144)    /* 1440 */
#define Q6K_ROW_WORDS (BLOCKS_PER_ROW * 210)    /* 2100 */
#define ATTN_OUT_BLOCKS (ATTN_OUT_K / 256)      /* = 1 */
#define ATTN_OUT_ROW_WORDS (ATTN_OUT_BLOCKS * 144)  /* = 144 */
#define EPS 1.0e-6f
#define ROPE_BASE 1000000.0f

float        x[K_DIM];
float        attn_norm_g[K_DIM];
float        attn_q_norm_g[HEAD_DIM];
float        attn_k_norm_g[HEAD_DIM];

unsigned int W_q[N_HEADS_SUB * HEAD_DIM * Q4K_ROW_WORDS];   /* [256, 2560] Q4_K */
unsigned int W_k[N_HEADS_SUB * HEAD_DIM * Q4K_ROW_WORDS];
unsigned int W_v[N_HEADS_SUB * HEAD_DIM * Q6K_ROW_WORDS];   /* [256, 2560] Q6_K */
unsigned int W_attn_out[M_OUT * ATTN_OUT_ROW_WORDS];        /* [2560, 256] Q4_K */
volatile int rope_pos_v;

float y[K_DIM];
float q[N_HEADS_SUB * HEAD_DIM];
float k[N_HEADS_SUB * HEAD_DIM];
float v[N_HEADS_SUB * HEAD_DIM];
float K_cache[N_HEADS_SUB][1][HEAD_DIM];
float V_cache[N_HEADS_SUB][1][HEAD_DIM];
float attn_concat[ATTN_OUT_K];      /* 256 */
float attn_out_v[M_OUT];            /* 2560 */
float x_post[M_OUT];                /* residual output */

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

static float q4k_block_dot(const unsigned int *Wp, int byte_off, const float *xb) {
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

static float q6k_block_dot(const unsigned int *Wp, int byte_off, const float *xb) {
    unsigned int d_bits = (Wp[byte_off + 208] & 0xff) | ((Wp[byte_off + 209] & 0xff) << 8);
    float d = fp16_to_fp32(d_bits);
    int ql_base = byte_off, qh_base = byte_off + 128, sc_base = byte_off + 192;
    float acc = 0.0f; int i;
    for (i = 0; i < 256; ++i) {
        int is = i / 16;
        unsigned int ql_b = Wp[ql_base + (i % 64) + 64 * (i / 128)] & 0xff;
        int q_lo = (ql_b >> (4 * ((i / 32) & 1))) & 0xF;
        unsigned int qh_b = Wp[qh_base + (i % 32) + 32 * (i / 128)] & 0xff;
        int q_hi = (qh_b >> (2 * ((i / 16) & 3))) & 0x3;
        unsigned int sc_b = Wp[sc_base + is] & 0xff;
        int sc = (sc_b & 0x80) ? (int)sc_b - 256 : (int)sc_b;
        acc += d * (float)sc * (float)((q_lo | (q_hi << 4)) - 32) * xb[i];
    }
    return acc;
}

static void gemv_q4k(const unsigned int *Wp, int M, int row_words, int blocks, float *out, const float *xv) {
    int r, blk;
    for (r = 0; r < M; ++r) {
        float a = 0.0f;
        int row_base = r * row_words;
        for (blk = 0; blk < blocks; ++blk)
            a += q4k_block_dot(Wp, row_base + blk * 144, xv + blk * 256);
        out[r] = a;
    }
}
static void gemv_q6k(const unsigned int *Wp, int M, int row_words, int blocks, float *out, const float *xv) {
    int r, blk;
    for (r = 0; r < M; ++r) {
        float a = 0.0f;
        int row_base = r * row_words;
        for (blk = 0; blk < blocks; ++blk)
            a += q6k_block_dot(Wp, row_base + blk * 210, xv + blk * 256);
        out[r] = a;
    }
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle (W_q+W_k Q4K + W_v Q6K + W_attn_out Q4K) */
    { volatile int w; for (w = 0; w < 8000000; ++w) ; }

    int i, h;

    /* 1) RMSNorm */
    float sum = 0.0f;
    for (i = 0; i < K_DIM; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum / (float)K_DIM + EPS);
    for (i = 0; i < K_DIM; ++i) y[i] = x[i] * inv_rms * attn_norm_g[i];

    /* 2-4) Q, K, V projections (2 heads × 128 = 256 floats each) */
    gemv_q4k(W_q, N_HEADS_SUB * HEAD_DIM, Q4K_ROW_WORDS, BLOCKS_PER_ROW, q, y);
    gemv_q4k(W_k, N_HEADS_SUB * HEAD_DIM, Q4K_ROW_WORDS, BLOCKS_PER_ROW, k, y);
    gemv_q6k(W_v, N_HEADS_SUB * HEAD_DIM, Q6K_ROW_WORDS, BLOCKS_PER_ROW, v, y);

    /* 5) per-head RMSNorm на каждый head в Q и K */
    for (h = 0; h < N_HEADS_SUB; ++h) {
        float qs = 0.0f, ks = 0.0f;
        for (i = 0; i < HEAD_DIM; ++i) {
            qs += q[h * HEAD_DIM + i] * q[h * HEAD_DIM + i];
            ks += k[h * HEAD_DIM + i] * k[h * HEAD_DIM + i];
        }
        float qi = 1.0f / sqrtf(qs / (float)HEAD_DIM + EPS);
        float ki = 1.0f / sqrtf(ks / (float)HEAD_DIM + EPS);
        for (i = 0; i < HEAD_DIM; ++i) {
            q[h * HEAD_DIM + i] = q[h * HEAD_DIM + i] * qi * attn_q_norm_g[i];
            k[h * HEAD_DIM + i] = k[h * HEAD_DIM + i] * ki * attn_k_norm_g[i];
        }
    }

    /* 6) RoPE на каждый head */
    int pos = rope_pos_v;
    for (h = 0; h < N_HEADS_SUB; ++h) {
        int base = h * HEAD_DIM;
        for (i = 0; i < HEAD_DIM; i += 2) {
            float theta = 1.0f / powf(ROPE_BASE, (float)i / (float)HEAD_DIM);
            float angle = (float)pos * theta;
            float c = cosf(angle), s = sinf(angle);
            float q0 = q[base + i], q1 = q[base + i + 1];
            float k0 = k[base + i], k1 = k[base + i + 1];
            q[base + i] = q0 * c - q1 * s; q[base + i + 1] = q0 * s + q1 * c;
            k[base + i] = k0 * c - k1 * s; k[base + i + 1] = k0 * s + k1 * c;
        }
    }

    /* 7) Append into KV cache (cache_len=1, slot 0) */
    for (h = 0; h < N_HEADS_SUB; ++h) {
        for (i = 0; i < HEAD_DIM; ++i) {
            K_cache[h][0][i] = k[h * HEAD_DIM + i];
            V_cache[h][0][i] = v[h * HEAD_DIM + i];
        }
    }

    /* 8) Attention per head: score = q · K_cache, softmax (trivial), attn = score*V */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (h = 0; h < N_HEADS_SUB; ++h) {
        float s_acc = 0.0f;
        for (i = 0; i < HEAD_DIM; ++i) s_acc += q[h * HEAD_DIM + i] * K_cache[h][0][i];
        s_acc *= scale;
        /* softmax trivial: 1.0 */
        for (i = 0; i < HEAD_DIM; ++i) attn_concat[h * HEAD_DIM + i] = V_cache[h][0][i];
    }

    /* 9) attn_output projection: M_OUT=2560, K=256 Q4_K */
    gemv_q4k(W_attn_out, M_OUT, ATTN_OUT_ROW_WORDS, ATTN_OUT_BLOCKS, attn_out_v, attn_concat);

    /* 10) Residual: x_post = x + attn_out */
    for (i = 0; i < M_OUT; ++i) x_post[i] = x[i] + attn_out_v[i];

    printf("NMC%d:%d FULL ATTN: pos=%d inv_rms=%f\n", cluster, core, pos, inv_rms);
    printf("  attn_out[0..3] = %f %f %f %f\n", attn_out_v[0], attn_out_v[1], attn_out_v[2], attn_out_v[3]);
    printf("  x_post[0..3]   = %f %f %f %f\n", x_post[0], x_post[1], x_post[2], x_post[3]);
    return 0;
}
