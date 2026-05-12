/* ============================================================================
 * nmc_qwen_layer.c — FULL Qwen3-4B attention sub-layer (1 head, decode step).
 *
 * Composes ВСЕ kernels в один NMC4 pass:
 *   1) RMSNorm on x[2560] with attn_norm
 *   2) Q-proj subset (1 head, head_dim=128) via Q4_K GEMV
 *   3) K-proj subset (1 KV head, 128) via Q4_K GEMV
 *   4) V-proj subset (1 KV head, 128) via Q6_K GEMV
 *   5) per-head RMSNorm on Q (attn_q_norm), on K (attn_k_norm), gamma[128]
 *   6) RoPE on Q[128] and K[128] at position pos
 *   7) Attention with cache_len tokens
 *      a) scores[t] = Q · K_cache[t]^T (scaled 1/sqrt(128))
 *      b) softmax(scores)
 *      c) attn_out[128] = sum_t scores[t] * V_cache[t]
 *   8) Output: attn_out (head result)
 *
 * For decode test: cache_len = 1 (current token only).
 * Validates structural composition of ALL 7+ kernels on real Qwen3-4B weights.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define K_DIM 2560
#define HEAD_DIM 128
#define BLOCKS_PER_ROW (K_DIM / 256)
#define Q4K_ROW_WORDS (BLOCKS_PER_ROW * 144)
#define Q6K_ROW_WORDS (BLOCKS_PER_ROW * 210)
#define CACHE_LEN 1
#define EPS 1.0e-6f
#define ROPE_BASE 1000000.0f

float        x[K_DIM];                                  /* residual */
float        attn_norm_g[K_DIM];
float        attn_q_norm_g[HEAD_DIM];
float        attn_k_norm_g[HEAD_DIM];
unsigned int W_q[HEAD_DIM * Q4K_ROW_WORDS];             /* 1 head subset */
unsigned int W_k[HEAD_DIM * Q4K_ROW_WORDS];             /* 1 KV head subset */
unsigned int W_v[HEAD_DIM * Q6K_ROW_WORDS];             /* 1 KV head subset Q6_K */
float        K_cache[CACHE_LEN][HEAD_DIM];
float        V_cache[CACHE_LEN][HEAD_DIM];
volatile int rope_pos_v;

float        y[K_DIM];
float        q_head[HEAD_DIM];
float        k_head[HEAD_DIM];
float        v_head[HEAD_DIM];
float        attn_out[HEAD_DIM];

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

static void gemv_q4k(const unsigned int *Wp, int M, float *out, const float *xv) {
    int r, blk;
    for (r = 0; r < M; ++r) {
        float a = 0.0f;
        int row_base = r * Q4K_ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk)
            a += q4k_block_dot(Wp, row_base + blk * 144, xv + blk * 256);
        out[r] = a;
    }
}
static void gemv_q6k(const unsigned int *Wp, int M, float *out, const float *xv) {
    int r, blk;
    for (r = 0; r < M; ++r) {
        float a = 0.0f;
        int row_base = r * Q6K_ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk)
            a += q6k_block_dot(Wp, row_base + blk * 210, xv + blk * 256);
        out[r] = a;
    }
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle (large weights) */
    { volatile int w; for (w = 0; w < 3000000; ++w) ; }

    /* 1) RMSNorm на x [2560] */
    float sum = 0.0f; int i;
    for (i = 0; i < K_DIM; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum / (float)K_DIM + EPS);
    for (i = 0; i < K_DIM; ++i) y[i] = x[i] * inv_rms * attn_norm_g[i];

    /* 2) Q-proj subset (1 head) */
    gemv_q4k(W_q, HEAD_DIM, q_head, y);
    /* 3) K-proj subset */
    gemv_q4k(W_k, HEAD_DIM, k_head, y);
    /* 4) V-proj Q6_K subset */
    gemv_q6k(W_v, HEAD_DIM, v_head, y);

    /* 5) per-head RMSNorm на Q и K */
    float qs = 0.0f, ks = 0.0f;
    for (i = 0; i < HEAD_DIM; ++i) { qs += q_head[i] * q_head[i]; ks += k_head[i] * k_head[i]; }
    float q_inv = 1.0f / sqrtf(qs / (float)HEAD_DIM + EPS);
    float k_inv = 1.0f / sqrtf(ks / (float)HEAD_DIM + EPS);
    for (i = 0; i < HEAD_DIM; ++i) {
        q_head[i] = q_head[i] * q_inv * attn_q_norm_g[i];
        k_head[i] = k_head[i] * k_inv * attn_k_norm_g[i];
    }

    /* 6) RoPE на Q, K (pair (2i, 2i+1) rotated by pos*theta_i) */
    int pos = rope_pos_v;
    for (i = 0; i < HEAD_DIM; i += 2) {
        float theta = 1.0f / powf(ROPE_BASE, (float)i / (float)HEAD_DIM);
        float angle = (float)pos * theta;
        float c = cosf(angle), s = sinf(angle);
        float q0 = q_head[i], q1 = q_head[i + 1];
        float k0 = k_head[i], k1 = k_head[i + 1];
        q_head[i] = q0 * c - q1 * s; q_head[i + 1] = q0 * s + q1 * c;
        k_head[i] = k0 * c - k1 * s; k_head[i + 1] = k0 * s + k1 * c;
    }

    /* 7) Append current K, V в cache (slot 0 of CACHE_LEN=1) */
    for (i = 0; i < HEAD_DIM; ++i) { K_cache[0][i] = k_head[i]; V_cache[0][i] = v_head[i]; }

    /* 7a) Attention scores = Q · K_cache^T (1 token, trivial softmax) */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float scores[CACHE_LEN];
    int t;
    for (t = 0; t < CACHE_LEN; ++t) {
        float s_acc = 0.0f;
        for (i = 0; i < HEAD_DIM; ++i) s_acc += q_head[i] * K_cache[t][i];
        scores[t] = s_acc * scale;
    }
    /* softmax (with cache_len=1 it's trivially 1.0) */
    float mx = scores[0];
    for (t = 1; t < CACHE_LEN; ++t) if (scores[t] > mx) mx = scores[t];
    float ssum = 0.0f;
    for (t = 0; t < CACHE_LEN; ++t) { scores[t] = expf(scores[t] - mx); ssum += scores[t]; }
    float inv_ssum = 1.0f / ssum;
    for (t = 0; t < CACHE_LEN; ++t) scores[t] *= inv_ssum;

    /* 7b) attn_out = sum_t scores[t] * V_cache[t] */
    for (i = 0; i < HEAD_DIM; ++i) attn_out[i] = 0.0f;
    for (t = 0; t < CACHE_LEN; ++t) {
        float w = scores[t];
        for (i = 0; i < HEAD_DIM; ++i) attn_out[i] += w * V_cache[t][i];
    }

    printf("NMC%d:%d 1-LAYER FORWARD: pos=%d inv_rms=%f\n", cluster, core, pos, inv_rms);
    printf("  q[0..3]    = %f %f %f %f\n", q_head[0], q_head[1], q_head[2], q_head[3]);
    printf("  k[0..3]    = %f %f %f %f\n", k_head[0], k_head[1], k_head[2], k_head[3]);
    printf("  v[0..3]    = %f %f %f %f\n", v_head[0], v_head[1], v_head[2], v_head[3]);
    printf("  attn[0..3] = %f %f %f %f\n", attn_out[0], attn_out[1], attn_out[2], attn_out[3]);
    return 0;
}
