/* ============================================================================
 * nmc_attn.c — Attention forward kernel for Qwen3 (one head, decode).
 *
 * Decode step:
 *   scores[t] = Q[head_dim] · K[t][head_dim]    для t = 0..cache_len-1
 *   softmax(scores)
 *   output[head_dim] = sum_t scores[t] * V[t][head_dim]
 *
 * Test: head_dim=128, cache_len=128 (Qwen3 single head, 128-token KV cache).
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include "nm6408load_nmc.h"

#define HEAD_DIM 128
#define CACHE_LEN 128

float attn_Q[HEAD_DIM];                    /* query vector */
float attn_K[CACHE_LEN][HEAD_DIM];         /* KV cache keys */
float attn_V[CACHE_LEN][HEAD_DIM];         /* KV cache values */
float attn_scores[CACHE_LEN];              /* scratch */
float attn_out[HEAD_DIM];                  /* output */

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle — large K/V arrays (64KB each), need longer wait */
    { volatile int w; for (w = 0; w < 2000000; ++w) ; }

    /* 1) scores = Q · K^T  +  scale by 1/sqrt(head_dim) */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    int t, i;
    for (t = 0; t < CACHE_LEN; ++t) {
        float s = 0.0f;
        for (i = 0; i < HEAD_DIM; ++i) s += attn_Q[i] * attn_K[t][i];
        attn_scores[t] = s * scale;
    }

    /* 2) softmax (stable) */
    float mx = attn_scores[0];
    for (t = 1; t < CACHE_LEN; ++t) if (attn_scores[t] > mx) mx = attn_scores[t];
    float sum = 0.0f;
    for (t = 0; t < CACHE_LEN; ++t) { attn_scores[t] = expf(attn_scores[t] - mx); sum += attn_scores[t]; }
    float inv = 1.0f / sum;
    for (t = 0; t < CACHE_LEN; ++t) attn_scores[t] *= inv;

    /* 3) output = V^T · scores  (weighted sum of value vectors) */
    for (i = 0; i < HEAD_DIM; ++i) attn_out[i] = 0.0f;
    for (t = 0; t < CACHE_LEN; ++t) {
        float w = attn_scores[t];
        for (i = 0; i < HEAD_DIM; ++i) attn_out[i] += w * attn_V[t][i];
    }

    printf("NMC%d:%d attn: head_dim=%d cache_len=%d scale=%f\n",
        cluster, core, HEAD_DIM, CACHE_LEN, scale);
    printf("  scores[0..3] = %f %f %f %f (sum=%f)\n",
        attn_scores[0], attn_scores[1], attn_scores[2], attn_scores[3], sum);
    printf("  out[0..3]    = %f %f %f %f\n",
        attn_out[0], attn_out[1], attn_out[2], attn_out[3]);
    return 0;
}
