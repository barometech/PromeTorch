/* ============================================================================
 * nmc_silu_softmax.c — SiLU + Softmax kernels for Qwen3.
 *
 * SiLU(x) = x / (1 + exp(-x))    — gate activation in SwiGLU FFN
 * Softmax(x) = exp(x - max) / sum(exp(...))  — attention weights
 *
 * Test sizes:
 *   SiLU:    K=8192 (Qwen3 intermediate_size for FFN gate)
 *   Softmax: K=128  (attn weights per query, T=128 context length)
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include "nm6408load_nmc.h"

#define K_SILU 8192
#define K_SM   128

float silu_in[K_SILU];
float silu_out[K_SILU];
float sm_in[K_SM];
float sm_out[K_SM];

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* SiLU forward */
    int i;
    for (i = 0; i < K_SILU; ++i) {
        silu_out[i] = silu(silu_in[i]);
    }

    /* Softmax forward (numerically stable) */
    float maxv = sm_in[0];
    for (i = 1; i < K_SM; ++i) if (sm_in[i] > maxv) maxv = sm_in[i];
    float sum = 0.0f;
    for (i = 0; i < K_SM; ++i) {
        float e = expf(sm_in[i] - maxv);
        sm_out[i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (i = 0; i < K_SM; ++i) sm_out[i] *= inv;

    printf("NMC%d:%d silu+softmax: silu_out[0..3]=%f %f %f %f\n",
        cluster, core, silu_out[0], silu_out[1], silu_out[2], silu_out[3]);
    printf("  sm_out[0..3]=%f %f %f %f (max=%f sum=%f)\n",
        sm_out[0], sm_out[1], sm_out[2], sm_out[3], maxv, sum);
    return 0;
}
