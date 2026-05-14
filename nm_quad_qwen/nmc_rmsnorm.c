/* ============================================================================
 * nmc_rmsnorm.c — RMSNorm kernel for Qwen3 на NMC4.
 *
 * Computes: y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
 * для i = 0..K-1.
 *
 * Qwen3-4B: K=2560 (hidden_size). Test K=2560 bit-exact против host.
 * Уровень: один token, один слой (Pre-RMSNorm перед attention или FFN).
 *
 * Built с -mnmc4-float -O3 -funroll-loops.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include "nm6408load_nmc.h"

#define K 2560
#define EPS 1.0e-6f   /* Qwen RMSNorm epsilon */

float rms_x[K];     /* input vector */
float rms_g[K];     /* gamma (learnable scale) */
float rms_y[K];     /* output */

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    for (volatile int w = 0; w < 100000; ++w);  /* DMA race fix */
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* mean(x^2) */
    float sum = 0.0f;
    int i;
    for (i = 0; i < K; ++i) {
        float xv = rms_x[i];
        sum += xv * xv;
    }
    float mean = sum / (float)K;
    float inv_rms = 1.0f / sqrtf(mean + EPS);

    /* y = x * inv_rms * gamma */
    for (i = 0; i < K; ++i) {
        rms_y[i] = rms_x[i] * inv_rms * rms_g[i];
    }

    printf("NMC%d:%d RMSNorm: K=%d sum=%f mean=%f inv_rms=%f\n",
        cluster, core, K, sum, mean, inv_rms);
    printf("  y[0..3] = %f %f %f %f\n", rms_y[0], rms_y[1], rms_y[2], rms_y[3]);
    return 0;
}
