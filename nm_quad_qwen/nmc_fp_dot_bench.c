/* ============================================================================
 * nmc_fp_dot_bench.c — pure FP dot product benchmark.
 * Measures NMC4 baseline FLOPS without Q4_K dequant overhead.
 *
 * Does M=32 × K=2560 = 81920 MAC = 32 dot256 × 10.
 * Compares fp dot perf to expected peak.
 * ============================================================================ */

#include <stdio.h>
#include "nm6408load_nmc.h"

#define M 32
#define K 2560

float A[M * K];   /* 32 × 2560 = 81920 floats */
float x[K];
float y[M];

static float fp_dot_256(const float *a, const float *b) {
    float acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;
    int i;
    for (i = 0; i < 256; i += 8) {
        acc0 += a[i+0] * b[i+0];
        acc1 += a[i+1] * b[i+1];
        acc2 += a[i+2] * b[i+2];
        acc3 += a[i+3] * b[i+3];
        acc4 += a[i+4] * b[i+4];
        acc5 += a[i+5] * b[i+5];
        acc6 += a[i+6] * b[i+6];
        acc7 += a[i+7] * b[i+7];
    }
    return (acc0+acc1)+(acc2+acc3)+(acc4+acc5)+(acc6+acc7);
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    int r, blk;
    for (r = 0; r < M; ++r) {
        float acc = 0.0f;
        const float *row = &A[r * K];
        for (blk = 0; blk < K; blk += 256) {
            acc += fp_dot_256(row + blk, x + blk);
        }
        y[r] = acc;
    }

    printf("NMC%d:%d FP dot bench: M=%d K=%d (no dequant)\n", cluster, core, M, K);
    printf("  y[0..3] = %f %f %f %f\n", y[0], y[1], y[2], y[3]);
    return 0;
}
