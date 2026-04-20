// sgemm_peak_probe.c — direct cblas_sgemm throughput probe, EML_MT on Эльбрус 8C2.
// Uses EML own threading (eml_SetNumThreads), NOT OpenMP.
//
// Compile on Эльбрус:
//   gcc -O2 -I/usr/include/eml -o sgemm_probe sgemm_peak_probe.c \
//       -L/usr/lib64 -leml_algebra -leml_ilp64 -lm
// Run:
//   PARALLEL=32 ./sgemm_probe
//
// Reference peak (4-chip E8C2):
//   6 channels VLIW × 128-bit SIMD FMA × 2 flops × 1.5 GHz = 72 GFLOPS/core
//   × 32 cores = 2304 GFLOPS theoretical.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include <eml_core.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench_one(int M, int N, int K) {
    size_t a_sz = (size_t)M * K;
    size_t b_sz = (size_t)K * N;
    size_t c_sz = (size_t)M * N;

    float *A, *B, *C;
    if (posix_memalign((void **)&A, 64, a_sz * sizeof(float))) { perror("A"); return; }
    if (posix_memalign((void **)&B, 64, b_sz * sizeof(float))) { perror("B"); return; }
    if (posix_memalign((void **)&C, 64, c_sz * sizeof(float))) { perror("C"); return; }

    for (size_t i = 0; i < a_sz; ++i) A[i] = 1.0f / (1.0f + (float)i);
    for (size_t i = 0; i < b_sz; ++i) B[i] = 1.0f / (2.0f + (float)i);
    memset(C, 0, c_sz * sizeof(float));

    // Warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double flops_per_iter = 2.0 * M * N * K;
    int iters = (int)(5.0e9 / flops_per_iter);
    if (iters < 1) iters = 1;
    if (iters > 200) iters = 200;

    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double t = now_sec() - t0;

    double gflops = (flops_per_iter * iters) / (t * 1e9);
    double pct = 100.0 * gflops / 2304.0;
    printf("sgemm  M=%5d N=%5d K=%5d  iters=%-3d  time=%7.3fs  GFLOPS=%8.1f  %.1f%% of 2304 peak\n",
           M, N, K, iters, t, gflops, pct);
    fflush(stdout);

    free(A); free(B); free(C);
}

int main(void) {
    eml_32s ncur = 0;
    eml_GetNumThreads(&ncur);
    printf("=== cblas_sgemm peak probe — E8C2 4-chip, 32 cores ===\n");
    printf("EML threads reported: %d (before set)\n", (int)ncur);

    // Force 32 threads explicitly.
    eml_SetNumThreads(32);
    eml_GetNumThreads(&ncur);
    printf("EML threads after set: %d\n", (int)ncur);
    printf("Reference peak: 2304 GFLOPS (6ch × 128b SIMD FMA × 1.5GHz × 32c)\n\n");

    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== Square M=N=K ===\n");
    for (int i = 0; i < n_sizes; ++i) {
        bench_one(sizes[i], sizes[i], sizes[i]);
    }

    // MNIST-like shapes (batch=64, layer dims)
    printf("\n=== MNIST-shaped (batch=64) ===\n");
    int mnist_dims[][3] = {
        { 64, 512, 784 },
        { 64, 256, 512 },
        { 64, 128, 256 },
        { 64,  10, 128 },
    };
    for (int i = 0; i < 4; ++i) {
        bench_one(mnist_dims[i][0], mnist_dims[i][1], mnist_dims[i][2]);
    }

    // Transformer qwen3:4b shapes
    printf("\n=== Transformer qwen3:4b per-layer ===\n");
    bench_one(1, 2560, 2560);
    bench_one(1, 6912, 2560);
    bench_one(1, 2560, 6912);
    bench_one(128, 2560, 2560);
    bench_one(128, 6912, 2560);
    bench_one(128, 2560, 6912);

    return 0;
}
