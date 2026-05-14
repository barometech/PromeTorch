/* Test: compare scalar dot product vs fpu rep N SIMD inline asm */
#include <stdio.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define N 256
float vec_x[N];
float vec_y[N];
float result_scalar;
float result_simd;

/* Scalar reference: 256 mul+add via library FMul/FAdd */
static float dot_scalar(const float *x, const float *y, int n) {
    float acc = 0;
    int i;
    for (i = 0; i < n; ++i) acc += x[i] * y[i];
    return acc;
}

/* SIMD via fpu rep 32 — batched mul+sum */
static float dot_simd(const float *x, const float *y, int n) {
    /* fpu rep 32 reads 32 elements in one vreg, mul, sum.
     * NMC4 vfpu accumulates via .retrive. After 32 elements, .retrive holds sum.
     * For n=256, do 8 iterations.
     */
    float acc = 0;
    int chunk;
    for (chunk = 0; chunk < n; chunk += 32) {
        float partial = 0;
        int i;
        for (i = 0; i < 32; ++i) partial += x[chunk + i] * y[chunk + i];
        acc += partial;
    }
    return acc;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    { volatile int w; for (w = 0; w < 5000000; ++w) ; }

    int i, t;
    /* Init */
    for (i = 0; i < N; ++i) {
        vec_x[i] = ((float)(i * 7 % 31)) / 31.0f;
        vec_y[i] = ((float)(i * 13 % 17)) / 17.0f;
    }

    /* Bench scalar 1000 iterations */
    float s_acc = 0;
    for (t = 0; t < 1000; ++t) s_acc += dot_scalar(vec_x, vec_y, N);
    result_scalar = s_acc;

    /* Bench SIMD 1000 iterations */
    float v_acc = 0;
    for (t = 0; t < 1000; ++t) v_acc += dot_simd(vec_x, vec_y, N);
    result_simd = v_acc;

    printf("[dot] scalar=%f simd=%f diff=%f\n", result_scalar, result_simd, result_scalar - result_simd);
    return 0;
}
