/* Test: 4-float dot product via vfpu inline asm vs scalar */
#include <stdio.h>
#include "nm6408load_nmc.h"

/* Global buffers for asm I/O */
float a_buf[8];
float b_buf[8];
float result_asm[4];

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    { volatile int w; for (w = 0; w < 5000000; ++w) ; }

    /* Setup test data */
    int i;
    for (i = 0; i < 8; ++i) {
        a_buf[i] = (float)(i + 1) * 0.5f;
        b_buf[i] = (float)(i + 1) * 0.25f;
    }

    /* Scalar reference: a[0..3] * b[0..3] */
    float ref[4];
    for (i = 0; i < 4; ++i) ref[i] = a_buf[i] * b_buf[i];

    /* SIMD inline asm — load 4 floats, mul, store
     * NMC4 vfpu: vreg = 4 floats wide for .float type.
     * rep 1 vreg0 = [ar0++]  → load 4 floats (one vreg)
     * rep 1 vreg1 = [ar1++]
     * fpu 0 .float vreg7 = vreg0 * vreg1
     * rep 1 [ar2++] = vreg7
     */
    asm volatile (
        "ar0 = %0;\n"
        "ar1 = %1;\n"
        "ar2 = %2;\n"
        "rep 1 vreg0 = [ar0++];\n"
        "rep 1 vreg1 = [ar1++];\n"
        "fpu 0 .float vreg7 = vreg0 * vreg1;\n"
        "rep 1 [ar2++] = vreg7;\n"
        :
        : "g"(a_buf), "g"(b_buf), "g"(result_asm)
    );

    printf("[ref ] %f %f %f %f\n", ref[0], ref[1], ref[2], ref[3]);
    printf("[asm ] %f %f %f %f\n", result_asm[0], result_asm[1], result_asm[2], result_asm[3]);

    int mismatches = 0;
    for (i = 0; i < 4; ++i) if (ref[i] != result_asm[i]) mismatches++;
    printf("[verify] mismatches=%d/4\n", mismatches);
    return 0;
}
