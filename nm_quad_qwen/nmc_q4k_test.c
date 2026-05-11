/* ============================================================================
 * nmc_q4k_test.c — Q4_K dequantization test kernel on NMC4.
 *
 * Block layout (144 bytes per 256 values):
 *   [0..2]   d (fp16)        — global delta
 *   [2..4]   dmin (fp16)     — global min
 *   [4..16]  scales[12]      — packed 6-bit per-sub-block scales/mins
 *   [16..144] qs[128]        — 256 × 4-bit quants
 *
 * Output: 256 fp32 values in `q4k_output[]` (EMI).
 * Host writes one Q4_K block to `q4k_block[]`, NMC dequantizes,
 * host reads back `q4k_output[]` to verify.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

/* EMI input/output buffers — host fills q4k_block, NMC fills q4k_output */
unsigned char q4k_block[144];
float         q4k_output[256];
volatile int  q4k_done;

/* fp16 → fp32 (IEEE 754 half-precision) */
static float fp16_to_fp32(unsigned int h) {
    unsigned int sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF; exp++;
            bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4);
    return f;
}

/* Unpack 6-bit scale/min from packed 12-byte scales array */
static void get_scale_min_k4(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* Dequantize one Q4_K block (144 bytes → 256 floats) */
static void dequant_q4k(const unsigned char *src, float *dst) {
    unsigned int d_bits, dmin_bits;
    /* On NMC4 char is 32-bit word, so read 2 bytes via shifts */
    d_bits    = src[0] | (src[1] << 8);
    dmin_bits = src[2] | (src[3] << 8);
    float d    = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    const unsigned char *scales = src + 4;
    const unsigned char *qs     = src + 16;

    int is = 0;
    int j;
    for (j = 0; j < 256; j += 64) {
        unsigned char sc, m;
        get_scale_min_k4(is, scales, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc;
        float m2 = dmin * (float)m;
        int l;
        for (l = 0; l < 32; ++l) {
            dst[j + l +  0] = d1 * (float)(qs[l] & 0xF) - m1;
            dst[j + l + 32] = d2 * (float)(qs[l] >> 4) - m2;
        }
        qs += 32;
        is += 2;
    }
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* Diagnostic: print first 16 bytes as NMC sees them */
    printf("NMC%d:%d block[0..15]: ", cluster, core);
    int kk;
    for (kk = 0; kk < 16; ++kk) printf("%02x ", (unsigned int)q4k_block[kk] & 0xff);
    printf("\n");
    unsigned int test_d = ((unsigned int)q4k_block[0] & 0xff) |
                          (((unsigned int)q4k_block[1] & 0xff) << 8);
    printf("NMC%d:%d test_d_bits=0x%x  fp16->fp32=%f\n",
        cluster, core, test_d, fp16_to_fp32(test_d));

    /* Dequantize */
    dequant_q4k(q4k_block, q4k_output);

    /* Compute stats for verification */
    float mn = q4k_output[0], mx = q4k_output[0], sum = 0.0f;
    int i;
    for (i = 0; i < 256; ++i) {
        if (q4k_output[i] < mn) mn = q4k_output[i];
        if (q4k_output[i] > mx) mx = q4k_output[i];
        sum += q4k_output[i];
    }
    float mean = sum / 256.0f;
    float var = 0.0f;
    for (i = 0; i < 256; ++i) {
        float d = q4k_output[i] - mean;
        var += d * d;
    }
    float std = sqrtf(var / 256.0f);

    printf("NMC%d:%d Q4_K test:\n", cluster, core);
    printf("  first 8: %f %f %f %f %f %f %f %f\n",
        q4k_output[0], q4k_output[1], q4k_output[2], q4k_output[3],
        q4k_output[4], q4k_output[5], q4k_output[6], q4k_output[7]);
    printf("  stats: min=%f max=%f mean=%f std=%f\n", mn, mx, mean, std);

    q4k_done = 1;
    return (int)(std * 10000.0f);
}
