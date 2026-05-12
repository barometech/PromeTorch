/* ============================================================================
 * nmc_q6k_test.c — Q6_K dequantization kernel for NMC4.
 *
 * Q6_K block layout (210 bytes per 256 values):
 *   [0..128]   ql[128]      — lower 4 bits of each 6-bit quant (256 nibbles)
 *   [128..192] qh[64]       — upper 2 bits, packed 4 per byte
 *   [192..208] scales[16]   — int8 per-32-block scale
 *   [208..210] d (fp16)     — super-block delta
 *
 * Dequant:
 *   For each of 16 sub-blocks of 16 values each:
 *     scale_i = d * (float)scales[i]   (signed int8)
 *     For l in 0..15: idx within sub-block
 *       qlow = (ql[low_off] >> low_shift) & 0xF
 *       qhi  = (qh[hi_off] >> hi_shift) & 0x3
 *       q6   = qlow | (qhi << 4) - 32   (signed 6-bit, range -32..31)
 *       y[i*16+l] = scale_i * q6
 *
 * Output: 256 fp32 values.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

unsigned int q6k_block[210];      /* host writes 210 bytes (as words) */
float        q6k_output[256];

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

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle */
    { volatile int w; for (w = 0; w < 200000; ++w) ; }

    /* Read d (fp16) at bytes 208..209 */
    unsigned int d_bits = (q6k_block[208] & 0xff) | ((q6k_block[209] & 0xff) << 8);
    float d = fp16_to_fp32(d_bits);

    /* For each pair of 64-element sub-groups (ggml layout):
     * Each 256-block split into 4 groups of 64. ql is 128 bytes = 256 nibbles.
     * The ggml Q6_K reference: 16 sub-blocks of 16 floats.
     * For sub-block is (0..15):
     *   scale = d * scales[is]    (signed int8)
     *   For l in 0..15:
     *     idx_4bit = is*16 + l in 0..255
     *     idx_2bit = is*16 + l
     *     ql_byte = ql[idx_4bit / 2]; nibble = (idx_4bit & 1) ? high : low
     *     qh_byte = qh[idx_2bit / 4]; bits = (qh_byte >> ((idx_2bit & 3) * 2)) & 0x3
     *     q6 = nibble | (bits << 4)
     *     signed_q6 = q6 - 32
     *     y[is*16+l] = scale * signed_q6
     *
     * But ggml has specific layout. Use ggml-style:
     *   y[i] = d * sc[i/16] * (((ql[i%64 + 64*(i/128)] >> (4*((i/32)&1))) & 0xF)
     *                          | (((qh[i%32 + 32*(i/128)] >> (2*((i/16)&3))) & 0x3) << 4)
     *                          - 32);
     */
    int i;
    for (i = 0; i < 256; ++i) {
        int is = i / 16;                            /* sub-block index 0..15 */
        int ql_idx = (i % 64) + 64 * (i / 128);     /* index into ql[0..127] */
        int ql_shift = 4 * ((i / 32) & 1);
        unsigned int ql_b = q6k_block[ql_idx] & 0xff;
        int q_lo = (ql_b >> ql_shift) & 0xF;

        int qh_idx = 128 + (i % 32) + 32 * (i / 128);
        int qh_shift = 2 * ((i / 16) & 3);
        unsigned int qh_b = q6k_block[qh_idx] & 0xff;
        int q_hi = (qh_b >> qh_shift) & 0x3;

        int q6 = q_lo | (q_hi << 4);
        int signed_q6 = q6 - 32;

        /* scale (int8, signed) at byte 192+is */
        unsigned int sc_b = q6k_block[192 + is] & 0xff;
        int sc = (sc_b & 0x80) ? (int)sc_b - 256 : (int)sc_b;   /* signed extend */
        float scale = d * (float)sc;

        q6k_output[i] = scale * (float)signed_q6;
    }

    float mn = q6k_output[0], mx = q6k_output[0], s = 0;
    int j;
    for (j = 0; j < 256; ++j) {
        if (q6k_output[j] < mn) mn = q6k_output[j];
        if (q6k_output[j] > mx) mx = q6k_output[j];
        s += q6k_output[j];
    }
    printf("NMC%d:%d Q6_K: d=%f first8=%f %f %f %f %f %f %f %f\n",
        cluster, core, d,
        q6k_output[0], q6k_output[1], q6k_output[2], q6k_output[3],
        q6k_output[4], q6k_output[5], q6k_output[6], q6k_output[7]);
    printf("  stats: min=%f max=%f mean=%f\n", mn, mx, s / 256.0f);
    return 0;
}
