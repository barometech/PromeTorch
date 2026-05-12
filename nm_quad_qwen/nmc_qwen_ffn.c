/* ============================================================================
 * nmc_qwen_ffn.c — Qwen3-4B FFN sub-layer (SwiGLU).
 *
 *   y[2560] = RMSNorm(x, ffn_norm)
 *   g[M] = Q4_K_GEMV(y, ffn_gate[M, 2560])
 *   u[M] = Q4_K_GEMV(y, ffn_up  [M, 2560])
 *   mul[M] = SiLU(g) * u
 *   out[N] = Q6_K_GEMV(mul, ffn_down[N, M])
 *
 * Real Qwen3-4B: ffn_dim=9728, hidden=2560. ffn_down is Q6_K.
 * Test: subset M=32 (gate/up out), N=32 (down out). Same K=2560 input.
 * Subset version validates structure; scaling = linear in M, N.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define K_DIM 2560
#define M 32    /* subset of 9728 ffn_dim */
#define N 32    /* subset of 2560 out dim */
#define BLOCKS_PER_ROW (K_DIM / 256)
#define Q4K_ROW_WORDS (BLOCKS_PER_ROW * 144)
#define BLOCKS_PER_ROW_DOWN (M / 256)
#define EPS 1.0e-6f

float        x[K_DIM];
float        ffn_norm_g[K_DIM];
unsigned int W_gate[M * Q4K_ROW_WORDS];
unsigned int W_up  [M * Q4K_ROW_WORDS];
/* W_down [N, M] in Q6_K — but M=32 doesn't form full 256-block; can't use Q6_K cleanly.
 * Workaround: use Q4_K_W_down which is M_FAKE=256 (full block) for one head.
 * Better: skip ffn_down here, replace with plain fp32 weights or another Q4_K test.
 * For this PoC just compute mul[M] and stop — validates gate+up+SiLU composing. */
float        y[K_DIM];
float        g_out[M];
float        u_out[M];
float        mul[M];

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

static void gemv_q4k(const unsigned int *Wp, int rows, float *out, const float *xv) {
    int r, blk;
    for (r = 0; r < rows; ++r) {
        float a = 0.0f;
        int row_base = r * Q4K_ROW_WORDS;
        for (blk = 0; blk < BLOCKS_PER_ROW; ++blk)
            a += q4k_block_dot(Wp, row_base + blk * 144, xv + blk * 256);
        out[r] = a;
    }
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    /* DMA settle */
    { volatile int w; for (w = 0; w < 2000000; ++w) ; }

    /* RMSNorm */
    float sum = 0.0f; int i;
    for (i = 0; i < K_DIM; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum / (float)K_DIM + EPS);
    for (i = 0; i < K_DIM; ++i) y[i] = x[i] * inv_rms * ffn_norm_g[i];

    /* gate, up projections */
    gemv_q4k(W_gate, M, g_out, y);
    gemv_q4k(W_up,   M, u_out, y);

    /* SwiGLU: mul = SiLU(gate) * up */
    for (i = 0; i < M; ++i) {
        float gv = g_out[i];
        float silu = gv / (1.0f + expf(-gv));
        mul[i] = silu * u_out[i];
    }

    printf("NMC%d:%d FFN substep: inv_rms=%f g[0..3]=%f %f %f %f u[0..3]=%f %f %f %f mul[0..3]=%f %f %f %f\n",
        cluster, core, inv_rms,
        g_out[0], g_out[1], g_out[2], g_out[3],
        u_out[0], u_out[1], u_out[2], u_out[3],
        mul[0], mul[1], mul[2], mul[3]);
    return 0;
}
