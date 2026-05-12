/* ============================================================================
 * nmc_rope.c — RoPE (Rotary Position Embedding) kernel for Qwen3.
 *
 * Для каждой пары (x[2i], x[2i+1]) и позиции pos:
 *   theta = 1 / (base^(2i / head_dim))
 *   angle = pos * theta
 *   c = cos(angle); s = sin(angle)
 *   x_new[2i]   = x[2i]   * c - x[2i+1] * s
 *   x_new[2i+1] = x[2i]   * s + x[2i+1] * c
 *
 * Qwen3-4B: head_dim=128, base=1000000.0 (rope_theta).
 * Test: rotate one Q vector head_dim=128 at pos=0..127.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include "nm6408load_nmc.h"

#define HEAD_DIM 128
#define BASE 1000000.0f

volatile int   rope_pos;
volatile float rope_x[HEAD_DIM];
float          rope_y[HEAD_DIM];

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    ncl_icache_ena();
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();

    int pos = rope_pos;
    int i;
    for (i = 0; i < HEAD_DIM; i += 2) {
        float exp_part = (float)i / (float)HEAD_DIM;
        float theta = 1.0f / powf(BASE, exp_part);
        float angle = (float)pos * theta;
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = rope_x[i];
        float x1 = rope_x[i + 1];
        rope_y[i]     = x0 * c - x1 * s;
        rope_y[i + 1] = x0 * s + x1 * c;
    }

    printf("NMC%d:%d RoPE: head_dim=%d pos=%d y[0..3]=%f %f %f %f\n",
        cluster, core, HEAD_DIM, pos, rope_y[0], rope_y[1], rope_y[2], rope_y[3]);
    return 0;
}
