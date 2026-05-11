/* ============================================================================
 * nanogpt_train_v3 — same as v2 + saves trained weights back to EMI
 * for the host to read via PL_ReadMemBlock and run inference.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define VOCAB 128
#define T     32
#define D     32
#define FF    64
#define EPS   1e-5f

#define DATA_SIZE (4 * 1024 * 1024)

/* EMI region: training input + loss output + saved weights */
char  training_data[DATA_SIZE];
volatile float core_loss_out;

/* All trained parameters flattened, in this order:
 *   [0]  Wtok[VOCAB*D] = 4096
 *   [4096]            Wpos[T*D] = 1024
 *   [5120]            Wqkv[D*3*D] = 3072
 *   [8192]            Wout[D*D] = 1024
 *   [9216]            Wfc1[D*FF] = 2048
 *   [11264]           Wfc2[FF*D] = 2048
 *   [13312]           Wunemb[D*VOCAB] = 4096
 *   [17408]           g1,b1 [2*D] = 64
 *   [17472]           g2,b2 [2*D] = 64
 *   [17536]           g3,b3 [2*D] = 64
 * Total: 17600 floats = 70.4 KB */
#define N_WEIGHTS 17600
#define N_CORES_TOTAL 16
/* Per-core slot — 16 cores share the cluster-local EMI, so we MUST avoid
 * writes at the same address (races overwrite each other). One slot per core. */
volatile float saved_weights_pool[N_CORES_TOTAL * N_WEIGHTS];

/* ---------- model parameters in IM ---------- */
static float Wtok[VOCAB][D], Wpos[T][D];
static float Wqkv[D][3 * D], Wout[D][D];
static float Wfc1[D][FF],    Wfc2[FF][D];
static float Wunemb[D][VOCAB];
static float g1[D], b1[D], g2[D], b2[D], g3[D], b3[D];

static float gWtok[VOCAB][D], gWpos[T][D];
static float gWqkv[D][3 * D], gWout[D][D];
static float gWfc1[D][FF],    gWfc2[FF][D];
static float gWunemb[D][VOCAB];
static float gg1[D], gb1[D], gg2[D], gb2[D], gg3[D], gb3[D];

static float mWtok[VOCAB][D], vWtok[VOCAB][D];
static float mWpos[T][D],     vWpos[T][D];
static float mWqkv[D][3 * D], vWqkv[D][3 * D];
static float mWout[D][D],     vWout[D][D];
static float mWfc1[D][FF],    vWfc1[D][FF];
static float mWfc2[FF][D],    vWfc2[FF][D];
static float mWunemb[D][VOCAB], vWunemb[D][VOCAB];
static float mg1[D], vg1[D], mb1[D], vb1[D];
static float mg2[D], vg2[D], mb2[D], vb2[D];
static float mg3[D], vg3[D], mb3[D], vb3[D];

static float a_x[T][D], a_ln1[T][D];
static float a_mu1[T], a_invstd1[T];
static float a_Q[T][D], a_K[T][D], a_V[T][D];
static float a_scores[T][T], a_attn[T][D], a_proj[T][D];
static float a_res1[T][D], a_ln2[T][D];
static float a_mu2[T], a_invstd2[T];
static float a_fc1pre[T][FF], a_fc1[T][FF], a_ffn[T][D];
static float a_res2[T][D], a_ln3[T][D];
static float a_mu3[T], a_invstd3[T];
static float a_logits[T][VOCAB], a_probs[T][VOCAB];

static float d_ln3[T][D], d_res2[T][D], d_ffn[T][D];
static float d_fc1[T][FF], d_ln2[T][D];
static float d_res1[T][D], d_proj[T][D], d_attn[T][D];
static float d_scores[T][T];
static float d_Q[T][D], d_K[T][D], d_V[T][D];
static float d_ln1[T][D], d_x[T][D];

static unsigned int rng;
static float frand(void) {
    rng = rng * 1103515245u + 12345u;
    return ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
}
static unsigned int rand_u32(void) {
    rng = rng * 1103515245u + 12345u;
    return rng;
}
static void init_mat(float *p, int n, float s) { int i; for (i = 0; i < n; ++i) p[i] = frand() * s; }
static void init_const(float *p, int n, float v) { int i; for (i = 0; i < n; ++i) p[i] = v; }
static void init_zero(float *p, int n) { int i; for (i = 0; i < n; ++i) p[i] = 0.0f; }

static void init_weights(unsigned int seed) {
    rng = seed ? seed : 0xC0FFEE;
    init_mat(&Wtok[0][0], VOCAB * D, 0.02f);
    init_mat(&Wpos[0][0], T * D,     0.02f);
    init_mat(&Wqkv[0][0], D * 3 * D, 1.0f / 5.65685f);
    init_mat(&Wout[0][0], D * D,     1.0f / 5.65685f);
    init_mat(&Wfc1[0][0], D * FF,    1.0f / 5.65685f);
    init_mat(&Wfc2[0][0], FF * D,    1.0f / 8.0f);
    init_mat(&Wunemb[0][0], D * VOCAB, 1.0f / 5.65685f);
    init_const(g1, D, 1.0f); init_const(b1, D, 0.0f);
    init_const(g2, D, 1.0f); init_const(b2, D, 0.0f);
    init_const(g3, D, 1.0f); init_const(b3, D, 0.0f);
    init_zero(&mWtok[0][0], VOCAB * D);   init_zero(&vWtok[0][0], VOCAB * D);
    init_zero(&mWpos[0][0], T * D);       init_zero(&vWpos[0][0], T * D);
    init_zero(&mWqkv[0][0], D * 3 * D);   init_zero(&vWqkv[0][0], D * 3 * D);
    init_zero(&mWout[0][0], D * D);       init_zero(&vWout[0][0], D * D);
    init_zero(&mWfc1[0][0], D * FF);      init_zero(&vWfc1[0][0], D * FF);
    init_zero(&mWfc2[0][0], FF * D);      init_zero(&vWfc2[0][0], FF * D);
    init_zero(&mWunemb[0][0], D * VOCAB); init_zero(&vWunemb[0][0], D * VOCAB);
    init_zero(mg1, D); init_zero(vg1, D); init_zero(mb1, D); init_zero(vb1, D);
    init_zero(mg2, D); init_zero(vg2, D); init_zero(mb2, D); init_zero(vb2, D);
    init_zero(mg3, D); init_zero(vg3, D); init_zero(mb3, D); init_zero(vb3, D);
}

static void ln_fwd(float *out, const float *in, const float *g, const float *b,
                   float *mu_out, float *invstd_out, int n) {
    int i; float mu = 0.0f, var = 0.0f;
    for (i = 0; i < n; ++i) mu += in[i]; mu /= (float)n;
    for (i = 0; i < n; ++i) { float d = in[i] - mu; var += d * d; } var /= (float)n;
    float inv = 1.0f / sqrtf(var + EPS); *mu_out = mu; *invstd_out = inv;
    for (i = 0; i < n; ++i) out[i] = (in[i] - mu) * inv * g[i] + b[i];
}

static void ln_bwd(float *d_in, const float *d_out, const float *in,
                   float mu, float invstd, const float *g, float *d_g, float *d_b, int n) {
    int i; float sum_dn = 0.0f, sum_dn_x = 0.0f;
    for (i = 0; i < n; ++i) {
        float norm_i = (in[i] - mu) * invstd; float dn = d_out[i] * g[i];
        sum_dn += dn; sum_dn_x += dn * norm_i;
        d_g[i] += d_out[i] * norm_i; d_b[i] += d_out[i];
    }
    float inv_n = 1.0f / (float)n;
    for (i = 0; i < n; ++i) {
        float norm_i = (in[i] - mu) * invstd; float dn = d_out[i] * g[i];
        d_in[i] = invstd * inv_n * ((float)n * dn - sum_dn - norm_i * sum_dn_x);
    }
}

static float forward(const int *tokens) {
    int t, d, h, k;
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d)
            a_x[t][d] = Wtok[tokens[t]][d] + Wpos[t][d];
    for (t = 0; t < T; ++t) ln_fwd(&a_ln1[t][0], &a_x[t][0], g1, b1, &a_mu1[t], &a_invstd1[t], D);
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float q = 0, kk = 0, v = 0;
            for (h = 0; h < D; ++h) {
                q  += a_ln1[t][h] * Wqkv[h][d];
                kk += a_ln1[t][h] * Wqkv[h][d + D];
                v  += a_ln1[t][h] * Wqkv[h][d + 2 * D];
            }
            a_Q[t][d] = q; a_K[t][d] = kk; a_V[t][d] = v;
        }
    {
        float scale = 1.0f / sqrtf((float)D);
        for (t = 0; t < T; ++t) {
            float maxs = -1e30f;
            for (k = 0; k <= t; ++k) {
                float s = 0; for (d = 0; d < D; ++d) s += a_Q[t][d] * a_K[k][d];
                s *= scale; a_scores[t][k] = s; if (s > maxs) maxs = s;
            }
            float sum = 0;
            for (k = 0; k <= t; ++k) { a_scores[t][k] = expf(a_scores[t][k] - maxs); sum += a_scores[t][k]; }
            float inv = 1.0f / sum;
            for (k = 0; k <= t; ++k) a_scores[t][k] *= inv;
            for (k = t + 1; k < T; ++k) a_scores[t][k] = 0.0f;
            for (d = 0; d < D; ++d) {
                float a = 0; for (k = 0; k <= t; ++k) a += a_scores[t][k] * a_V[k][d];
                a_attn[t][d] = a;
            }
        }
    }
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float s = 0; for (h = 0; h < D; ++h) s += a_attn[t][h] * Wout[h][d];
            a_proj[t][d] = s;
        }
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) a_res1[t][d] = a_x[t][d] + a_proj[t][d];
    for (t = 0; t < T; ++t) ln_fwd(&a_ln2[t][0], &a_res1[t][0], g2, b2, &a_mu2[t], &a_invstd2[t], D);
    for (t = 0; t < T; ++t) {
        for (h = 0; h < FF; ++h) {
            float s = 0; for (d = 0; d < D; ++d) s += a_ln2[t][d] * Wfc1[d][h];
            a_fc1pre[t][h] = s; a_fc1[t][h] = (s > 0) ? s : 0;
        }
        for (d = 0; d < D; ++d) {
            float s = 0; for (h = 0; h < FF; ++h) s += a_fc1[t][h] * Wfc2[h][d];
            a_ffn[t][d] = s;
        }
    }
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) a_res2[t][d] = a_res1[t][d] + a_ffn[t][d];
    for (t = 0; t < T; ++t) ln_fwd(&a_ln3[t][0], &a_res2[t][0], g3, b3, &a_mu3[t], &a_invstd3[t], D);
    for (t = 0; t < T; ++t)
        for (d = 0; d < VOCAB; ++d) {
            float s = 0; for (h = 0; h < D; ++h) s += a_ln3[t][h] * Wunemb[h][d];
            a_logits[t][d] = s;
        }
    float loss = 0.0f; int valid = 0;
    for (t = 0; t < T - 1; ++t) {
        int tgt = tokens[t + 1];
        float maxs = a_logits[t][0];
        for (d = 1; d < VOCAB; ++d) if (a_logits[t][d] > maxs) maxs = a_logits[t][d];
        float sum = 0;
        for (d = 0; d < VOCAB; ++d) { a_probs[t][d] = expf(a_logits[t][d] - maxs); sum += a_probs[t][d]; }
        float inv = 1.0f / sum;
        for (d = 0; d < VOCAB; ++d) a_probs[t][d] *= inv;
        loss += -logf(a_probs[t][tgt] + 1e-30f); ++valid;
    }
    return loss / (float)valid;
}

static void zero_grads(void) {
    init_zero(&gWtok[0][0], VOCAB * D); init_zero(&gWpos[0][0], T * D);
    init_zero(&gWqkv[0][0], D * 3 * D); init_zero(&gWout[0][0], D * D);
    init_zero(&gWfc1[0][0], D * FF);    init_zero(&gWfc2[0][0], FF * D);
    init_zero(&gWunemb[0][0], D * VOCAB);
    init_zero(gg1, D); init_zero(gb1, D); init_zero(gg2, D); init_zero(gb2, D);
    init_zero(gg3, D); init_zero(gb3, D);
}

static void backward(const int *tokens) {
    int t, d, h, k;
    zero_grads();
    float inv_n = 1.0f / (float)(T - 1);
    for (t = 0; t < T - 1; ++t) {
        int tgt = tokens[t + 1];
        for (d = 0; d < VOCAB; ++d)
            a_probs[t][d] = (a_probs[t][d] - ((d == tgt) ? 1.0f : 0.0f)) * inv_n;
    }
    for (d = 0; d < VOCAB; ++d) a_probs[T - 1][d] = 0.0f;
    for (t = 0; t < T; ++t)
        for (h = 0; h < D; ++h) {
            float s = 0; for (d = 0; d < VOCAB; ++d) s += a_probs[t][d] * Wunemb[h][d];
            d_ln3[t][h] = s;
        }
    for (h = 0; h < D; ++h)
        for (d = 0; d < VOCAB; ++d) {
            float s = 0; for (t = 0; t < T; ++t) s += a_ln3[t][h] * a_probs[t][d];
            gWunemb[h][d] += s;
        }
    for (t = 0; t < T; ++t)
        ln_bwd(&d_res2[t][0], &d_ln3[t][0], &a_res2[t][0], a_mu3[t], a_invstd3[t], g3, gg3, gb3, D);
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) { d_ffn[t][d] = d_res2[t][d]; d_res1[t][d] = d_res2[t][d]; }
    for (t = 0; t < T; ++t)
        for (h = 0; h < FF; ++h) {
            float s = 0; for (d = 0; d < D; ++d) s += d_ffn[t][d] * Wfc2[h][d];
            d_fc1[t][h] = (a_fc1pre[t][h] > 0) ? s : 0.0f;
        }
    for (h = 0; h < FF; ++h)
        for (d = 0; d < D; ++d) {
            float s = 0; for (t = 0; t < T; ++t) s += a_fc1[t][h] * d_ffn[t][d];
            gWfc2[h][d] += s;
        }
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float s = 0; for (h = 0; h < FF; ++h) s += d_fc1[t][h] * Wfc1[d][h];
            d_ln2[t][d] = s;
        }
    for (d = 0; d < D; ++d)
        for (h = 0; h < FF; ++h) {
            float s = 0; for (t = 0; t < T; ++t) s += a_ln2[t][d] * d_fc1[t][h];
            gWfc1[d][h] += s;
        }
    { float tmp[D];
      for (t = 0; t < T; ++t) {
        ln_bwd(tmp, &d_ln2[t][0], &a_res1[t][0], a_mu2[t], a_invstd2[t], g2, gg2, gb2, D);
        for (d = 0; d < D; ++d) d_res1[t][d] += tmp[d];
      } }
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) { d_proj[t][d] = d_res1[t][d]; d_x[t][d] = d_res1[t][d]; }
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float s = 0; for (h = 0; h < D; ++h) s += d_proj[t][h] * Wout[d][h];
            d_attn[t][d] = s;
        }
    for (h = 0; h < D; ++h)
        for (d = 0; d < D; ++d) {
            float s = 0; for (t = 0; t < T; ++t) s += a_attn[t][h] * d_proj[t][d];
            gWout[h][d] += s;
        }
    { float scale = 1.0f / sqrtf((float)D);
      for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) { d_Q[t][d] = d_K[t][d] = d_V[t][d] = 0; }
      for (t = 0; t < T; ++t) for (k = 0; k < T; ++k) d_scores[t][k] = 0;
      for (t = 0; t < T; ++t) {
          for (k = 0; k <= t; ++k) {
              float ds = 0; for (d = 0; d < D; ++d) ds += d_attn[t][d] * a_V[k][d];
              d_scores[t][k] = ds;
              float w = a_scores[t][k];
              for (d = 0; d < D; ++d) d_V[k][d] += w * d_attn[t][d];
          }
      }
      for (t = 0; t < T; ++t) {
          float dot = 0; for (k = 0; k <= t; ++k) dot += a_scores[t][k] * d_scores[t][k];
          for (k = 0; k <= t; ++k) d_scores[t][k] = a_scores[t][k] * (d_scores[t][k] - dot);
          for (k = t + 1; k < T; ++k) d_scores[t][k] = 0.0f;
      }
      for (t = 0; t < T; ++t)
          for (k = 0; k <= t; ++k) {
              float ds = d_scores[t][k] * scale;
              for (d = 0; d < D; ++d) { d_Q[t][d] += ds * a_K[k][d]; d_K[k][d] += ds * a_Q[t][d]; }
          } }
    for (t = 0; t < T; ++t)
        for (h = 0; h < D; ++h) {
            float s = 0;
            for (d = 0; d < D; ++d) {
                s += d_Q[t][d] * Wqkv[h][d];
                s += d_K[t][d] * Wqkv[h][d + D];
                s += d_V[t][d] * Wqkv[h][d + 2 * D];
            }
            d_ln1[t][h] = s;
        }
    for (h = 0; h < D; ++h)
        for (d = 0; d < D; ++d) {
            float sq = 0, sk = 0, sv = 0;
            for (t = 0; t < T; ++t) {
                sq += a_ln1[t][h] * d_Q[t][d];
                sk += a_ln1[t][h] * d_K[t][d];
                sv += a_ln1[t][h] * d_V[t][d];
            }
            gWqkv[h][d] += sq; gWqkv[h][d + D] += sk; gWqkv[h][d + 2 * D] += sv;
        }
    { float tmp[D];
      for (t = 0; t < T; ++t) {
        ln_bwd(tmp, &d_ln1[t][0], &a_x[t][0], a_mu1[t], a_invstd1[t], g1, gg1, gb1, D);
        for (d = 0; d < D; ++d) d_x[t][d] += tmp[d];
      } }
    for (t = 0; t < T; ++t) {
        int tk = tokens[t];
        for (d = 0; d < D; ++d) {
            gWtok[tk][d] += d_x[t][d];
            gWpos[t][d]  += d_x[t][d];
        }
    }
}

static void adam_apply(float *p, float *m, float *v, const float *g, int n, float lr, float b1c, float b2c) {
    int i;
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    const float weight_decay = 0.01f;   /* L2 reg to prevent overfit */
    for (i = 0; i < n; ++i) {
        float gi = g[i];
        if (gi >  1.0f) gi =  1.0f;
        if (gi < -1.0f) gi = -1.0f;
        m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        float mh = m[i] / b1c, vh = v[i] / b2c;
        /* AdamW: decoupled weight decay */
        p[i] -= lr * (mh / (sqrtf(vh) + eps) + weight_decay * p[i]);
    }
}

static void adam_step(int step, float lr) {
    float b1c = 1.0f - powf(0.9f, (float)step), b2c = 1.0f - powf(0.999f, (float)step);
    adam_apply(&Wtok[0][0], &mWtok[0][0], &vWtok[0][0], &gWtok[0][0], VOCAB*D, lr, b1c, b2c);
    adam_apply(&Wpos[0][0], &mWpos[0][0], &vWpos[0][0], &gWpos[0][0], T*D, lr, b1c, b2c);
    adam_apply(&Wqkv[0][0], &mWqkv[0][0], &vWqkv[0][0], &gWqkv[0][0], D*3*D, lr, b1c, b2c);
    adam_apply(&Wout[0][0], &mWout[0][0], &vWout[0][0], &gWout[0][0], D*D, lr, b1c, b2c);
    adam_apply(&Wfc1[0][0], &mWfc1[0][0], &vWfc1[0][0], &gWfc1[0][0], D*FF, lr, b1c, b2c);
    adam_apply(&Wfc2[0][0], &mWfc2[0][0], &vWfc2[0][0], &gWfc2[0][0], FF*D, lr, b1c, b2c);
    adam_apply(&Wunemb[0][0], &mWunemb[0][0], &vWunemb[0][0], &gWunemb[0][0], D*VOCAB, lr, b1c, b2c);
    adam_apply(g1, mg1, vg1, gg1, D, lr, b1c, b2c); adam_apply(b1, mb1, vb1, gb1, D, lr, b1c, b2c);
    adam_apply(g2, mg2, vg2, gg2, D, lr, b1c, b2c); adam_apply(b2, mb2, vb2, gb2, D, lr, b1c, b2c);
    adam_apply(g3, mg3, vg3, gg3, D, lr, b1c, b2c); adam_apply(b3, mb3, vb3, gb3, D, lr, b1c, b2c);
}

/* Copy all params to EMI saved_weights_pool[shard_id] for host to read */
static void save_weights_to_emi(int shard_id) {
    volatile float *w = saved_weights_pool + shard_id * N_WEIGHTS;
    int off = 0, i;
    for (i = 0; i < VOCAB*D; ++i)   w[off++] = ((float*)&Wtok[0][0])[i];
    for (i = 0; i < T*D; ++i)       w[off++] = ((float*)&Wpos[0][0])[i];
    for (i = 0; i < D*3*D; ++i)     w[off++] = ((float*)&Wqkv[0][0])[i];
    for (i = 0; i < D*D; ++i)       w[off++] = ((float*)&Wout[0][0])[i];
    for (i = 0; i < D*FF; ++i)      w[off++] = ((float*)&Wfc1[0][0])[i];
    for (i = 0; i < FF*D; ++i)      w[off++] = ((float*)&Wfc2[0][0])[i];
    for (i = 0; i < D*VOCAB; ++i)   w[off++] = ((float*)&Wunemb[0][0])[i];
    for (i = 0; i < D; ++i) w[off++] = g1[i];
    for (i = 0; i < D; ++i) w[off++] = b1[i];
    for (i = 0; i < D; ++i) w[off++] = g2[i];
    for (i = 0; i < D; ++i) w[off++] = b2[i];
    for (i = 0; i < D; ++i) w[off++] = g3[i];
    for (i = 0; i < D; ++i) w[off++] = b3[i];
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();
    unsigned int seed = 0xC0FFEEu ^ (cluster * 4 + core);
    init_weights(seed);

    int shard_id   = cluster * 4 + core;             /* 0..15 — slot for output */
    /* All cores see the FULL 1 MB dataset.  Per-core diversity comes from
     * different seeds (different init + different sampling sequence), not
     * different shards.  Avoids per-core memorization observed at 64 KB. */
    int shard_off  = 0;
    int valid_end  = DATA_SIZE - T - 1;

    int tokens[T]; int step, last_loss_q = 0;
    float last_loss = 0.0f;

    const int N_STEPS = 2000;
    const float LR_INIT = 5e-4f;
    const float LR_MIN  = 1e-5f;

    printf("NMC%d:%d v3 train+save: shard=[%d..%d] VOCAB=%d T=%d D=%d FF=%d steps=%d LR=%g→%g\n",
        cluster, core, shard_off, valid_end, VOCAB, T, D, FF, N_STEPS, LR_INIT, LR_MIN);

    for (step = 1; step <= N_STEPS; ++step) {
        unsigned int r = rand_u32();
        int span = valid_end - shard_off; if (span < 1) span = 1;
        int start = shard_off + (int)((r >> 8) & 0xFFFFFF) % span;
        int i;
        for (i = 0; i < T; ++i) tokens[i] = (int)training_data[start + i] & 0x7F;

        last_loss = forward(tokens);
        backward(tokens);
        /* Cosine LR decay from LR_INIT down to LR_MIN over N_STEPS */
        float frac = (float)step / (float)N_STEPS;
        float cos_v = 0.5f * (1.0f + cosf(3.14159265f * frac));
        float lr_now = LR_MIN + (LR_INIT - LR_MIN) * cos_v;
        adam_step(step, lr_now);

        if (step == 1 || step % 100 == 0 || step == N_STEPS) {
            int q = (int)(last_loss * 1000.0f); last_loss_q = q;
            printf("NMC%d:%d step=%d loss=%f\n", cluster, core, step, last_loss);
        }
    }

    core_loss_out = last_loss;
    save_weights_to_emi(shard_id);
    printf("NMC%d:%d weights saved to EMI slot %d (%d floats)\n",
        cluster, core, shard_id, N_WEIGHTS);
    return last_loss_q;
}
