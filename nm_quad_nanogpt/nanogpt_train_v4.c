/* ============================================================================
 * nanogpt_train_v4 — 2-LAYER transformer for NMC4 (NM Quad).
 *
 * Same as v3 but adds a second transformer block. Per layer:
 *   - LayerNorm → QKV proj → causal attention → output proj → residual
 *   - LayerNorm → FFN(linear-ReLU-linear) → residual
 * After 2 layers: final LayerNorm → unembed → logits.
 *
 * Settings (byte-level, fits in single NMC4 EMI):
 *   VOCAB=128, T=32, D=32, FF=64, n_layers=2.
 *   Params: 2*8320 + (4096+1024+4096+64) = 25920 floats = 104 KB.
 *
 * Trained with AdamW (weight_decay=0.01) + grad clip + cosine LR decay.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "nm6408load_nmc.h"

#define VOCAB 128
#define T     32
#define D     32
#define FF    64
#define L     2          /* number of transformer layers */
#define EPS   1e-5f
#define DATA_SIZE (4 * 1024 * 1024)

char  training_data[DATA_SIZE];
volatile float core_loss_out;

/* Flattened layout of all params, in this order:
 *   Wtok[VOCAB*D] | Wpos[T*D] |
 *   per layer i in 0..L-1: Wqkv_i | Wout_i | Wfc1_i | Wfc2_i | g1_i b1_i g2_i b2_i |
 *   Wunemb[D*VOCAB] | gF[D] bF[D]
 */
#define PER_LAYER_PARAMS  (D*3*D + D*D + D*FF + FF*D + 4*D)   /* 8320 */
#define N_WEIGHTS         (VOCAB*D + T*D + L*PER_LAYER_PARAMS + D*VOCAB + 2*D)
#define N_CORES_TOTAL     16
volatile float saved_weights_pool[N_CORES_TOTAL * N_WEIGHTS];

/* ---------- model params (per-layer arrays) ---------- */
static float Wtok[VOCAB][D], Wpos[T][D];
static float Wqkv[L][D][3 * D];
static float Wout[L][D][D];
static float Wfc1[L][D][FF];
static float Wfc2[L][FF][D];
static float g1[L][D], b1[L][D];   /* pre-attn LN */
static float g2[L][D], b2[L][D];   /* pre-ffn LN */
static float Wunemb[D][VOCAB];
static float gF[D], bF[D];         /* final LN */

static float gWtok[VOCAB][D], gWpos[T][D];
static float gWqkv[L][D][3*D], gWout[L][D][D];
static float gWfc1[L][D][FF],  gWfc2[L][FF][D];
static float gg1[L][D], gb1[L][D], gg2[L][D], gb2[L][D];
static float gWunemb[D][VOCAB];
static float ggF[D], gbF[D];

static float mWtok[VOCAB][D], vWtok[VOCAB][D];
static float mWpos[T][D],     vWpos[T][D];
static float mWqkv[L][D][3*D], vWqkv[L][D][3*D];
static float mWout[L][D][D],   vWout[L][D][D];
static float mWfc1[L][D][FF],  vWfc1[L][D][FF];
static float mWfc2[L][FF][D],  vWfc2[L][FF][D];
static float mWunemb[D][VOCAB], vWunemb[D][VOCAB];
static float mg1[L][D], vg1[L][D], mb1[L][D], vb1[L][D];
static float mg2[L][D], vg2[L][D], mb2[L][D], vb2[L][D];
static float mgF[D], vgF[D], mbF[D], vbF[D];

/* ---------- forward activations ---------- */
static float a_emb[T][D];                  /* token+pos embedding */
static float a_in[L][T][D];                /* input to each layer (residual stream before this layer) */
static float a_ln1[L][T][D];               /* pre-attn LN output */
static float a_mu1[L][T], a_invstd1[L][T];
static float a_Q[L][T][D], a_K[L][T][D], a_V[L][T][D];
static float a_scores[L][T][T];            /* softmax probs */
static float a_attn[L][T][D];
static float a_proj[L][T][D];
static float a_res1[L][T][D];              /* after attn+residual */
static float a_ln2[L][T][D];
static float a_mu2[L][T], a_invstd2[L][T];
static float a_fc1pre[L][T][FF];
static float a_fc1[L][T][FF];
static float a_ffn[L][T][D];
static float a_out[L][T][D];               /* output of layer (residual stream after) */
static float a_ln3[T][D];                  /* final LN */
static float a_muF[T], a_invstdF[T];
static float a_logits[T][VOCAB];
static float a_probs[T][VOCAB];

/* ---------- backward scratch ---------- */
static float d_ln3[T][D], d_resF[T][D];
static float d_layer_out[L][T][D];   /* gradient flowing into output of layer L */
static float d_ffn[T][D], d_fc1[T][FF];
static float d_ln2_t[T][D];
static float d_proj_t[T][D];
static float d_attn[T][D];
static float d_scores[T][T];
static float d_Q[T][D], d_K[T][D], d_V[T][D];
static float d_ln1_t[T][D];

/* ---------- RNG ---------- */
static unsigned int rng;
static float frand(void) {
    rng = rng * 1103515245u + 12345u;
    return ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
}
static unsigned int rand_u32(void) { rng = rng * 1103515245u + 12345u; return rng; }
static void init_mat(float *p, int n, float s) { int i; for (i = 0; i < n; ++i) p[i] = frand() * s; }
static void init_const(float *p, int n, float v) { int i; for (i = 0; i < n; ++i) p[i] = v; }
static void init_zero(float *p, int n) { int i; for (i = 0; i < n; ++i) p[i] = 0.0f; }

static void init_weights(unsigned int seed) {
    int l;
    rng = seed ? seed : 0xC0FFEE;
    init_mat(&Wtok[0][0], VOCAB * D, 0.02f);
    init_mat(&Wpos[0][0], T * D,     0.02f);
    for (l = 0; l < L; ++l) {
        init_mat(&Wqkv[l][0][0], D * 3 * D, 1.0f / 5.65685f);
        init_mat(&Wout[l][0][0], D * D,     1.0f / 5.65685f);
        init_mat(&Wfc1[l][0][0], D * FF,    1.0f / 5.65685f);
        init_mat(&Wfc2[l][0][0], FF * D,    1.0f / 8.0f);
        init_const(g1[l], D, 1.0f); init_const(b1[l], D, 0.0f);
        init_const(g2[l], D, 1.0f); init_const(b2[l], D, 0.0f);
    }
    init_mat(&Wunemb[0][0], D * VOCAB, 1.0f / 5.65685f);
    init_const(gF, D, 1.0f); init_const(bF, D, 0.0f);

    init_zero(&mWtok[0][0], VOCAB * D);   init_zero(&vWtok[0][0], VOCAB * D);
    init_zero(&mWpos[0][0], T * D);       init_zero(&vWpos[0][0], T * D);
    init_zero(&mWqkv[0][0][0], L * D * 3 * D); init_zero(&vWqkv[0][0][0], L * D * 3 * D);
    init_zero(&mWout[0][0][0], L * D * D);     init_zero(&vWout[0][0][0], L * D * D);
    init_zero(&mWfc1[0][0][0], L * D * FF);    init_zero(&vWfc1[0][0][0], L * D * FF);
    init_zero(&mWfc2[0][0][0], L * FF * D);    init_zero(&vWfc2[0][0][0], L * FF * D);
    init_zero(&mWunemb[0][0], D * VOCAB); init_zero(&vWunemb[0][0], D * VOCAB);
    init_zero(&mg1[0][0], L * D); init_zero(&vg1[0][0], L * D);
    init_zero(&mb1[0][0], L * D); init_zero(&vb1[0][0], L * D);
    init_zero(&mg2[0][0], L * D); init_zero(&vg2[0][0], L * D);
    init_zero(&mb2[0][0], L * D); init_zero(&vb2[0][0], L * D);
    init_zero(mgF, D); init_zero(vgF, D); init_zero(mbF, D); init_zero(vbF, D);
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

/* ---------- per-layer forward block ---------- */
static void layer_fwd(int l, const float in[T][D]) {
    int t, d, h, k;
    /* Save input for residual + ln1 */
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) a_in[l][t][d] = in[t][d];
    /* LN1 */
    for (t = 0; t < T; ++t)
        ln_fwd(&a_ln1[l][t][0], &a_in[l][t][0], g1[l], b1[l], &a_mu1[l][t], &a_invstd1[l][t], D);
    /* QKV */
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float q=0, kk=0, v=0;
            for (h = 0; h < D; ++h) {
                q  += a_ln1[l][t][h] * Wqkv[l][h][d];
                kk += a_ln1[l][t][h] * Wqkv[l][h][d + D];
                v  += a_ln1[l][t][h] * Wqkv[l][h][d + 2 * D];
            }
            a_Q[l][t][d] = q; a_K[l][t][d] = kk; a_V[l][t][d] = v;
        }
    /* Causal attention */
    {
        float scale = 1.0f / sqrtf((float)D);
        for (t = 0; t < T; ++t) {
            float maxs = -1e30f;
            for (k = 0; k <= t; ++k) {
                float s = 0; for (d = 0; d < D; ++d) s += a_Q[l][t][d] * a_K[l][k][d];
                s *= scale; a_scores[l][t][k] = s; if (s > maxs) maxs = s;
            }
            float sum = 0;
            for (k = 0; k <= t; ++k) {
                a_scores[l][t][k] = expf(a_scores[l][t][k] - maxs); sum += a_scores[l][t][k];
            }
            float inv = 1.0f / sum;
            for (k = 0; k <= t; ++k) a_scores[l][t][k] *= inv;
            for (k = t + 1; k < T; ++k) a_scores[l][t][k] = 0.0f;
            for (d = 0; d < D; ++d) {
                float a = 0; for (k = 0; k <= t; ++k) a += a_scores[l][t][k] * a_V[l][k][d];
                a_attn[l][t][d] = a;
            }
        }
    }
    /* Output proj */
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float s = 0; for (h = 0; h < D; ++h) s += a_attn[l][t][h] * Wout[l][h][d];
            a_proj[l][t][d] = s;
        }
    /* res1 = in + proj */
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) a_res1[l][t][d] = a_in[l][t][d] + a_proj[l][t][d];
    /* LN2 */
    for (t = 0; t < T; ++t)
        ln_fwd(&a_ln2[l][t][0], &a_res1[l][t][0], g2[l], b2[l], &a_mu2[l][t], &a_invstd2[l][t], D);
    /* FFN */
    for (t = 0; t < T; ++t) {
        for (h = 0; h < FF; ++h) {
            float s = 0; for (d = 0; d < D; ++d) s += a_ln2[l][t][d] * Wfc1[l][d][h];
            a_fc1pre[l][t][h] = s; a_fc1[l][t][h] = (s > 0) ? s : 0;
        }
        for (d = 0; d < D; ++d) {
            float s = 0; for (h = 0; h < FF; ++h) s += a_fc1[l][t][h] * Wfc2[l][h][d];
            a_ffn[l][t][d] = s;
        }
    }
    /* out = res1 + ffn */
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) a_out[l][t][d] = a_res1[l][t][d] + a_ffn[l][t][d];
}

/* ---------- per-layer backward block; d_out_layer is grad flowing IN, fills d_in_layer */
static void layer_bwd(int l, const float d_out_layer[T][D], float d_in_layer[T][D]) {
    int t, d, h, k;
    /* out = res1 + ffn → d_res1 += d_out, d_ffn = d_out */
    static float d_res1[T][D];
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) {
        d_ffn[t][d] = d_out_layer[t][d];
        d_res1[t][d] = d_out_layer[t][d];
    }
    /* FFN bwd: d_fc1 = d_ffn @ Wfc2^T, with ReLU mask; gWfc2 += fc1^T @ d_ffn */
    for (t = 0; t < T; ++t)
        for (h = 0; h < FF; ++h) {
            float s = 0; for (d = 0; d < D; ++d) s += d_ffn[t][d] * Wfc2[l][h][d];
            d_fc1[t][h] = (a_fc1pre[l][t][h] > 0) ? s : 0.0f;
        }
    for (h = 0; h < FF; ++h)
        for (d = 0; d < D; ++d) {
            float s = 0; for (t = 0; t < T; ++t) s += a_fc1[l][t][h] * d_ffn[t][d];
            gWfc2[l][h][d] += s;
        }
    /* d_ln2 = d_fc1 @ Wfc1^T;  gWfc1 += ln2^T @ d_fc1 */
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float s = 0; for (h = 0; h < FF; ++h) s += d_fc1[t][h] * Wfc1[l][d][h];
            d_ln2_t[t][d] = s;
        }
    for (d = 0; d < D; ++d)
        for (h = 0; h < FF; ++h) {
            float s = 0; for (t = 0; t < T; ++t) s += a_ln2[l][t][d] * d_fc1[t][h];
            gWfc1[l][d][h] += s;
        }
    /* LN2 bwd: → d_res1 += */
    {
        float tmp[D];
        for (t = 0; t < T; ++t) {
            ln_bwd(tmp, &d_ln2_t[t][0], &a_res1[l][t][0], a_mu2[l][t], a_invstd2[l][t],
                   g2[l], gg2[l], gb2[l], D);
            for (d = 0; d < D; ++d) d_res1[t][d] += tmp[d];
        }
    }
    /* res1 = in + proj → d_in += d_res1, d_proj = d_res1 */
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) {
        d_proj_t[t][d] = d_res1[t][d];
        d_in_layer[t][d] = d_res1[t][d];
    }
    /* proj bwd: d_attn = d_proj @ Wout^T;  gWout += attn^T @ d_proj */
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float s = 0; for (h = 0; h < D; ++h) s += d_proj_t[t][h] * Wout[l][d][h];
            d_attn[t][d] = s;
        }
    for (h = 0; h < D; ++h)
        for (d = 0; d < D; ++d) {
            float s = 0; for (t = 0; t < T; ++t) s += a_attn[l][t][h] * d_proj_t[t][d];
            gWout[l][h][d] += s;
        }
    /* Attention bwd */
    {
        float scale = 1.0f / sqrtf((float)D);
        for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) { d_Q[t][d] = d_K[t][d] = d_V[t][d] = 0; }
        for (t = 0; t < T; ++t) for (k = 0; k < T; ++k) d_scores[t][k] = 0;
        for (t = 0; t < T; ++t) {
            for (k = 0; k <= t; ++k) {
                float ds = 0; for (d = 0; d < D; ++d) ds += d_attn[t][d] * a_V[l][k][d];
                d_scores[t][k] = ds;
                float w = a_scores[l][t][k];
                for (d = 0; d < D; ++d) d_V[k][d] += w * d_attn[t][d];
            }
        }
        for (t = 0; t < T; ++t) {
            float dot = 0; for (k = 0; k <= t; ++k) dot += a_scores[l][t][k] * d_scores[t][k];
            for (k = 0; k <= t; ++k)
                d_scores[t][k] = a_scores[l][t][k] * (d_scores[t][k] - dot);
            for (k = t + 1; k < T; ++k) d_scores[t][k] = 0.0f;
        }
        for (t = 0; t < T; ++t)
            for (k = 0; k <= t; ++k) {
                float ds = d_scores[t][k] * scale;
                for (d = 0; d < D; ++d) { d_Q[t][d] += ds * a_K[l][k][d]; d_K[k][d] += ds * a_Q[l][t][d]; }
            }
    }
    /* QKV bwd → d_ln1 */
    for (t = 0; t < T; ++t)
        for (h = 0; h < D; ++h) {
            float s = 0;
            for (d = 0; d < D; ++d) {
                s += d_Q[t][d] * Wqkv[l][h][d];
                s += d_K[t][d] * Wqkv[l][h][d + D];
                s += d_V[t][d] * Wqkv[l][h][d + 2 * D];
            }
            d_ln1_t[t][h] = s;
        }
    for (h = 0; h < D; ++h)
        for (d = 0; d < D; ++d) {
            float sq=0, sk=0, sv=0;
            for (t = 0; t < T; ++t) {
                sq += a_ln1[l][t][h] * d_Q[t][d];
                sk += a_ln1[l][t][h] * d_K[t][d];
                sv += a_ln1[l][t][h] * d_V[t][d];
            }
            gWqkv[l][h][d]         += sq;
            gWqkv[l][h][d + D]     += sk;
            gWqkv[l][h][d + 2 * D] += sv;
        }
    /* LN1 bwd → d_in += */
    {
        float tmp[D];
        for (t = 0; t < T; ++t) {
            ln_bwd(tmp, &d_ln1_t[t][0], &a_in[l][t][0], a_mu1[l][t], a_invstd1[l][t],
                   g1[l], gg1[l], gb1[l], D);
            for (d = 0; d < D; ++d) d_in_layer[t][d] += tmp[d];
        }
    }
}

static float forward(const int *tokens) {
    int t, d, h;
    /* embed */
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d)
            a_emb[t][d] = Wtok[tokens[t]][d] + Wpos[t][d];

    /* layer 0..L-1 */
    {
        float cur[T][D];
        for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) cur[t][d] = a_emb[t][d];
        int l;
        for (l = 0; l < L; ++l) {
            layer_fwd(l, cur);
            for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) cur[t][d] = a_out[l][t][d];
        }
        /* Final LN over cur */
        for (t = 0; t < T; ++t)
            ln_fwd(&a_ln3[t][0], cur[t], gF, bF, &a_muF[t], &a_invstdF[t], D);
    }
    /* Unembed */
    for (t = 0; t < T; ++t)
        for (d = 0; d < VOCAB; ++d) {
            float s = 0; for (h = 0; h < D; ++h) s += a_ln3[t][h] * Wunemb[h][d];
            a_logits[t][d] = s;
        }
    /* Cross-entropy */
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
    init_zero(&gWtok[0][0], VOCAB * D);
    init_zero(&gWpos[0][0], T * D);
    init_zero(&gWqkv[0][0][0], L * D * 3 * D);
    init_zero(&gWout[0][0][0], L * D * D);
    init_zero(&gWfc1[0][0][0], L * D * FF);
    init_zero(&gWfc2[0][0][0], L * FF * D);
    init_zero(&gWunemb[0][0], D * VOCAB);
    init_zero(&gg1[0][0], L * D); init_zero(&gb1[0][0], L * D);
    init_zero(&gg2[0][0], L * D); init_zero(&gb2[0][0], L * D);
    init_zero(ggF, D); init_zero(gbF, D);
}

static void backward(const int *tokens) {
    int t, d, h;
    zero_grads();
    /* d_logits = (probs - 1_target)/N stored in a_probs */
    float inv_n = 1.0f / (float)(T - 1);
    for (t = 0; t < T - 1; ++t) {
        int tgt = tokens[t + 1];
        for (d = 0; d < VOCAB; ++d)
            a_probs[t][d] = (a_probs[t][d] - ((d == tgt) ? 1.0f : 0.0f)) * inv_n;
    }
    for (d = 0; d < VOCAB; ++d) a_probs[T - 1][d] = 0.0f;
    /* d_ln3 = d_logits @ Wunemb^T;  gWunemb += ln3^T @ d_logits */
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
    /* Final LN bwd: d_ln3 → d_resF (which is grad into a_out[L-1]) */
    for (t = 0; t < T; ++t)
        ln_bwd(&d_resF[t][0], &d_ln3[t][0], &a_out[L-1][t][0],
               a_muF[t], a_invstdF[t], gF, ggF, gbF, D);

    /* Backward through layers, last to first */
    {
        float d_layer_in[T][D];
        /* Top layer's d_out = d_resF */
        for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) d_layer_out[L-1][t][d] = d_resF[t][d];
        int l;
        for (l = L - 1; l >= 0; --l) {
            layer_bwd(l, (const float (*)[D])d_layer_out[l], d_layer_in);
            if (l > 0) {
                for (t = 0; t < T; ++t) for (d = 0; d < D; ++d)
                    d_layer_out[l-1][t][d] = d_layer_in[t][d];
            } else {
                /* d_layer_in goes back to embed */
                for (t = 0; t < T; ++t) {
                    int tk = tokens[t];
                    for (d = 0; d < D; ++d) {
                        gWtok[tk][d] += d_layer_in[t][d];
                        gWpos[t][d]  += d_layer_in[t][d];
                    }
                }
            }
        }
    }
}

static void adam_apply(float *p, float *m, float *v, const float *g, int n,
                       float lr, float b1c, float b2c)
{
    int i;
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    const float wd = 0.01f;
    for (i = 0; i < n; ++i) {
        float gi = g[i];
        if (gi >  1.0f) gi =  1.0f;
        if (gi < -1.0f) gi = -1.0f;
        m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        float mh = m[i] / b1c, vh = v[i] / b2c;
        p[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
    }
}

static void adam_step(int step, float lr) {
    float b1c = 1.0f - powf(0.9f,   (float)step);
    float b2c = 1.0f - powf(0.999f, (float)step);
    adam_apply(&Wtok[0][0],   &mWtok[0][0],   &vWtok[0][0],   &gWtok[0][0],   VOCAB*D, lr, b1c, b2c);
    adam_apply(&Wpos[0][0],   &mWpos[0][0],   &vWpos[0][0],   &gWpos[0][0],   T*D,     lr, b1c, b2c);
    adam_apply(&Wqkv[0][0][0], &mWqkv[0][0][0], &vWqkv[0][0][0], &gWqkv[0][0][0], L*D*3*D, lr, b1c, b2c);
    adam_apply(&Wout[0][0][0], &mWout[0][0][0], &vWout[0][0][0], &gWout[0][0][0], L*D*D,   lr, b1c, b2c);
    adam_apply(&Wfc1[0][0][0], &mWfc1[0][0][0], &vWfc1[0][0][0], &gWfc1[0][0][0], L*D*FF,  lr, b1c, b2c);
    adam_apply(&Wfc2[0][0][0], &mWfc2[0][0][0], &vWfc2[0][0][0], &gWfc2[0][0][0], L*FF*D,  lr, b1c, b2c);
    adam_apply(&Wunemb[0][0], &mWunemb[0][0], &vWunemb[0][0], &gWunemb[0][0], D*VOCAB, lr, b1c, b2c);
    adam_apply(&g1[0][0], &mg1[0][0], &vg1[0][0], &gg1[0][0], L*D, lr, b1c, b2c);
    adam_apply(&b1[0][0], &mb1[0][0], &vb1[0][0], &gb1[0][0], L*D, lr, b1c, b2c);
    adam_apply(&g2[0][0], &mg2[0][0], &vg2[0][0], &gg2[0][0], L*D, lr, b1c, b2c);
    adam_apply(&b2[0][0], &mb2[0][0], &vb2[0][0], &gb2[0][0], L*D, lr, b1c, b2c);
    adam_apply(gF, mgF, vgF, ggF, D, lr, b1c, b2c);
    adam_apply(bF, mbF, vbF, gbF, D, lr, b1c, b2c);
}

static void save_weights_to_emi(int shard_id) {
    volatile float *w = saved_weights_pool + shard_id * N_WEIGHTS;
    int off = 0, i;
    for (i = 0; i < VOCAB*D; ++i)    w[off++] = ((float*)&Wtok[0][0])[i];
    for (i = 0; i < T*D; ++i)        w[off++] = ((float*)&Wpos[0][0])[i];
    int l;
    for (l = 0; l < L; ++l) {
        for (i = 0; i < D*3*D; ++i)  w[off++] = ((float*)&Wqkv[l][0][0])[i];
        for (i = 0; i < D*D; ++i)    w[off++] = ((float*)&Wout[l][0][0])[i];
        for (i = 0; i < D*FF; ++i)   w[off++] = ((float*)&Wfc1[l][0][0])[i];
        for (i = 0; i < FF*D; ++i)   w[off++] = ((float*)&Wfc2[l][0][0])[i];
        for (i = 0; i < D; ++i) w[off++] = g1[l][i];
        for (i = 0; i < D; ++i) w[off++] = b1[l][i];
        for (i = 0; i < D; ++i) w[off++] = g2[l][i];
        for (i = 0; i < D; ++i) w[off++] = b2[l][i];
    }
    for (i = 0; i < D*VOCAB; ++i)    w[off++] = ((float*)&Wunemb[0][0])[i];
    for (i = 0; i < D; ++i) w[off++] = gF[i];
    for (i = 0; i < D; ++i) w[off++] = bF[i];
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    int cluster = ncl_getClusterID();
    int core    = ncl_getCoreID();
    unsigned int seed = 0xC0FFEEu ^ (cluster * 4 + core);
    init_weights(seed);

    int shard_id = cluster * 4 + core;
    int shard_off = 0;
    int valid_end = DATA_SIZE - T - 1;

    int tokens[T]; int step, last_loss_q = 0;
    float last_loss = 0.0f;
    float best_loss = 1e30f;   /* track best-seen loss to checkpoint best snapshot */
    const int N_STEPS = 2000;
    const float LR_INIT = 5e-4f, LR_MIN = 1e-5f;   /* same as v4 best run */

    printf("NMC%d:%d v4 (%d layers) train+save: VOCAB=%d T=%d D=%d FF=%d steps=%d LR=%g→%g\n",
        cluster, core, L, VOCAB, T, D, FF, N_STEPS, LR_INIT, LR_MIN);

    for (step = 1; step <= N_STEPS; ++step) {
        unsigned int r = rand_u32();
        int span = valid_end - shard_off; if (span < 1) span = 1;
        int start = shard_off + (int)((r >> 8) & 0xFFFFFF) % span;
        int i;
        for (i = 0; i < T; ++i) tokens[i] = (int)training_data[start + i] & 0x7F;

        last_loss = forward(tokens);
        backward(tokens);
        float frac = (float)step / (float)N_STEPS;
        float cos_v = 0.5f * (1.0f + cosf(3.14159265f * frac));
        float lr_now = LR_MIN + (LR_INIT - LR_MIN) * cos_v;
        adam_step(step, lr_now);

        if (step == 1 || step % 100 == 0 || step == N_STEPS) {
            int q = (int)(last_loss * 1000.0f); last_loss_q = q;
            printf("NMC%d:%d step=%d loss=%f\n", cluster, core, step, last_loss);
        }
        /* Save best snapshot in HEALTHY regime only (1.5..4.0).
         * Below 1.5 = overfit basin (degenerate solution). Above 4.0 = barely
         * trained / NaN'd. */
        if (step > 100 && last_loss < best_loss && last_loss > 1.5f && last_loss < 4.0f) {
            best_loss = last_loss;
            save_weights_to_emi(shard_id);
        }
    }
    core_loss_out = best_loss < 1e29f ? best_loss : last_loss;
    printf("NMC%d:%d best_loss=%f (final %f) (%d floats)\n",
        cluster, core, best_loss, last_loss, N_WEIGHTS);
    return (int)(core_loss_out * 1000.0f);
}
