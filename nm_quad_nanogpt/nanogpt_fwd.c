/* ============================================================================
 * Minimal 1-layer transformer FORWARD pass for NMC4 (NM Quad).
 * Byte-level (vocab=128 ASCII), T=32, D=32, FFN=64, 1 head.
 *
 * Memory budget per core (NMC4 IM = 512KB):
 *   params:  ~25 KB  (kept in IM)
 *   buffers: ~22 KB
 *   total:   ~47 KB  (room for ~10× this once backward is added)
 *
 * This is FORWARD only — backward + Adam will be added next.
 * ============================================================================ */

#include <stdio.h>
#include <math.h>
#include "nm6408load_nmc.h"

#define VOCAB 128
#define T     32
#define D     32
#define FF    64

/* ---------- model parameters (all FP32) ---------- */
static float token_emb[VOCAB][D];   /* 128 * 32 * 4 =  16 KB */
static float pos_emb  [T][D];       /*  32 * 32 * 4 =   4 KB */
static float W_qkv    [D][3 * D];   /*  32 * 96 * 4 =  12 KB */
static float W_out    [D][D];       /*  32 * 32 * 4 =   4 KB */
static float W_fc1    [D][FF];      /*  32 * 64 * 4 =   8 KB */
static float W_fc2    [FF][D];      /*  64 * 32 * 4 =   8 KB */
static float W_unemb  [D][VOCAB];   /*  32 *128 * 4 =  16 KB */
static float ln1_g[D], ln1_b[D];
static float ln2_g[D], ln2_b[D];
static float ln3_g[D], ln3_b[D];

/* ---------- forward activation buffers ---------- */
static float x       [T][D];
static float ln1_out [T][D];
static float Q[T][D], K[T][D], V[T][D];
static float scores  [T][T];
static float attn_out[T][D];
static float proj_out[T][D];
static float res1    [T][D];
static float ln2_out [T][D];
static float ffn_h   [T][FF];
static float ffn_out [T][D];
static float res2    [T][D];
static float ln3_out [T][D];
static float logits  [T][VOCAB];

/* ---------- simple linear-congruential PRNG (no libc rand_r on NMC4) ---------- */
static unsigned int rng_state;
static float frand(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return ((float)((rng_state >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
}

/* Xavier-ish init: scale by 1/sqrt(fan_in) */
static void init_matrix(float *p, int rows, int cols, float scale) {
    int i;
    int n = rows * cols;
    for (i = 0; i < n; ++i) p[i] = frand() * scale;
}

static void init_vec_const(float *p, int n, float val) {
    int i;
    for (i = 0; i < n; ++i) p[i] = val;
}

static void init_weights(unsigned int seed) {
    rng_state = seed ? seed : 0xC0FFEEu;
    init_matrix(&token_emb[0][0], VOCAB, D,    0.02f);
    init_matrix(&pos_emb  [0][0], T,     D,    0.02f);
    init_matrix(&W_qkv    [0][0], D,     3 * D, 1.0f / 5.65685f);   /* 1/sqrt(D) */
    init_matrix(&W_out    [0][0], D,     D,    1.0f / 5.65685f);
    init_matrix(&W_fc1    [0][0], D,     FF,   1.0f / 5.65685f);
    init_matrix(&W_fc2    [0][0], FF,    D,    1.0f / 8.0f);        /* 1/sqrt(FF) */
    init_matrix(&W_unemb  [0][0], D,     VOCAB, 1.0f / 5.65685f);
    init_vec_const(ln1_g, D, 1.0f); init_vec_const(ln1_b, D, 0.0f);
    init_vec_const(ln2_g, D, 1.0f); init_vec_const(ln2_b, D, 0.0f);
    init_vec_const(ln3_g, D, 1.0f); init_vec_const(ln3_b, D, 0.0f);
}

/* ---------- layernorm: out[i] = (x[i]-mu)/sqrt(var+eps) * g[i] + b[i] ---------- */
static void layernorm(float *out, const float *in, const float *g, const float *b, int n) {
    int i;
    float mu = 0.0f, var = 0.0f;
    for (i = 0; i < n; ++i) mu += in[i];
    mu /= (float)n;
    for (i = 0; i < n; ++i) { float d = in[i] - mu; var += d * d; }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (i = 0; i < n; ++i) out[i] = (in[i] - mu) * inv * g[i] + b[i];
}

/* ---------- forward: returns mean cross-entropy loss ---------- */
static float forward(const int *tokens) {
    int t, d, h, k;

    /* 1. embed: x[t] = tok_emb[tok[t]] + pos_emb[t] */
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d)
            x[t][d] = token_emb[tokens[t]][d] + pos_emb[t][d];

    /* 2. ln1 */
    for (t = 0; t < T; ++t) layernorm(&ln1_out[t][0], &x[t][0], ln1_g, ln1_b, D);

    /* 3. Q,K,V = ln1 @ W_qkv (fused; W_qkv columns: [0..D)=Q, [D..2D)=K, [2D..3D)=V) */
    for (t = 0; t < T; ++t) {
        for (d = 0; d < D; ++d) {
            float q = 0.0f, kk = 0.0f, v = 0.0f;
            for (h = 0; h < D; ++h) {
                q  += ln1_out[t][h] * W_qkv[h][d];
                kk += ln1_out[t][h] * W_qkv[h][d + D];
                v  += ln1_out[t][h] * W_qkv[h][d + 2 * D];
            }
            Q[t][d] = q; K[t][d] = kk; V[t][d] = v;
        }
    }

    /* 4. Causal self-attention */
    {
        float scale = 1.0f / sqrtf((float)D);
        for (t = 0; t < T; ++t) {
            /* compute scores[t][0..t] = Q[t] . K[k] * scale */
            float maxs = -1e30f;
            for (k = 0; k <= t; ++k) {
                float s = 0.0f;
                for (d = 0; d < D; ++d) s += Q[t][d] * K[k][d];
                s *= scale;
                scores[t][k] = s;
                if (s > maxs) maxs = s;
            }
            /* softmax */
            float sum = 0.0f;
            for (k = 0; k <= t; ++k) {
                scores[t][k] = expf(scores[t][k] - maxs);
                sum += scores[t][k];
            }
            float inv = 1.0f / sum;
            for (k = 0; k <= t; ++k) scores[t][k] *= inv;
            for (k = t + 1; k < T; ++k) scores[t][k] = 0.0f;
            /* attn_out[t] = sum_k scores[t][k] * V[k] */
            for (d = 0; d < D; ++d) {
                float a = 0.0f;
                for (k = 0; k <= t; ++k) a += scores[t][k] * V[k][d];
                attn_out[t][d] = a;
            }
        }
    }

    /* 5. proj = attn_out @ W_out */
    for (t = 0; t < T; ++t)
        for (d = 0; d < D; ++d) {
            float s = 0.0f;
            for (h = 0; h < D; ++h) s += attn_out[t][h] * W_out[h][d];
            proj_out[t][d] = s;
        }

    /* 6. residual 1 */
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) res1[t][d] = x[t][d] + proj_out[t][d];

    /* 7. ln2 */
    for (t = 0; t < T; ++t) layernorm(&ln2_out[t][0], &res1[t][0], ln2_g, ln2_b, D);

    /* 8. FFN: relu(ln2 @ W_fc1) @ W_fc2 */
    for (t = 0; t < T; ++t) {
        for (h = 0; h < FF; ++h) {
            float s = 0.0f;
            for (d = 0; d < D; ++d) s += ln2_out[t][d] * W_fc1[d][h];
            ffn_h[t][h] = (s > 0.0f) ? s : 0.0f;
        }
        for (d = 0; d < D; ++d) {
            float s = 0.0f;
            for (h = 0; h < FF; ++h) s += ffn_h[t][h] * W_fc2[h][d];
            ffn_out[t][d] = s;
        }
    }

    /* 9. residual 2 */
    for (t = 0; t < T; ++t) for (d = 0; d < D; ++d) res2[t][d] = res1[t][d] + ffn_out[t][d];

    /* 10. final ln */
    for (t = 0; t < T; ++t) layernorm(&ln3_out[t][0], &res2[t][0], ln3_g, ln3_b, D);

    /* 11. logits = ln3 @ W_unemb */
    for (t = 0; t < T; ++t)
        for (d = 0; d < VOCAB; ++d) {
            float s = 0.0f;
            for (h = 0; h < D; ++h) s += ln3_out[t][h] * W_unemb[h][d];
            logits[t][d] = s;
        }

    /* 12. cross-entropy: predict tokens[t+1] from logits[t] */
    {
        float total = 0.0f;
        int valid = 0;
        for (t = 0; t < T - 1; ++t) {
            int target = tokens[t + 1];
            float maxs = logits[t][0];
            for (d = 1; d < VOCAB; ++d) if (logits[t][d] > maxs) maxs = logits[t][d];
            float sum = 0.0f;
            for (d = 0; d < VOCAB; ++d) sum += expf(logits[t][d] - maxs);
            float lse = maxs + logf(sum);
            total += lse - logits[t][target];
            ++valid;
        }
        return total / (float)valid;
    }
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    init_weights(0xC0FFEE);

    /* Tokens: ASCII bytes of "Hello, world! This is a test." padded. */
    static const char prompt[] = "Hello, world! This is a test.   ";
    int tokens[T];
    int i;
    for (i = 0; i < T; ++i) tokens[i] = (int)prompt[i] & 0x7F;

    /* Expected initial loss for VOCAB=128 random: ln(128) ≈ 4.852 */
    float loss = forward(tokens);
    printf("NMC%d:%d nanogpt-fwd: VOCAB=%d T=%d D=%d FF=%d  loss=%f  (uniform=%f)\n",
        ncl_getClusterID(), ncl_getCoreID(),
        VOCAB, T, D, FF, loss, logf((float)VOCAB));

    /* Return loss * 1000 as exit code so host can see it via -v */
    return (int)(loss * 1000.0f);
}
