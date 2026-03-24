// ============================================================================
// dispatcher_nmquad_v3.cpp — ROW-PARALLEL fused transformer for NM QUAD
// ============================================================================
// Architecture: EACH core runs the FULL fused forward/backward for its own
// slice of the batch. NO inter-core coordination. NO coordinator/worker split.
//
// Host dispatches to ALL 16 cores simultaneously:
//   Core 0: rows 0..B_mine-1
//   Core 1: rows B_mine..2*B_mine-1
//   ...
//   Core 15: rows 15*B_mine..B-1
//
// Each core reads SHARED weights (read-only) and writes to NON-OVERLAPPING
// output regions. Completely independent — no barriers, no sync, no DDR races.
//
// Key difference from v2 (coordinator+workers):
//   v2: Core 0 orchestrates, cores 1-15 are dumb matmul workers → HANGS
//   v3: All cores are equal, each runs full transformer → SIMPLE & FAST
//
// Build:
//   nmc-g++ -std=gnu++11 -O2 -o dispatcher_nmquad_v3.abs dispatcher_nmquad_v3.cpp
//     -Wl,--whole-archive -l nm6408load_nmc -lnmpp-nm6408
//     -Wl,--no-whole-archive -T nm6408brd.lds

#include "nm6408load_nmc.h"

// nmpp SIMD matmul
extern "C" {
    void nmppmMul_mm_32f(float* A, int nHeight1, int nStride1,
                         float* B, int nWidth1, int nStride2,
                         float* C, int nWidth2, int nStrideDst, int bPlusDst);
}

#define DDR_BASE      0x00340000u
#define CMD_BLOCK_SIZE 32   // 32 words per core

#define OP_NOP           0
#define OP_MATMUL        1
#define OP_FUSED_FORWARD_ROWPAR  32  // NEW: row-parallel fused forward
#define OP_FUSED_BACKWARD_ROWPAR 33  // NEW: row-parallel fused backward
#define OP_EXIT          255

#define STATUS_ADDR    30
#define WATCHDOG_ADDR  31

static volatile unsigned int* mem;
static int core_index;

// ============================================================
// Utility functions
// ============================================================

static void fused_rmsnorm(float* x, float* g, float* out, int batch, int D) {
    for (int b = 0; b < batch; b++) {
        float ss = 0;
        for (int d = 0; d < D; d++) ss += x[b*D+d] * x[b*D+d];
        float rms = ss / D + 1e-5f;
        float inv = 1.0f;
        for (int i = 0; i < 5; i++) inv = inv * (1.5f - 0.5f * rms * inv * inv);
        for (int d = 0; d < D; d++) out[b*D+d] = x[b*D+d] * inv * g[d];
    }
}

static void fused_relu(float* x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

static void fused_add(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

static float fast_invsqrt(float x) {
    float g = 1.0f;
    for (int i = 0; i < 5; i++) g = g * (1.5f - 0.5f * x * g * g);
    return g;
}

// ============================================================
// ROW-PARALLEL FUSED FORWARD
// ============================================================
// Each core processes B_mine rows of the batch independently.
// Reads shared weights, writes to its own output slice.
//
// CMD block layout:
//   [1]  B_mine      — number of batch rows this core handles
//   [2]  T           — sequence length
//   [3]  D           — model dimension
//   [4]  H           — number of heads
//   [5]  FF          — FFN hidden dim
//   [6]  V           — vocab size
//   [7]  L           — number of layers
//   [8]  tokens_addr — this core's token slice [B_mine*T]
//   [9]  wte_addr    — shared embedding weights [V*D] (read-only)
//   [10] wpe_addr    — shared position embeddings [T*D] (read-only)
//   [11] layers_addr — shared packed layer weights (read-only)
//   [12] lm_head_addr— shared lm_head [D*V] (read-only)
//   [13] logits_addr — this core's logits output [B_mine*T*V]
//   [14] h_out_addr  — this core's h cache output
//   [15] scratch_addr— this core's scratch area

static void op_fused_forward_rowpar() {
    int Bm = (int)mem[1];   // B_mine: rows for THIS core
    int T  = (int)mem[2];
    int D  = (int)mem[3];
    int H  = (int)mem[4];
    int FF = (int)mem[5];
    int V  = (int)mem[6];
    int L  = (int)mem[7];
    unsigned int* tokens = (unsigned int*)mem[8];
    float* wte         = (float*)mem[9];    // shared, read-only
    float* wpe         = (float*)mem[10];   // shared, read-only
    float* layers_base = (float*)mem[11];   // shared, read-only
    float* lm_head     = (float*)mem[12];   // shared, read-only
    float* logits_out  = (float*)mem[13];   // this core's output
    float* h_out       = (float*)mem[14];   // this core's cache
    float* scratch     = (float*)mem[15];   // this core's scratch

    if (Bm <= 0) return;

    int HD = D / H;
    int BT = Bm * T;    // this core's BT
    int BH = Bm * H;

    // Scratch layout — all private to this core
    float* h       = scratch;
    float* hn      = h + BT * D;
    float* Q       = hn + BT * D;
    float* K_buf   = Q + BT * D;
    float* V_buf   = K_buf + BT * D;
    float* Kt      = V_buf + BT * D;
    float* scores  = Kt + BH * HD * T;
    float* attn_out= scores + BH * T * T;
    float* proj    = attn_out + BT * D;
    float* ff1     = proj + BT * D;
    float* ff2     = ff1 + BT * FF;
    float* Q_tmp   = ff2 + BT * D;
    float* V_bh    = Q_tmp + BH * T * HD;

    // Cache areas in h_out
    float* h_cache   = h_out;
    float* hn_cache  = h_cache + (L+1) * BT * D;
    float* ff1r_cache= hn_cache + L * BT * D;

    // === Embedding ===
    for (int b = 0; b < Bm; b++)
        for (int t = 0; t < T; t++) {
            int tok = tokens[b * T + t];
            for (int d = 0; d < D; d++)
                h[b*T*D + t*D + d] = wte[tok*D + d] + wpe[t*D + d];
        }

    // Save initial h
    for (int i = 0; i < BT*D; i++) h_cache[i] = h[i];

    int layer_size = 4*D*D + 2*D*FF + D;

    for (int li = 0; li < L; li++) {
        float* lw = layers_base + li * layer_size;
        float* Wq = lw;
        float* Wk = lw + D*D;
        float* Wv = lw + 2*D*D;
        float* Wo = lw + 3*D*D;
        float* W1 = lw + 4*D*D;
        float* W2 = lw + 4*D*D + D*FF;
        float* g  = lw + 4*D*D + 2*D*FF;

        // RMSNorm
        fused_rmsnorm(h, g, hn, BT, D);
        for (int i = 0; i < BT*D; i++) hn_cache[li*BT*D + i] = hn[i];

        // QKV projections — nmpp SIMD matmul
        nmppmMul_mm_32f(hn, BT, D, Wq, D, D, Q, D, D, 0);
        nmppmMul_mm_32f(hn, BT, D, Wk, D, D, K_buf, D, D, 0);
        nmppmMul_mm_32f(hn, BT, D, Wv, D, D, V_buf, D, D, 0);

        // Reshape Q→[BH,T,HD], K→Kt[BH,HD,T], V→V_bh[BH,T,HD]
        for (int b = 0; b < Bm; b++)
            for (int hh = 0; hh < H; hh++)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < HD; d++) {
                        int src = b*T*D + t*D + hh*HD + d;
                        int bh  = b*H + hh;
                        Q_tmp[bh*T*HD + t*HD + d] = Q[src];
                        Kt[bh*HD*T + d*T + t]     = K_buf[src];
                        V_bh[bh*T*HD + t*HD + d]  = V_buf[src];
                    }

        // Attention scores: per head Q @ Kt → [BH, T, T]
        for (int bh = 0; bh < BH; bh++)
            nmppmMul_mm_32f(Q_tmp + bh*T*HD, T, HD,
                           Kt + bh*HD*T, HD, T,
                           scores + bh*T*T, T, T, 0);

        // Causal softmax with 1/sqrt(HD) scaling
        float scale = fast_invsqrt((float)HD);

        for (int bh = 0; bh < BH; bh++)
            for (int i = 0; i < T; i++) {
                float* row = scores + bh*T*T + i*T;
                float mx = -1e9f;
                for (int j = 0; j <= i; j++) {
                    row[j] *= scale;
                    if (row[j] > mx) mx = row[j];
                }
                float sm = 0;
                for (int j = 0; j <= i; j++) {
                    float v = row[j] - mx;
                    float e = 1.0f + v + v*v*0.5f + v*v*v*0.1666667f + v*v*v*v*0.0416667f;
                    if (e < 0) e = 0;
                    row[j] = e;
                    sm += e;
                }
                float inv = (sm > 0) ? 1.0f / sm : 0;
                for (int j = 0; j <= i; j++) row[j] *= inv;
                for (int j = i+1; j < T; j++) row[j] = 0;
            }

        // Attention output: scores @ V_bh per head
        for (int bh = 0; bh < BH; bh++)
            nmppmMul_mm_32f(scores + bh*T*T, T, T,
                           V_bh + bh*T*HD, T, HD,
                           attn_out + bh*T*HD, HD, HD, 0);

        // Unreshape [BH, T, HD] → [BT, D]
        for (int b = 0; b < Bm; b++)
            for (int hh = 0; hh < H; hh++)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < HD; d++)
                        proj[b*T*D + t*D + hh*HD + d] = attn_out[(b*H+hh)*T*HD + t*HD + d];

        // Output projection + residual
        nmppmMul_mm_32f(proj, BT, D, Wo, D, D, hn, D, D, 0);
        fused_add(h, hn, BT*D);

        // FFN: RMSNorm → W1 → ReLU → W2 → residual
        fused_rmsnorm(h, g, hn, BT, D);
        nmppmMul_mm_32f(hn, BT, D, W1, D, FF, ff1, FF, FF, 0);
        fused_relu(ff1, BT*FF);
        for (int i = 0; i < BT*FF; i++) ff1r_cache[li*BT*FF + i] = ff1[i];
        nmppmMul_mm_32f(ff1, BT, FF, W2, FF, D, ff2, D, D, 0);
        fused_add(h, ff2, BT*D);

        // Cache h after layer
        for (int i = 0; i < BT*D; i++) h_cache[(li+1)*BT*D + i] = h[i];
    }

    // Final h to output
    for (int i = 0; i < BT*D; i++) h_out[i] = h[i];

    // LM head: h @ lm_head → logits [BT, V]
    nmppmMul_mm_32f(h, BT, D, lm_head, D, V, logits_out, V, V, 0);
}

// ============================================================
// ROW-PARALLEL FUSED BACKWARD + SGD
// ============================================================
// Each core computes gradients for its own batch rows.
// Weight updates: each core accumulates its OWN gradient into a per-core
// gradient buffer. HOST sums gradients across cores and applies update.
//
// ALTERNATIVE (used here): each core applies lr/N_cores scaled update directly.
// This is mathematically equivalent to averaged gradient + full lr.
//
// CMD block layout:
//   [1]  B_mine      — batch rows for this core
//   [2]  T
//   [3]  D
//   [4]  H
//   [5]  FF
//   [6]  V
//   [7]  L
//   [8]  dlogits_addr — this core's dlogits [B_mine*T*V]
//   [9]  tokens_addr  — this core's tokens [B_mine*T]
//   [10] wte_addr     — shared wte (WRITTEN: each core updates its rows atomically)
//   [11] layers_addr  — shared layer weights (WRITTEN: scaled lr)
//   [12] lm_head_addr — shared lm_head (WRITTEN: scaled lr)
//   [13] h_cache_addr — this core's h cache
//   [14] hn_cache_addr— this core's hn cache
//   [15] ff1r_cache   — this core's ff1r cache
//   [16] lr_bits      — learning rate (float as uint32)
//   [17] scratch_addr — this core's scratch
//   [18] n_cores      — total number of cores (for lr scaling)

static void fused_sgd(float* W, float* dW, int n, float lr) {
    for (int i = 0; i < n; i++) {
        float g = dW[i];
        if (g > 1.0f) g = 1.0f;
        if (g < -1.0f) g = -1.0f;
        W[i] -= lr * g;
    }
}

static void op_fused_backward_rowpar() {
    int Bm = (int)mem[1];
    int T  = (int)mem[2];
    int D  = (int)mem[3];
    int H  = (int)mem[4];
    int FF = (int)mem[5];
    int V  = (int)mem[6];
    int L  = (int)mem[7];
    float* dlogits    = (float*)mem[8];
    unsigned int* tokens = (unsigned int*)mem[9];
    float* wte        = (float*)mem[10];
    float* layers_base= (float*)mem[11];
    float* lm_head    = (float*)mem[12];
    float* h_cache    = (float*)mem[13];
    float* hn_cache   = (float*)mem[14];
    float* ff1r_cache = (float*)mem[15];
    float lr;
    { unsigned int b = mem[16]; float* f = (float*)&b; lr = *f; }
    float* scratch    = (float*)mem[17];
    int n_cores       = (int)mem[18];

    if (Bm <= 0) return;

    // Each core processes independent data, so use full lr.
    // Host normalizes dlogits by total batch, so gradients are already averaged.
    float lr_scaled = lr;

    int HD = D / H, BT = Bm * T, BH = Bm * H;
    int lsz = 4*D*D + 2*D*FF + D;

    // Scratch layout
    float* dW    = scratch;
    float* dx    = dW + D * V;  // max(D*V, D*D, D*FF, FF*D)
    float* temp1 = dx + BT * D;
    float* temp2 = temp1 + BT * FF;
    float* temp3 = temp2 + BT * FF;

    // 1. LM head backward: dW_lm = h_final.T @ dlogits
    float* hf = h_cache + L * BT * D;
    for (int i = 0; i < BT; i++)
        for (int d = 0; d < D; d++)
            temp1[d * BT + i] = hf[i * D + d];  // transpose h_final

    nmppmMul_mm_32f(temp1, D, BT, dlogits, BT, V, dW, V, V, 0);
    fused_sgd(lm_head, dW, D * V, lr_scaled);

    // 2. dx = dlogits @ lm_head.T
    for (int i = 0; i < D; i++)
        for (int j = 0; j < V; j++)
            temp1[j * D + i] = lm_head[i * V + j];
    nmppmMul_mm_32f(dlogits, BT, V, temp1, V, D, dx, D, D, 0);

    // 3. Layer backward (reverse order)
    for (int li = L - 1; li >= 0; li--) {
        float* lw = layers_base + li * lsz;
        float* Wq = lw, *Wk = lw+D*D, *Wv = lw+2*D*D, *Wo = lw+3*D*D;
        float* W1 = lw+4*D*D, *W2 = lw+4*D*D+D*FF;
        float* hn  = hn_cache + li * BT * D;
        float* ff1r= ff1r_cache + li * BT * FF;

        // --- FFN backward ---
        // dW2 = ff1r.T @ dx
        for (int i = 0; i < BT; i++)
            for (int j = 0; j < FF; j++)
                temp2[j * BT + i] = ff1r[i * FF + j];
        nmppmMul_mm_32f(temp2, FF, BT, dx, BT, D, dW, D, D, 0);
        fused_sgd(W2, dW, FF * D, lr_scaled);

        // dff1r = dx @ W2.T
        for (int i = 0; i < FF; i++)
            for (int j = 0; j < D; j++)
                temp1[j * FF + i] = W2[i * D + j];
        nmppmMul_mm_32f(dx, BT, D, temp1, D, FF, temp2, FF, FF, 0);

        // ReLU backward
        for (int i = 0; i < BT * FF; i++)
            if (ff1r[i] <= 0) temp2[i] = 0;

        // dW1 = hn.T @ dff1
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d];
        nmppmMul_mm_32f(temp1, D, BT, temp2, BT, FF, dW, FF, FF, 0);
        fused_sgd(W1, dW, D * FF, lr_scaled);

        // dx += dff1 @ W1.T (residual backward)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < FF; j++)
                temp1[j * D + i] = W1[i * FF + j];
        nmppmMul_mm_32f(temp2, BT, FF, temp1, FF, D, temp3, D, D, 0);
        fused_add(dx, temp3, BT * D);

        // --- Attention backward (simplified: Wo + QKV) ---
        // dWo = hn.T @ dx
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d];
        nmppmMul_mm_32f(temp1, D, BT, dx, BT, D, dW, D, D, 0);
        fused_sgd(Wo, dW, D * D, lr_scaled);

        // dx += dx @ Wo.T (residual)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                temp1[j * D + i] = Wo[i * D + j];
        nmppmMul_mm_32f(dx, BT, D, temp1, D, D, temp3, D, D, 0);
        fused_add(dx, temp3, BT * D);

        // === FULL ATTENTION BACKWARD ===
        // Recompute Q, K, V from hn
        float* Q_full = temp3 + BT * D;
        float* K_full = Q_full + BT * D;
        float* V_full = K_full + BT * D;
        float* d_O    = V_full + BT * D;

        nmppmMul_mm_32f(hn, BT, D, Wq, D, D, Q_full, D, D, 0);
        nmppmMul_mm_32f(hn, BT, D, Wk, D, D, K_full, D, D, 0);
        nmppmMul_mm_32f(hn, BT, D, Wv, D, D, V_full, D, D, 0);

        // d_O = dx @ Wo.T (already have temp3 from above — but was overwritten)
        // Recompute Wo.T @ dx
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                temp1[j * D + i] = Wo[i * D + j];
        nmppmMul_mm_32f(dx, BT, D, temp1, D, D, d_O, D, D, 0);

        // d_Q, d_K, d_V accumulators
        float* d_Q = d_O + BT * D;
        float* d_K = d_Q + BT * D;
        float* d_V = d_K + BT * D;
        for (int i = 0; i < BT*D; i++) { d_Q[i] = 0; d_K[i] = 0; d_V[i] = 0; }

        float scale_val = fast_invsqrt((float)HD);

        // Per-head attention backward
        // Use stack arrays — safe for small T (max 64) and HD (max 64)
        for (int bh = 0; bh < BH; bh++) {
            int b = bh / H, hh = bh % H;

            // Extract per-head slices
            float Q_h[64*64], K_h[64*64], V_h[64*64], dO_h[64*64];
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++) {
                    int idx = b*T*D + t*D + hh*HD + d;
                    Q_h[t*HD+d]  = Q_full[idx];
                    K_h[t*HD+d]  = K_full[idx];
                    V_h[t*HD+d]  = V_full[idx];
                    dO_h[t*HD+d] = d_O[idx];
                }

            // Recompute scores = Q @ K.T / sqrt(HD)
            float K_t[64*64];
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++)
                    K_t[d*T+t] = K_h[t*HD+d];

            float sc[64*64];
            nmppmMul_mm_32f(Q_h, T, HD, K_t, HD, T, sc, T, T, 0);

            // Causal softmax
            float attn_h[64*64];
            for (int i = 0; i < T; i++) {
                float mx = -1e9f;
                for (int j = 0; j <= i; j++) {
                    sc[i*T+j] *= scale_val;
                    if (sc[i*T+j] > mx) mx = sc[i*T+j];
                }
                float sm = 0;
                for (int j = 0; j <= i; j++) {
                    float v = sc[i*T+j] - mx;
                    float e = 1.0f + v + v*v*0.5f + v*v*v*0.1666667f + v*v*v*v*0.0416667f;
                    if (e < 0) e = 0;
                    attn_h[i*T+j] = e;
                    sm += e;
                }
                float inv = (sm > 0) ? 1.0f / sm : 0;
                for (int j = 0; j <= i; j++) attn_h[i*T+j] *= inv;
                for (int j = i+1; j < T; j++) attn_h[i*T+j] = 0;
            }

            // d_attn = dO_h @ V.T
            float V_t[64*64];
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++)
                    V_t[d*T+t] = V_h[t*HD+d];
            float d_attn[64*64];
            nmppmMul_mm_32f(dO_h, T, HD, V_t, HD, T, d_attn, T, T, 0);

            // d_V = attn.T @ dO_h
            float attn_t[64*64];
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    attn_t[j*T+i] = attn_h[i*T+j];
            float dV_h[64*64];
            nmppmMul_mm_32f(attn_t, T, T, dO_h, T, HD, dV_h, HD, HD, 0);

            // Softmax backward: d_sc = attn * (d_attn - sum(d_attn * attn, dim=-1))
            float d_sc[64*64];
            for (int i = 0; i < T; i++) {
                float dot = 0;
                for (int j = 0; j < T; j++) dot += d_attn[i*T+j] * attn_h[i*T+j];
                for (int j = 0; j < T; j++)
                    d_sc[i*T+j] = attn_h[i*T+j] * (d_attn[i*T+j] - dot) * scale_val;
            }

            // d_Q = d_sc @ K, d_K = d_sc.T @ Q
            float dQ_h[64*64], dK_h[64*64];
            nmppmMul_mm_32f(d_sc, T, T, K_h, T, HD, dQ_h, HD, HD, 0);

            float d_sc_t[64*64];
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    d_sc_t[j*T+i] = d_sc[i*T+j];
            nmppmMul_mm_32f(d_sc_t, T, T, Q_h, T, HD, dK_h, HD, HD, 0);

            // Scatter back to [BT, D]
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++) {
                    int idx = b*T*D + t*D + hh*HD + d;
                    d_Q[idx] += dQ_h[t*HD+d];
                    d_K[idx] += dK_h[t*HD+d];
                    d_V[idx] += dV_h[t*HD+d];
                }
        }

        // dWq = hn.T @ d_Q
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d];
        nmppmMul_mm_32f(temp1, D, BT, d_Q, BT, D, dW, D, D, 0);
        fused_sgd(Wq, dW, D * D, lr_scaled);

        nmppmMul_mm_32f(temp1, D, BT, d_K, BT, D, dW, D, D, 0);
        fused_sgd(Wk, dW, D * D, lr_scaled);

        nmppmMul_mm_32f(temp1, D, BT, d_V, BT, D, dW, D, D, 0);
        fused_sgd(Wv, dW, D * D, lr_scaled);

        // dx += d_Q @ Wq.T + d_K @ Wk.T + d_V @ Wv.T
        float* Wt = d_V + BT * D;
        float* dx_add = Wt + D * D;

        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                Wt[j*D+i] = Wq[i*D+j];
        nmppmMul_mm_32f(d_Q, BT, D, Wt, D, D, dx_add, D, D, 0);
        fused_add(dx, dx_add, BT * D);

        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                Wt[j*D+i] = Wk[i*D+j];
        nmppmMul_mm_32f(d_K, BT, D, Wt, D, D, dx_add, D, D, 0);
        fused_add(dx, dx_add, BT * D);

        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                Wt[j*D+i] = Wv[i*D+j];
        nmppmMul_mm_32f(d_V, BT, D, Wt, D, D, dx_add, D, D, 0);
        fused_add(dx, dx_add, BT * D);
    }

    // 4. Embedding gradient: wte[tok] -= lr_scaled * dx[bt]
    for (int bt = 0; bt < BT; bt++) {
        int tok = tokens[bt];
        for (int d = 0; d < D; d++)
            wte[tok * D + d] -= lr_scaled * dx[bt * D + d];
    }
}

// ============================================================
// MAIN — each core runs independently
// ============================================================
int main() {
    int core_id = ncl_getCoreID();
    int cluster_id = ncl_getClusterID();
    core_index = (cluster_id << 2) + core_id;

    mem = (volatile unsigned int*)(DDR_BASE + (core_index << 5));

    // Signal ready
    mem[STATUS_ADDR] = 1;
    mem[WATCHDOG_ADDR] = 0;
    mem[0] = OP_NOP;

    unsigned int watchdog = 0;

    while (1) {
        watchdog++;
        mem[WATCHDOG_ADDR] = watchdog;

        unsigned int op = mem[0];
        if (op == OP_NOP) continue;

        if (op == OP_EXIT) {
            mem[STATUS_ADDR] = 1;
            mem[0] = OP_NOP;
            break;
        }

        mem[STATUS_ADDR] = 0;  // busy

        switch (op) {
            case OP_MATMUL: {
                unsigned int M = mem[1], K = mem[2], N = mem[3];
                float* A = (float*)mem[4];
                float* B = (float*)mem[5];
                float* C = (float*)mem[6];
                nmppmMul_mm_32f(A, M, K, B, K, N, C, N, N, 0);
                break;
            }
            case OP_FUSED_FORWARD_ROWPAR:
                op_fused_forward_rowpar();
                break;
            case OP_FUSED_BACKWARD_ROWPAR:
                op_fused_backward_rowpar();
                break;
            default:
                mem[STATUS_ADDR] = 2;  // error: unknown op
                mem[0] = OP_NOP;
                continue;
        }

        mem[STATUS_ADDR] = 1;  // done
        mem[0] = OP_NOP;       // ready for next
    }

    return 0;
}
