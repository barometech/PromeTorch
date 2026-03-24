// ============================================================================
// dispatcher_nmquad_v3_scalar.cpp — ROW-PARALLEL fused transformer for NM QUAD
// ============================================================================
// Scalar matmul version (no nmpp dependency). Each core runs FULL fused
// forward/backward for its own batch rows. NO inter-core coordination.
//
// Build:
//   nmc-g++ -std=c++11 -mnmc4 -O2 -o dispatcher_nmquad_v3.abs dispatcher_nmquad_v3_scalar.cpp
//     -I/usr/local/rc_module/board-nm_quad/include
//     -Wl,--whole-archive -L/usr/local/rc_module/board-nm_quad/lib -lnm6408load_nmc
//     -Wl,--no-whole-archive -T nm6408brd.lds

#include "nm6408load_nmc.h"

#define DDR_BASE      0x00340000u
#define CMD_BLOCK_SIZE 32

#define OP_NOP           0
#define OP_MATMUL        1
#define OP_FUSED_FORWARD_ROWPAR  32
#define OP_FUSED_BACKWARD_ROWPAR 33
#define OP_EXIT          255

#define STATUS_ADDR    30
#define WATCHDOG_ADDR  31

static volatile unsigned int* mem;

// ============================================================
// Scalar matmul: C[M,N] = A[M,K] @ B[K,N]
// ============================================================
static void sgemm(float* A, int M, int K, float* B, int N, float* C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

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
// CMD: [B_mine, T, D, H, FF, V, L, tokens, wte, wpe, layers, lm_head, logits_out, h_out, scratch]
static void op_fused_forward_rowpar() {
    int Bm = (int)mem[1];
    int T  = (int)mem[2];
    int D  = (int)mem[3];
    int H  = (int)mem[4];
    int FF = (int)mem[5];
    int V  = (int)mem[6];
    int L  = (int)mem[7];
    unsigned int* tokens = (unsigned int*)mem[8];
    float* wte         = (float*)mem[9];
    float* wpe         = (float*)mem[10];
    float* layers_base = (float*)mem[11];
    float* lm_head     = (float*)mem[12];
    float* logits_out  = (float*)mem[13];
    float* h_out       = (float*)mem[14];
    float* scratch     = (float*)mem[15];

    if (Bm <= 0) return;

    int HD = D / H;
    int BT = Bm * T;
    int BH = Bm * H;

    // Scratch layout
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

    // Embedding
    for (int b = 0; b < Bm; b++)
        for (int t = 0; t < T; t++) {
            int tok = tokens[b * T + t];
            for (int d = 0; d < D; d++)
                h[b*T*D + t*D + d] = wte[tok*D + d] + wpe[t*D + d];
        }

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

        // QKV projections
        sgemm(hn, BT, D, Wq, D, Q);
        sgemm(hn, BT, D, Wk, D, K_buf);
        sgemm(hn, BT, D, Wv, D, V_buf);

        // Reshape Q, K->Kt, V
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

        // Attention scores per head
        for (int bh = 0; bh < BH; bh++)
            sgemm(Q_tmp + bh*T*HD, T, HD, Kt + bh*HD*T, T, scores + bh*T*T);

        // Causal softmax
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

        // Attn output per head
        for (int bh = 0; bh < BH; bh++)
            sgemm(scores + bh*T*T, T, T, V_bh + bh*T*HD, HD, attn_out + bh*T*HD);

        // Unreshape [BH,T,HD] -> [BT,D]
        for (int b = 0; b < Bm; b++)
            for (int hh = 0; hh < H; hh++)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < HD; d++)
                        proj[b*T*D + t*D + hh*HD + d] = attn_out[(b*H+hh)*T*HD + t*HD + d];

        // Output projection + residual
        sgemm(proj, BT, D, Wo, D, hn);
        fused_add(h, hn, BT*D);

        // FFN: RMSNorm -> W1 -> ReLU -> W2 -> residual
        fused_rmsnorm(h, g, hn, BT, D);
        sgemm(hn, BT, D, W1, FF, ff1);
        fused_relu(ff1, BT*FF);
        for (int i = 0; i < BT*FF; i++) ff1r_cache[li*BT*FF + i] = ff1[i];
        sgemm(ff1, BT, FF, W2, D, ff2);
        fused_add(h, ff2, BT*D);

        // Cache h after layer
        for (int i = 0; i < BT*D; i++) h_cache[(li+1)*BT*D + i] = h[i];
    }

    // Final h
    for (int i = 0; i < BT*D; i++) h_out[i] = h[i];

    // LM head
    sgemm(h, BT, D, lm_head, V, logits_out);
}

// ============================================================
// ROW-PARALLEL FUSED BACKWARD + SGD
// ============================================================
// CMD: [B_mine, T, D, H, FF, V, L, dlogits, tokens, wte, layers, lm_head,
//       h_cache, hn_cache, ff1r_cache, lr_bits, scratch, n_cores]
static void fused_sgd(float* W, float* dW, int n, float lr) {
    for (int i = 0; i < n; i++) {
        float g = dW[i];
        if (g > 1.0f) g = 1.0f;
        if (g < -1.0f) g = -1.0f;
        W[i] -= lr * g;
    }
}

static void op_fused_backward_rowpar() {
    int Bm = (int)mem[1], T = (int)mem[2], D = (int)mem[3];
    int H = (int)mem[4], FF = (int)mem[5], V = (int)mem[6], L = (int)mem[7];
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
    float lr_s = lr;  // each core has independent data, host normalizes dlogits

    int HD = D / H, BT = Bm * T, BH = Bm * H;
    int lsz = 4*D*D + 2*D*FF + D;

    // Scratch layout
    // Total: D*V + BT*D + 2*BT*FF + 8*BT*D (temp3 + 7 attn backward buffers)
    float* dW    = scratch;
    float* dx    = dW + D * V;
    float* temp1 = dx + BT * D;
    float* temp2 = temp1 + BT * FF;
    float* temp3 = temp2 + BT * FF;

    // LM head backward: dW_lm = h_final.T @ dlogits
    float* hf = h_cache + L * BT * D;
    for (int i = 0; i < BT; i++)
        for (int d = 0; d < D; d++)
            temp1[d * BT + i] = hf[i * D + d];
    sgemm(temp1, D, BT, dlogits, V, dW);
    fused_sgd(lm_head, dW, D * V, lr_s);

    // dx = dlogits @ lm_head.T
    for (int i = 0; i < D; i++)
        for (int j = 0; j < V; j++)
            temp1[j * D + i] = lm_head[i * V + j];
    sgemm(dlogits, BT, V, temp1, D, dx);

    // Layer backward (reverse)
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
        sgemm(temp2, FF, BT, dx, D, dW);
        fused_sgd(W2, dW, FF * D, lr_s);

        // dff1 = dx @ W2.T
        for (int i = 0; i < FF; i++)
            for (int j = 0; j < D; j++)
                temp1[j * FF + i] = W2[i * D + j];
        sgemm(dx, BT, D, temp1, FF, temp2);

        // ReLU backward
        for (int i = 0; i < BT * FF; i++)
            if (ff1r[i] <= 0) temp2[i] = 0;

        // dW1 = hn.T @ dff1
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d];
        sgemm(temp1, D, BT, temp2, FF, dW);
        fused_sgd(W1, dW, D * FF, lr_s);

        // dx += dff1 @ W1.T
        for (int i = 0; i < D; i++)
            for (int j = 0; j < FF; j++)
                temp1[j * D + i] = W1[i * FF + j];
        sgemm(temp2, BT, FF, temp1, D, temp3);
        fused_add(dx, temp3, BT * D);

        // --- Attention backward (FULL per-head computation) ---
        // IMPORTANT: compute d_O BEFORE modifying dx with Wo residual
        // d_O = dx (the incoming gradient is d_O for the attention output proj)

        // Stack arrays for per-head work (max T=64, HD=64)
        float Q_h[64*64], K_h[64*64], V_h[64*64];
        float scores_h[64*64], dO_h[64*64];
        float dAttn[64*64], dScores[64*64];
        float dQ_h[64*64], dK_h[64*64], dV_h[64*64];
        float Kt_h[64*64];

        // Recompute Q, K, V from hn and weights into temp3 area
        // We need Q[BT,D], K[BT,D], V[BT,D] — use scratch after temp3
        float* Q_full = temp3;                    // [BT, D]
        float* K_full = Q_full + BT * D;          // [BT, D]
        float* V_full = K_full + BT * D;          // [BT, D]
        float* dO_full = V_full + BT * D;         // [BT, D] — copy of dx before Wo residual
        float* dQ_full = dO_full + BT * D;        // [BT, D]
        float* dK_full = dQ_full + BT * D;        // [BT, D]
        float* dV_full = dK_full + BT * D;        // [BT, D]

        sgemm(hn, BT, D, Wq, D, Q_full);
        sgemm(hn, BT, D, Wk, D, K_full);
        sgemm(hn, BT, D, Wv, D, V_full);

        // d_proj = dx @ Wo.T  (gradient through output projection)
        // But first, dWo = proj.T @ dx, where proj was the pre-Wo input
        // We don't have proj cached, but proj = attn_concat @ Wo produced hn_add
        // Actually: the attention block is h += proj @ Wo, where proj = concat(heads)
        // So dx flows to both residual AND through Wo back to proj.
        // d_proj = dx @ Wo.T, dWo = proj.T @ dx
        // We need proj = recomputed attn output. But we can compute dWo differently:
        // The forward was: attn_concat -> Wo projection -> add to residual
        // We need to recompute attn_concat to get dWo.

        // Save dx as d_O (gradient into the Wo projection output)
        for (int i = 0; i < BT * D; i++) dO_full[i] = dx[i];

        // d_proj = dx @ Wo.T  — gradient flowing back through Wo to attn_concat
        float WoT[64*64]; // max D=64 => D*D = 4096, fits in 64*64
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                WoT[j * D + i] = Wo[i * D + j];
        // temp3 is no longer safe to use (Q_full etc are there), use stack for d_proj
        // Actually let's reorganize: d_proj goes into dQ_full temporarily
        float* d_proj = dQ_full; // will be overwritten later with actual dQ
        sgemm(dx, BT, D, WoT, D, d_proj); // d_proj[BT, D]

        // Now update Wo: dWo = recomputed_proj.T @ dx
        // We need to recompute proj (attn concat output) — do it per head below
        // For now, we need the full attention output. Recompute it:
        float* attn_concat = dK_full; // temporary reuse, will be overwritten
        // Zero attn_concat
        for (int i = 0; i < BT * D; i++) attn_concat[i] = 0;

        float scale = fast_invsqrt((float)HD);

        // Initialize dQ, dK, dV to zero
        // (We'll accumulate per-head results into these)
        // But first let's use them after computing attn_concat for dWo
        // So we do TWO passes: first recompute attn for dWo, then do full backward

        // === Pass 1: Recompute attention output for dWo ===
        for (int bh = 0; bh < BH; bh++) {
            int b = bh / H, hh = bh % H;
            // Extract Q_h, K_h, V_h for this head
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++) {
                    int src = b*T*D + t*D + hh*HD + d;
                    Q_h[t*HD + d] = Q_full[src];
                    K_h[t*HD + d] = K_full[src];
                    V_h[t*HD + d] = V_full[src];
                }
            // K_h transposed
            for (int d = 0; d < HD; d++)
                for (int t = 0; t < T; t++)
                    Kt_h[d*T + t] = K_h[t*HD + d];
            // scores = Q @ K.T
            sgemm(Q_h, T, HD, Kt_h, T, scores_h);
            // Causal softmax
            for (int i = 0; i < T; i++) {
                float* row = scores_h + i*T;
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
            // attn_out = softmax @ V
            sgemm(scores_h, T, T, V_h, HD, dO_h); // reuse dO_h as attn_out_h
            // Scatter back to attn_concat
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++)
                    attn_concat[b*T*D + t*D + hh*HD + d] = dO_h[t*HD + d];
        }

        // dWo = attn_concat.T @ dx
        // temp1 = attn_concat.T [D, BT]
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = attn_concat[i * D + d];
        sgemm(temp1, D, BT, dx, D, dW);
        fused_sgd(Wo, dW, D * D, lr_s);

        // === Pass 2: Full attention backward per head ===
        // d_proj was computed above (dx @ Wo.T)
        // Now zero out dQ_full, dK_full, dV_full for accumulation
        // d_proj is in a temp location, copy it out first
        // Actually d_proj aliases dQ_full which we're about to zero. Save it.
        // Use attn_concat (dK_full) area is also about to be zeroed.
        // Let's copy d_proj to a safe place — reuse temp1 temporarily
        // Actually temp1 is [max(BT*FF, D*BT)] — big enough for BT*D
        for (int i = 0; i < BT * D; i++) temp1[i] = d_proj[i];
        // Now d_proj points to temp1
        // Zero dQ, dK, dV
        for (int i = 0; i < BT * D; i++) { dQ_full[i] = 0; dK_full[i] = 0; dV_full[i] = 0; }

        for (int bh = 0; bh < BH; bh++) {
            int b = bh / H, hh = bh % H;
            // Extract Q_h, K_h, V_h, dO_h for this head
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++) {
                    int src = b*T*D + t*D + hh*HD + d;
                    Q_h[t*HD + d] = Q_full[src];
                    K_h[t*HD + d] = K_full[src];
                    V_h[t*HD + d] = V_full[src];
                    dO_h[t*HD + d] = temp1[src]; // d_proj per head
                }
            // Recompute K.T
            for (int d = 0; d < HD; d++)
                for (int t = 0; t < T; t++)
                    Kt_h[d*T + t] = K_h[t*HD + d];
            // Recompute scores = Q @ K.T, then causal softmax
            sgemm(Q_h, T, HD, Kt_h, T, scores_h);
            for (int i = 0; i < T; i++) {
                float* row = scores_h + i*T;
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
            // scores_h now contains attn weights (softmax output) [T, T]

            // dAttn = dO_h @ V_h.T  [T, T]
            // V_h.T [HD, T]
            float VhT[64*64];
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++)
                    VhT[d*T + t] = V_h[t*HD + d];
            sgemm(dO_h, T, HD, VhT, T, dAttn); // [T, T]

            // dV_h = attn.T @ dO_h  [T, HD]
            // attn.T [T, T]
            float attnT[64*64];
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    attnT[j*T + i] = scores_h[i*T + j];
            sgemm(attnT, T, T, dO_h, HD, dV_h); // [T, HD]

            // Softmax backward: dScores = attn * (dAttn - sum(dAttn * attn, dim=-1))
            for (int i = 0; i < T; i++) {
                float dot = 0;
                for (int j = 0; j < T; j++)
                    dot += dAttn[i*T + j] * scores_h[i*T + j];
                for (int j = 0; j < T; j++)
                    dScores[i*T + j] = scores_h[i*T + j] * (dAttn[i*T + j] - dot);
            }

            // Apply scale (scores were scaled before softmax, so dScores needs scale too)
            for (int i = 0; i < T * T; i++) dScores[i] *= scale;

            // dQ_h = dScores @ K_h  [T, HD]
            sgemm(dScores, T, T, K_h, HD, dQ_h); // [T, HD]

            // dK_h = dScores.T @ Q_h  [T, HD]
            float dScoresT[64*64];
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    dScoresT[j*T + i] = dScores[i*T + j];
            sgemm(dScoresT, T, T, Q_h, HD, dK_h); // [T, HD]

            // Scatter dQ_h, dK_h, dV_h back to [BT, D]
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++) {
                    int dst = b*T*D + t*D + hh*HD + d;
                    dQ_full[dst] += dQ_h[t*HD + d];
                    dK_full[dst] += dK_h[t*HD + d];
                    dV_full[dst] += dV_h[t*HD + d];
                }
        }

        // Update Wq: dWq = hn.T @ dQ_full
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d]; // hn.T
        sgemm(temp1, D, BT, dQ_full, D, dW);
        fused_sgd(Wq, dW, D * D, lr_s);

        // Update Wk: dWk = hn.T @ dK_full
        sgemm(temp1, D, BT, dK_full, D, dW);
        fused_sgd(Wk, dW, D * D, lr_s);

        // Update Wv: dWv = hn.T @ dV_full
        sgemm(temp1, D, BT, dV_full, D, dW);
        fused_sgd(Wv, dW, D * D, lr_s);

        // Backprop through QKV projections to dx:
        // dx_attn = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
        // Plus the residual: dx already has the FFN gradient, add Wo residual + QKV
        // First: dx += d_proj @ Wo.T was already handled? No — we need the FULL chain:
        // The forward residual is: h += attn_out @ Wo
        // So dx passes through to both residual (already in dx) and through Wo -> attn -> QKV -> hn
        // The QKV backward gives us d_hn_attn = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
        // But hn was produced by RMSNorm(h_prev), so this goes back to dx via RMSNorm backward
        // For simplicity (matching the v3 nmpp version), we add the QKV contribution to dx directly
        // (RMSNorm backward is approximated as pass-through, same as the FFN path)

        // dx += dQ @ Wq.T
        float WqT[64*64];
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                WqT[j * D + i] = Wq[i * D + j];
        sgemm(dQ_full, BT, D, WqT, D, temp3);
        fused_add(dx, temp3, BT * D);

        // dx += dK @ Wk.T
        float WkT[64*64];
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                WkT[j * D + i] = Wk[i * D + j];
        sgemm(dK_full, BT, D, WkT, D, temp3);
        fused_add(dx, temp3, BT * D);

        // dx += dV @ Wv.T
        float WvT[64*64];
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                WvT[j * D + i] = Wv[i * D + j];
        sgemm(dV_full, BT, D, WvT, D, temp3);
        fused_add(dx, temp3, BT * D);
    }

    // Embedding gradient
    for (int bt = 0; bt < BT; bt++) {
        int tok = tokens[bt];
        for (int d = 0; d < D; d++)
            wte[tok * D + d] -= lr_s * dx[bt * D + d];
    }
}

// ============================================================
// MAIN
// ============================================================
int main() {
    int core_id = ncl_getCoreID();
    int cluster_id = ncl_getClusterID();
    int core_index = (cluster_id << 2) + core_id;

    mem = (volatile unsigned int*)(DDR_BASE + (core_index << 5));
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

        mem[STATUS_ADDR] = 0;

        switch (op) {
            case OP_MATMUL: {
                unsigned int M = mem[1], K = mem[2], N = mem[3];
                sgemm((float*)mem[4], M, K, (float*)mem[5], N, (float*)mem[6]);
                break;
            }
            case OP_FUSED_FORWARD_ROWPAR:
                op_fused_forward_rowpar();
                break;
            case OP_FUSED_BACKWARD_ROWPAR:
                op_fused_backward_rowpar();
                break;
            default:
                mem[STATUS_ADDR] = 2;
                mem[0] = OP_NOP;
                continue;
        }

        mem[STATUS_ADDR] = 1;
        mem[0] = OP_NOP;
    }
    return 0;
}
