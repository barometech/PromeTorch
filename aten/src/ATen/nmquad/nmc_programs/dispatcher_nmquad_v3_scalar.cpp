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
    float lr_s = lr / (float)n_cores;

    int HD = D / H, BT = Bm * T, BH = Bm * H;
    int lsz = 4*D*D + 2*D*FF + D;

    // Scratch layout
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

        // --- Attention backward (simplified: Wo + QKV) ---
        // dWo = hn.T @ dx
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d];
        sgemm(temp1, D, BT, dx, D, dW);
        fused_sgd(Wo, dW, D * D, lr_s);

        // dx += dx @ Wo.T
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                temp1[j * D + i] = Wo[i * D + j];
        sgemm(dx, BT, D, temp1, D, temp3);
        fused_add(dx, temp3, BT * D);

        // QKV updates (simplified)
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d];
        sgemm(temp1, D, BT, temp3, D, dW);
        fused_sgd(Wq, dW, D * D, lr_s);
        sgemm(temp1, D, BT, dx, D, dW);
        fused_sgd(Wk, dW, D * D, lr_s);
        fused_sgd(Wv, dW, D * D, lr_s);
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
