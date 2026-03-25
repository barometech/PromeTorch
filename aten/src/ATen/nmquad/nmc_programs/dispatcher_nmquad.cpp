// ============================================================================
// dispatcher_nmquad.cpp — Multi-core DDR-polling dispatcher for NM QUAD
// ============================================================================
// Exact same pattern as NMCard dispatcher_mc.cpp:
// - Each core polls its own cmd block in DDR (no PL_Sync!)
// - Host writes opcode → core executes → sets status=1 → clears opcode
// - Float32 native (NM6408 has FPU, no Q16.16 needed)
//
// Build:
//   nmc-g++ -std=gnu++11 -O2 -o dispatcher_nmquad.abs dispatcher_nmquad.cpp
//     -Wl,--whole-archive -l nm6408load_nmc -Wl,--no-whole-archive
//     -L C:\Module\NM_Quad\lib -T nm6408brd.lds

#include "nm6408load_nmc.h"

// nmpp SIMD matmul (from libnmpp-nm6408.a)
// C[M,N] = A[M,K] @ B[K,N]
// Signature: (A, nHeight1, nStride1, B, nWidth1, nStride2, C, nWidth2, nStrideDst, bPlusDst)
extern "C" {
    void nmppmMul_mm_32f(float* A, int nHeight1, int nStride1,
                         float* B, int nWidth1, int nStride2,
                         float* C, int nWidth2, int nStrideDst, int bPlusDst);
}

#define DDR_BASE      0x00340000
#define CMD_BLOCK_SIZE 32   // 32 words per core

#define OP_NOP           0
#define OP_MATMUL        1
#define OP_ADD           2
#define OP_MUL           3
#define OP_RELU          4
#define OP_SIGMOID       5
#define OP_SOFTMAX       6
#define OP_RMSNORM       7
#define OP_ELEM_SUB      8
#define OP_SCALE         9
#define OP_TRANSPOSE     10  // transpose 2D matrix
#define OP_CAUSAL_SOFTMAX 11 // softmax with causal mask + scale
#define OP_RESHAPE_HEADS 12  // [B*T,D] → [B*H,T,HD] with head interleave
#define OP_UNRESHAPE_HEADS 13 // [B*H,T,HD] → [B*T,D]
#define OP_EMBEDDING     14  // token indices → embedding vectors
#define OP_MATMUL_PARTIAL 22
#define OP_EXIT          255

#define STATUS_ADDR    30
#define WATCHDOG_ADDR  31

volatile unsigned int* mem;

// ============================================================
// Float32 operations (native FPU on NM6408)
// ============================================================

void op_matmul() {
    // args: [M, K, N, addr_A, addr_B, addr_C]
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    float* A = (float*)mem[4];
    float* B = (float*)mem[5];
    float* C = (float*)mem[6];

    // nmpp SIMD matmul: C[M,N] = A[M,K] @ B[K,N]
    // nmppmMul_mm_32f(A, Height=M, StrideA=K, B, Width=K, StrideB=N, C, Width2=N, StrideC=N, bPlus=0)
    nmppmMul_mm_32f(A, M, K, B, K, N, C, N, N, 0);
}

void op_matmul_partial() {
    // Batch-slice matmul with nmpp SIMD
    // args: [M_slice, K, N, addr_A_slice, addr_B, addr_C_slice]
    // Each core gets its own row-slice of A and C, shared B
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    float* A = (float*)mem[4];
    float* B = (float*)mem[5];
    float* C = (float*)mem[6];

    nmppmMul_mm_32f(A, M, K, B, K, N, C, N, N, 0);
}

void op_add() {
    unsigned int count = mem[1];
    float* a = (float*)mem[2];
    float* b = (float*)mem[3];
    float* out = (float*)mem[4];
    for (unsigned int i = 0; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}

void op_mul() {
    unsigned int count = mem[1];
    float* a = (float*)mem[2];
    float* b = (float*)mem[3];
    float* out = (float*)mem[4];
    for (unsigned int i = 0; i < count; i++) {
        out[i] = a[i] * b[i];
    }
}

void op_relu() {
    unsigned int count = mem[1];
    float* x = (float*)mem[2];
    float* y = (float*)mem[3];
    for (unsigned int i = 0; i < count; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void op_scale() {
    unsigned int count = mem[1];
    float* x = (float*)mem[2];
    float* y = (float*)mem[3];
    float alpha = *(float*)&mem[4];
    for (unsigned int i = 0; i < count; i++) {
        y[i] = alpha * x[i];
    }
}

void op_elem_sub() {
    unsigned int count = mem[1];
    float* a = (float*)mem[2];
    float* b = (float*)mem[3];
    float* out = (float*)mem[4];
    for (unsigned int i = 0; i < count; i++) {
        out[i] = a[i] - b[i];
    }
}

void op_rmsnorm() {
    unsigned int batch = mem[1];
    unsigned int hidden = mem[2];
    float* input = (float*)mem[3];
    float* output = (float*)mem[4];
    float* gamma = (float*)mem[5];

    for (unsigned int b = 0; b < batch; b++) {
        float* x = input + b * hidden;
        float* y = output + b * hidden;

        float sum_sq = 0.0f;
        for (unsigned int i = 0; i < hidden; i++) {
            sum_sq += x[i] * x[i];
        }
        float rms = sum_sq / hidden + 1e-5f;
        // Simple inverse sqrt
        float inv_rms = 1.0f;
        // Newton's method: inv_rms = 1/sqrt(rms)
        float guess = 1.0f;
        for (int iter = 0; iter < 5; iter++) {
            guess = guess * (1.5f - 0.5f * rms * guess * guess);
        }
        inv_rms = guess;

        for (unsigned int i = 0; i < hidden; i++) {
            y[i] = x[i] * inv_rms * gamma[i];
        }
    }
}

void op_softmax() {
    unsigned int batch = mem[1];
    unsigned int dim = mem[2];
    float* input = (float*)mem[3];
    float* output = (float*)mem[4];

    for (unsigned int b = 0; b < batch; b++) {
        float* x = input + b * dim;
        float* y = output + b * dim;

        float max_val = x[0];
        for (unsigned int i = 1; i < dim; i++) {
            if (x[i] > max_val) max_val = x[i];
        }

        float sum = 0.0f;
        for (unsigned int i = 0; i < dim; i++) {
            float val = x[i] - max_val;
            // Simple exp approximation
            if (val < -10.0f) { y[i] = 0.0f; continue; }
            float e = 1.0f + val + val*val*0.5f + val*val*val*0.1666667f + val*val*val*val*0.0416667f;
            if (e < 0.0f) e = 0.0f;
            y[i] = e;
            sum += e;
        }

        if (sum > 0.0f) {
            float inv_sum = 1.0f / sum;
            for (unsigned int i = 0; i < dim; i++) {
                y[i] *= inv_sum;
            }
        }
    }
}

void op_transpose() {
    // Batched transpose: batch × [M,N] → batch × [N,M]
    // args: [batch, M, N, addr_in, addr_out]
    unsigned int batch = mem[1];
    unsigned int M = mem[2];
    unsigned int N = mem[3];
    float* in = (float*)mem[4];
    float* out = (float*)mem[5];
    for (unsigned int b = 0; b < batch; b++) {
        float* src = in + b * M * N;
        float* dst = out + b * N * M;
        for (unsigned int i = 0; i < M; i++)
            for (unsigned int j = 0; j < N; j++)
                dst[j * M + i] = src[i * N + j];
    }
}

void op_causal_softmax() {
    // Softmax with causal mask and scale
    // args: [batch, T, addr_in, addr_out, scale_bits]
    unsigned int batch = mem[1];
    unsigned int T = mem[2];
    float* input = (float*)mem[3];
    float* output = (float*)mem[4];
    float scale = *(float*)&mem[5];

    for (unsigned int b = 0; b < batch; b++) {
        for (unsigned int i = 0; i < T; i++) {
            float* row_in = input + b * T * T + i * T;
            float* row_out = output + b * T * T + i * T;

            // Apply scale and find max (causal: only j <= i)
            float mx = -1e9f;
            for (unsigned int j = 0; j <= i; j++) {
                row_in[j] *= scale;
                if (row_in[j] > mx) mx = row_in[j];
            }

            // Exp and sum
            float sm = 0.0f;
            for (unsigned int j = 0; j <= i; j++) {
                float v = row_in[j] - mx;
                float e = 1.0f + v + v*v*0.5f + v*v*v*0.1666667f + v*v*v*v*0.0416667f;
                if (e < 0.0f) e = 0.0f;
                row_out[j] = e;
                sm += e;
            }

            // Normalize
            float inv = (sm > 0.0f) ? 1.0f / sm : 0.0f;
            for (unsigned int j = 0; j <= i; j++) row_out[j] *= inv;
            for (unsigned int j = i + 1; j < T; j++) row_out[j] = 0.0f;
        }
    }
}

void op_reshape_heads() {
    // [B*T, D] → [B*H, T, HD]
    // args: [B, T, H, HD, addr_in, addr_out]
    unsigned int B = mem[1];
    unsigned int T = mem[2];
    unsigned int H = mem[3];
    unsigned int HD = mem[4];
    float* in = (float*)mem[5];
    float* out = (float*)mem[6];
    unsigned int D = H * HD;

    for (unsigned int b = 0; b < B; b++)
        for (unsigned int h = 0; h < H; h++)
            for (unsigned int t = 0; t < T; t++)
                for (unsigned int d = 0; d < HD; d++)
                    out[(b*H+h)*T*HD + t*HD + d] = in[b*T*D + t*D + h*HD + d];
}

void op_unreshape_heads() {
    // [B*H, T, HD] → [B*T, D]
    // args: [B, T, H, HD, addr_in, addr_out]
    unsigned int B = mem[1];
    unsigned int T = mem[2];
    unsigned int H = mem[3];
    unsigned int HD = mem[4];
    float* in = (float*)mem[5];
    float* out = (float*)mem[6];
    unsigned int D = H * HD;

    for (unsigned int b = 0; b < B; b++)
        for (unsigned int h = 0; h < H; h++)
            for (unsigned int t = 0; t < T; t++)
                for (unsigned int d = 0; d < HD; d++)
                    out[b*T*D + t*D + h*HD + d] = in[(b*H+h)*T*HD + t*HD + d];
}

void op_embedding() {
    // Lookup: out[B*T, D] = wte[tokens[b*T+t], :] + wpe[t, :]
    // args: [B, T, D, addr_tokens(uint32), addr_wte, addr_wpe, addr_out]
    unsigned int B = mem[1];
    unsigned int T = mem[2];
    unsigned int D = mem[3];
    unsigned int* tokens = (unsigned int*)mem[4];
    float* wte = (float*)mem[5];
    float* wpe = (float*)mem[6];
    float* out = (float*)mem[7];

    for (unsigned int b = 0; b < B; b++)
        for (unsigned int t = 0; t < T; t++) {
            unsigned int tok = tokens[b * T + t];
            for (unsigned int d = 0; d < D; d++)
                out[(b*T+t)*D + d] = wte[tok*D + d] + wpe[t*D + d];
        }
}

// ============================================================
// FUSED FORWARD: entire transformer forward in ONE call
// ============================================================
// args: [B, T, D, H, FF, V, L,
//        addr_tokens, addr_wte, addr_wpe,
//        addr_layers_start,  // packed: Wq,Wk,Wv,Wo,W1,W2,g per layer
//        addr_lm_head,
//        addr_logits_out,
//        addr_h_out]         // final hidden states for backward
#define OP_FUSED_FORWARD 30

// Scratch area for fused forward (after cmd blocks, before data)
#define SCRATCH_BASE (DDR_BASE + 16 * CMD_BLOCK_SIZE + 0x10000)

static void fused_rmsnorm(float* x, float* g, float* out, int batch, int D) {
    for (int b = 0; b < batch; b++) {
        float ss = 0;
        for (int d = 0; d < D; d++) ss += x[b*D+d] * x[b*D+d];
        float inv = 1.0f;
        float rms = ss / D + 1e-5f;
        float guess = 1.0f;
        for (int i = 0; i < 5; i++) guess = guess * (1.5f - 0.5f * rms * guess * guess);
        inv = guess;
        for (int d = 0; d < D; d++) out[b*D+d] = x[b*D+d] * inv * g[d];
    }
}

static void fused_relu(float* x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

static void fused_add(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

void op_fused_forward() {
    int B = (int)mem[1];
    int T = (int)mem[2];
    int D = (int)mem[3];
    int H = (int)mem[4];
    int FF = (int)mem[5];
    int V = (int)mem[6];
    int L = (int)mem[7];
    unsigned int* tokens = (unsigned int*)mem[8];
    float* wte = (float*)mem[9];
    float* wpe = (float*)mem[10];
    float* layers_base = (float*)mem[11]; // packed weights
    float* lm_head = (float*)mem[12];
    float* logits_out = (float*)mem[13];
    float* h_out = (float*)mem[14];

    int HD = D / H;
    int BT = B * T;
    int BH = B * H;

    // Scratch base passed as last arg
    float* scratch = (float*)mem[15];
    float* h = scratch;
    float* hn = h + BT * D;
    float* Q = hn + BT * D;
    float* K_buf = Q + BT * D;
    float* V_buf = K_buf + BT * D;
    float* Kt = V_buf + BT * D;
    float* scores = Kt + BH * HD * T;
    float* attn_out = scores + BH * T * T;
    float* proj = attn_out + BT * D;
    float* ff1 = proj + BT * D;
    float* ff2 = ff1 + BT * FF;
    // Total scratch: BT*D*7 + BH*HD*T + BH*T*T + BT*FF

    // Embedding
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int tok = tokens[b*T+t];
            for (int d = 0; d < D; d++)
                h[b*T*D+t*D+d] = wte[tok*D+d] + wpe[t*D+d];
        }

    // Layer weights are packed: [Wq(D*D), Wk(D*D), Wv(D*D), Wo(D*D), W1(D*FF), W2(FF*D), g(D)] per layer
    int layer_size = 4*D*D + 2*D*FF + D;

    // Cache areas (passed via h_out extended region)
    // h_cache: h_out + BT*D points to cache area
    // Layout: h_cache[L+1][BT*D], hn_cache[L][BT*D], ff1r_cache[L][BT*FF]
    float* h_cache = h_out;  // h_cache[0..L] starts at h_out
    float* hn_cache = h_cache + (L+1)*BT*D;
    float* ff1r_cache = hn_cache + L*BT*D;

    // Save initial h
    for (int i = 0; i < BT*D; i++) h_cache[i] = h[i];

    for (int li = 0; li < L; li++) {
        float* lw = layers_base + li * layer_size;
        float* Wq = lw;
        float* Wk = lw + D*D;
        float* Wv = lw + 2*D*D;
        float* Wo = lw + 3*D*D;
        float* W1 = lw + 4*D*D;
        float* W2 = lw + 4*D*D + D*FF;
        float* g = lw + 4*D*D + 2*D*FF;

        // RMSNorm
        fused_rmsnorm(h, g, hn, BT, D);

        // Cache normalized h for backward
        for (int i = 0; i < BT*D; i++) hn_cache[li*BT*D + i] = hn[i];

        // QKV matmul (nmpp SIMD!)
        nmppmMul_mm_32f(hn, BT, D, Wq, D, D, Q, D, D, 0);
        nmppmMul_mm_32f(hn, BT, D, Wk, D, D, K_buf, D, D, 0);
        nmppmMul_mm_32f(hn, BT, D, Wv, D, D, V_buf, D, D, 0);

        // Reshape [BT,D] → [BH,T,HD] and transpose K
        for (int b = 0; b < B; b++)
            for (int hh = 0; hh < H; hh++)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < HD; d++) {
                        int src = b*T*D + t*D + hh*HD + d;
                        int bh = b*H + hh;
                        // K transposed: [bh, d, t]
                        Kt[bh*HD*T + d*T + t] = K_buf[src];
                    }

        // Reshape Q and V for batched matmul
        // Q_bh[BH*T, HD]
        float* Q_bh = Q; // reuse buffer, reshape in-place
        float* V_bh = V_buf;
        // Need to reshape Q to [BH, T, HD] layout
        // Q is [BT, D] = [B, T, H, HD]
        // Q_bh should be [BH, T, HD] = [B, H, T, HD]
        // temp buffer for reshaping
        float* Q_tmp = ff2; // reuse ff2 as temp (not needed yet)
        for (int b = 0; b < B; b++)
            for (int hh = 0; hh < H; hh++)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < HD; d++) {
                        Q_tmp[(b*H+hh)*T*HD + t*HD + d] = Q[b*T*D + t*D + hh*HD + d];
                        V_bh[(b*H+hh)*T*HD + t*HD + d] = V_buf[b*T*D + t*D + hh*HD + d];
                    }

        // scores = Q_bh @ Kt: [BH, T, T] — nmpp SIMD batched
        // Treat as [BH*T, HD] @ [BH*HD, T] — but need per-batch
        for (int bh = 0; bh < BH; bh++) {
            nmppmMul_mm_32f(Q_tmp + bh*T*HD, T, HD,
                           Kt + bh*HD*T, HD, T,
                           scores + bh*T*T, T, T, 0);
        }

        // Causal softmax
        float scale = 1.0f;
        // Simple inverse sqrt
        {
            float hd_f = (float)HD;
            float g2 = 1.0f;
            for (int i = 0; i < 5; i++) g2 = g2 * (1.5f - 0.5f * hd_f * g2 * g2);
            scale = g2;
        }

        for (int bh = 0; bh < BH; bh++)
            for (int i = 0; i < T; i++) {
                float* row = scores + bh*T*T + i*T;
                float mx = -1e9f;
                for (int j = 0; j <= i; j++) { row[j] *= scale; if (row[j] > mx) mx = row[j]; }
                float sm = 0;
                for (int j = 0; j <= i; j++) {
                    float v = row[j] - mx;
                    float e = 1.0f + v + v*v*0.5f + v*v*v*0.1666667f + v*v*v*v*0.0416667f;
                    if (e < 0) e = 0;
                    row[j] = e; sm += e;
                }
                float inv = (sm > 0) ? 1.0f / sm : 0;
                for (int j = 0; j <= i; j++) row[j] *= inv;
                for (int j = i+1; j < T; j++) row[j] = 0;
            }

        // attn_out = scores @ V_bh: per head
        for (int bh = 0; bh < BH; bh++) {
            nmppmMul_mm_32f(scores + bh*T*T, T, T,
                           V_bh + bh*T*HD, T, HD,
                           attn_out + bh*T*HD, HD, HD, 0);
        }

        // Unreshape [BH, T, HD] → [BT, D]
        for (int b = 0; b < B; b++)
            for (int hh = 0; hh < H; hh++)
                for (int t = 0; t < T; t++)
                    for (int d = 0; d < HD; d++)
                        proj[b*T*D + t*D + hh*HD + d] = attn_out[(b*H+hh)*T*HD + t*HD + d];

        // proj = attn_reshaped @ Wo
        nmppmMul_mm_32f(proj, BT, D, Wo, D, D, hn, D, D, 0);
        // Use hn as temp for proj result

        // Residual
        fused_add(h, hn, BT*D);

        // FFN: RMSNorm → W1 → ReLU → W2 → residual
        fused_rmsnorm(h, g, hn, BT, D);
        nmppmMul_mm_32f(hn, BT, D, W1, D, FF, ff1, FF, FF, 0);
        fused_relu(ff1, BT*FF);

        // Cache relu output for backward
        for (int i = 0; i < BT*FF; i++) ff1r_cache[li*BT*FF + i] = ff1[i];
        nmppmMul_mm_32f(ff1, BT, FF, W2, FF, D, ff2, D, D, 0);
        fused_add(h, ff2, BT*D);

        // Cache h after this layer
        for (int i = 0; i < BT*D; i++) h_cache[(li+1)*BT*D + i] = h[i];
    }

    // h_out stores final h AND is also h_cache[L]
    for (int i = 0; i < BT*D; i++) h_out[i] = h[i];

    // Note: h_cache, hn_cache, ff1r_cache are filled during forward
    // h_cache[li] = h before layer li (stored in DDR by host)
    // For now h_out = h_cache[L] (final)

    // LM head
    nmppmMul_mm_32f(h, BT, D, lm_head, D, V, logits_out, V, V, 0);
}

// ============================================================
// FUSED BACKWARD + SGD: full gradient through all layers
// ============================================================
// args: [B, T, D, H, FF, V, L,
//        addr_dlogits,     // [BT, V] softmax grad from host
//        addr_tokens,      // [BT] for embedding update
//        addr_wte,         // embedding weights (updated in-place)
//        addr_layers_start,// packed weights (updated in-place)
//        addr_lm_head,     // lm_head (updated in-place)
//        addr_h_cache,     // [L+1, BT, D] cached h per layer from forward
//        addr_hn_cache,    // [L, BT, D] cached normalized h per layer
//        addr_ff1r_cache,  // [L, BT, FF] cached relu(ff1) per layer
//        lr_bits,          // learning rate as float bits
//        addr_scratch]
#define OP_FUSED_BACKWARD 31

static void fused_sgd_update(float* W, float* dW, int n, float lr) {
    for (int i = 0; i < n; i++) {
        float g = dW[i];
        if (g > 1.0f) g = 1.0f;
        if (g < -1.0f) g = -1.0f;
        W[i] -= lr * g;
    }
}

void op_fused_backward() {
    int B = (int)mem[1], T = (int)mem[2], D = (int)mem[3];
    int H = (int)mem[4], FF = (int)mem[5], V = (int)mem[6], L = (int)mem[7];
    float* dlogits = (float*)mem[8];
    unsigned int* tokens = (unsigned int*)mem[9];
    float* wte = (float*)mem[10];
    float* layers_base = (float*)mem[11];
    float* lm_head = (float*)mem[12];
    float* h_cache = (float*)mem[13];     // [L+1][BT*D] — h[0]=input, h[L]=final
    float* hn_cache = (float*)mem[14];    // [L][BT*D]
    float* ff1r_cache = (float*)mem[15];  // [L][BT*FF]
    float lr; { unsigned int b = mem[16]; float* f = (float*)&b; lr = *f; }
    float* scratch = (float*)mem[17];

    int HD = D / H, BT = B * T, BH = B * H;
    int layer_size = 4*D*D + 2*D*FF + D;

    // Scratch layout
    float* dW = scratch;
    float* dx = dW + D * V;       // reuse for various grads
    float* temp1 = dx + BT * D;
    float* temp2 = temp1 + BT * FF;

    // 1. dW_lm = h_final.T @ dlogits  [D, V]
    float* h_final = h_cache + L * BT * D;
    // Transpose h_final → [D, BT]
    for (int i = 0; i < BT; i++)
        for (int d = 0; d < D; d++)
            temp1[d * BT + i] = h_final[i * D + d];

    nmppmMul_mm_32f(temp1, D, BT, dlogits, BT, V, dW, V, V, 0);
    fused_sgd_update(lm_head, dW, D * V, lr);

    // 2. dx = dlogits @ lm_head.T  [BT, D]
    // Transpose lm_head [D,V] → [V,D]
    for (int i = 0; i < D; i++)
        for (int j = 0; j < V; j++)
            temp1[j * D + i] = lm_head[i * V + j];
    nmppmMul_mm_32f(dlogits, BT, V, temp1, V, D, dx, D, D, 0);

    // 3. Backprop through layers (reverse)
    for (int li = L - 1; li >= 0; li--) {
        float* lw = layers_base + li * layer_size;
        float* Wq = lw, *Wk = lw+D*D, *Wv = lw+2*D*D, *Wo = lw+3*D*D;
        float* W1 = lw+4*D*D, *W2 = lw+4*D*D+D*FF;
        float* g = lw + 4*D*D + 2*D*FF;
        float* h_in = h_cache + li * BT * D;
        float* hn = hn_cache + li * BT * D;
        float* ff1r = ff1r_cache + li * BT * FF;

        // --- FFN backward ---
        // dx is gradient flowing in
        // ff2 = relu(hn @ W1) @ W2, residual: h += ff2
        // dff2 = dx (from residual)

        // dW2 = ff1r.T @ dx  [FF, D]
        for (int i = 0; i < BT; i++)
            for (int j = 0; j < FF; j++)
                temp2[j * BT + i] = ff1r[i * FF + j]; // transpose ff1r
        nmppmMul_mm_32f(temp2, FF, BT, dx, BT, D, dW, D, D, 0);
        fused_sgd_update(W2, dW, FF * D, lr);

        // dff1r = dx @ W2.T  [BT, FF]
        for (int i = 0; i < FF; i++)
            for (int j = 0; j < D; j++)
                temp1[j * FF + i] = W2[i * D + j]; // transpose W2
        nmppmMul_mm_32f(dx, BT, D, temp1, D, FF, temp2, FF, FF, 0);

        // ReLU backward
        for (int i = 0; i < BT * FF; i++)
            if (ff1r[i] <= 0) temp2[i] = 0;

        // dW1 = hn.T @ dff1_masked  [D, FF]
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d]; // transpose hn
        nmppmMul_mm_32f(temp1, D, BT, temp2, BT, FF, dW, FF, FF, 0);
        fused_sgd_update(W1, dW, D * FF, lr);

        // dx += dff1_masked @ W1.T  (residual backward through FFN)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < FF; j++)
                temp1[j * D + i] = W1[i * FF + j]; // transpose W1
        float* dffn_back = scratch + D*V + BT*D + BT*FF + BT*FF; // need more scratch
        nmppmMul_mm_32f(temp2, BT, FF, temp1, FF, D, dffn_back, D, D, 0);
        for (int i = 0; i < BT * D; i++) dx[i] += dffn_back[i];

        // --- Attention backward (simplified: update Wo, Wq, Wk, Wv) ---
        // proj = attn_reshape @ Wo, residual: h += proj
        // dWo = hn.T @ dx  [D, D]  (using hn from attention norm)
        nmppmMul_mm_32f(temp1, D, BT, dx, BT, D, dW, D, D, 0); // temp1 still has hn.T
        fused_sgd_update(Wo, dW, D * D, lr);

        // dx += (dx @ Wo.T) for residual backward through attention
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                temp1[j * D + i] = Wo[i * D + j]; // transpose Wo
        nmppmMul_mm_32f(dx, BT, D, temp1, D, D, dffn_back, D, D, 0);
        for (int i = 0; i < BT * D; i++) dx[i] += dffn_back[i];

        // === FULL ATTENTION BACKWARD ===
        // Forward was:
        //   Q = hn @ Wq, K = hn @ Wk, V = hn @ Wv        [BT, D]
        //   Q_bh = reshape(Q, [BH, T, HD])
        //   K_bh = reshape(K, [BH, T, HD])
        //   V_bh = reshape(V, [BH, T, HD])
        //   scores = Q_bh @ K_bh.T / sqrt(HD)             [BH, T, T]
        //   attn = causal_softmax(scores)                  [BH, T, T]
        //   O_bh = attn @ V_bh                            [BH, T, HD]
        //   O = unreshape(O_bh, [BT, D])
        //   proj = O @ Wo                                  [BT, D]
        //   h += proj
        //
        // Backward (dx = gradient from above):
        //   d_proj = dx
        //   dWo = O.T @ d_proj                            [D, D]
        //   d_O = d_proj @ Wo.T                            [BT, D]
        //   d_O_bh = reshape(d_O, [BH, T, HD])
        //   d_attn = d_O_bh @ V_bh.T                      [BH, T, T]
        //   d_V_bh = attn.T @ d_O_bh                      [BH, T, HD]
        //   d_scores = d_attn * attn * (1 - sum reduction) — softmax backward
        //            = attn * (d_attn - sum(d_attn * attn, dim=-1, keepdim))
        //   d_scores /= sqrt(HD)
        //   d_Q_bh = d_scores @ K_bh                      [BH, T, HD]
        //   d_K_bh = d_scores.T @ Q_bh                    [BH, T, HD]
        //   d_Q = unreshape(d_Q_bh)                        [BT, D]
        //   d_K = unreshape(d_K_bh)                        [BT, D]
        //   d_V = unreshape(d_V_bh)                        [BT, D]
        //   dWq = hn.T @ d_Q, dWk = hn.T @ d_K, dWv = hn.T @ d_V
        //   dx += d_Q @ Wq.T + d_K @ Wk.T + d_V @ Wv.T   (residual through QKV)

        // Need scratch for attention backward:
        // We'll reuse scratch areas carefully
        // Q, K, V, scores, attn were computed in forward but not cached
        // Recompute them (or cache in forward — TODO for speed)

        // Recompute Q, K, V from hn and weights
        float* Q_re = dffn_back; // reuse [BT*D]
        float* K_re = Q_re + BT * D;  // need more scratch...
        // Actually we need a lot of scratch. Use areas beyond current scratch.
        float* attn_bk_scratch = scratch + D*V + BT*D + BT*FF + BT*FF + BT*D;
        float* Q_full = attn_bk_scratch;
        float* K_full = Q_full + BT * D;
        float* V_full = K_full + BT * D;
        float* d_O = V_full + BT * D;
        float* scores_re = d_O + BT * D;
        float* attn_re = scores_re + BH * T * T;
        float* d_Q = attn_re + BH * T * T;
        float* d_K = d_Q + BT * D;
        float* d_V = d_K + BT * D;

        // Recompute Q, K, V
        nmppmMul_mm_32f(hn, BT, D, Wq, D, D, Q_full, D, D, 0);
        nmppmMul_mm_32f(hn, BT, D, Wk, D, D, K_full, D, D, 0);
        nmppmMul_mm_32f(hn, BT, D, Wv, D, D, V_full, D, D, 0);

        // d_O = dx @ Wo.T  (dffn_back already computed this above)
        // Actually dffn_back = dx @ Wo.T from the residual backward
        // Copy it to d_O
        for (int i = 0; i < BT * D; i++) d_O[i] = dffn_back[i];

        // Reshape Q, K, V, d_O to [BH, T, HD]
        float* Q_bh = scores_re; // reuse temporarily
        float* K_bh_re = Q_bh + BH * T * HD;
        float* V_bh_re = K_bh_re + BH * T * HD;
        float* d_O_bh = V_bh_re + BH * T * HD;

        // Actually we're running out of named scratch. Simplify: use flat arrays.
        // Q_bh, K_bh, V_bh, d_O_bh are same size as Q_full etc.
        // Reshape in-place by reindexing:

        // Zero d_Q, d_K, d_V before accumulation
        for (int i = 0; i < BT*D; i++) { d_Q[i] = 0; d_K[i] = 0; d_V[i] = 0; }

        // Per-head attention backward
        for (int bh = 0; bh < BH; bh++) {
            int b = bh / H, hh = bh % H;

            // Extract Q_h, K_h, V_h, d_O_h for this head [T, HD]
            float Q_h[64*32], K_h[64*32], V_h[64*32], dO_h[64*32]; // max T=64, HD=32
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++) {
                    Q_h[t*HD+d] = Q_full[b*T*D + t*D + hh*HD + d];
                    K_h[t*HD+d] = K_full[b*T*D + t*D + hh*HD + d];
                    V_h[t*HD+d] = V_full[b*T*D + t*D + hh*HD + d];
                    dO_h[t*HD+d] = d_O[b*T*D + t*D + hh*HD + d];
                }

            // Recompute scores = Q_h @ K_h.T / sqrt(HD)
            float K_t[32*64]; // [HD, T]
            for (int t = 0; t < T; t++)
                for (int d = 0; d < HD; d++)
                    K_t[d*T+t] = K_h[t*HD+d];

            float sc[64*64]; // [T, T]
            nmppmMul_mm_32f(Q_h, T, HD, K_t, HD, T, sc, T, T, 0);

            // Causal softmax
            float scale_val = 1.0f;
            { float hdf = (float)HD; float g2 = 1.0f;
              for (int i = 0; i < 5; i++) g2 = g2*(1.5f-0.5f*hdf*g2*g2);
              scale_val = g2; }

            float attn_h[64*64]; // [T, T]
            for (int i = 0; i < T; i++) {
                float mx = -1e9f;
                for (int j = 0; j <= i; j++) { sc[i*T+j] *= scale_val; if (sc[i*T+j]>mx) mx=sc[i*T+j]; }
                float sm = 0;
                for (int j = 0; j <= i; j++) {
                    float v = sc[i*T+j]-mx;
                    float e = 1.0f+v+v*v*0.5f+v*v*v*0.1666667f+v*v*v*v*0.0416667f;
                    if (e<0) e=0; attn_h[i*T+j]=e; sm+=e;
                }
                float inv = (sm>0)?1.0f/sm:0;
                for (int j=0;j<=i;j++) attn_h[i*T+j]*=inv;
                for (int j=i+1;j<T;j++) attn_h[i*T+j]=0;
            }

            // d_attn = d_O_h @ V_h.T  [T, T]
            float V_t[32*64]; // [HD, T]
            for (int t=0;t<T;t++) for (int d=0;d<HD;d++) V_t[d*T+t]=V_h[t*HD+d];
            float d_attn[64*64];
            nmppmMul_mm_32f(dO_h, T, HD, V_t, HD, T, d_attn, T, T, 0);

            // d_V_h = attn.T @ d_O_h  [T, HD]
            float attn_t[64*64]; // [T, T] transposed
            for (int i=0;i<T;i++) for (int j=0;j<T;j++) attn_t[j*T+i]=attn_h[i*T+j];
            float dV_h[64*32];
            nmppmMul_mm_32f(attn_t, T, T, dO_h, T, HD, dV_h, HD, HD, 0);

            // Softmax backward: d_scores = attn * (d_attn - row_sum(d_attn * attn))
            float d_sc[64*64];
            for (int i=0;i<T;i++) {
                float dot = 0;
                for (int j=0;j<T;j++) dot += d_attn[i*T+j]*attn_h[i*T+j];
                for (int j=0;j<T;j++)
                    d_sc[i*T+j] = attn_h[i*T+j] * (d_attn[i*T+j] - dot);
                // Scale backward
                for (int j=0;j<T;j++) d_sc[i*T+j] *= scale_val;
            }

            // d_Q_h = d_scores @ K_h  [T, HD]
            float dQ_h[64*32];
            nmppmMul_mm_32f(d_sc, T, T, K_h, T, HD, dQ_h, HD, HD, 0);

            // d_K_h = d_scores.T @ Q_h  [T, HD]
            float d_sc_t[64*64];
            for (int i=0;i<T;i++) for (int j=0;j<T;j++) d_sc_t[j*T+i]=d_sc[i*T+j];
            float dK_h[64*32];
            nmppmMul_mm_32f(d_sc_t, T, T, Q_h, T, HD, dK_h, HD, HD, 0);

            // Scatter back to [BT, D]
            for (int t=0;t<T;t++) for (int d=0;d<HD;d++) {
                d_Q[b*T*D+t*D+hh*HD+d] += dQ_h[t*HD+d];
                d_K[b*T*D+t*D+hh*HD+d] += dK_h[t*HD+d];
                d_V[b*T*D+t*D+hh*HD+d] += dV_h[t*HD+d];
            }
        }

        // dWq = hn.T @ d_Q  [D, D]
        // hn.T already in temp1
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                temp1[d * BT + i] = hn[i * D + d];

        nmppmMul_mm_32f(temp1, D, BT, d_Q, BT, D, dW, D, D, 0);
        fused_sgd_update(Wq, dW, D * D, lr);

        // dWk = hn.T @ d_K
        nmppmMul_mm_32f(temp1, D, BT, d_K, BT, D, dW, D, D, 0);
        fused_sgd_update(Wk, dW, D * D, lr);

        // dWv = hn.T @ d_V
        nmppmMul_mm_32f(temp1, D, BT, d_V, BT, D, dW, D, D, 0);
        fused_sgd_update(Wv, dW, D * D, lr);

        // dx += d_Q @ Wq.T + d_K @ Wk.T + d_V @ Wv.T
        // Wq.T
        float* Wt = dffn_back; // reuse
        for (int i=0;i<D;i++) for (int j=0;j<D;j++) Wt[j*D+i]=Wq[i*D+j];
        float* dx_add = Wt + D*D;
        nmppmMul_mm_32f(d_Q, BT, D, Wt, D, D, dx_add, D, D, 0);
        fused_add(dx, dx_add, BT*D);

        for (int i=0;i<D;i++) for (int j=0;j<D;j++) Wt[j*D+i]=Wk[i*D+j];
        nmppmMul_mm_32f(d_K, BT, D, Wt, D, D, dx_add, D, D, 0);
        fused_add(dx, dx_add, BT*D);

        for (int i=0;i<D;i++) for (int j=0;j<D;j++) Wt[j*D+i]=Wv[i*D+j];
        nmppmMul_mm_32f(d_V, BT, D, Wt, D, D, dx_add, D, D, 0);
        fused_add(dx, dx_add, BT*D);
    }

    // 4. Embedding gradient: wte[tok] -= lr * dx[bt]
    for (int bt = 0; bt < BT; bt++) {
        int tok = tokens[bt];
        for (int d = 0; d < D; d++)
            wte[tok * D + d] -= lr * dx[bt * D + d];
    }
}

// ============================================================
// Main — DDR polling loop (same as NMCard dispatcher_mc.cpp)
// ============================================================
int main() {
    // Determine core index
    int core_id = ncl_getCoreID();
    int cluster_id = ncl_getClusterID();
    unsigned int core_index = (unsigned int)((cluster_id << 2) + core_id);

    // Point to this core's cmd block
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
            case OP_MATMUL:         op_matmul(); break;
            case OP_MATMUL_PARTIAL: op_matmul_partial(); break;
            case OP_ADD:            op_add(); break;
            case OP_MUL:            op_mul(); break;
            case OP_RELU:           op_relu(); break;
            case OP_SCALE:          op_scale(); break;
            case OP_ELEM_SUB:       op_elem_sub(); break;
            case OP_RMSNORM:        op_rmsnorm(); break;
            case OP_SOFTMAX:        op_softmax(); break;
            case OP_TRANSPOSE:     op_transpose(); break;
            case OP_CAUSAL_SOFTMAX: op_causal_softmax(); break;
            case OP_RESHAPE_HEADS: op_reshape_heads(); break;
            case OP_UNRESHAPE_HEADS: op_unreshape_heads(); break;
            case OP_EMBEDDING:     op_embedding(); break;
            case OP_FUSED_FORWARD: op_fused_forward(); break;
            case OP_FUSED_BACKWARD: op_fused_backward(); break;
            default:
                mem[STATUS_ADDR] = 2;  // error
                mem[0] = OP_NOP;
                continue;
        }

        mem[STATUS_ADDR] = 1;  // done
        mem[0] = OP_NOP;       // ready for next
    }

    return 0;
}
