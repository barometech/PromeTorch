// ============================================================
// dispatcher_float.cpp — Full FLOAT dispatcher for NMC4
// ============================================================
// ALL ops in native IEEE 754 float. No Q16.16.
// Weights preloaded to DDR — only activations transferred per step.
// Ops: matmul, rmsnorm, relu, softmax, elem_add, fused_layer
// ============================================================

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

#define OP_NOP          0
#define OP_MATMUL       1
#define OP_RMSNORM      2
#define OP_SOFTMAX      3
#define OP_RELU         4
#define OP_ELEM_ADD     10
#define OP_CAUSAL_ATTN  20
#define OP_FUSED_LAYER  30
#define OP_EXIT         255
#define STATUS_ADDR     30
#define WATCHDOG_ADDR   31

// ============================================================
// MatMul: C[M,N] = A[M,K] @ B[K,N]
// args: [M, K, N, addr_A, addr_B, addr_C]
// ============================================================
void op_matmul() {
    unsigned int M = mem[1]; unsigned int K = mem[2]; unsigned int N = mem[3];
    float* A = (float*)mem[4]; float* B = (float*)mem[5]; float* C = (float*)mem[6];
    unsigned int ci = 0;
    for (unsigned int i = 0; i < M; i++) {
        unsigned int ar = i * K;
        for (unsigned int j = 0; j < N; j++) {
            float s = 0.0f; unsigned int bc = j;
            for (unsigned int k = 0; k < K; k++) { s += A[ar+k] * B[bc]; bc += N; }
            C[ci++] = s;
        }
    }
}

// ============================================================
// RMSNorm: y[i] = x[i] / rms * gamma[i]
// args: [T, D, addr_x, addr_gamma, addr_y]
// ============================================================
void op_rmsnorm() {
    unsigned int T = mem[1]; unsigned int D = mem[2];
    float* x = (float*)mem[3]; float* g = (float*)mem[4]; float* y = (float*)mem[5];
    for (unsigned int t = 0; t < T; t++) {
        float* xr = x + t*D; float* yr = y + t*D;
        float ss = 0.0f;
        for (unsigned int d = 0; d < D; d++) ss += xr[d] * xr[d];
        float inv = 1.0f;
        // sqrt approximation: Newton's method (2 iterations)
        float rms2 = ss / (float)D + 1e-6f;
        float est = rms2 * 0.5f;
        if (est > 0.0f) {
            // 1/sqrt via Newton: x = x * (1.5 - 0.5*a*x*x)
            float x0 = 1.0f;
            x0 = x0 * (1.5f - 0.5f * rms2 * x0 * x0);
            x0 = x0 * (1.5f - 0.5f * rms2 * x0 * x0);
            x0 = x0 * (1.5f - 0.5f * rms2 * x0 * x0);
            inv = x0;
        }
        for (unsigned int d = 0; d < D; d++) yr[d] = xr[d] * inv * g[d];
    }
}

// ============================================================
// Softmax: y = exp(x - max) / sum(exp(x - max))
// args: [batch, dim, addr_in, addr_out]
// ============================================================
void op_softmax() {
    unsigned int batch = mem[1]; unsigned int dim = mem[2];
    float* in = (float*)mem[3]; float* out = (float*)mem[4];
    for (unsigned int b = 0; b < batch; b++) {
        float* x = in + b*dim; float* y = out + b*dim;
        float mx = x[0];
        for (unsigned int i = 1; i < dim; i++) if (x[i] > mx) mx = x[i];
        float s = 0.0f;
        for (unsigned int i = 0; i < dim; i++) {
            float diff = x[i] - mx;
            // exp approximation: 1 + x + x^2/2 + x^3/6 + x^4/24
            // clamp to avoid overflow
            if (diff < -10.0f) diff = -10.0f;
            float e = 1.0f + diff + diff*diff*0.5f + diff*diff*diff*0.16667f + diff*diff*diff*diff*0.04167f;
            if (e < 0.0f) e = 0.0f;
            y[i] = e; s += e;
        }
        if (s > 0.0f) { float is = 1.0f / s; for (unsigned int i = 0; i < dim; i++) y[i] *= is; }
    }
}

// ============================================================
// ReLU: y = max(0, x)
// args: [count, addr_in, addr_out]
// ============================================================
void op_relu() {
    unsigned int n = mem[1]; float* x = (float*)mem[2]; float* y = (float*)mem[3];
    for (unsigned int i = 0; i < n; i++) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// ============================================================
// Element-wise add: out = a + b
// args: [count, addr_a, addr_b, addr_out]
// ============================================================
void op_elem_add() {
    unsigned int n = mem[1]; float* a = (float*)mem[2]; float* b = (float*)mem[3]; float* o = (float*)mem[4];
    for (unsigned int i = 0; i < n; i++) o[i] = a[i] + b[i];
}

// ============================================================
// Causal self-attention (single-head slice)
// args: [T, HD, addr_Q, addr_K, addr_V, addr_out, addr_scratch]
// Q,K,V = [T, HD], out = [T, HD]
// ============================================================
void op_causal_attn() {
    unsigned int T = mem[1]; unsigned int HD = mem[2];
    float* Q = (float*)mem[3]; float* K = (float*)mem[4];
    float* V = (float*)mem[5]; float* O = (float*)mem[6];
    float* scores = (float*)mem[7]; // scratch [T] for softmax

    float scale = 1.0f;
    // 1/sqrt(HD) approximation
    if (HD == 16) scale = 0.25f;
    else if (HD == 8) scale = 0.354f;
    else if (HD == 32) scale = 0.177f;

    for (unsigned int t = 0; t < T; t++) {
        float* qt = Q + t * HD;
        // Compute scores[t2] = Q[t] . K[t2] * scale, for t2 <= t
        float mx = -1e9f;
        for (unsigned int t2 = 0; t2 <= t; t2++) {
            float* kt = K + t2 * HD;
            float dot = 0.0f;
            for (unsigned int d = 0; d < HD; d++) dot += qt[d] * kt[d];
            dot *= scale;
            scores[t2] = dot;
            if (dot > mx) mx = dot;
        }
        // Softmax
        float s = 0.0f;
        for (unsigned int t2 = 0; t2 <= t; t2++) {
            float diff = scores[t2] - mx;
            if (diff < -10.0f) diff = -10.0f;
            float e = 1.0f + diff + diff*diff*0.5f + diff*diff*diff*0.16667f + diff*diff*diff*diff*0.04167f;
            if (e < 0.0f) e = 0.0f;
            scores[t2] = e; s += e;
        }
        if (s > 0.0f) { float is = 1.0f / s; for (unsigned int t2 = 0; t2 <= t; t2++) scores[t2] *= is; }
        // Weighted sum of V
        float* ot = O + t * HD;
        for (unsigned int d = 0; d < HD; d++) {
            float acc = 0.0f;
            for (unsigned int t2 = 0; t2 <= t; t2++) acc += scores[t2] * V[t2 * HD + d];
            ot[d] = acc;
        }
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    mem[STATUS_ADDR] = 0;
    mem[WATCHDOG_ADDR] = 0;
    unsigned int watchdog = 0;

    while (1) {
        watchdog++;
        mem[WATCHDOG_ADDR] = watchdog;

        unsigned int op = mem[0];
        if (op == OP_NOP) continue;
        if (op == OP_EXIT) { mem[STATUS_ADDR] = 1; mem[0] = OP_NOP; break; }

        mem[STATUS_ADDR] = 0;

        switch (op) {
            case OP_MATMUL:      op_matmul(); break;
            case OP_RMSNORM:     op_rmsnorm(); break;
            case OP_SOFTMAX:     op_softmax(); break;
            case OP_RELU:        op_relu(); break;
            case OP_ELEM_ADD:    op_elem_add(); break;
            case OP_CAUSAL_ATTN: op_causal_attn(); break;
            default: mem[STATUS_ADDR] = 2; mem[0] = OP_NOP; continue;
        }

        mem[STATUS_ADDR] = 1;
        mem[0] = OP_NOP;
    }
    return 0;
}
