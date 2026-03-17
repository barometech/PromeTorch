// ============================================================
// dispatcher_float_vec.cpp — VECTORIZED FLOAT dispatcher for NMC4
// ============================================================
// Same as dispatcher_float.cpp but MATMUL uses nmppmMul_mm_32f
// from nmpp library → runs on NMC4 VECTOR PIPELINE (not scalar RISC)
// Expected speedup: 10-100x for matmul operations.
//
// All other ops remain scalar (relu, softmax, etc.) — they are
// element-wise and fast enough. Matmul is 90% of training time.
// ============================================================

// nmpp float matmul declaration
extern "C" {
    void nmppmMul_mm_32f(float* pSrcMtr1, int nHeight1, int nStride1,
                          float* pSrcMtr2, int nWidth1, int nStride2,
                          float* pDstMtr, int nWidth2, int nStrideDst, int bPlusDst);
}

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

#define OP_NOP          0
#define OP_MATMUL       1
#define OP_MATMUL_AT    2
#define OP_MATMUL_BT    3
#define OP_RELU         4
#define OP_RELU_BWD     5
#define OP_RMSNORM      6
#define OP_SOFTMAX      7
#define OP_ELEM_ADD     10
#define OP_SGD          20
#define OP_CAUSAL_ATTN  21
#define OP_EXIT         255
#define STATUS_ADDR     30
#define WATCHDOG_ADDR   31

// ============================================================
// MatMul: C[M,N] = A[M,K] @ B[K,N] — VECTORIZED via nmpp
// args: [M, K, N, addr_A, addr_B, addr_C]
// ============================================================
void op_matmul() {
    int M = (int)mem[1]; int K = (int)mem[2]; int N = (int)mem[3];
    float* A = (float*)mem[4]; float* B = (float*)mem[5]; float* C = (float*)mem[6];
    // nmppmMul_mm_32f(A, nHeight1=M, nStride1=K, B, nWidth1=K, nStride2=N,
    //                 C, nWidth2=N, nStrideDst=N, bPlusDst=0)
    nmppmMul_mm_32f(A, M, K, B, K, N, C, N, N, 0);
}

// ============================================================
// MatMul A^T: C[K,N] = A^T[K,M] @ B[M,N]
// A is [M,K], transposed. For dW = x^T @ dout
// nmpp doesn't have direct transpose matmul for float,
// so we do scalar (backward is less frequent than forward)
// ============================================================
void op_matmul_at() {
    unsigned int M=mem[1];unsigned int K=mem[2];unsigned int N=mem[3];
    float*A=(float*)mem[4];float*B=(float*)mem[5];float*C=(float*)mem[6];
    unsigned int ci=0;
    for(unsigned int k=0;k<K;k++){
        for(unsigned int n=0;n<N;n++){
            float s=0.0f;
            for(unsigned int i=0;i<M;i++) s+=A[i*K+k]*B[i*N+n];
            C[ci++]=s;
        }
    }
}

// ============================================================
// MatMul B^T: C[M,K] = A[M,N] @ B^T[K,N]
// B is [K,N], transposed. For dx = dout @ W^T
// ============================================================
void op_matmul_bt() {
    unsigned int M=mem[1];unsigned int K=mem[2];unsigned int N=mem[3];
    float*A=(float*)mem[4];float*B=(float*)mem[5];float*C=(float*)mem[6];
    unsigned int ci=0;
    for(unsigned int m=0;m<M;m++){
        for(unsigned int k=0;k<K;k++){
            float s=0.0f;
            for(unsigned int n=0;n<N;n++) s+=A[m*N+n]*B[k*N+n];
            C[ci++]=s;
        }
    }
}

// ============================================================
// RMSNorm: y[i] = x[i] / rms * gamma[i]
// ============================================================
void op_rmsnorm() {
    unsigned int T = mem[1]; unsigned int D = mem[2];
    float* x = (float*)mem[3]; float* g = (float*)mem[4]; float* y = (float*)mem[5];
    for (unsigned int t = 0; t < T; t++) {
        float* xr = x + t*D; float* yr = y + t*D;
        float ss = 0.0f;
        for (unsigned int d = 0; d < D; d++) ss += xr[d] * xr[d];
        float inv = 1.0f;
        float rms2 = ss / (float)D + 1e-6f;
        float est = rms2 * 0.5f;
        if (est > 0.0f) {
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
            if (diff < -10.0f) diff = -10.0f;
            float e = 1.0f + diff + diff*diff*0.5f + diff*diff*diff*0.16667f + diff*diff*diff*diff*0.04167f;
            if (e < 0.0f) e = 0.0f;
            y[i] = e; s += e;
        }
        if (s > 0.0f) { float is = 1.0f / s; for (unsigned int i = 0; i < dim; i++) y[i] *= is; }
    }
}

// ============================================================
// ReLU, ReLU backward, element-wise add, SGD
// ============================================================
void op_relu() {
    unsigned int n = mem[1]; float* x = (float*)mem[2]; float* y = (float*)mem[3];
    for (unsigned int i = 0; i < n; i++) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void op_relu_bwd() {
    unsigned int n=mem[1];float*dy=(float*)mem[2];float*x=(float*)mem[3];float*dx=(float*)mem[4];
    for(unsigned int i=0;i<n;i++) dx[i]=x[i]>0.0f?dy[i]:0.0f;
}

void op_elem_add() {
    unsigned int n=mem[1];float*a=(float*)mem[2];float*b=(float*)mem[3];float*o=(float*)mem[4];
    for(unsigned int i=0;i<n;i++) o[i]=a[i]+b[i];
}

void op_sgd() {
    unsigned int n=mem[1];float*w=(float*)mem[2];float*g=(float*)mem[3];
    unsigned int lr_u=mem[4];float*lr_p=(float*)&lr_u;float lr=*lr_p;
    for(unsigned int i=0;i<n;i++) w[i]-=lr*g[i];
}

// ============================================================
// Causal self-attention (single-head slice)
// ============================================================
void op_causal_attn() {
    unsigned int T = mem[1]; unsigned int HD = mem[2];
    float* Q = (float*)mem[3]; float* K = (float*)mem[4];
    float* V = (float*)mem[5]; float* O = (float*)mem[6];
    float* scores = (float*)mem[7];

    float scale = 1.0f;
    if (HD == 16) scale = 0.25f;
    else if (HD == 8) scale = 0.354f;
    else if (HD == 32) scale = 0.177f;

    for (unsigned int t = 0; t < T; t++) {
        float* qt = Q + t * HD;
        float mx = -1e9f;
        for (unsigned int t2 = 0; t2 <= t; t2++) {
            float* kt = K + t2 * HD;
            float dot = 0.0f;
            for (unsigned int d = 0; d < HD; d++) dot += qt[d] * kt[d];
            dot *= scale;
            scores[t2] = dot;
            if (dot > mx) mx = dot;
        }
        float s = 0.0f;
        for (unsigned int t2 = 0; t2 <= t; t2++) {
            float diff = scores[t2] - mx;
            if (diff < -10.0f) diff = -10.0f;
            float e = 1.0f + diff + diff*diff*0.5f + diff*diff*diff*0.16667f + diff*diff*diff*diff*0.04167f;
            if (e < 0.0f) e = 0.0f;
            scores[t2] = e; s += e;
        }
        if (s > 0.0f) { float is = 1.0f / s; for (unsigned int t2 = 0; t2 <= t; t2++) scores[t2] *= is; }
        float* ot = O + t * HD;
        for (unsigned int d = 0; d < HD; d++) {
            float acc = 0.0f;
            for (unsigned int t2 = 0; t2 <= t; t2++) acc += scores[t2] * V[t2 * HD + d];
            ot[d] = acc;
        }
    }
}

// ============================================================
// Main dispatcher loop
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
            case OP_MATMUL:      op_matmul(); break;      // ← VECTORIZED via nmpp!
            case OP_MATMUL_AT:   op_matmul_at(); break;
            case OP_MATMUL_BT:   op_matmul_bt(); break;
            case OP_RELU:        op_relu(); break;
            case OP_RELU_BWD:    op_relu_bwd(); break;
            case OP_RMSNORM:     op_rmsnorm(); break;
            case OP_SOFTMAX:     op_softmax(); break;
            case OP_ELEM_ADD:    op_elem_add(); break;
            case OP_SGD:         op_sgd(); break;
            case OP_CAUSAL_ATTN: op_causal_attn(); break;
            default: mem[STATUS_ADDR] = 2; mem[0] = OP_NOP; continue;
        }

        mem[STATUS_ADDR] = 1;
        mem[0] = OP_NOP;
    }
    return 0;
}
