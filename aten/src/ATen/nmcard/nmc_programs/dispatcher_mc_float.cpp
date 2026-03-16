// ============================================================
// dispatcher_mc_float.cpp — 16-core FLOAT dispatcher for NMC4
// ============================================================
// Each core polls its own CMD block: DDR_BASE + core_index * CMD_BLOCK_SIZE
// core_index = cluster_id * 4 + core_id (0..15)
// Host sends same op to all 16 cores with different data slices.
//
// Float mode: links with libgcc_float.a for FMul/FAdd.
// No Q16.16, no mymath.h. Native IEEE 754.
// ============================================================

// NMC4 multi-core identification
extern "C" {
    int ncl_getCoreID(void);
    int ncl_getClusterID(void);
}

// Stubs for missing libgcc symbols (required by nm6408load_nmc)
extern "C" unsigned int LShift32(unsigned int a, unsigned int b) {
    unsigned int r = a;
    for (unsigned int i = 0; i < b; i++) r <<= 1;
    return r;
}
extern "C" unsigned int RShift32(unsigned int a, unsigned int b) {
    unsigned int r = a;
    for (unsigned int i = 0; i < b; i++) r >>= 1;
    return r;
}
extern "C" int Mul32(int a, int b) {
    // Simple shift-add multiply
    int result = 0;
    int sign = 1;
    if (a < 0) { a = -a; sign = -sign; }
    if (b < 0) { b = -b; sign = -sign; }
    unsigned int ua = (unsigned int)a;
    unsigned int ub = (unsigned int)b;
    while (ub) {
        if (ub & 1) result += ua;
        ua <<= 1;
        ub >>= 1;
    }
    return sign < 0 ? -result : result;
}

#define DDR_BASE 0x00340000
#define CMD_BLOCK_SIZE 32
#define OP_NOP         0
#define OP_MATMUL      1
#define OP_MATMUL_AT   2    // C = A^T @ B (for dW = x^T @ dout)
#define OP_MATMUL_BT   3    // C = A @ B^T (for dx = dout @ W^T)
#define OP_RELU        4
#define OP_RELU_BWD    5
#define OP_ELEM_ADD    10
#define OP_SGD         20
#define OP_EXIT        255
#define STATUS_ADDR 30
#define WATCHDOG_ADDR 31

// Per-core memory pointer (set in main based on core_index)
volatile unsigned int* mem;

// C[M,N] = A[M,K] @ B[K,N]
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

// C[K,N] = A^T[K,M] @ B[M,N]  (A is [M,K], transposed)
// Used for: dW = x^T @ dout
void op_matmul_at() {
    unsigned int M = mem[1]; unsigned int K = mem[2]; unsigned int N = mem[3];
    float* A = (float*)mem[4]; float* B = (float*)mem[5]; float* C = (float*)mem[6];
    // A^T[k,i] = A[i,k], so C[k,n] = sum_i A[i,k] * B[i,n]
    unsigned int ci = 0;
    for (unsigned int k = 0; k < K; k++) {
        for (unsigned int n = 0; n < N; n++) {
            float s = 0.0f;
            for (unsigned int i = 0; i < M; i++) {
                s += A[i * K + k] * B[i * N + n];
            }
            C[ci++] = s;
        }
    }
}

// C[M,K] = A[M,N] @ B^T[K,N]  (B is [K,N], transposed)
// Used for: dx = dout @ W^T
void op_matmul_bt() {
    unsigned int M = mem[1]; unsigned int K = mem[2]; unsigned int N = mem[3];
    float* A = (float*)mem[4]; float* B = (float*)mem[5]; float* C = (float*)mem[6];
    // B^T[n,k] = B[k,n], so C[m,k] = sum_n A[m,n] * B[k,n]
    unsigned int ci = 0;
    for (unsigned int m = 0; m < M; m++) {
        for (unsigned int k = 0; k < K; k++) {
            float s = 0.0f;
            for (unsigned int n = 0; n < N; n++) {
                s += A[m * N + n] * B[k * N + n];
            }
            C[ci++] = s;
        }
    }
}

// ReLU: y = max(0, x)
void op_relu() {
    unsigned int n = mem[1]; float* x = (float*)mem[2]; float* y = (float*)mem[3];
    for (unsigned int i = 0; i < n; i++) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// ReLU backward: dx = dy * (x > 0)
void op_relu_bwd() {
    unsigned int n = mem[1]; float* dy = (float*)mem[2]; float* x = (float*)mem[3]; float* dx = (float*)mem[4];
    for (unsigned int i = 0; i < n; i++) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}

// Element-wise add: out = a + b
void op_elem_add() {
    unsigned int n = mem[1]; float* a = (float*)mem[2]; float* b = (float*)mem[3]; float* o = (float*)mem[4];
    for (unsigned int i = 0; i < n; i++) o[i] = a[i] + b[i];
}

// SGD update: w -= lr * grad
void op_sgd() {
    unsigned int n = mem[1]; float* w = (float*)mem[2]; float* g = (float*)mem[3];
    float lr_bits; unsigned int lr_u = mem[4];
    // reinterpret uint as float
    float* lr_ptr = (float*)&lr_u;
    float lr = *lr_ptr;
    for (unsigned int i = 0; i < n; i++) w[i] -= lr * g[i];
}

int main() {
    int core_id = ncl_getCoreID();
    int cluster_id = ncl_getClusterID();
    unsigned int core_index = (unsigned int)((cluster_id << 2) + core_id);

    // Point to this core's CMD block
    mem = (volatile unsigned int*)(DDR_BASE + (core_index << 5));

    mem[STATUS_ADDR] = 1;  // ready
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
            case OP_MATMUL:    op_matmul(); break;
            case OP_MATMUL_AT: op_matmul_at(); break;
            case OP_MATMUL_BT: op_matmul_bt(); break;
            case OP_RELU:      op_relu(); break;
            case OP_RELU_BWD:  op_relu_bwd(); break;
            case OP_ELEM_ADD:  op_elem_add(); break;
            case OP_SGD:       op_sgd(); break;
            default:
                mem[STATUS_ADDR] = 2;
                mem[0] = OP_NOP;
                continue;
        }

        mem[STATUS_ADDR] = 1;  // done
        mem[0] = OP_NOP;
    }
    return 0;
}
