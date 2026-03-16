// ============================================================
// dispatcher_float.cpp — FLOAT mode dispatcher for NMC4
// ============================================================
// Uses NMC4 float VPU — no Q16.16, no mymath.h.
// Native IEEE 754 float arithmetic.
// Links with MullMatrix_f.asm for vectorized float matmul.
// ============================================================

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

#define OP_NOP     0
#define OP_MATMUL  1
#define OP_EXIT    255
#define STATUS_ADDR 30
#define WATCHDOG_ADDR 31

// Pure C float matmul — NMC4 float VPU auto-vectorizes this
void op_matmul() {
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    float* A = (float*)mem[4];
    float* B = (float*)mem[5];
    float* C = (float*)mem[6];

    // C[i,j] = sum_k A[i,k] * B[k,j]
    unsigned int c_idx = 0;
    for (unsigned int i = 0; i < M; i++) {
        unsigned int a_row = i * K;
        for (unsigned int j = 0; j < N; j++) {
            float sum = 0.0f;
            unsigned int b_col = j;
            for (unsigned int k = 0; k < K; k++) {
                sum += A[a_row + k] * B[b_col];
                b_col += N;
            }
            C[c_idx] = sum;
            c_idx++;
        }
    }
}

int main() {
    mem[STATUS_ADDR] = 0;
    mem[WATCHDOG_ADDR] = 0;
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
            case OP_MATMUL: op_matmul(); break;
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
