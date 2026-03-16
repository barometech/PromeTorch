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

// nmpp float matmul
extern "C" {
    void nmppmMul_mm_32f(float* A, int M, int strideA,
                          float* B, int K, int strideB,
                          float* C, int N, int strideC, int bPlusDst);
}

void op_matmul() {
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    float* A = (float*)mem[4];
    float* B = (float*)mem[5];
    float* C = (float*)mem[6];

    // Direct float matmul on NMC4 float VPU
    nmppmMul_mm_32f(A, (int)M, (int)K, B, (int)K, (int)N, C, (int)N, (int)N, 0);
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
