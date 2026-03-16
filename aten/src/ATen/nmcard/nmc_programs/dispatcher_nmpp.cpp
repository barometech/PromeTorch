// ============================================================
// dispatcher_nmpp.cpp — Optimized dispatcher using nmpp BLAS
// ============================================================
// Uses nmppmMul_mm_32s32s from libnmpps-nmc4.a for vectorized
// matrix multiplication on NMC4 vector pipeline.
// 10-100x faster than scalar mul_fixed loop.
//
// Data format: Q16.16 fixed-point (nm32s = signed 32-bit)
// Input/output in DDR: IEEE 754 float → converted to Q16.16 on-card
// ============================================================

#include "mymath.h"

// nmpp matrix multiply declarations
extern "C" {
    void nmppmMul_mm_32s32s(int* pSrcMtr1, int nHeight1, int nWidth1,
                             int* pSrcMtr2, int* pDstMtr, int nWidth2);
    void nmppmMul_mm_32f(float* pSrcMtr1, int nHeight1, int nStride1,
                          float* pSrcMtr2, int nWidth1, int nStride2,
                          float* pDstMtr, int nWidth2, int nStrideDst, int bPlusDst);
}

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

#define OP_NOP           0
#define OP_MATMUL        1
#define OP_MATMUL_NMPP   25   // vectorized matmul via nmpp
#define OP_ELEM_ADD      10
#define OP_ELEM_MUL      11
#define OP_EXIT          255
#define STATUS_ADDR      30
#define WATCHDOG_ADDR    31

// ============================================================
// Q8.8 scaling for nmpp (avoids 32-bit overflow)
// nmpp does integer C = sum(A_int * B_int)
// With Q8.8: A_int = round(A * 256), B_int = round(B * 256)
// C_int = 256^2 * sum(A*B), so C_float = C_int / 65536
// ============================================================
#define NMPP_S 256
#define NMPP_S2 65536

void convert_float_to_q8(unsigned int* data, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 f = float_to_fixed(data[i]);  // IEEE float → Q16.16
        int val = ((int)f) >> 8;              // Q16.16 → Q8.8 (= int * 256)
        data[i] = (unsigned int)val;
    }
}

// ============================================================
// MatMul via nmpp FLOAT: C[M,N] = A[M,K] @ B[K,N]
// Input/output: IEEE 754 float DIRECTLY — no Q16.16 conversion!
// ============================================================
void op_matmul_nmpp() {
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    float* A = (float*)mem[4];
    float* B = (float*)mem[5];
    float* C = (float*)mem[6];

    // Direct float matmul — nmpp handles FP32 natively!
    // nmppmMul_mm_32f(A, M, K, B, K, N, C, N, N, 0)
    // stride1=K (row stride of A), stride2=N (row stride of B), strideDst=N, bPlusDst=0
    nmppmMul_mm_32f(A, (int)M, (int)K, B, (int)K, (int)N, C, (int)N, (int)N, 0);
}

// ============================================================
// Scalar matmul fallback (same as original dispatcher)
// ============================================================
void op_matmul_scalar() {
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    unsigned int* A = (unsigned int*)mem[4];
    unsigned int* B = (unsigned int*)mem[5];
    unsigned int* C = (unsigned int*)mem[6];

    unsigned int c_idx = 0;
    for (unsigned int i = 0; i < M; i++) {
        unsigned int a_row = mul_u32(i, K);
        for (unsigned int j = 0; j < N; j++) {
            fixed32 sum = 0;
            unsigned int b_col = j;
            for (unsigned int k = 0; k < K; k++) {
                fixed32 a_val = float_to_fixed(A[a_row + k]);
                fixed32 b_val = float_to_fixed(B[b_col]);
                sum = add_fixed(sum, mul_fixed(a_val, b_val));
                b_col += N;
            }
            C[c_idx] = fixed_to_float(sum);
            c_idx++;
        }
    }
}

// ============================================================
// Element-wise ops
// ============================================================
void op_elem_add() {
    unsigned int count = mem[1];
    unsigned int* a = (unsigned int*)mem[2];
    unsigned int* b = (unsigned int*)mem[3];
    unsigned int* out = (unsigned int*)mem[4];
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        fixed32 vb = float_to_fixed(b[i]);
        out[i] = fixed_to_float(add_fixed(va, vb));
    }
}

void op_elem_mul() {
    unsigned int count = mem[1];
    unsigned int* a = (unsigned int*)mem[2];
    unsigned int* b = (unsigned int*)mem[3];
    unsigned int* out = (unsigned int*)mem[4];
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        fixed32 vb = float_to_fixed(b[i]);
        out[i] = fixed_to_float(mul_fixed(va, vb));
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
        if (op == OP_EXIT) {
            mem[STATUS_ADDR] = 1;
            mem[0] = OP_NOP;
            break;
        }

        mem[STATUS_ADDR] = 0;

        switch (op) {
            case OP_MATMUL:       op_matmul_scalar(); break;
            case OP_MATMUL_NMPP:  op_matmul_nmpp(); break;
            case OP_ELEM_ADD:     op_elem_add(); break;
            case OP_ELEM_MUL:     op_elem_mul(); break;
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
