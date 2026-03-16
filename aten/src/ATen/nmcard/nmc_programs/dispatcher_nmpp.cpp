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

// nmpp matrix multiply declaration
extern "C" {
    void nmppmMul_mm_32s32s(int* pSrcMtr1, int nHeight1, int nWidth1,
                             int* pSrcMtr2, int* pDstMtr, int nWidth2);
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
// Convert float array in DDR to Q16.16 in-place
// ============================================================
void convert_float_to_q16(unsigned int* data, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        data[i] = (unsigned int)float_to_fixed(data[i]);
    }
}

// ============================================================
// Convert Q16.16 array in DDR to float in-place
// ============================================================
void convert_q16_to_float(unsigned int* data, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        data[i] = fixed_to_float((fixed32)data[i]);
    }
}

// ============================================================
// MatMul via nmpp: C[M,N] = A[M,K] @ B[K,N]
// args: [M, K, N, addr_A, addr_B, addr_C]
// Input A,B in IEEE float. Output C in IEEE float.
// ============================================================
void op_matmul_nmpp() {
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    unsigned int* A = (unsigned int*)mem[4];
    unsigned int* B = (unsigned int*)mem[5];
    unsigned int* C = (unsigned int*)mem[6];

    // Step 1: Convert A,B from float to Q16.16
    convert_float_to_q16(A, mul_u32(M, K));
    convert_float_to_q16(B, mul_u32(K, N));

    // Step 2: Vectorized matmul via nmpp
    // nmppmMul_mm_32s32s(A, M, K, B, C, N)
    nmppmMul_mm_32s32s((int*)A, (int)M, (int)K, (int*)B, (int*)C, (int)N);

    // Step 3: Convert C from Q16.16 to float
    // Note: nmpp result is in 32s32s = 32-bit accumulation of 32-bit inputs
    // The result needs to be scaled by 1/FIXED_ONE since both inputs were Q16.16
    // Result = sum(a_q16 * b_q16) which is Q32.32 truncated to 32 bits
    // For Q16.16 * Q16.16 = Q32.32, we need to right-shift by 16
    unsigned int MN = mul_u32(M, N);
    for (unsigned int i = 0; i < MN; i++) {
        int val = (int)C[i];
        // Right shift by 16 to go from Q32.32 to Q16.16
        val = (val >> 16);  // This is the Q16.16 result
        C[i] = fixed_to_float((fixed32)val);
    }
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
