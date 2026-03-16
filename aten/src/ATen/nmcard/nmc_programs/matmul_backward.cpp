// matmul_backward.cpp - Backward pass for matrix multiplication on NMC4
// Forward: C = A @ B where A[M,K], B[K,N], C[M,N]
// Backward:
//   dA = dC @ B^T  (gradient w.r.t. input)
//   dB = A^T @ dC  (gradient w.r.t. weights)
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"
#include "mymath_backward.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = backward_input, 2 = backward_weights, 3 = both)
// [1] = M
// [2] = K
// [3] = N
// [4] = addr_A (input activations, for dB calculation)
// [5] = addr_B (weights, for dA calculation)
// [6] = addr_grad_C (gradient from next layer)
// [7] = addr_grad_A (output: gradient w.r.t. input)
// [8] = addr_grad_B (output: gradient w.r.t. weights)
// [9] = status (0 = busy, 1 = done)

// dA = dC @ B^T
// dC[M,N] @ B[K,N]^T = dA[M,K]
void matmul_backward_input(
    volatile unsigned int* A,      // not used for dA
    volatile unsigned int* B,      // [K,N]
    volatile unsigned int* grad_C, // [M,N]
    volatile unsigned int* grad_A, // [M,K] output
    unsigned int M, unsigned int K, unsigned int N
) {
    // dA[i,k] = sum_j(dC[i,j] * B[k,j])
    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int k = 0; k < K; k++) {
            fixed32 sum = 0;
            for (unsigned int j = 0; j < N; j++) {
                fixed32 dc = float_to_fixed(grad_C[i * N + j]);
                fixed32 b = float_to_fixed(B[k * N + j]);  // B[k,j]
                sum = add_fixed(sum, mul_fixed(dc, b));
            }
            grad_A[i * K + k] = fixed_to_float(sum);
        }
    }
}

// dB = A^T @ dC
// A[M,K]^T @ dC[M,N] = dB[K,N]
void matmul_backward_weights(
    volatile unsigned int* A,      // [M,K]
    volatile unsigned int* B,      // not used for dB
    volatile unsigned int* grad_C, // [M,N]
    volatile unsigned int* grad_B, // [K,N] output
    unsigned int M, unsigned int K, unsigned int N
) {
    // dB[k,n] = sum_i(A[i,k] * dC[i,n])
    for (unsigned int k = 0; k < K; k++) {
        for (unsigned int n = 0; n < N; n++) {
            fixed32 sum = 0;
            for (unsigned int i = 0; i < M; i++) {
                fixed32 a = float_to_fixed(A[i * K + k]);  // A[i,k]
                fixed32 dc = float_to_fixed(grad_C[i * N + n]);
                sum = add_fixed(sum, mul_fixed(a, dc));
            }
            grad_B[k * N + n] = fixed_to_float(sum);
        }
    }
}

int main() {
    mem[9] = 0;  // status = ready

    while (1) {
        unsigned int cmd = mem[0];
        // OP_EXIT = 255 - безопасный выход!
        if (cmd == 255) { mem[9] = 1; break; }
        if (cmd == 0) continue;

        mem[9] = 0;  // busy

        unsigned int M = mem[1];
        unsigned int K = mem[2];
        unsigned int N = mem[3];

        volatile unsigned int* A = (volatile unsigned int*)mem[4];
        volatile unsigned int* B = (volatile unsigned int*)mem[5];
        volatile unsigned int* grad_C = (volatile unsigned int*)mem[6];
        volatile unsigned int* grad_A = (volatile unsigned int*)mem[7];
        volatile unsigned int* grad_B = (volatile unsigned int*)mem[8];

        if (cmd == 1 || cmd == 3) {
            // Backward input: dA = dC @ B^T
            matmul_backward_input(A, B, grad_C, grad_A, M, K, N);
        }

        if (cmd == 2 || cmd == 3) {
            // Backward weights: dB = A^T @ dC
            matmul_backward_weights(A, B, grad_C, grad_B, M, K, N);
        }

        mem[9] = 1;  // done
        mem[0] = 0;  // ready for next
    }

    return 0;
}
