// MatMul one-shot - no infinite loop, just execute and return
// Параметры и данные уже в памяти, программа читает их, выполняет matmul, пишет результат

#include "mymath.h"

#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = M
// [1] = K
// [2] = N
// [3] = addr_A
// [4] = addr_B
// [5] = addr_C
// [6] = status (0=начало, 1=готово)
// [7] = debug

int main() {
    // Читаем параметры
    unsigned int M = mem[0];
    unsigned int K = mem[1];
    unsigned int N = mem[2];
    unsigned int addr_A = mem[3];
    unsigned int addr_B = mem[4];
    unsigned int addr_C = mem[5];

    volatile unsigned int* A = (volatile unsigned int*)addr_A;
    volatile unsigned int* B = (volatile unsigned int*)addr_B;
    volatile unsigned int* C = (volatile unsigned int*)addr_C;

    // Debug: записываем что программа запустилась
    mem[7] = 0xAAAA;

    // Выполняем MatMul: C[M,N] = A[M,K] * B[K,N]
    // Используем инкрементальные индексы чтобы избежать Mul32

    unsigned int a_row_idx = 0;  // = i * K
    unsigned int c_row_idx = 0;  // = i * N

    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            fixed32 sum = 0;

            unsigned int b_col_idx = j;  // = k * N + j, начинаем с j

            for (unsigned int k = 0; k < K; k++) {
                // a_idx = i*K + k = a_row_idx + k
                // b_idx = k*N + j = b_col_idx

                fixed32 a_fixed = float_to_fixed(A[a_row_idx + k]);
                fixed32 b_fixed = float_to_fixed(B[b_col_idx]);

                sum = add_fixed(sum, mul_fixed(a_fixed, b_fixed));

                b_col_idx += N;  // переход к следующей строке B
            }

            // c_idx = i*N + j = c_row_idx + j
            C[c_row_idx + j] = fixed_to_float(sum);
        }

        a_row_idx += K;  // переход к следующей строке A
        c_row_idx += N;  // переход к следующей строке C
    }

    // Debug: записываем сколько элементов посчитали
    mem[7] = mul_u32(M, N);  // используем наш mul без Mul32

    // Готово!
    mem[6] = 1;

    return 0;
}
