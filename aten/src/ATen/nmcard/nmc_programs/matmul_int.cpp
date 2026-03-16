// Integer MatMul for NMC4 - без float операций для тестирования

#define DDR_BASE 0x00340000

// Memory layout:
// [0] = cmd (1 = matmul, 0 = idle)
// [1] = M
// [2] = K
// [3] = N
// [4] = addr_A
// [5] = addr_B
// [6] = addr_C
// [7] = status (0 = busy, 1 = done)
// [8] = debug: iterations done

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

#define CMD      0
#define M_DIM    1
#define K_DIM    2
#define N_DIM    3
#define ADDR_A   4
#define ADDR_B   5
#define ADDR_C   6
#define STATUS   7
#define DBG_ITER 8

// Integer MatMul: C[M,N] = A[M,K] * B[K,N]
// Matrices stored as int, row-major
void matmul_int(int* A, int* B, int* C, unsigned int M, unsigned int K, unsigned int N) {
    unsigned int iter = 0;
    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            int sum = 0;
            for (unsigned int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
            iter++;
        }
    }
    mem[DBG_ITER] = iter;
}

int main() {
    // Initialize
    mem[STATUS] = 0;
    mem[CMD] = 0;
    mem[DBG_ITER] = 0;

    // Main loop
    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[CMD] == 255) { mem[STATUS] = 1; break; }
        if (mem[CMD] == 1) {
            mem[STATUS] = 0;  // busy

            unsigned int M = mem[M_DIM];
            unsigned int K = mem[K_DIM];
            unsigned int N = mem[N_DIM];

            int* A = (int*)mem[ADDR_A];
            int* B = (int*)mem[ADDR_B];
            int* C = (int*)mem[ADDR_C];

            matmul_int(A, B, C, M, K, N);

            mem[STATUS] = 1;  // done
            mem[CMD] = 0;     // ready
        }
    }

    return 0;
}
