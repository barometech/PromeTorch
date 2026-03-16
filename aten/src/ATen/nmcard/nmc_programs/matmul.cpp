// MatMul for NMC4
// NMC4 использует СЛОВАРНУЮ адресацию! Каждый адрес = 32-bit слово.

// DDR base - словарные адреса
#define DDR_BASE 0x00340000

// Структура команды от хоста (8 слов)
struct MatMulCmd {
    unsigned int cmd;        // 0: слово 0 - команда (1 = matmul, 0 = idle)
    unsigned int M;          // 1: слово 1 - rows of A
    unsigned int K;          // 2: слово 2 - cols of A, rows of B
    unsigned int N;          // 3: слово 3 - cols of B
    unsigned int addr_A;     // 4: слово 4 - адрес матрицы A (словарный!)
    unsigned int addr_B;     // 5: слово 5 - адрес матрицы B (словарный!)
    unsigned int addr_C;     // 6: слово 6 - адрес результата C (словарный!)
    unsigned int status;     // 7: слово 7 - статус (0 = busy, 1 = done)
};

// Указатель на command block в начале DDR
volatile MatMulCmd* cmd_block = (volatile MatMulCmd*)DDR_BASE;

// Базовый MatMul: C[M,N] = A[M,K] * B[K,N]
// Все указатели - словарные адреса в DDR
void matmul_basic(volatile float* A, volatile float* B, volatile float* C,
                  unsigned int M, unsigned int K, unsigned int N) {
    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < K; k++) {
                // Row-major: A[i,k] = A[i*K + k], B[k,j] = B[k*N + j]
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Главный цикл - ждёт команды от хоста
int main() {
    // Сигнал что программа запустилась
    cmd_block->status = 0;
    cmd_block->cmd = 0;

    // Бесконечный цикл обработки команд
    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (cmd_block->cmd == 255) { cmd_block->status = 1; break; }
        // Ждём команду matmul (cmd == 1)
        if (cmd_block->cmd != 1) {
            continue;
        }

        // Получили команду - устанавливаем busy
        cmd_block->status = 0;

        // Читаем параметры
        unsigned int M = cmd_block->M;
        unsigned int K = cmd_block->K;
        unsigned int N = cmd_block->N;

        // Адреса матриц (уже словарные)
        volatile float* A = (volatile float*)cmd_block->addr_A;
        volatile float* B = (volatile float*)cmd_block->addr_B;
        volatile float* C = (volatile float*)cmd_block->addr_C;

        // Выполняем matmul
        matmul_basic(A, B, C, M, K, N);

        // Готово
        cmd_block->status = 1;
        cmd_block->cmd = 0;  // ready for next
    }

    return 0;
}
