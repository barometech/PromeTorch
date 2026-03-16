// dispatcher_safe.cpp - БЕЗОПАСНЫЙ dispatcher с OP_EXIT
// НЕ ЗАВИСАЕТ! Всегда можно завершить записью OP_EXIT в память.

#include "mymath.h"

// DDR base address (словарная адресация)
#define DDR_BASE 0x00340000
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// mem[0]  = CMD (команда от хоста)
// mem[1]  = ARG0
// mem[2]  = ARG1
// mem[3]  = ARG2
// mem[4]  = ARG3
// mem[5]  = ARG4
// mem[6]  = ARG5
// mem[7]  = STATUS (0=busy, 1=done, 2=error)
// mem[8]  = RESULT
// mem[16+] = DATA

#define CMD_ADDR    0
#define ARG0_ADDR   1
#define ARG1_ADDR   2
#define ARG2_ADDR   3
#define ARG3_ADDR   4
#define ARG4_ADDR   5
#define ARG5_ADDR   6
#define STATUS_ADDR 7
#define RESULT_ADDR 8
#define DATA_ADDR   16

// Коды операций
#define OP_NOP      0    // Нет операции (ждём)
#define OP_MATMUL   1    // MatMul
#define OP_RMSNORM  2    // RMSNorm
#define OP_SILU     3    // SiLU
#define OP_SOFTMAX  4    // Softmax
#define OP_ROPE     5    // RoPE
#define OP_ADD      6    // Elementwise Add
#define OP_MUL      7    // Elementwise Mul
#define OP_ATTENTION 8   // Attention
#define OP_PING     100  // Тест связи
#define OP_EXIT     255  // ВЫХОД! Завершить программу

// Статусы
#define STATUS_BUSY  0
#define STATUS_DONE  1
#define STATUS_ERROR 2

// Forward declarations
void do_matmul();
void do_rmsnorm();
void do_silu();
void do_softmax();
void do_add();
void do_mul();

int main() {
    // Инициализация - сигнал что dispatcher готов
    mem[STATUS_ADDR] = STATUS_DONE;
    mem[RESULT_ADDR] = 0xBEAD1;  // Ready marker

    // Главный цикл - ОБЯЗАТЕЛЬНО с условием выхода!
    while (1) {
        unsigned int cmd = mem[CMD_ADDR];

        // OP_EXIT - немедленный выход!
        if (cmd == OP_EXIT) {
            mem[STATUS_ADDR] = STATUS_DONE;
            mem[RESULT_ADDR] = 0xB1EB1E;
            break;  // ВЫХОД ИЗ ЦИКЛА!
        }

        // OP_NOP - просто ждём
        if (cmd == OP_NOP) {
            continue;
        }

        // Есть команда - выполняем
        mem[STATUS_ADDR] = STATUS_BUSY;

        switch (cmd) {
            case OP_PING:
                // Простой тест - возвращаем ARG0 + 1
                mem[RESULT_ADDR] = mem[ARG0_ADDR] + 1;
                mem[STATUS_ADDR] = STATUS_DONE;
                break;

            case OP_MATMUL:
                do_matmul();
                break;

            case OP_RMSNORM:
                do_rmsnorm();
                break;

            case OP_SILU:
                do_silu();
                break;

            case OP_SOFTMAX:
                do_softmax();
                break;

            case OP_ADD:
                do_add();
                break;

            case OP_MUL:
                do_mul();
                break;

            default:
                // Неизвестная команда
                mem[RESULT_ADDR] = cmd;
                mem[STATUS_ADDR] = STATUS_ERROR;
                break;
        }

        // Сбрасываем команду - готовы к следующей
        mem[CMD_ADDR] = OP_NOP;
    }

    // Программа завершается корректно!
    return 0;
}

// ============================================
// Реализации операций
// ============================================

void do_matmul() {
    // ARG0 = M, ARG1 = K, ARG2 = N
    // ARG3 = offset A, ARG4 = offset B, ARG5 = offset C
    unsigned int M = mem[ARG0_ADDR];
    unsigned int K = mem[ARG1_ADDR];
    unsigned int N = mem[ARG2_ADDR];
    unsigned int offA = mem[ARG3_ADDR];
    unsigned int offB = mem[ARG4_ADDR];
    unsigned int offC = mem[ARG5_ADDR];

    volatile int* A = (volatile int*)(mem + DATA_ADDR + offA);
    volatile int* B = (volatile int*)(mem + DATA_ADDR + offB);
    volatile int* C = (volatile int*)(mem + DATA_ADDR + offC);

    // C[i,j] = sum_k A[i,k] * B[k,j]
    unsigned int idx_c = 0;
    for (unsigned int i = 0; i < M; i++) {
        unsigned int idx_a_row = i * K;
        for (unsigned int j = 0; j < N; j++) {
            int sum = 0;
            unsigned int idx_a = idx_a_row;
            unsigned int idx_b = j;
            for (unsigned int k = 0; k < K; k++) {
                sum += mul_fixed(A[idx_a], B[idx_b]);
                idx_a++;
                idx_b += N;
            }
            C[idx_c] = sum;
            idx_c++;
        }
    }

    mem[RESULT_ADDR] = M * N;  // Количество элементов
    mem[STATUS_ADDR] = STATUS_DONE;
}

void do_rmsnorm() {
    // ARG0 = size, ARG1 = offset X, ARG2 = offset gamma, ARG3 = offset Y
    unsigned int size = mem[ARG0_ADDR];
    unsigned int offX = mem[ARG1_ADDR];
    unsigned int offG = mem[ARG2_ADDR];
    unsigned int offY = mem[ARG3_ADDR];

    volatile int* X = (volatile int*)(mem + DATA_ADDR + offX);
    volatile int* gamma = (volatile int*)(mem + DATA_ADDR + offG);
    volatile int* Y = (volatile int*)(mem + DATA_ADDR + offY);

    // Compute sum of squares
    int sum_sq = 0;
    for (unsigned int i = 0; i < size; i++) {
        sum_sq += mul_fixed(X[i], X[i]);
    }

    // Mean of squares
    int mean_sq = div_fixed(sum_sq, INT_TO_FIXED(size));

    // RMS = sqrt(mean_sq + eps)
    int eps = 1;  // Small epsilon in Q16.16
    int rms = sqrt_fixed(mean_sq + eps);

    // Normalize: Y = X / RMS * gamma
    int inv_rms = div_fixed(FIXED_ONE, rms);
    for (unsigned int i = 0; i < size; i++) {
        int norm = mul_fixed(X[i], inv_rms);
        Y[i] = mul_fixed(norm, gamma[i]);
    }

    mem[RESULT_ADDR] = size;
    mem[STATUS_ADDR] = STATUS_DONE;
}

void do_silu() {
    // ARG0 = size, ARG1 = offset X, ARG2 = offset Y
    unsigned int size = mem[ARG0_ADDR];
    unsigned int offX = mem[ARG1_ADDR];
    unsigned int offY = mem[ARG2_ADDR];

    volatile int* X = (volatile int*)(mem + DATA_ADDR + offX);
    volatile int* Y = (volatile int*)(mem + DATA_ADDR + offY);

    for (unsigned int i = 0; i < size; i++) {
        Y[i] = silu_fixed(X[i]);
    }

    mem[RESULT_ADDR] = size;
    mem[STATUS_ADDR] = STATUS_DONE;
}

void do_softmax() {
    // ARG0 = size, ARG1 = offset X, ARG2 = offset Y
    unsigned int size = mem[ARG0_ADDR];
    unsigned int offX = mem[ARG1_ADDR];
    unsigned int offY = mem[ARG2_ADDR];

    volatile int* X = (volatile int*)(mem + DATA_ADDR + offX);
    volatile int* Y = (volatile int*)(mem + DATA_ADDR + offY);

    // Find max
    int max_val = X[0];
    for (unsigned int i = 1; i < size; i++) {
        if (X[i] > max_val) max_val = X[i];
    }

    // Compute exp(x - max) and sum
    int sum = 0;
    for (unsigned int i = 0; i < size; i++) {
        Y[i] = exp_fixed(X[i] - max_val);
        sum += Y[i];
    }

    // Normalize
    if (sum > 0) {
        for (unsigned int i = 0; i < size; i++) {
            Y[i] = div_fixed(Y[i], sum);
        }
    }

    mem[RESULT_ADDR] = size;
    mem[STATUS_ADDR] = STATUS_DONE;
}

void do_add() {
    // ARG0 = size, ARG1 = offset A, ARG2 = offset B, ARG3 = offset C
    unsigned int size = mem[ARG0_ADDR];
    unsigned int offA = mem[ARG1_ADDR];
    unsigned int offB = mem[ARG2_ADDR];
    unsigned int offC = mem[ARG3_ADDR];

    volatile int* A = (volatile int*)(mem + DATA_ADDR + offA);
    volatile int* B = (volatile int*)(mem + DATA_ADDR + offB);
    volatile int* C = (volatile int*)(mem + DATA_ADDR + offC);

    for (unsigned int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }

    mem[RESULT_ADDR] = size;
    mem[STATUS_ADDR] = STATUS_DONE;
}

void do_mul() {
    // ARG0 = size, ARG1 = offset A, ARG2 = offset B, ARG3 = offset C
    unsigned int size = mem[ARG0_ADDR];
    unsigned int offA = mem[ARG1_ADDR];
    unsigned int offB = mem[ARG2_ADDR];
    unsigned int offC = mem[ARG3_ADDR];

    volatile int* A = (volatile int*)(mem + DATA_ADDR + offA);
    volatile int* B = (volatile int*)(mem + DATA_ADDR + offB);
    volatile int* C = (volatile int*)(mem + DATA_ADDR + offC);

    for (unsigned int i = 0; i < size; i++) {
        C[i] = mul_fixed(A[i], B[i]);
    }

    mem[RESULT_ADDR] = size;
    mem[STATUS_ADDR] = STATUS_DONE;
}
