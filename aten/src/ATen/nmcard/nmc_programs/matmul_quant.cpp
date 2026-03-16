// matmul_quant.cpp - MatMul with INT8 quantized weights for NM Card Mini
// Input: Q16.16 activations
// Weights: Packed INT8 with per-channel Q16.16 scales
// Output: Q16.16
//
// Memory savings: 4x compared to full Q16.16 weights
// Arithmetic: Same Q16.16 as matmul_custom.cpp

// Type definitions (NMC4 compiler doesn't have <stdint.h>)
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned int uint32_t;

#include "mymath.h"
#include "dequant.h"

// ============================================================================
// Memory Layout
// ============================================================================

#define DDR_BASE 0x00340000

// Command buffer at start of DDR
volatile uint32_t* cmd = (volatile uint32_t*)(DDR_BASE);

// Command structure offsets
#define CMD_OP        cmd[0]    // Operation code (unused for single kernel)
#define CMD_STATUS    cmd[1]    // 0=busy, 1=done, 2=error
#define CMD_M         cmd[2]    // Rows of A, rows of C
#define CMD_N         cmd[3]    // Cols of B, cols of C
#define CMD_K         cmd[4]    // Cols of A, rows of B
#define CMD_A_PTR     cmd[5]    // Input A: Q16.16 [M, K] - word offset from DDR_BASE
#define CMD_B_PTR     cmd[6]    // Weights B: packed INT8 [N, K] - word offset
#define CMD_C_PTR     cmd[7]    // Output C: Q16.16 [M, N] - word offset
#define CMD_SCALES    cmd[8]    // Scales: Q16.16 [N] - word offset

// ============================================================================
// MatMul with INT8 Weights
// ============================================================================

// C[M,N] = A[M,K] @ B[K,N]^T
// Where B is stored as [N, K] in INT8 packed format (row-major)
// Each row of B has its own scale

void matmul_int8_weights() {
    // Read parameters
    uint32_t M = CMD_M;
    uint32_t N = CMD_N;
    uint32_t K = CMD_K;

    // Calculate pointers (word-addressed)
    volatile fixed32* A = (volatile fixed32*)(DDR_BASE + CMD_A_PTR);
    volatile uint32_t* B_packed = (volatile uint32_t*)(DDR_BASE + CMD_B_PTR);
    volatile fixed32* C = (volatile fixed32*)(DDR_BASE + CMD_C_PTR);
    volatile fixed32* scales = (volatile fixed32*)(DDR_BASE + CMD_SCALES);

    // Packed row size: K INT8 values = (K+3)/4 uint32 words
    uint32_t packed_row_words = (K + 3) >> 2;

    // Main computation
    // For each output row i
    for (uint32_t i = 0; i < M; i++) {
        // For each output column j (which is row j of B)
        for (uint32_t j = 0; j < N; j++) {
            // Get scale for this weight row
            fixed32 scale_j = scales[j];

            // Pointer to packed INT8 row j of B
            volatile uint32_t* B_row = B_packed + j * packed_row_words;

            // Compute dot product using fused dequant
            // A[i,:] dot B[j,:] with B dequantized on-the-fly
            fixed32 sum = dot_int8_q16(
                B_row,           // Packed INT8 weights
                A + i * K,       // Q16.16 input row
                scale_j,         // Scale for this weight row
                K                // Vector length
            );

            // Store result
            C[i * N + j] = sum;
        }
    }

    // Signal completion
    CMD_STATUS = 1;
}

// ============================================================================
// Alternative: Dequantize row first, then standard MatMul
// Uses more memory but may be faster for small K
// ============================================================================

// Static buffer for dequantized weight row
// Adjust size based on max expected K
#define MAX_K 4096
static fixed32 weight_buffer[MAX_K];

void matmul_int8_weights_buffered() {
    uint32_t M = CMD_M;
    uint32_t N = CMD_N;
    uint32_t K = CMD_K;

    if (K > MAX_K) {
        CMD_STATUS = 2;  // Error: K too large
        return;
    }

    volatile fixed32* A = (volatile fixed32*)(DDR_BASE + CMD_A_PTR);
    volatile uint32_t* B_packed = (volatile uint32_t*)(DDR_BASE + CMD_B_PTR);
    volatile fixed32* C = (volatile fixed32*)(DDR_BASE + CMD_C_PTR);
    volatile fixed32* scales = (volatile fixed32*)(DDR_BASE + CMD_SCALES);

    uint32_t packed_row_words = (K + 3) >> 2;

    // For each weight row (output column)
    for (uint32_t j = 0; j < N; j++) {
        // Dequantize entire weight row to buffer
        dequant_row(
            B_packed + j * packed_row_words,
            scales[j],
            weight_buffer,
            K
        );

        // Compute all output elements for this column
        for (uint32_t i = 0; i < M; i++) {
            fixed32 sum = 0;

            // Standard dot product in Q16.16
            for (uint32_t k = 0; k < K; k++) {
                sum = add_fixed(sum, mul_fixed(A[i * K + k], weight_buffer[k]));
            }

            C[i * N + j] = sum;
        }
    }

    CMD_STATUS = 1;
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main() {
    // Wait for command
    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (CMD_OP == 255) { CMD_STATUS = 1; break; }

        // Check for new command (status = 0 means new command pending)
        if (CMD_STATUS == 0) {
            // Execute MatMul with INT8 weights
            matmul_int8_weights();
        }

        // Small delay to avoid busy-waiting too hard
        for (volatile int i = 0; i < 100; i++);
    }

    return 0;
}
