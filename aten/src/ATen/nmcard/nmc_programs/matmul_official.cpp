// MatMul kernel using OFFICIAL sync API
// This is the correct way according to RC Module documentation
//
// Comparison:
// - OUR approach: polling DDR memory at 0x00340000
// - OFFICIAL: using ncl_hostSyncArray() with IPC registers at 0x18018000

#include "nm6408load_nmc.h"
#include "mymath.h"

// Command codes (must match host)
#define CMD_EXIT 0
#define CMD_MATMUL 1
#define CMD_RESULT 2

// Max buffer size (in NMMB local memory - 512KB total)
// Each word = 4 bytes, so 32KB = 8192 words
#define MAX_BUFFER_SIZE 8192

// Buffers in local memory (NMMB)
static unsigned int A_buffer[MAX_BUFFER_SIZE];
static unsigned int B_buffer[MAX_BUFFER_SIZE];
static unsigned int C_buffer[MAX_BUFFER_SIZE];

// Command structure
struct MatMulCmd {
    unsigned int M;
    unsigned int K;
    unsigned int N;
};

// Custom matmul (no library calls)
void matmul(unsigned int* A, unsigned int* B, unsigned int* C,
            unsigned int M, unsigned int K, unsigned int N) {

    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            fixed32 sum = 0;
            for (unsigned int k = 0; k < K; k++) {
                unsigned int a_idx = mul_u32(i, K) + k;
                unsigned int b_idx = mul_u32(k, N) + j;
                fixed32 a_val = float_to_fixed(A[a_idx]);
                fixed32 b_val = float_to_fixed(B[b_idx]);
                sum = add_fixed(sum, mul_fixed(a_val, b_val));
            }
            C[mul_u32(i, N) + j] = fixed_to_float(sum);
        }
    }
}

int main() {
    MatMulCmd cmd;
    int sync_value;
    void* in_array;
    unsigned int in_len;

    while (1) {
        // Wait for command from host (blocking call)
        // ncl_hostSyncArray returns the value sent by host
        // and receives array address/length if provided
        sync_value = ncl_hostSyncArray(
            CMD_MATMUL,           // value we send
            &cmd,                 // our output array (command struct)
            sizeof(cmd) / 4,      // size in words
            &in_array,            // input array from host (NULL = none)
            &in_len               // input length from host
        );

        // Check for exit command
        if (sync_value == CMD_EXIT) {
            break;
        }

        // Receive matrix A
        sync_value = ncl_hostSyncArray(
            1, A_buffer, cmd.M * cmd.K, NULL, NULL
        );
        if (sync_value == CMD_EXIT) break;

        // Receive matrix B
        sync_value = ncl_hostSyncArray(
            2, B_buffer, cmd.K * cmd.N, NULL, NULL
        );
        if (sync_value == CMD_EXIT) break;

        // Execute matmul
        matmul(A_buffer, B_buffer, C_buffer, cmd.M, cmd.K, cmd.N);

        // Send result back
        sync_value = ncl_hostSyncArray(
            CMD_RESULT,           // value indicating result ready
            C_buffer,             // result array
            cmd.M * cmd.N,        // result size
            NULL, NULL
        );
        if (sync_value == CMD_EXIT) break;
    }

    return 0;
}
