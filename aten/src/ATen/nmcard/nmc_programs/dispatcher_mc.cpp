// Multi-Core Dispatcher for NMC4
// Same program loaded on all 16 cores. Each core determines its ID
// via ncl_getCoreID()/ncl_getClusterID() and polls its own cmd block.
//
// DDR layout: DDR_BASE + core_index*32 = cmd block for core_index
// core_index = cluster_id * 4 + core_id (0..15)

#include "mymath.h"
#include "nm6408load_nmc.h"

// Stub for LShift32 - required by libnm6408load_nmc.a abort handler
// Never called at runtime, just satisfies linker
extern "C" unsigned int LShift32(unsigned int a, unsigned int b) {
    return my_lshift(a, b);
}
extern "C" unsigned int RShift32(unsigned int a, unsigned int b) {
    return my_rshift(a, b);
}
extern "C" int Mul32(int a, int b) {
    return (int)mul_i32(a, b);
}

#define DDR_BASE 0x00340000
#define CMD_BLOCK_SIZE 32

// Each core's cmd block is at DDR_BASE + core_index * CMD_BLOCK_SIZE
// mem pointer is set per-core in main()
volatile unsigned int* mem;

// ============================================================
// Operation codes (same as single-core dispatcher)
// ============================================================
#define OP_NOP          0
#define OP_MATMUL       1
#define OP_RMSNORM      2
#define OP_SOFTMAX      3
#define OP_SILU         4
#define OP_ROPE         5
#define OP_ATTENTION    6
#define OP_ELEM_ADD     10
#define OP_ELEM_MUL     11
#define OP_ELEM_SUB     12
#define OP_GATE_MUL     13
#define OP_MUL_SCALAR   14
#define OP_GELU         15
#define OP_LAYERNORM    16
#define OP_MATMUL_DDR   20
#define OP_RMSNORM_DDR  21

// Multi-core specific
#define OP_MATMUL_PARTIAL 22  // matmul computing only a column range

#define OP_EXIT         255

#define STATUS_ADDR 30
#define WATCHDOG_ADDR 31

// ============================================================
// MatMul: C[M,N] = A[M,K] @ B[K,N]  (full)
// args: [M, K, N, addr_A, addr_B, addr_C]
// ============================================================
void op_matmul() {
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    unsigned int* A = (unsigned int*)mem[4];
    unsigned int* B = (unsigned int*)mem[5];
    unsigned int* C = (unsigned int*)mem[6];

    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            fixed32 sum = 0;
            unsigned int a_idx = mul_u32(i, K);
            unsigned int b_idx = j;
            for (unsigned int k = 0; k < K; k++) {
                fixed32 a_val = float_to_fixed(A[a_idx + k]);
                fixed32 b_val = float_to_fixed(B[b_idx]);
                sum = add_fixed(sum, mul_fixed(a_val, b_val));
                b_idx += N;
            }
            C[mul_u32(i, N) + j] = fixed_to_float(sum);
        }
    }
}

// ============================================================
// MatMul Partial: compute only columns [col_start, col_end) of C
// args: [M, K, N, addr_A, addr_B, addr_C, col_start, col_end]
// All cores share same A, B, C arrays in DDR.
// Each core writes to C[i, col_start..col_end-1].
// ============================================================
void op_matmul_partial() {
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    unsigned int* A = (unsigned int*)mem[4];
    unsigned int* B = (unsigned int*)mem[5];
    unsigned int* C = (unsigned int*)mem[6];
    unsigned int col_start = mem[7];
    unsigned int col_end = mem[8];

    for (unsigned int i = 0; i < M; i++) {
        unsigned int a_off = mul_u32(i, K);
        unsigned int c_off = mul_u32(i, N);

        for (unsigned int j = col_start; j < col_end; j++) {
            fixed32 sum = 0;
            unsigned int b_off = j;  // B[0, j]

            for (unsigned int k = 0; k < K; k++) {
                fixed32 a_val = float_to_fixed(A[a_off + k]);
                fixed32 b_val = float_to_fixed(B[b_off]);
                sum = add_fixed(sum, mul_fixed(a_val, b_val));
                b_off += N;  // B[k+1, j]
            }
            C[c_off + j] = fixed_to_float(sum);
        }
    }
}

// ============================================================
// RMSNorm
// args: [batch, hidden, addr_in, addr_out, addr_gamma]
// ============================================================
void op_rmsnorm() {
    unsigned int batch = mem[1];
    unsigned int hidden = mem[2];
    unsigned int* input = (unsigned int*)mem[3];
    unsigned int* output = (unsigned int*)mem[4];
    unsigned int* gamma = (unsigned int*)mem[5];
    fixed32 eps = 1;

    for (unsigned int b = 0; b < batch; b++) {
        unsigned int off = mul_u32(b, hidden);
        unsigned int* x = input + off;
        unsigned int* y = output + off;

        fixed32 sum_sq = 0;
        for (unsigned int i = 0; i < hidden; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            sum_sq = add_fixed(sum_sq, mul_fixed(xi, xi));
        }
        fixed32 rms = sqrt_fixed(add_fixed(div_fixed(sum_sq, INT_TO_FIXED(hidden)), eps));
        fixed32 inv_rms = div_fixed(FIXED_ONE, rms);

        for (unsigned int i = 0; i < hidden; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            fixed32 g = float_to_fixed(gamma[i]);
            y[i] = fixed_to_float(mul_fixed(mul_fixed(xi, inv_rms), g));
        }
    }
}

// ============================================================
// Softmax
// args: [batch, dim, addr_in, addr_out]
// ============================================================
void op_softmax() {
    unsigned int batch = mem[1];
    unsigned int dim = mem[2];
    unsigned int* input = (unsigned int*)mem[3];
    unsigned int* output = (unsigned int*)mem[4];

    for (unsigned int b = 0; b < batch; b++) {
        unsigned int off = mul_u32(b, dim);
        unsigned int* x = input + off;
        unsigned int* y = output + off;

        fixed32 max_val = float_to_fixed(x[0]);
        for (unsigned int i = 1; i < dim; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            if (xi > max_val) max_val = xi;
        }

        fixed32 exp_sum = 0;
        for (unsigned int i = 0; i < dim; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            fixed32 exp_val = exp_fixed_lut(sub_fixed(xi, max_val));
            y[i] = (unsigned int)exp_val;
            exp_sum = add_fixed(exp_sum, exp_val);
        }

        if (exp_sum == 0) exp_sum = 1;
        for (unsigned int i = 0; i < dim; i++) {
            fixed32 exp_val = (fixed32)y[i];
            y[i] = fixed_to_float(div_fixed(exp_val, exp_sum));
        }
    }
}

// ============================================================
// SiLU: y = x * sigmoid(x)
// args: [count, addr_in, addr_out]
// ============================================================
void op_silu() {
    unsigned int count = mem[1];
    unsigned int* input = (unsigned int*)mem[2];
    unsigned int* output = (unsigned int*)mem[3];

    for (unsigned int i = 0; i < count; i++) {
        fixed32 x = float_to_fixed(input[i]);
        output[i] = fixed_to_float(silu_fixed(x));
    }
}

// ============================================================
// RoPE
// args: [seq_len, head_dim, pos_offset, addr_in, addr_out, addr_freqs]
// ============================================================
void op_rope() {
    unsigned int seq_len = mem[1];
    unsigned int head_dim = mem[2];
    unsigned int pos_offset = mem[3];
    unsigned int* input = (unsigned int*)mem[4];
    unsigned int* output = (unsigned int*)mem[5];
    unsigned int* freqs = (unsigned int*)mem[6];

    unsigned int half_dim = head_dim >> 1;

    for (unsigned int pos = 0; pos < seq_len; pos++) {
        unsigned int m = pos + pos_offset;
        unsigned int row = mul_u32(pos, head_dim);

        for (unsigned int i = 0; i < half_dim; i++) {
            fixed32 inv_freq = float_to_fixed(freqs[i]);
            fixed32 angle = mul_fixed(INT_TO_FIXED(m), inv_freq);
            fixed32 cos_v = cos_fixed(angle);
            fixed32 sin_v = sin_fixed(angle);

            unsigned int i0 = i << 1;
            unsigned int i1 = i0 + 1;

            fixed32 x0 = float_to_fixed(input[row + i0]);
            fixed32 x1 = float_to_fixed(input[row + i1]);

            output[row + i0] = fixed_to_float(sub_fixed(mul_fixed(x0, cos_v), mul_fixed(x1, sin_v)));
            output[row + i1] = fixed_to_float(add_fixed(mul_fixed(x0, sin_v), mul_fixed(x1, cos_v)));
        }
    }
}

// ============================================================
// Elementwise ops
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

void op_gate_mul() {
    unsigned int count = mem[1];
    unsigned int* a = (unsigned int*)mem[2];
    unsigned int* b = (unsigned int*)mem[3];
    unsigned int* out = (unsigned int*)mem[4];
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        fixed32 vb = float_to_fixed(b[i]);
        fixed32 silu_b = silu_fixed(vb);
        out[i] = fixed_to_float(mul_fixed(va, silu_b));
    }
}

// ============================================================
// MatMul DDR: A is float, B is pre-loaded Q16.16
// args: [M, K, N, addr_A, addr_B_q16, addr_C, addr_tmp]
// ============================================================
void op_matmul_ddr() {
    unsigned int M = mem[1];
    unsigned int K = mem[2];
    unsigned int N = mem[3];
    unsigned int* A = (unsigned int*)mem[4];
    fixed32* B = (fixed32*)mem[5];
    unsigned int* C = (unsigned int*)mem[6];
    fixed32* a_buf = (fixed32*)mem[7];

    unsigned int a_off = 0;
    unsigned int c_off = 0;

    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int k = 0; k < K; k++) {
            a_buf[k] = float_to_fixed(A[a_off + k]);
        }
        for (unsigned int j = 0; j < N; j++) {
            fixed32 sum = 0;
            unsigned int b_off = j;
            for (unsigned int k = 0; k < K; k++) {
                sum = add_fixed(sum, mul_fixed(a_buf[k], B[b_off]));
                b_off += N;
            }
            C[c_off + j] = fixed_to_float(sum);
        }
        a_off += K;
        c_off += N;
    }
}

// ============================================================
// RMSNorm DDR: gamma is pre-loaded Q16.16
// args: [batch, hidden, addr_in, addr_out, addr_gamma_q16]
// ============================================================
void op_rmsnorm_ddr() {
    unsigned int batch = mem[1];
    unsigned int hidden = mem[2];
    unsigned int* input = (unsigned int*)mem[3];
    unsigned int* output = (unsigned int*)mem[4];
    fixed32* gamma = (fixed32*)mem[5];
    fixed32 eps = 1;

    for (unsigned int b = 0; b < batch; b++) {
        unsigned int off = mul_u32(b, hidden);
        unsigned int* x = input + off;
        unsigned int* y = output + off;

        fixed32 sum_sq = 0;
        for (unsigned int i = 0; i < hidden; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            sum_sq = add_fixed(sum_sq, mul_fixed(xi, xi));
        }
        fixed32 rms = sqrt_fixed(add_fixed(div_fixed(sum_sq, INT_TO_FIXED(hidden)), eps));
        fixed32 inv_rms = div_fixed(FIXED_ONE, rms);

        for (unsigned int i = 0; i < hidden; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            fixed32 g = gamma[i];
            y[i] = fixed_to_float(mul_fixed(mul_fixed(xi, inv_rms), g));
        }
    }
}

// ============================================================
// Main dispatcher loop (multi-core)
// ============================================================
int main() {
    // Determine which core we are
    int core_id = ncl_getCoreID();       // 0-3 within cluster
    int cluster_id = ncl_getClusterID(); // 0-3

    // Global core index: 0..15
    // cluster_id * 4 + core_id, using shifts to avoid Mul32
    unsigned int core_index = (unsigned int)((cluster_id << 2) + core_id);

    // Point mem to this core's cmd block
    // DDR_BASE + core_index * 32, using shift: core_index << 5
    mem = (volatile unsigned int*)(DDR_BASE + (core_index << 5));

    // Signal ready
    mem[STATUS_ADDR] = 1;
    mem[WATCHDOG_ADDR] = 0;
    mem[0] = OP_NOP;

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

        mem[STATUS_ADDR] = 0;  // busy

        switch (op) {
            case OP_MATMUL:         op_matmul(); break;
            case OP_MATMUL_PARTIAL: op_matmul_partial(); break;
            case OP_RMSNORM:        op_rmsnorm(); break;
            case OP_SOFTMAX:        op_softmax(); break;
            case OP_SILU:           op_silu(); break;
            case OP_ROPE:           op_rope(); break;
            case OP_ELEM_ADD:       op_elem_add(); break;
            case OP_ELEM_MUL:       op_elem_mul(); break;
            case OP_GATE_MUL:       op_gate_mul(); break;
            case OP_MATMUL_DDR:     op_matmul_ddr(); break;
            case OP_RMSNORM_DDR:    op_rmsnorm_ddr(); break;
            default:
                mem[STATUS_ADDR] = 2;  // error
                mem[0] = OP_NOP;
                continue;
        }

        mem[STATUS_ADDR] = 1;  // done
        mem[0] = OP_NOP;       // ready for next
    }

    return 0;
}
