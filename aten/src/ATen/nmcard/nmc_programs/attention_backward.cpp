// attention_backward.cpp - Backward pass for Attention on NMC4
// Forward: attn = softmax(Q @ K^T / sqrt(d)) @ V
// Backward: computes dQ, dK, dV from dO (grad w.r.t. output)
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"
#include "mymath_backward.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout (extended to 16 words):
// [0] = cmd (1 = attention_backward)
// [1] = seq_len
// [2] = head_dim
// [3] = addr_Q
// [4] = addr_K
// [5] = addr_V
// [6] = addr_attn_weights (softmax output from forward)
// [7] = addr_grad_output (dO)
// [8] = addr_grad_Q (output)
// [9] = addr_grad_K (output)
// [10] = addr_grad_V (output)
// [11] = status (0 = busy, 1 = done)

// Temporary buffer for intermediate gradients
static fixed32 grad_scores[64];  // For small seq_len
static fixed32 d_attn[64];

void apply_attention_backward(
    volatile unsigned int* Q,
    volatile unsigned int* K,
    volatile unsigned int* V,
    volatile unsigned int* attn_weights,  // [seq_len, seq_len]
    volatile unsigned int* grad_O,        // [seq_len, head_dim]
    volatile unsigned int* grad_Q,        // [seq_len, head_dim]
    volatile unsigned int* grad_K,        // [seq_len, head_dim]
    volatile unsigned int* grad_V,        // [seq_len, head_dim]
    unsigned int seq_len,
    unsigned int head_dim
) {
    // Scale factor: 1/sqrt(head_dim)
    fixed32 scale = div_fixed(FIXED_ONE, sqrt_fixed(INT_TO_FIXED(head_dim)));

    // Step 1: dV = attn_weights^T @ dO
    // dV[j, d] = sum_i(attn_weights[i,j] * dO[i,d])
    unsigned int idx_j = 0;  // j * head_dim
    for (unsigned int j = 0; j < seq_len; j++) {
        for (unsigned int d = 0; d < head_dim; d++) {
            fixed32 sum = 0;
            unsigned int idx_i = 0;  // i * seq_len for attn, i * head_dim for grad_O
            unsigned int idx_io = 0;
            for (unsigned int i = 0; i < seq_len; i++) {
                fixed32 a = float_to_fixed(attn_weights[idx_i + j]);
                fixed32 go = float_to_fixed(grad_O[idx_io + d]);
                sum = add_fixed(sum, mul_fixed(a, go));
                idx_i += seq_len;
                idx_io += head_dim;
            }
            grad_V[idx_j + d] = fixed_to_float(sum);
        }
        idx_j += head_dim;
    }

    // Step 2: Compute gradients for Q and K
    unsigned int idx_i_s = 0;  // i * seq_len
    unsigned int idx_i_h = 0;  // i * head_dim
    for (unsigned int i = 0; i < seq_len; i++) {
        // Compute d_attn[j] = sum_d(dO[i,d] * V[j,d])
        unsigned int idx_j_h = 0;  // j * head_dim
        for (unsigned int j = 0; j < seq_len; j++) {
            fixed32 sum = 0;
            for (unsigned int d = 0; d < head_dim; d++) {
                fixed32 go = float_to_fixed(grad_O[idx_i_h + d]);
                fixed32 v = float_to_fixed(V[idx_j_h + d]);
                sum = add_fixed(sum, mul_fixed(go, v));
            }
            d_attn[j] = sum;
            idx_j_h += head_dim;
        }

        // Softmax backward for row i
        fixed32 dot = 0;
        for (unsigned int j = 0; j < seq_len; j++) {
            fixed32 a = float_to_fixed(attn_weights[idx_i_s + j]);
            dot = add_fixed(dot, mul_fixed(a, d_attn[j]));
        }

        for (unsigned int j = 0; j < seq_len; j++) {
            fixed32 a = float_to_fixed(attn_weights[idx_i_s + j]);
            grad_scores[j] = mul_fixed(a, sub_fixed(d_attn[j], dot));
        }

        // dQ[i, :] = sum_j(d_scores[j] * K[j,:]) * scale
        for (unsigned int d = 0; d < head_dim; d++) {
            fixed32 sum = 0;
            unsigned int idx_j_k = 0;
            for (unsigned int j = 0; j < seq_len; j++) {
                fixed32 k = float_to_fixed(K[idx_j_k + d]);
                sum = add_fixed(sum, mul_fixed(grad_scores[j], k));
                idx_j_k += head_dim;
            }
            grad_Q[idx_i_h + d] = fixed_to_float(mul_fixed(sum, scale));
        }

        // Accumulate dK: dK[j, :] += d_scores[j] * Q[i,:] * scale
        unsigned int idx_j_dk = 0;
        for (unsigned int j = 0; j < seq_len; j++) {
            fixed32 ds = mul_fixed(grad_scores[j], scale);
            for (unsigned int d = 0; d < head_dim; d++) {
                fixed32 q = float_to_fixed(Q[idx_i_h + d]);
                fixed32 dk = float_to_fixed(grad_K[idx_j_dk + d]);
                dk = add_fixed(dk, mul_fixed(ds, q));
                grad_K[idx_j_dk + d] = fixed_to_float(dk);
            }
            idx_j_dk += head_dim;
        }

        idx_i_s += seq_len;
        idx_i_h += head_dim;
    }
}

int main() {
    mem[11] = 0;

    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[0] == 255) { mem[11] = 1; break; }
        if (mem[0] == 1) {
            mem[11] = 0;  // busy

            unsigned int seq_len = mem[1];
            unsigned int head_dim = mem[2];
            volatile unsigned int* Q = (volatile unsigned int*)mem[3];
            volatile unsigned int* K = (volatile unsigned int*)mem[4];
            volatile unsigned int* V = (volatile unsigned int*)mem[5];
            volatile unsigned int* attn_weights = (volatile unsigned int*)mem[6];
            volatile unsigned int* grad_O = (volatile unsigned int*)mem[7];
            volatile unsigned int* grad_Q = (volatile unsigned int*)mem[8];
            volatile unsigned int* grad_K = (volatile unsigned int*)mem[9];
            volatile unsigned int* grad_V = (volatile unsigned int*)mem[10];

            // Zero out grad_K for accumulation
            unsigned int total = 0;
            for (unsigned int i = 0; i < seq_len; i++) {
                for (unsigned int d = 0; d < head_dim; d++) {
                    grad_K[total] = 0;
                    total++;
                }
            }

            apply_attention_backward(Q, K, V, attn_weights, grad_O,
                                     grad_Q, grad_K, grad_V, seq_len, head_dim);

            mem[11] = 1;  // done
            mem[0] = 0;
        }
    }

    return 0;
}
