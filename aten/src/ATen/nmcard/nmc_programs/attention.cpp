// Scaled Dot-Product Attention for NMC4
//
// Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(d_k)) @ V
//
// For causal (autoregressive) models, applies mask where
// positions can only attend to previous positions.
//
// All custom, no library calls - uses fixed-point Q16.16

#include "mymath.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = attention)
// [1] = seq_len (sequence length)
// [2] = head_dim (dimension per head)
// [3] = addr_Q (shape: [seq_len, head_dim])
// [4] = addr_K (shape: [seq_len, head_dim])
// [5] = addr_V (shape: [seq_len, head_dim])
// [6] = addr_output (shape: [seq_len, head_dim])
// [7] = status (0 = busy, 1 = done)
// [8] = flags (bit 0: causal mask)
// [9] = addr_scratch (temp buffer for attention weights, size: seq_len * seq_len)

// Note: This is single-head attention.
// Multi-head is done by calling this kernel multiple times.

void attention(unsigned int* Q, unsigned int* K, unsigned int* V,
               unsigned int* output, unsigned int* scratch,
               unsigned int seq_len, unsigned int head_dim,
               int causal) {

    // scale = 1 / sqrt(head_dim)
    fixed32 head_dim_fixed = INT_TO_FIXED(head_dim);
    fixed32 scale = div_fixed(FIXED_ONE, sqrt_fixed(head_dim_fixed));

    // Step 1: Compute attention scores: S = Q @ K^T
    // S[i,j] = sum_k(Q[i,k] * K[j,k]) / sqrt(d)
    for (unsigned int i = 0; i < seq_len; i++) {
        unsigned int q_offset = mul_u32(i, head_dim);

        for (unsigned int j = 0; j < seq_len; j++) {
            // Causal mask: if j > i, mask it out
            if (causal && j > i) {
                // Very negative value -> softmax will make it ~0
                // -10 in Q16.16 = -655360
                scratch[mul_u32(i, seq_len) + j] = (unsigned int)(-655360);
                continue;
            }

            unsigned int k_offset = mul_u32(j, head_dim);

            // Dot product Q[i] @ K[j]
            fixed32 dot = 0;
            for (unsigned int k = 0; k < head_dim; k++) {
                fixed32 q_val = float_to_fixed(Q[q_offset + k]);
                fixed32 k_val = float_to_fixed(K[k_offset + k]);
                dot = add_fixed(dot, mul_fixed(q_val, k_val));
            }

            // Scale
            fixed32 score = mul_fixed(dot, scale);

            // Store as fixed (will apply softmax next)
            scratch[mul_u32(i, seq_len) + j] = (unsigned int)score;
        }
    }

    // Step 2: Softmax over each row
    for (unsigned int i = 0; i < seq_len; i++) {
        unsigned int row_offset = mul_u32(i, seq_len);

        // Find max for numerical stability
        fixed32 max_val = (fixed32)scratch[row_offset];
        unsigned int len = causal ? (i + 1) : seq_len;

        for (unsigned int j = 1; j < len; j++) {
            fixed32 val = (fixed32)scratch[row_offset + j];
            if (val > max_val) max_val = val;
        }

        // Compute exp(score - max) and sum
        fixed32 exp_sum = 0;
        for (unsigned int j = 0; j < seq_len; j++) {
            if (causal && j > i) {
                scratch[row_offset + j] = 0;  // masked position
                continue;
            }

            fixed32 score = (fixed32)scratch[row_offset + j];
            fixed32 shifted = sub_fixed(score, max_val);
            fixed32 exp_val = exp_fixed_lut(shifted);

            scratch[row_offset + j] = (unsigned int)exp_val;
            exp_sum = add_fixed(exp_sum, exp_val);
        }

        // Normalize
        if (exp_sum == 0) exp_sum = 1;
        for (unsigned int j = 0; j < seq_len; j++) {
            if (causal && j > i) continue;

            fixed32 exp_val = (fixed32)scratch[row_offset + j];
            fixed32 attn_weight = div_fixed(exp_val, exp_sum);
            scratch[row_offset + j] = (unsigned int)attn_weight;
        }
    }

    // Step 3: Output = Attention_weights @ V
    // output[i,k] = sum_j(attn[i,j] * V[j,k])
    for (unsigned int i = 0; i < seq_len; i++) {
        unsigned int out_offset = mul_u32(i, head_dim);
        unsigned int attn_row = mul_u32(i, seq_len);

        for (unsigned int k = 0; k < head_dim; k++) {
            fixed32 sum = 0;

            for (unsigned int j = 0; j < seq_len; j++) {
                fixed32 attn_weight = (fixed32)scratch[attn_row + j];
                fixed32 v_val = float_to_fixed(V[mul_u32(j, head_dim) + k]);
                sum = add_fixed(sum, mul_fixed(attn_weight, v_val));
            }

            output[out_offset + k] = fixed_to_float(sum);
        }
    }
}

int main() {
    mem[7] = 0;

    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[0] == 255) { mem[7] = 1; break; }
        if (mem[0] == 1) {
            mem[7] = 0;

            unsigned int seq_len = mem[1];
            unsigned int head_dim = mem[2];
            unsigned int* Q = (unsigned int*)mem[3];
            unsigned int* K = (unsigned int*)mem[4];
            unsigned int* V = (unsigned int*)mem[5];
            unsigned int* output = (unsigned int*)mem[6];
            int causal = mem[8] & 1;
            unsigned int* scratch = (unsigned int*)mem[9];

            attention(Q, K, V, output, scratch, seq_len, head_dim, causal);

            mem[7] = 1;
            mem[0] = 0;
        }
    }

    return 0;
}
