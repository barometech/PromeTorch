// rope_backward.cpp - Backward pass for Rotary Position Embedding on NMC4
// RoPE forward: x_rotated = x * cos + rotate_half(x) * sin
// RoPE backward: grad_input = grad_output * cos + rotate_half(grad_output) * sin
// (same formula because rotation is orthogonal)
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = rope_backward)
// [1] = seq_len
// [2] = head_dim
// [3] = addr_grad_output (gradient from next layer)
// [4] = addr_cos (precomputed cos values)
// [5] = addr_sin (precomputed sin values)
// [6] = addr_grad_input (output gradient)
// [7] = status (0 = busy, 1 = done)

void apply_rope_backward(
    volatile unsigned int* grad_output,
    volatile unsigned int* cos_vals,
    volatile unsigned int* sin_vals,
    volatile unsigned int* grad_input,
    unsigned int seq_len,
    unsigned int head_dim
) {
    unsigned int half_dim = head_dim >> 1;

    for (unsigned int pos = 0; pos < seq_len; pos++) {
        // For each position, apply inverse rotation
        // Since rotation is orthogonal, transpose = inverse
        // grad_x = grad_y * cos + rotate_half(grad_y) * sin

        for (unsigned int i = 0; i < half_dim; i++) {
            // Get cos/sin for this position and dimension
            fixed32 c = float_to_fixed(cos_vals[pos * half_dim + i]);
            fixed32 s = float_to_fixed(sin_vals[pos * half_dim + i]);

            // grad_output indices
            unsigned int idx1 = pos * head_dim + i;           // first half
            unsigned int idx2 = pos * head_dim + half_dim + i; // second half

            fixed32 g1 = float_to_fixed(grad_output[idx1]);
            fixed32 g2 = float_to_fixed(grad_output[idx2]);

            // rotate_half for grad: [-g2, g1]
            // grad_input = grad_output * cos + rotate_half(grad_output) * sin
            // grad_input[i] = g1 * c + (-g2) * s = g1*c - g2*s
            // grad_input[i+half] = g2 * c + g1 * s

            fixed32 gi1 = sub_fixed(mul_fixed(g1, c), mul_fixed(g2, s));
            fixed32 gi2 = add_fixed(mul_fixed(g2, c), mul_fixed(g1, s));

            grad_input[idx1] = fixed_to_float(gi1);
            grad_input[idx2] = fixed_to_float(gi2);
        }
    }
}

int main() {
    mem[7] = 0;

    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[0] == 255) { mem[7] = 1; break; }
        if (mem[0] == 1) {
            mem[7] = 0;  // busy

            unsigned int seq_len = mem[1];
            unsigned int head_dim = mem[2];
            volatile unsigned int* grad_output = (volatile unsigned int*)mem[3];
            volatile unsigned int* cos_vals = (volatile unsigned int*)mem[4];
            volatile unsigned int* sin_vals = (volatile unsigned int*)mem[5];
            volatile unsigned int* grad_input = (volatile unsigned int*)mem[6];

            apply_rope_backward(grad_output, cos_vals, sin_vals, grad_input, seq_len, head_dim);

            mem[7] = 1;  // done
            mem[0] = 0;
        }
    }

    return 0;
}
