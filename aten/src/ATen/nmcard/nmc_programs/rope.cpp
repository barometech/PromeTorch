// RoPE (Rotary Position Embedding) for NMC4 - used in Llama attention
//
// For position m and dimension pair (2i, 2i+1):
//   theta_i = base^(-2i/d)
//   x'[2i]   = x[2i] * cos(m*theta_i) - x[2i+1] * sin(m*theta_i)
//   x'[2i+1] = x[2i] * sin(m*theta_i) + x[2i+1] * cos(m*theta_i)
//
// All custom, no library calls - uses fixed-point Q16.16

#include "mymath.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = rope)
// [1] = seq_len (number of positions)
// [2] = head_dim (dimension of each head, must be even)
// [3] = position_offset (starting position for this batch)
// [4] = addr_input (shape: [seq_len, head_dim])
// [5] = addr_output
// [6] = addr_freqs (precomputed frequencies, shape: [head_dim/2])
// [7] = status (0 = busy, 1 = done)
// [8] = base (default 10000) as float bits

// Precompute inv_freq[i] = 1 / (base^(2i/d))
// This is done on host for efficiency

void apply_rope(unsigned int* input, unsigned int* output,
                unsigned int* freqs,
                unsigned int seq_len, unsigned int head_dim,
                unsigned int pos_offset) {

    unsigned int half_dim = head_dim >> 1;  // head_dim / 2

    for (unsigned int pos = 0; pos < seq_len; pos++) {
        unsigned int m = pos + pos_offset;  // actual position

        unsigned int row_offset = mul_u32(pos, head_dim);
        unsigned int* x = input + row_offset;
        unsigned int* y = output + row_offset;

        for (unsigned int i = 0; i < half_dim; i++) {
            // Get the frequency for this dimension pair
            // freqs[i] = 1 / base^(2i/head_dim) in float
            fixed32 inv_freq = float_to_fixed(freqs[i]);

            // angle = m * inv_freq
            fixed32 angle = mul_fixed(INT_TO_FIXED(m), inv_freq);

            // Compute sin and cos
            fixed32 cos_val = cos_fixed(angle);
            fixed32 sin_val = sin_fixed(angle);

            // Get input pair
            unsigned int idx0 = i << 1;       // 2*i
            unsigned int idx1 = idx0 + 1;     // 2*i + 1

            fixed32 x0 = float_to_fixed(x[idx0]);
            fixed32 x1 = float_to_fixed(x[idx1]);

            // Apply rotation
            // y[2i]   = x[2i] * cos - x[2i+1] * sin
            // y[2i+1] = x[2i] * sin + x[2i+1] * cos
            fixed32 y0 = sub_fixed(mul_fixed(x0, cos_val), mul_fixed(x1, sin_val));
            fixed32 y1 = add_fixed(mul_fixed(x0, sin_val), mul_fixed(x1, cos_val));

            y[idx0] = fixed_to_float(y0);
            y[idx1] = fixed_to_float(y1);
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
            unsigned int pos_offset = mem[3];
            unsigned int* input = (unsigned int*)mem[4];
            unsigned int* output = (unsigned int*)mem[5];
            unsigned int* freqs = (unsigned int*)mem[6];

            apply_rope(input, output, freqs, seq_len, head_dim, pos_offset);

            mem[7] = 1;
            mem[0] = 0;
        }
    }

    return 0;
}
