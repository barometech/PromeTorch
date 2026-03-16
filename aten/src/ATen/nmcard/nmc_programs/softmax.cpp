// Softmax for NMC4
// y[i] = exp(x[i] - max) / sum(exp(x - max))
// Numerical stability: subtract max before exp
// All custom, no library calls - uses fixed-point Q16.16

#include "mymath.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = softmax)
// [1] = batch_size (number of vectors)
// [2] = dim_size (dimension to apply softmax)
// [3] = addr_input
// [4] = addr_output
// [5] = axis (0 = last dim, 1 = specified - for now always last)
// [6] = reserved
// [7] = status (0 = busy, 1 = done)

void softmax(unsigned int* input, unsigned int* output,
             unsigned int batch_size, unsigned int dim_size) {

    for (unsigned int b = 0; b < batch_size; b++) {
        unsigned int offset = mul_u32(b, dim_size);
        unsigned int* x = input + offset;
        unsigned int* y = output + offset;

        // Step 1: Find max for numerical stability
        fixed32 max_val = float_to_fixed(x[0]);
        for (unsigned int i = 1; i < dim_size; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            if (xi > max_val) max_val = xi;
        }

        // Step 2: Compute exp(x - max) and sum
        fixed32 exp_sum = 0;

        // Temporary storage for exp values (reuse output buffer)
        for (unsigned int i = 0; i < dim_size; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            fixed32 shifted = sub_fixed(xi, max_val);

            // Use LUT-based exp for better accuracy
            fixed32 exp_val = exp_fixed_lut(shifted);

            // Store temporarily as fixed-point in output
            y[i] = (unsigned int)exp_val;
            exp_sum = add_fixed(exp_sum, exp_val);
        }

        // Step 3: Normalize by sum
        if (exp_sum == 0) exp_sum = 1;  // prevent div by zero

        for (unsigned int i = 0; i < dim_size; i++) {
            fixed32 exp_val = (fixed32)y[i];  // retrieve stored exp
            fixed32 softmax_val = div_fixed(exp_val, exp_sum);

            // Convert back to float
            y[i] = fixed_to_float(softmax_val);
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

            unsigned int batch_size = mem[1];
            unsigned int dim_size = mem[2];
            unsigned int* input = (unsigned int*)mem[3];
            unsigned int* output = (unsigned int*)mem[4];

            softmax(input, output, batch_size, dim_size);

            mem[7] = 1;
            mem[0] = 0;
        }
    }

    return 0;
}
