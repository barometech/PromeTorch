// softmax_backward.cpp - Backward pass for Softmax on NMC4
// For softmax output y: grad_input_i = y_i * (grad_output_i - sum_j(y_j * grad_output_j))
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"
#include "mymath_backward.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = softmax_backward)
// [1] = size (number of elements)
// [2] = addr_y (softmax output from forward pass)
// [3] = addr_grad_output (gradient from next layer)
// [4] = addr_grad_input (output gradient)
// [5] = reserved
// [6] = reserved
// [7] = status (0 = busy, 1 = done)

void apply_softmax_backward(
    volatile unsigned int* y,
    volatile unsigned int* grad_output,
    volatile unsigned int* grad_input,
    unsigned int size
) {
    // Step 1: Compute dot product sum_j(y_j * grad_output_j)
    fixed32 dot = 0;
    for (unsigned int i = 0; i < size; i++) {
        fixed32 yi = float_to_fixed(y[i]);
        fixed32 gi = float_to_fixed(grad_output[i]);
        dot = add_fixed(dot, mul_fixed(yi, gi));
    }

    // Step 2: grad_input_i = y_i * (grad_output_i - dot)
    for (unsigned int i = 0; i < size; i++) {
        fixed32 yi = float_to_fixed(y[i]);
        fixed32 gi = float_to_fixed(grad_output[i]);
        fixed32 diff = sub_fixed(gi, dot);
        fixed32 grad_in = mul_fixed(yi, diff);
        grad_input[i] = fixed_to_float(grad_in);
    }
}

int main() {
    mem[7] = 0;

    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[0] == 255) { mem[7] = 1; break; }
        if (mem[0] == 1) {
            mem[7] = 0;  // busy

            unsigned int size = mem[1];
            volatile unsigned int* y = (volatile unsigned int*)mem[2];
            volatile unsigned int* grad_output = (volatile unsigned int*)mem[3];
            volatile unsigned int* grad_input = (volatile unsigned int*)mem[4];

            apply_softmax_backward(y, grad_output, grad_input, size);

            mem[7] = 1;  // done
            mem[0] = 0;
        }
    }

    return 0;
}
