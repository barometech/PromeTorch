// silu_backward.cpp - Backward pass for SiLU activation on NMC4
// SiLU(x) = x * sigmoid(x)
// SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"
#include "mymath_backward.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = silu_backward)
// [1] = count (number of elements)
// [2] = addr_input (forward input x)
// [3] = addr_grad_output (gradient from next layer)
// [4] = addr_grad_input (output gradient)
// [5] = reserved
// [6] = reserved
// [7] = status (0 = busy, 1 = done)

void apply_silu_backward(
    volatile unsigned int* input,
    volatile unsigned int* grad_output,
    volatile unsigned int* grad_input,
    unsigned int count
) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 x = float_to_fixed(input[i]);
        fixed32 grad_out = float_to_fixed(grad_output[i]);
        fixed32 grad_in = silu_backward(x, grad_out);
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

            unsigned int count = mem[1];
            volatile unsigned int* input = (volatile unsigned int*)mem[2];
            volatile unsigned int* grad_output = (volatile unsigned int*)mem[3];
            volatile unsigned int* grad_input = (volatile unsigned int*)mem[4];

            apply_silu_backward(input, grad_output, grad_input, count);

            mem[7] = 1;  // done
            mem[0] = 0;
        }
    }

    return 0;
}
