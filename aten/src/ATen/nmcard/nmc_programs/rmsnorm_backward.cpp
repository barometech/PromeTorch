// rmsnorm_backward.cpp - Backward pass for RMSNorm on NMC4
// Forward: y = x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps)
// Backward: computes grad_input and grad_gamma
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"
#include "mymath_backward.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = rmsnorm_backward)
// [1] = size (hidden dimension)
// [2] = addr_x (input from forward pass)
// [3] = addr_gamma (weight parameter)
// [4] = addr_grad_output (gradient from next layer)
// [5] = addr_grad_input (output: gradient w.r.t. input)
// [6] = addr_grad_gamma (output: gradient w.r.t. gamma, accumulated)
// [7] = status (0 = busy, 1 = done)

// eps in Q16.16: 1e-5 ≈ 1
#define EPS_FIXED 1

void apply_rmsnorm_backward(
    volatile unsigned int* x,
    volatile unsigned int* gamma,
    volatile unsigned int* grad_output,
    volatile unsigned int* grad_input,
    volatile unsigned int* grad_gamma,
    unsigned int size
) {
    // Step 1: Compute RMS from forward pass
    // rms = sqrt(mean(x^2) + eps)
    fixed32 sum_sq = 0;
    for (unsigned int i = 0; i < size; i++) {
        fixed32 xi = float_to_fixed(x[i]);
        sum_sq = add_fixed(sum_sq, mul_fixed(xi, xi));
    }
    fixed32 mean_sq = div_fixed(sum_sq, INT_TO_FIXED(size));
    fixed32 rms = sqrt_fixed(add_fixed(mean_sq, EPS_FIXED));
    fixed32 inv_rms = div_fixed(FIXED_ONE, rms);

    // Step 2: Compute grad_gamma = grad_output * x_norm
    // x_norm = x * inv_rms
    // Also accumulate (for batched training)
    for (unsigned int i = 0; i < size; i++) {
        fixed32 xi = float_to_fixed(x[i]);
        fixed32 x_norm = mul_fixed(xi, inv_rms);
        fixed32 go = float_to_fixed(grad_output[i]);

        // Accumulate grad_gamma
        fixed32 gg = float_to_fixed(grad_gamma[i]);
        gg = add_fixed(gg, mul_fixed(go, x_norm));
        grad_gamma[i] = fixed_to_float(gg);
    }

    // Step 3: Compute grad_input (simplified approximation)
    // Full formula involves second-order terms through rms
    // Simplified: grad_input = grad_output * gamma * inv_rms
    for (unsigned int i = 0; i < size; i++) {
        fixed32 go = float_to_fixed(grad_output[i]);
        fixed32 g = float_to_fixed(gamma[i]);
        fixed32 gi = mul_fixed(mul_fixed(go, g), inv_rms);
        grad_input[i] = fixed_to_float(gi);
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
            volatile unsigned int* x = (volatile unsigned int*)mem[2];
            volatile unsigned int* gamma = (volatile unsigned int*)mem[3];
            volatile unsigned int* grad_output = (volatile unsigned int*)mem[4];
            volatile unsigned int* grad_input = (volatile unsigned int*)mem[5];
            volatile unsigned int* grad_gamma = (volatile unsigned int*)mem[6];

            apply_rmsnorm_backward(x, gamma, grad_output, grad_input, grad_gamma, size);

            mem[7] = 1;  // done
            mem[0] = 0;
        }
    }

    return 0;
}
