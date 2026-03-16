// LayerNorm for NMC4
// y = (x - mean) / sqrt(variance + eps) * gamma + beta
// All custom, no library calls - uses fixed-point Q16.16

#include "mymath.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = layernorm)
// [1] = batch_size (number of vectors)
// [2] = hidden_size (dimension to normalize over)
// [3] = addr_input
// [4] = addr_output
// [5] = addr_gamma (scale)
// [6] = addr_beta (bias)
// [7] = status (0 = busy, 1 = done)
// [8] = epsilon as float bits (default: 1e-5 = 0x3727c5ac)

// Epsilon in fixed-point (1e-5 << 16 ≈ 0.655)
#define DEFAULT_EPS 1

void layernorm(unsigned int* input, unsigned int* output,
               unsigned int* gamma, unsigned int* beta,
               unsigned int batch_size, unsigned int hidden_size,
               fixed32 eps) {

    for (unsigned int b = 0; b < batch_size; b++) {
        unsigned int offset = mul_u32(b, hidden_size);
        unsigned int* x = input + offset;
        unsigned int* y = output + offset;

        // Step 1: Compute mean
        fixed32 sum = 0;
        for (unsigned int i = 0; i < hidden_size; i++) {
            sum = add_fixed(sum, float_to_fixed(x[i]));
        }
        fixed32 mean = div_fixed(sum, INT_TO_FIXED(hidden_size));

        // Step 2: Compute variance = E[(x - mean)^2]
        fixed32 var_sum = 0;
        for (unsigned int i = 0; i < hidden_size; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            fixed32 diff = sub_fixed(xi, mean);
            fixed32 diff_sq = mul_fixed(diff, diff);
            var_sum = add_fixed(var_sum, diff_sq);
        }
        fixed32 variance = div_fixed(var_sum, INT_TO_FIXED(hidden_size));

        // Step 3: inv_std = 1 / sqrt(variance + eps)
        fixed32 var_eps = add_fixed(variance, eps);
        fixed32 std = sqrt_fixed(var_eps);
        fixed32 inv_std = div_fixed(FIXED_ONE, std);

        // Step 4: Normalize and apply gamma/beta
        for (unsigned int i = 0; i < hidden_size; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            fixed32 normalized = mul_fixed(sub_fixed(xi, mean), inv_std);

            // Apply scale (gamma) and bias (beta)
            fixed32 g = float_to_fixed(gamma[i]);
            fixed32 bias = float_to_fixed(beta[i]);
            fixed32 result = add_fixed(mul_fixed(normalized, g), bias);

            y[i] = fixed_to_float(result);
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
            unsigned int hidden_size = mem[2];
            unsigned int* input = (unsigned int*)mem[3];
            unsigned int* output = (unsigned int*)mem[4];
            unsigned int* gamma = (unsigned int*)mem[5];
            unsigned int* beta = (unsigned int*)mem[6];

            // Epsilon: use provided or default
            fixed32 eps = DEFAULT_EPS;
            if (mem[8] != 0) {
                eps = float_to_fixed(mem[8]);
                if (eps < 1) eps = 1;  // minimum epsilon
            }

            layernorm(input, output, gamma, beta, batch_size, hidden_size, eps);

            mem[7] = 1;
            mem[0] = 0;
        }
    }

    return 0;
}
