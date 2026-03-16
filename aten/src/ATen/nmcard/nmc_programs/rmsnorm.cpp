// RMSNorm for NMC4 (used in Llama/TinyLlama)
// y = x * gamma / sqrt(mean(x^2) + eps)
// Simpler than LayerNorm: no mean subtraction, no beta
// All custom, no library calls - uses fixed-point Q16.16

#include "mymath.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = rmsnorm)
// [1] = batch_size (number of vectors)
// [2] = hidden_size (dimension to normalize over)
// [3] = addr_input
// [4] = addr_output
// [5] = addr_gamma (scale weights)
// [6] = reserved
// [7] = status (0 = busy, 1 = done)
// [8] = epsilon as float bits (default: 1e-6)

// Epsilon in fixed-point
#define DEFAULT_EPS 1

void rmsnorm(unsigned int* input, unsigned int* output,
             unsigned int* gamma,
             unsigned int batch_size, unsigned int hidden_size,
             fixed32 eps) {

    for (unsigned int b = 0; b < batch_size; b++) {
        unsigned int offset = mul_u32(b, hidden_size);
        unsigned int* x = input + offset;
        unsigned int* y = output + offset;

        // Step 1: Compute mean of squares = sum(x^2) / n
        fixed32 sum_sq = 0;
        for (unsigned int i = 0; i < hidden_size; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            fixed32 xi_sq = mul_fixed(xi, xi);
            sum_sq = add_fixed(sum_sq, xi_sq);
        }
        fixed32 mean_sq = div_fixed(sum_sq, INT_TO_FIXED(hidden_size));

        // Step 2: RMS = sqrt(mean_sq + eps)
        fixed32 rms = sqrt_fixed(add_fixed(mean_sq, eps));

        // Step 3: inv_rms = 1 / rms
        fixed32 inv_rms = div_fixed(FIXED_ONE, rms);

        // Step 4: Normalize and apply gamma
        // y = x * gamma / rms = x * inv_rms * gamma
        for (unsigned int i = 0; i < hidden_size; i++) {
            fixed32 xi = float_to_fixed(x[i]);
            fixed32 g = float_to_fixed(gamma[i]);

            // normalized = xi * inv_rms
            fixed32 normalized = mul_fixed(xi, inv_rms);

            // result = normalized * gamma
            fixed32 result = mul_fixed(normalized, g);

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

            // Epsilon
            fixed32 eps = DEFAULT_EPS;
            if (mem[8] != 0) {
                eps = float_to_fixed(mem[8]);
                if (eps < 1) eps = 1;
            }

            rmsnorm(input, output, gamma, batch_size, hidden_size, eps);

            mem[7] = 1;
            mem[0] = 0;
        }
    }

    return 0;
}
