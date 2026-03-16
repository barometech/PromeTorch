// SiLU (Swish) activation for NMC4 - used in Llama FFN
// SiLU(x) = x * sigmoid(x)
// All custom, no library calls - uses fixed-point Q16.16

#include "mymath.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = silu)
// [1] = count (number of elements)
// [2] = addr_input
// [3] = addr_output
// [4] = reserved
// [5] = reserved
// [6] = reserved
// [7] = status (0 = busy, 1 = done)

void apply_silu(unsigned int* input, unsigned int* output, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 x = float_to_fixed(input[i]);
        fixed32 result = silu_fixed(x);
        output[i] = fixed_to_float(result);
    }
}

int main() {
    mem[7] = 0;

    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[0] == 255) { mem[7] = 1; break; }
        if (mem[0] == 1) {
            mem[7] = 0;

            unsigned int count = mem[1];
            unsigned int* input = (unsigned int*)mem[2];
            unsigned int* output = (unsigned int*)mem[3];

            apply_silu(input, output, count);

            mem[7] = 1;
            mem[0] = 0;
        }
    }

    return 0;
}
