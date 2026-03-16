// cross_entropy.cpp - Cross-entropy loss and backward on NMC4
// loss = -log(pred[target])
// grad = pred - one_hot(target)
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"
#include "mymath_backward.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = loss, 2 = backward, 3 = both)
// [1] = vocab_size
// [2] = target_class
// [3] = addr_pred (softmax output)
// [4] = addr_grad (output gradient, only for backward)
// [5] = loss_value (output, Q16.16)
// [6] = reserved
// [7] = status (0 = busy, 1 = done)

int main() {
    mem[7] = 0;

    while (1) {
        unsigned int cmd = mem[0];
        // OP_EXIT = 255 - безопасный выход!
        if (cmd == 255) { mem[7] = 1; break; }
        if (cmd == 0) continue;

        mem[7] = 0;  // busy

        unsigned int vocab_size = mem[1];
        int target_class = (int)mem[2];
        volatile unsigned int* pred = (volatile unsigned int*)mem[3];
        volatile unsigned int* grad = (volatile unsigned int*)mem[4];

        if (cmd == 1 || cmd == 3) {
            // Compute cross-entropy loss = -log(pred[target])
            fixed32 p = float_to_fixed(pred[target_class]);

            // Clamp to avoid log(0)
            if (p < 655) p = 655;  // ~0.01 in Q16.16

            // log(p) approximation: log(p) ≈ 2 * (p - 1) / (p + 1)
            fixed32 num = sub_fixed(p, FIXED_ONE);
            fixed32 denom = add_fixed(p, FIXED_ONE);
            fixed32 log_approx = mul_fixed(num << 1, div_fixed(FIXED_ONE, denom));

            // loss = -log(p)
            fixed32 loss = -log_approx;
            mem[5] = (unsigned int)loss;
        }

        if (cmd == 2 || cmd == 3) {
            // Backward: grad[i] = pred[i] - (i == target ? 1 : 0)
            for (unsigned int i = 0; i < vocab_size; i++) {
                fixed32 p = float_to_fixed(pred[i]);
                fixed32 g;
                if ((int)i == target_class) {
                    g = sub_fixed(p, FIXED_ONE);
                } else {
                    g = p;
                }
                grad[i] = fixed_to_float(g);
            }
        }

        mem[7] = 1;  // done
        mem[0] = 0;
    }

    return 0;
}
