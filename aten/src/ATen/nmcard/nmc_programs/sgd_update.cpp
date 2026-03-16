// sgd_update.cpp - SGD optimizer update on NMC4
// weight = weight - lr * grad
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"
#include "mymath_backward.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = sgd, 2 = sgd_momentum)
// [1] = count (number of weights)
// [2] = addr_weights
// [3] = addr_grads
// [4] = addr_velocity (for momentum, 0 if unused)
// [5] = learning_rate (Q16.16 as uint32)
// [6] = momentum (Q16.16 as uint32, e.g. 0.9 = 58982)
// [7] = status (0 = busy, 1 = done)

void sgd_step(
    volatile unsigned int* weights,
    volatile unsigned int* grads,
    fixed32 lr,
    unsigned int count
) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 w = float_to_fixed(weights[i]);
        fixed32 g = float_to_fixed(grads[i]);
        fixed32 update = mul_fixed(lr, g);
        w = sub_fixed(w, update);
        weights[i] = fixed_to_float(w);
    }
}

void sgd_momentum_step(
    volatile unsigned int* weights,
    volatile unsigned int* grads,
    volatile unsigned int* velocity,
    fixed32 lr,
    fixed32 momentum,
    unsigned int count
) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 w = float_to_fixed(weights[i]);
        fixed32 g = float_to_fixed(grads[i]);
        fixed32 v = float_to_fixed(velocity[i]);

        // v = momentum * v + grad
        v = add_fixed(mul_fixed(momentum, v), g);

        // w = w - lr * v
        w = sub_fixed(w, mul_fixed(lr, v));

        weights[i] = fixed_to_float(w);
        velocity[i] = fixed_to_float(v);
    }
}

int main() {
    mem[7] = 0;

    while (1) {
        unsigned int cmd = mem[0];
        // OP_EXIT = 255 - безопасный выход!
        if (cmd == 255) { mem[7] = 1; break; }
        if (cmd == 0) continue;

        mem[7] = 0;  // busy

        unsigned int count = mem[1];
        volatile unsigned int* weights = (volatile unsigned int*)mem[2];
        volatile unsigned int* grads = (volatile unsigned int*)mem[3];
        volatile unsigned int* velocity = (volatile unsigned int*)mem[4];
        fixed32 lr = (fixed32)mem[5];
        fixed32 momentum = (fixed32)mem[6];

        if (cmd == 1) {
            sgd_step(weights, grads, lr, count);
        } else if (cmd == 2) {
            sgd_momentum_step(weights, grads, velocity, lr, momentum, count);
        }

        mem[7] = 1;  // done
        mem[0] = 0;
    }

    return 0;
}
