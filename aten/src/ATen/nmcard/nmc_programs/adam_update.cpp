// adam_update.cpp - Adam optimizer on NMC4
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// m_hat = m / (1 - beta1^t)  (bias correction)
// v_hat = v / (1 - beta2^t)  (bias correction)
// weight = weight - lr * m_hat / (sqrt(v_hat) + eps)
// Uses Q16.16 fixed-point arithmetic

#include "mymath.h"
#include "mymath_backward.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = adam_update, 2 = adam_with_bias_correction)
// [1] = count (number of weights)
// [2] = addr_weights
// [3] = addr_grads
// [4] = addr_m (first moment)
// [5] = addr_v (second moment)
// [6] = learning_rate (Q16.16)
// [7] = beta1 (Q16.16, typically 0.9 = 58982)
// [8] = beta2 (Q16.16, typically 0.999 = 65470)
// [9] = eps (Q16.16, typically 1e-8 ≈ 1)
// [10] = timestep (for bias correction, integer)
// [11] = status (0 = busy, 1 = done)

// Constants in Q16.16
#define BETA1_DEFAULT 58982   // 0.9
#define BETA2_DEFAULT 65470   // 0.999
#define EPS_DEFAULT 1         // ~1e-8

void adam_step(
    volatile unsigned int* weights,
    volatile unsigned int* grads,
    volatile unsigned int* m,
    volatile unsigned int* v,
    fixed32 lr,
    fixed32 beta1,
    fixed32 beta2,
    fixed32 eps,
    unsigned int count
) {
    fixed32 one_minus_beta1 = sub_fixed(FIXED_ONE, beta1);
    fixed32 one_minus_beta2 = sub_fixed(FIXED_ONE, beta2);

    for (unsigned int i = 0; i < count; i++) {
        fixed32 w = float_to_fixed(weights[i]);
        fixed32 g = float_to_fixed(grads[i]);
        fixed32 mi = float_to_fixed(m[i]);
        fixed32 vi = float_to_fixed(v[i]);

        // m = beta1 * m + (1 - beta1) * grad
        mi = add_fixed(mul_fixed(beta1, mi), mul_fixed(one_minus_beta1, g));

        // v = beta2 * v + (1 - beta2) * grad^2
        fixed32 g_sq = mul_fixed(g, g);
        vi = add_fixed(mul_fixed(beta2, vi), mul_fixed(one_minus_beta2, g_sq));

        // weight = weight - lr * m / (sqrt(v) + eps)
        fixed32 denom = add_fixed(sqrt_fixed(vi), eps);
        fixed32 update = mul_fixed(lr, div_fixed(mi, denom));
        w = sub_fixed(w, update);

        weights[i] = fixed_to_float(w);
        m[i] = fixed_to_float(mi);
        v[i] = fixed_to_float(vi);
    }
}

void adam_step_bias_corrected(
    volatile unsigned int* weights,
    volatile unsigned int* grads,
    volatile unsigned int* m,
    volatile unsigned int* v,
    fixed32 lr,
    fixed32 beta1,
    fixed32 beta2,
    fixed32 eps,
    unsigned int timestep,
    unsigned int count
) {
    fixed32 one_minus_beta1 = sub_fixed(FIXED_ONE, beta1);
    fixed32 one_minus_beta2 = sub_fixed(FIXED_ONE, beta2);

    // Compute bias correction factors
    // beta1^t and beta2^t via repeated multiplication
    fixed32 beta1_t = FIXED_ONE;
    fixed32 beta2_t = FIXED_ONE;
    for (unsigned int t = 0; t < timestep; t++) {
        beta1_t = mul_fixed(beta1_t, beta1);
        beta2_t = mul_fixed(beta2_t, beta2);
    }
    fixed32 bias_corr1 = sub_fixed(FIXED_ONE, beta1_t);  // 1 - beta1^t
    fixed32 bias_corr2 = sub_fixed(FIXED_ONE, beta2_t);  // 1 - beta2^t

    for (unsigned int i = 0; i < count; i++) {
        fixed32 w = float_to_fixed(weights[i]);
        fixed32 g = float_to_fixed(grads[i]);
        fixed32 mi = float_to_fixed(m[i]);
        fixed32 vi = float_to_fixed(v[i]);

        // m = beta1 * m + (1 - beta1) * grad
        mi = add_fixed(mul_fixed(beta1, mi), mul_fixed(one_minus_beta1, g));

        // v = beta2 * v + (1 - beta2) * grad^2
        fixed32 g_sq = mul_fixed(g, g);
        vi = add_fixed(mul_fixed(beta2, vi), mul_fixed(one_minus_beta2, g_sq));

        // Bias correction
        fixed32 m_hat = div_fixed(mi, bias_corr1);
        fixed32 v_hat = div_fixed(vi, bias_corr2);

        // weight = weight - lr * m_hat / (sqrt(v_hat) + eps)
        fixed32 denom = add_fixed(sqrt_fixed(v_hat), eps);
        fixed32 update = mul_fixed(lr, div_fixed(m_hat, denom));
        w = sub_fixed(w, update);

        weights[i] = fixed_to_float(w);
        m[i] = fixed_to_float(mi);
        v[i] = fixed_to_float(vi);
    }
}

int main() {
    mem[11] = 0;

    while (1) {
        unsigned int cmd = mem[0];
        // OP_EXIT = 255 - безопасный выход!
        if (cmd == 255) { mem[11] = 1; break; }
        if (cmd == 0) continue;

        mem[11] = 0;  // busy

        unsigned int count = mem[1];
        volatile unsigned int* weights = (volatile unsigned int*)mem[2];
        volatile unsigned int* grads = (volatile unsigned int*)mem[3];
        volatile unsigned int* m = (volatile unsigned int*)mem[4];
        volatile unsigned int* v = (volatile unsigned int*)mem[5];
        fixed32 lr = (fixed32)mem[6];
        fixed32 beta1 = (fixed32)mem[7];
        fixed32 beta2 = (fixed32)mem[8];
        fixed32 eps = (fixed32)mem[9];
        unsigned int timestep = mem[10];

        if (cmd == 1) {
            adam_step(weights, grads, m, v, lr, beta1, beta2, eps, count);
        } else if (cmd == 2) {
            adam_step_bias_corrected(weights, grads, m, v, lr, beta1, beta2, eps, timestep, count);
        }

        mem[11] = 1;  // done
        mem[0] = 0;
    }

    return 0;
}
