// mymath_backward.h - Backward pass functions for training on NMC4
// Градиенты активаций и операций для backpropagation
// Дополняет mymath.h для полного training pipeline

#ifndef MYMATH_BACKWARD_H
#define MYMATH_BACKWARD_H

#include "mymath.h"

// ============================================================
// Градиенты активаций
// ============================================================

// ReLU backward: grad_input = grad_output * (x > 0 ? 1 : 0)
inline fixed32 relu_backward(fixed32 x, fixed32 grad_output) {
    return x > 0 ? grad_output : 0;
}

// Sigmoid backward: grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
inline fixed32 sigmoid_backward(fixed32 x, fixed32 grad_output) {
    fixed32 sig = sigmoid_fixed(x);
    fixed32 one_minus_sig = sub_fixed(FIXED_ONE, sig);
    fixed32 local_grad = mul_fixed(sig, one_minus_sig);
    return mul_fixed(grad_output, local_grad);
}

// SiLU backward:
// silu(x) = x * sigmoid(x)
// silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//          = sigmoid(x) * (1 + x - x * sigmoid(x))
inline fixed32 silu_backward(fixed32 x, fixed32 grad_output) {
    fixed32 sig = sigmoid_fixed(x);
    fixed32 one_minus_sig = sub_fixed(FIXED_ONE, sig);
    fixed32 x_term = mul_fixed(x, one_minus_sig);
    fixed32 local_grad = mul_fixed(sig, add_fixed(FIXED_ONE, x_term));
    return mul_fixed(grad_output, local_grad);
}

// GELU backward (using sigmoid approximation):
// gelu(x) ≈ x * sigmoid(1.702 * x)
// gelu'(x) ≈ sigmoid(1.702*x) + 1.702*x * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
inline fixed32 gelu_backward(fixed32 x, fixed32 grad_output) {
    const fixed32 GELU_COEF = 111543;  // 1.702 in Q16.16
    fixed32 scaled = mul_fixed(x, GELU_COEF);
    fixed32 sig = sigmoid_fixed(scaled);
    fixed32 one_minus_sig = sub_fixed(FIXED_ONE, sig);
    fixed32 term2 = mul_fixed(scaled, mul_fixed(sig, one_minus_sig));
    fixed32 local_grad = add_fixed(sig, term2);
    return mul_fixed(grad_output, local_grad);
}

// Tanh backward: grad_input = grad_output * (1 - tanh(x)^2)
inline fixed32 tanh_backward(fixed32 x, fixed32 grad_output) {
    fixed32 t = tanh_fixed(x);
    fixed32 t_sq = mul_fixed(t, t);
    fixed32 local_grad = sub_fixed(FIXED_ONE, t_sq);
    return mul_fixed(grad_output, local_grad);
}

// ============================================================
// Softmax backward (given output and grad_output)
// For softmax output y: grad_input_i = y_i * (grad_output_i - sum_j(y_j * grad_output_j))
// ============================================================
inline void softmax_backward(
    fixed32* y,           // softmax output [size]
    fixed32* grad_output, // incoming gradient [size]
    fixed32* grad_input,  // output gradient [size]
    int size
) {
    // Compute dot product: sum_j(y_j * grad_output_j)
    fixed32 dot = 0;
    for (int i = 0; i < size; i++) {
        dot = add_fixed(dot, mul_fixed(y[i], grad_output[i]));
    }

    // grad_input_i = y_i * (grad_output_i - dot)
    for (int i = 0; i < size; i++) {
        fixed32 diff = sub_fixed(grad_output[i], dot);
        grad_input[i] = mul_fixed(y[i], diff);
    }
}

// ============================================================
// RMSNorm backward
// y = x / sqrt(mean(x^2) + eps) * gamma
// Gradient is more complex, simplified version:
// ============================================================
inline void rmsnorm_backward(
    fixed32* x,           // input [size]
    fixed32* gamma,       // weight [size]
    fixed32* grad_output, // incoming gradient [size]
    fixed32* grad_input,  // output: gradient wrt input [size]
    fixed32* grad_gamma,  // output: gradient wrt gamma [size]
    int size,
    fixed32 eps
) {
    // Forward: compute rms
    fixed32 sum_sq = 0;
    for (int i = 0; i < size; i++) {
        sum_sq = add_fixed(sum_sq, mul_fixed(x[i], x[i]));
    }
    fixed32 mean_sq = div_fixed(sum_sq, INT_TO_FIXED(size));
    fixed32 rms = sqrt_fixed(add_fixed(mean_sq, eps));
    fixed32 inv_rms = div_fixed(FIXED_ONE, rms);

    // Compute normalized x
    // x_norm = x * inv_rms

    // grad_gamma = sum(grad_output * x_norm)
    // (accumulate across batch in practice)
    for (int i = 0; i < size; i++) {
        fixed32 x_norm = mul_fixed(x[i], inv_rms);
        grad_gamma[i] = add_fixed(grad_gamma[i], mul_fixed(grad_output[i], x_norm));
    }

    // grad_input (simplified, ignoring second-order terms through rms)
    // Full formula is complex; this is an approximation
    for (int i = 0; i < size; i++) {
        grad_input[i] = mul_fixed(mul_fixed(grad_output[i], gamma[i]), inv_rms);
    }
}

// ============================================================
// Cross-Entropy Loss (with softmax)
// loss = -sum(target * log(pred))
// For one-hot target with class c: loss = -log(pred[c])
// ============================================================
inline fixed32 cross_entropy_loss(
    fixed32* pred,    // softmax output [vocab_size]
    int target_class, // correct class index
    int vocab_size
) {
    // loss = -log(pred[target_class])
    // log in fixed point is tricky; use approximation
    fixed32 p = pred[target_class];

    // Clamp to avoid log(0)
    if (p < 655) p = 655;  // ~0.01 in Q16.16

    // log(x) ≈ (x - 1) - (x-1)^2/2 + (x-1)^3/3 for x near 1
    // Or use: log(x) = log(2) * log2(x)
    // Simplified: log(p) ≈ (p - 1) for p near 1
    // Better: use iterative or lookup table

    // For now, simple approximation:
    // log(p) ≈ 2 * (p - 1) / (p + 1) for p in (0, 2)
    fixed32 num = sub_fixed(p, FIXED_ONE);
    fixed32 denom = add_fixed(p, FIXED_ONE);
    fixed32 log_approx = mul_fixed(num << 1, div_fixed(FIXED_ONE, denom));

    return -log_approx;  // return -log(p)
}

// Cross-entropy backward: grad = pred - one_hot(target)
// For class c: grad[i] = pred[i] - (i == c ? 1 : 0)
inline void cross_entropy_backward(
    fixed32* pred,        // softmax output [vocab_size]
    int target_class,     // correct class
    fixed32* grad_output, // output gradient [vocab_size]
    int vocab_size
) {
    for (int i = 0; i < vocab_size; i++) {
        if (i == target_class) {
            grad_output[i] = sub_fixed(pred[i], FIXED_ONE);
        } else {
            grad_output[i] = pred[i];
        }
    }
}

// ============================================================
// MSE Loss: loss = mean((pred - target)^2)
// ============================================================
inline fixed32 mse_loss(
    fixed32* pred,
    fixed32* target,
    int size
) {
    fixed32 sum = 0;
    for (int i = 0; i < size; i++) {
        fixed32 diff = sub_fixed(pred[i], target[i]);
        sum = add_fixed(sum, mul_fixed(diff, diff));
    }
    return div_fixed(sum, INT_TO_FIXED(size));
}

// MSE backward: grad = 2 * (pred - target) / size
inline void mse_backward(
    fixed32* pred,
    fixed32* target,
    fixed32* grad_output,
    int size
) {
    fixed32 scale = div_fixed(FIXED_ONE << 1, INT_TO_FIXED(size));  // 2/size
    for (int i = 0; i < size; i++) {
        fixed32 diff = sub_fixed(pred[i], target[i]);
        grad_output[i] = mul_fixed(diff, scale);
    }
}

// ============================================================
// SGD Optimizer: weight = weight - lr * grad
// ============================================================
inline void sgd_update(
    fixed32* weights,
    fixed32* grads,
    fixed32 lr,  // learning rate in Q16.16
    int size
) {
    for (int i = 0; i < size; i++) {
        fixed32 update = mul_fixed(lr, grads[i]);
        weights[i] = sub_fixed(weights[i], update);
    }
}

// ============================================================
// SGD with Momentum:
// v = momentum * v + grad
// weight = weight - lr * v
// ============================================================
inline void sgd_momentum_update(
    fixed32* weights,
    fixed32* grads,
    fixed32* velocity,  // momentum buffer
    fixed32 lr,
    fixed32 momentum,   // e.g., 0.9 = 58982 in Q16.16
    int size
) {
    for (int i = 0; i < size; i++) {
        // v = momentum * v + grad
        velocity[i] = add_fixed(mul_fixed(momentum, velocity[i]), grads[i]);
        // w = w - lr * v
        weights[i] = sub_fixed(weights[i], mul_fixed(lr, velocity[i]));
    }
}

// ============================================================
// Adam Optimizer (simplified, without bias correction)
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// weight = weight - lr * m / (sqrt(v) + eps)
// ============================================================
inline void adam_update(
    fixed32* weights,
    fixed32* grads,
    fixed32* m,         // first moment
    fixed32* v,         // second moment
    fixed32 lr,
    fixed32 beta1,      // 0.9 = 58982
    fixed32 beta2,      // 0.999 = 65470
    fixed32 eps,        // 1e-8 ≈ 0 in Q16.16, use small value like 1
    int size
) {
    fixed32 one_minus_beta1 = sub_fixed(FIXED_ONE, beta1);
    fixed32 one_minus_beta2 = sub_fixed(FIXED_ONE, beta2);

    for (int i = 0; i < size; i++) {
        // m = beta1 * m + (1 - beta1) * grad
        m[i] = add_fixed(
            mul_fixed(beta1, m[i]),
            mul_fixed(one_minus_beta1, grads[i])
        );

        // v = beta2 * v + (1 - beta2) * grad^2
        fixed32 grad_sq = mul_fixed(grads[i], grads[i]);
        v[i] = add_fixed(
            mul_fixed(beta2, v[i]),
            mul_fixed(one_minus_beta2, grad_sq)
        );

        // weight = weight - lr * m / (sqrt(v) + eps)
        fixed32 denom = add_fixed(sqrt_fixed(v[i]), eps);
        fixed32 update = mul_fixed(lr, div_fixed(m[i], denom));
        weights[i] = sub_fixed(weights[i], update);
    }
}

// ============================================================
// MatMul backward
// Forward: C = A @ B where A[M,K], B[K,N], C[M,N]
// Backward:
//   dA = dC @ B^T
//   dB = A^T @ dC
// ============================================================

// Note: These are implemented as separate kernels in matmul_backward.cpp

#endif // MYMATH_BACKWARD_H
