// qemu_backward_test.cpp - Test backward functions in QEMU
// Tests gradient computations for training

#include "mymath.h"
#include "mymath_backward.h"

// Helper to check if values are close
inline int is_close(fixed32 a, fixed32 b, int tolerance) {
    int diff = a - b;
    if (diff < 0) diff = -diff;
    return diff <= tolerance;
}

int main() {
    // ========================================
    // Test 1: ReLU backward
    // ========================================
    // relu'(x) = 1 if x > 0, else 0
    fixed32 grad_out = FIXED_ONE;

    // Positive input
    fixed32 grad = relu_backward(INT_TO_FIXED(5), grad_out);
    if (grad != FIXED_ONE) return 1;

    // Negative input
    grad = relu_backward(INT_TO_FIXED(-5), grad_out);
    if (grad != 0) return 2;

    // Zero input
    grad = relu_backward(0, grad_out);
    if (grad != 0) return 3;

    // ========================================
    // Test 2: Sigmoid backward
    // ========================================
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    // At x=0: sigmoid(0)=0.5, gradient = 0.5 * 0.5 = 0.25

    grad = sigmoid_backward(0, FIXED_ONE);
    // 0.25 in Q16.16 = 16384
    if (!is_close(grad, 16384, 2000)) return 4;

    // ========================================
    // Test 3: SiLU backward
    // ========================================
    // silu'(x) = sigmoid(x) + x * sigmoid'(x)
    // At x=0: silu'(0) = 0.5 + 0 = 0.5

    grad = silu_backward(0, FIXED_ONE);
    // 0.5 in Q16.16 = 32768
    if (!is_close(grad, 32768, 3000)) return 5;

    // ========================================
    // Test 4: SGD update (using array)
    // ========================================
    // w = w - lr * grad
    // w=1.0, grad=0.5, lr=0.1 => w = 1.0 - 0.1*0.5 = 0.95

    fixed32 weights[1] = {FIXED_ONE};  // 1.0
    fixed32 grads[1] = {0x8000};       // 0.5
    fixed32 lr = 6554;                  // 0.1 in Q16.16

    sgd_update(weights, grads, lr, 1);
    // 0.95 in Q16.16 = 62259
    if (!is_close(weights[0], 62259, 500)) return 6;

    // ========================================
    // Test 5: Cross-entropy backward (using array)
    // ========================================
    // CE gradient for softmax output is: pred - target_one_hot
    // If pred[i] = 0.7 and target = i, gradient = 0.7 - 1.0 = -0.3

    // vocab_size = 3, target = 0
    fixed32 pred[3] = {52428, 6554, 6554};  // [0.8, 0.1, 0.1] approx
    fixed32 ce_grads[3];

    cross_entropy_backward(pred, 0, ce_grads, 3);

    // pred[0] is target: grad = 0.8 - 1.0 = -0.2 = -13107
    if (!is_close(ce_grads[0], -13107, 2000)) return 7;

    // pred[1] is not target: grad = 0.1 = 6554
    if (!is_close(ce_grads[1], 6554, 1000)) return 8;

    // ========================================
    // Test 6: Tanh backward
    // ========================================
    // tanh'(x) = 1 - tanh(x)^2
    // At x=0: tanh(0)=0, gradient = 1 - 0 = 1

    grad = tanh_backward(0, FIXED_ONE);
    if (!is_close(grad, FIXED_ONE, 3000)) return 9;

    // ========================================
    // Test 7: GELU backward
    // ========================================
    // At x=0, GELU'(0) = 0.5 (similar to sigmoid at 0)
    grad = gelu_backward(0, FIXED_ONE);
    // 0.5 in Q16.16 = 32768
    if (!is_close(grad, 32768, 5000)) return 10;

    // ========================================
    // Test 8: Softmax backward
    // ========================================
    // y = [0.7, 0.2, 0.1], grad_out = [1, 0, 0]
    // dot = 0.7 * 1 + 0.2 * 0 + 0.1 * 0 = 0.7
    // grad_in[0] = 0.7 * (1 - 0.7) = 0.21
    // grad_in[1] = 0.2 * (0 - 0.7) = -0.14
    // grad_in[2] = 0.1 * (0 - 0.7) = -0.07

    fixed32 softmax_y[3] = {45875, 13107, 6554};  // [0.7, 0.2, 0.1]
    fixed32 softmax_grad_out[3] = {FIXED_ONE, 0, 0};
    fixed32 softmax_grad_in[3];

    softmax_backward(softmax_y, softmax_grad_out, softmax_grad_in, 3);

    // grad_in[0] ≈ 0.21 = 13762 in Q16.16
    if (!is_close(softmax_grad_in[0], 13762, 2000)) return 11;

    // grad_in[1] ≈ -0.14 = -9175 in Q16.16
    if (!is_close(softmax_grad_in[1], -9175, 2000)) return 12;

    // ========================================
    // Test 9: Adam optimizer
    // ========================================
    fixed32 adam_w[1] = {FIXED_ONE};  // 1.0
    fixed32 adam_g[1] = {0x8000};     // 0.5
    fixed32 adam_m[1] = {0};
    fixed32 adam_v[1] = {0};
    fixed32 adam_lr = 6554;            // 0.1
    fixed32 beta1 = 58982;             // 0.9
    fixed32 beta2 = 65470;             // 0.999
    fixed32 eps = 1;                   // small epsilon

    adam_update(adam_w, adam_g, adam_m, adam_v, adam_lr, beta1, beta2, eps, 1);

    // After one step, weight should decrease slightly
    // m = 0.1 * 0.5 = 0.05
    // v = 0.001 * 0.25 = 0.00025
    // update = 0.1 * 0.05 / sqrt(0.00025) ≈ 0.316
    // But this is just first step, exact value depends on implementation
    if (adam_w[0] >= FIXED_ONE) return 13;  // Should have decreased

    // ========================================
    // All tests passed!
    // ========================================
    return 0;
}
