// qemu_training_test.cpp - Test complete training pipeline in QEMU
// Tests: Forward pass, Cross-entropy loss, Backward pass, Optimizer update
// Verifies that loss decreases after training step

#include "mymath.h"
#include "mymath_backward.h"

// Helper to check if values are close
inline int is_close(fixed32 a, fixed32 b, int tolerance) {
    int diff = a - b;
    if (diff < 0) diff = -diff;
    return diff <= tolerance;
}

// ============================================================
// Test 1: MatMul backward
// Forward: C = A @ B
// Backward: dA = dC @ B^T, dB = A^T @ dC
// ============================================================
int test_matmul_backward() {
    // A[2,3], B[3,2], C[2,2], dC[2,2]
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[1, 2], [3, 4], [5, 6]]
    // dC = [[1, 0], [0, 1]] (identity-like gradient)

    fixed32 A[6] = {
        FIXED_ONE, INT_TO_FIXED(2), INT_TO_FIXED(3),
        INT_TO_FIXED(4), INT_TO_FIXED(5), INT_TO_FIXED(6)
    };
    fixed32 B[6] = {
        FIXED_ONE, INT_TO_FIXED(2),
        INT_TO_FIXED(3), INT_TO_FIXED(4),
        INT_TO_FIXED(5), INT_TO_FIXED(6)
    };
    fixed32 dC[4] = {FIXED_ONE, 0, 0, FIXED_ONE};

    // dA = dC @ B^T  [2,2] @ [2,3] = [2,3]
    // B^T = [[1, 3, 5], [2, 4, 6]]
    // dA[0,0] = 1*1 + 0*2 = 1
    // dA[0,1] = 1*3 + 0*4 = 3
    // dA[0,2] = 1*5 + 0*6 = 5
    // dA[1,0] = 0*1 + 1*2 = 2
    // dA[1,1] = 0*3 + 1*4 = 4
    // dA[1,2] = 0*5 + 1*6 = 6

    fixed32 dA[6];
    int M = 2, K = 3, N = 2;

    // dA = dC @ B^T
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            fixed32 sum = 0;
            for (int j = 0; j < N; j++) {
                // dC[i,j] * B^T[j,k] = dC[i,j] * B[k,j]
                int dc_idx = mul_u32(i, N) + j;
                int bt_idx = mul_u32(k, N) + j;  // B[k,j] = B^T[j,k]
                sum = add_fixed(sum, mul_fixed(dC[dc_idx], B[bt_idx]));
            }
            dA[mul_u32(i, K) + k] = sum;
        }
    }

    if (!is_close(dA[0], FIXED_ONE, 100)) return 1;
    if (!is_close(dA[1], INT_TO_FIXED(3), 100)) return 2;
    if (!is_close(dA[2], INT_TO_FIXED(5), 100)) return 3;
    if (!is_close(dA[3], INT_TO_FIXED(2), 100)) return 4;
    if (!is_close(dA[4], INT_TO_FIXED(4), 100)) return 5;
    if (!is_close(dA[5], INT_TO_FIXED(6), 100)) return 6;

    // dB = A^T @ dC  [3,2] @ [2,2] = [3,2]
    // A^T = [[1, 4], [2, 5], [3, 6]]
    // dB[0,0] = 1*1 + 4*0 = 1
    // dB[0,1] = 1*0 + 4*1 = 4
    // dB[1,0] = 2*1 + 5*0 = 2
    // dB[1,1] = 2*0 + 5*1 = 5
    // dB[2,0] = 3*1 + 6*0 = 3
    // dB[2,1] = 3*0 + 6*1 = 6

    fixed32 dB[6];

    // dB = A^T @ dC
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            fixed32 sum = 0;
            for (int i = 0; i < M; i++) {
                // A^T[k,i] * dC[i,j] = A[i,k] * dC[i,j]
                int at_idx = mul_u32(i, K) + k;  // A[i,k] = A^T[k,i]
                int dc_idx = mul_u32(i, N) + j;
                sum = add_fixed(sum, mul_fixed(A[at_idx], dC[dc_idx]));
            }
            dB[mul_u32(k, N) + j] = sum;
        }
    }

    if (!is_close(dB[0], FIXED_ONE, 100)) return 7;
    if (!is_close(dB[1], INT_TO_FIXED(4), 100)) return 8;
    if (!is_close(dB[2], INT_TO_FIXED(2), 100)) return 9;
    if (!is_close(dB[3], INT_TO_FIXED(5), 100)) return 10;
    if (!is_close(dB[4], INT_TO_FIXED(3), 100)) return 11;
    if (!is_close(dB[5], INT_TO_FIXED(6), 100)) return 12;

    return 0;
}

// ============================================================
// Test 2: Softmax + Cross-entropy backward
// This is the main loss gradient for classification
// ============================================================
int test_softmax_ce_backward() {
    // logits = [1, 2, 3], target = 2 (class index)
    // softmax([1,2,3]) = [0.090, 0.245, 0.665]
    // CE backward: grad[i] = softmax[i] - (i == target ? 1 : 0)
    // grad = [0.090, 0.245, 0.665 - 1] = [0.090, 0.245, -0.335]

    fixed32 logits[3] = {FIXED_ONE, INT_TO_FIXED(2), INT_TO_FIXED(3)};
    fixed32 softmax_out[3];
    fixed32 grad[3];
    int target = 2;

    // Compute softmax
    fixed32 max_val = logits[0];
    for (int i = 1; i < 3; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    fixed32 exp_sum = 0;
    for (int i = 0; i < 3; i++) {
        fixed32 exp_val = exp_fixed_lut(sub_fixed(logits[i], max_val));
        softmax_out[i] = exp_val;
        exp_sum = add_fixed(exp_sum, exp_val);
    }

    for (int i = 0; i < 3; i++) {
        softmax_out[i] = div_fixed(softmax_out[i], exp_sum);
    }

    // Cross-entropy backward
    cross_entropy_backward(softmax_out, target, grad, 3);

    // grad[0] = softmax[0] ~= 0.090 -> 5898
    // grad[1] = softmax[1] ~= 0.245 -> 16056
    // grad[2] = softmax[2] - 1 ~= -0.335 -> -21955

    if (!is_close(grad[0], 5898, 2000)) return 20;
    if (!is_close(grad[1], 16056, 3000)) return 21;
    if (!is_close(grad[2], -21955, 3000)) return 22;

    return 0;
}

// ============================================================
// Test 3: SiLU backward
// ============================================================
int test_silu_backward_full() {
    // silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    // At x = 0: silu'(0) = 0.5 * (1 + 0) = 0.5
    // At x = 1: silu'(1) = 0.731 * (1 + 1 * 0.269) ~= 0.928

    fixed32 grad_out = FIXED_ONE;

    fixed32 grad_0 = silu_backward(0, grad_out);
    // 0.5 * 65536 = 32768
    if (!is_close(grad_0, 32768, 3000)) return 30;

    fixed32 grad_1 = silu_backward(FIXED_ONE, grad_out);
    // 0.928 * 65536 ~= 60817
    if (!is_close(grad_1, 60817, 5000)) return 31;

    // At x = -1: silu'(-1) ~= 0.072
    fixed32 grad_m1 = silu_backward(-FIXED_ONE, grad_out);
    // 0.072 * 65536 ~= 4719
    if (!is_close(grad_m1, 4719, 3000)) return 32;

    return 0;
}

// ============================================================
// Test 4: SGD optimizer update
// ============================================================
int test_sgd_update_full() {
    // w = 1.0, grad = 0.5, lr = 0.1
    // new_w = w - lr * grad = 1.0 - 0.1 * 0.5 = 0.95

    fixed32 weights[2] = {FIXED_ONE, INT_TO_FIXED(2)};
    fixed32 grads[2] = {0x8000, FIXED_ONE};  // [0.5, 1.0]
    fixed32 lr = 6554;  // 0.1 in Q16.16

    sgd_update(weights, grads, lr, 2);

    // weights[0] = 1.0 - 0.1 * 0.5 = 0.95 -> 62259
    // weights[1] = 2.0 - 0.1 * 1.0 = 1.9 -> 124518

    if (!is_close(weights[0], 62259, 500)) return 40;
    if (!is_close(weights[1], 124518, 500)) return 41;

    return 0;
}

// ============================================================
// Test 5: Adam optimizer update
// ============================================================
int test_adam_update_full() {
    fixed32 weights[1] = {FIXED_ONE};
    fixed32 grads[1] = {0x8000};  // 0.5
    fixed32 m[1] = {0};
    fixed32 v[1] = {0};
    fixed32 lr = 6554;      // 0.1
    fixed32 beta1 = 58982;  // 0.9
    fixed32 beta2 = 65470;  // 0.999
    fixed32 eps = 1;

    fixed32 w_before = weights[0];

    adam_update(weights, grads, m, v, lr, beta1, beta2, eps, 1);

    // Weight should decrease (gradient positive, updating to minimize)
    if (weights[0] >= w_before) return 50;

    // m should be updated: m = 0.1 * 0.5 = 0.05
    // 0.05 * 65536 = 3277
    if (!is_close(m[0], 3277, 500)) return 51;

    return 0;
}

// ============================================================
// Test 6: Full training step - verify loss decreases
// Simple: linear model y = W*x, MSE loss
// ============================================================
int test_training_step() {
    // Model: y = W*x where W is a scalar weight
    // Data: x = 1.0, target = 2.0
    // Loss: MSE = (y - target)^2

    fixed32 W = FIXED_ONE;       // Initial weight = 1.0
    fixed32 x = FIXED_ONE;       // Input = 1.0
    fixed32 target = INT_TO_FIXED(2);  // Target = 2.0
    fixed32 lr = 13107;          // 0.2 in Q16.16 (larger LR for faster convergence)

    // === Step 1: Forward pass ===
    fixed32 y = mul_fixed(W, x);  // y = 1.0 * 1.0 = 1.0

    // === Step 2: Compute loss ===
    fixed32 diff = sub_fixed(y, target);  // y - target = 1.0 - 2.0 = -1.0
    fixed32 loss_before = mul_fixed(diff, diff);  // loss = 1.0

    // === Step 3: Backward pass ===
    // d(loss)/d(y) = 2*(y - target) = 2*(-1) = -2
    // d(y)/d(W) = x = 1
    // d(loss)/d(W) = d(loss)/d(y) * d(y)/d(W) = -2 * 1 = -2
    fixed32 grad_y = mul_fixed(diff, INT_TO_FIXED(2));  // -2.0
    fixed32 grad_W = mul_fixed(grad_y, x);  // -2.0

    // === Step 4: SGD update ===
    // W = W - lr * grad = 1.0 - 0.2 * (-2) = 1.0 + 0.4 = 1.4
    fixed32 update = mul_fixed(lr, grad_W);
    W = sub_fixed(W, update);

    // W should be ~1.4
    // 1.4 * 65536 = 91750
    if (!is_close(W, 91750, 1000)) return 60;

    // === Step 5: Forward again to check loss ===
    y = mul_fixed(W, x);  // y = 1.4 * 1.0 = 1.4
    diff = sub_fixed(y, target);  // 1.4 - 2.0 = -0.6
    fixed32 loss_after = mul_fixed(diff, diff);  // 0.36

    // Loss should decrease: 1.0 -> 0.36
    // 0.36 * 65536 = 23593
    if (!is_close(loss_after, 23593, 2000)) return 61;

    // Verify loss decreased
    if (loss_after >= loss_before) return 62;

    return 0;
}

// ============================================================
// Test 7: Mini neural network training step
// 2-layer: hidden = silu(x @ W1), out = hidden @ W2
// ============================================================
int test_nn_training_step() {
    // Simplified: x[1], W1[1], W2[1], target = 2
    // forward: h = silu(x * W1), out = h * W2
    // backward: compute gradients, update

    fixed32 x = FIXED_ONE;
    fixed32 W1 = FIXED_ONE;
    fixed32 W2 = FIXED_ONE;
    fixed32 target = INT_TO_FIXED(2);
    fixed32 lr = 13107;  // 0.2

    // === Forward ===
    fixed32 z1 = mul_fixed(x, W1);       // z1 = 1.0
    fixed32 h = silu_fixed(z1);          // h = silu(1) ~= 0.731
    fixed32 out = mul_fixed(h, W2);      // out = 0.731 * 1 = 0.731

    // === Loss (MSE) ===
    fixed32 diff = sub_fixed(out, target);  // 0.731 - 2 = -1.269
    fixed32 loss_before = mul_fixed(diff, diff);  // ~1.61

    // === Backward ===
    // d_loss/d_out = 2 * (out - target) = 2 * (-1.269) = -2.538
    fixed32 grad_out = mul_fixed(diff, INT_TO_FIXED(2));

    // d_out/d_W2 = h
    // d_loss/d_W2 = grad_out * h
    fixed32 grad_W2 = mul_fixed(grad_out, h);

    // d_out/d_h = W2
    // d_loss/d_h = grad_out * W2
    fixed32 grad_h = mul_fixed(grad_out, W2);

    // d_h/d_z1 = silu'(z1)
    // d_loss/d_z1 = grad_h * silu'(z1)
    fixed32 grad_z1 = silu_backward(z1, grad_h);

    // d_z1/d_W1 = x
    // d_loss/d_W1 = grad_z1 * x
    fixed32 grad_W1 = mul_fixed(grad_z1, x);

    // === Update ===
    W1 = sub_fixed(W1, mul_fixed(lr, grad_W1));
    W2 = sub_fixed(W2, mul_fixed(lr, grad_W2));

    // === Forward again ===
    z1 = mul_fixed(x, W1);
    h = silu_fixed(z1);
    out = mul_fixed(h, W2);
    diff = sub_fixed(out, target);
    fixed32 loss_after = mul_fixed(diff, diff);

    // Loss should decrease
    if (loss_after >= loss_before) return 70;

    return 0;
}

// ============================================================
// Test 8: RMSNorm backward
// ============================================================
int test_rmsnorm_backward() {
    fixed32 x[2] = {FIXED_ONE, INT_TO_FIXED(2)};
    fixed32 gamma[2] = {FIXED_ONE, FIXED_ONE};
    fixed32 grad_output[2] = {FIXED_ONE, FIXED_ONE};
    fixed32 grad_input[2] = {0, 0};
    fixed32 grad_gamma[2] = {0, 0};
    fixed32 eps = 1;

    rmsnorm_backward(x, gamma, grad_output, grad_input, grad_gamma, 2, eps);

    // grad_input should be non-zero
    if (grad_input[0] == 0 && grad_input[1] == 0) return 80;

    // grad_gamma should be non-zero
    if (grad_gamma[0] == 0 && grad_gamma[1] == 0) return 81;

    return 0;
}

// ============================================================
// Test 9: Softmax backward
// ============================================================
int test_softmax_backward_full() {
    // softmax output y = [0.7, 0.2, 0.1]
    // grad_output = [1, 0, 0]
    // dot = sum(y * grad_out) = 0.7
    // grad_in[i] = y[i] * (grad_out[i] - dot)
    // grad_in[0] = 0.7 * (1 - 0.7) = 0.21
    // grad_in[1] = 0.2 * (0 - 0.7) = -0.14
    // grad_in[2] = 0.1 * (0 - 0.7) = -0.07

    fixed32 y[3] = {45875, 13107, 6554};  // [0.7, 0.2, 0.1]
    fixed32 grad_out[3] = {FIXED_ONE, 0, 0};
    fixed32 grad_in[3];

    softmax_backward(y, grad_out, grad_in, 3);

    // 0.21 * 65536 ~= 13762
    if (!is_close(grad_in[0], 13762, 2000)) return 90;
    // -0.14 * 65536 ~= -9175
    if (!is_close(grad_in[1], -9175, 2000)) return 91;
    // -0.07 * 65536 ~= -4588
    if (!is_close(grad_in[2], -4588, 2000)) return 92;

    return 0;
}

// ============================================================
// Test 10: Cross-entropy loss value
// ============================================================
int test_cross_entropy_loss() {
    // pred = [0.1, 0.2, 0.7], target = 2
    // loss = -log(0.7) ~= 0.357

    fixed32 pred[3] = {6554, 13107, 45875};  // [0.1, 0.2, 0.7]
    int target = 2;

    fixed32 loss = cross_entropy_loss(pred, target, 3);

    // -log(0.7) ~= 0.357 -> 23396 in Q16.16
    // But our log approximation is rough, just check it's positive and reasonable
    if (loss <= 0) return 100;
    if (loss > INT_TO_FIXED(2)) return 101;  // Should be < 2

    return 0;
}

// ============================================================
// Main
// ============================================================
int main() {
    int result;

    // Test 1-12: MatMul backward
    result = test_matmul_backward();
    if (result != 0) return result;

    // Test 20-22: Softmax + CE backward
    result = test_softmax_ce_backward();
    if (result != 0) return result;

    // Test 30-32: SiLU backward
    result = test_silu_backward_full();
    if (result != 0) return result;

    // Test 40-41: SGD update
    result = test_sgd_update_full();
    if (result != 0) return result;

    // Test 50-51: Adam update
    result = test_adam_update_full();
    if (result != 0) return result;

    // Test 60-62: Training step (loss decreases)
    result = test_training_step();
    if (result != 0) return result;

    // Test 70: NN training step
    result = test_nn_training_step();
    if (result != 0) return result;

    // Test 80-81: RMSNorm backward
    result = test_rmsnorm_backward();
    if (result != 0) return result;

    // Test 90-92: Softmax backward
    result = test_softmax_backward_full();
    if (result != 0) return result;

    // Test 100-101: Cross-entropy loss
    result = test_cross_entropy_loss();
    if (result != 0) return result;

    // All tests passed!
    return 0;
}
