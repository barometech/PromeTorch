// QEMU-safe diagnostic test
// Tests all math operations + simulated memory patterns
// No DDR_BASE access - purely computational

#include "mymath.h"
#include "mymath_backward.h"

// Test counters
static int tests_run = 0;
static int tests_passed = 0;

void check(bool condition, const char* name) {
    tests_run++;
    if (condition) {
        tests_passed++;
    }
}

int abs_diff(int a, int b) {
    int d = a - b;
    return d < 0 ? -d : d;
}

int main() {
    // ===== PART 1: Basic Math =====

    // Integer operations
    check(mul_u32(7, 8) == 56, "mul_u32");
    check(mul_i32(-5, 6) == -30, "mul_i32");
    check(mul_i32(-3, -4) == 12, "mul_i32 neg*neg");

    // Fixed point basics
    fixed32 one = INT_TO_FIXED(1);
    fixed32 two = INT_TO_FIXED(2);
    fixed32 half = one >> 1;  // 0.5

    check(add_fixed(one, one) == two, "add_fixed");
    check(sub_fixed(two, one) == one, "sub_fixed");
    check(mul_fixed(two, half) == one, "mul_fixed");

    // Float conversion (this was buggy before!)
    fixed32 f_1_5 = float_to_fixed(1.5f);
    check(abs_diff(f_1_5, one + half) < 100, "float_to_fixed 1.5");

    fixed32 f_0_25 = float_to_fixed(0.25f);
    check(abs_diff(f_0_25, one >> 2) < 100, "float_to_fixed 0.25");

    fixed32 f_neg = float_to_fixed(-2.0f);
    check(abs_diff(f_neg, -two) < 100, "float_to_fixed -2.0");

    // Division
    fixed32 four = INT_TO_FIXED(4);
    fixed32 div_result = div_fixed(four, two);
    check(abs_diff(div_result, two) < 100, "div_fixed 4/2");

    // Sqrt
    fixed32 sqrt_4 = sqrt_fixed(four);
    check(abs_diff(sqrt_4, two) < 1000, "sqrt_fixed 4");

    // Exp
    fixed32 exp_0 = exp_fixed(0);
    check(abs_diff(exp_0, one) < 1000, "exp_fixed 0");

    // ===== PART 2: Activations =====

    // ReLU
    check(relu_fixed(one) == one, "relu pos");
    check(relu_fixed(-one) == 0, "relu neg");

    // Sigmoid (at x=0, should be ~0.5)
    fixed32 sig_0 = sigmoid_fixed(0);
    check(abs_diff(sig_0, half) < 5000, "sigmoid 0");

    // SiLU (at x=0, should be 0)
    fixed32 silu_0 = silu_fixed(0);
    check(abs_diff(silu_0, 0) < 1000, "silu 0");

    // GELU
    fixed32 gelu_0 = gelu_fixed(0);
    check(abs_diff(gelu_0, 0) < 1000, "gelu 0");

    // ===== PART 3: Backward Pass =====

    // ReLU backward
    fixed32 grad = one;
    check(relu_backward(one, grad) == grad, "relu_backward pos");
    check(relu_backward(-one, grad) == 0, "relu_backward neg");

    // Sigmoid backward (at x=0: sig=0.5, grad = 0.5 * 0.5 = 0.25)
    fixed32 sig_grad = sigmoid_backward(0, one);
    fixed32 expected_sig_grad = one >> 2;  // 0.25
    check(abs_diff(sig_grad, expected_sig_grad) < 5000, "sigmoid_backward 0");

    // SiLU backward (at x=0: silu'(0) = 0.5)
    fixed32 silu_grad = silu_backward(0, one);
    check(abs_diff(silu_grad, half) < 5000, "silu_backward 0");

    // ===== PART 4: MatMul simulation =====

    // 2x2 @ 2x2 = 2x2
    fixed32 A[4] = {one, two, INT_TO_FIXED(3), four};  // [[1,2],[3,4]]
    fixed32 B[4] = {one, 0, 0, one};  // Identity
    fixed32 C[4] = {0, 0, 0, 0};

    // Manual matmul
    C[0] = add_fixed(mul_fixed(A[0], B[0]), mul_fixed(A[1], B[2]));
    C[1] = add_fixed(mul_fixed(A[0], B[1]), mul_fixed(A[1], B[3]));
    C[2] = add_fixed(mul_fixed(A[2], B[0]), mul_fixed(A[3], B[2]));
    C[3] = add_fixed(mul_fixed(A[2], B[1]), mul_fixed(A[3], B[3]));

    // A @ I = A
    check(abs_diff(C[0], A[0]) < 100, "matmul [0,0]");
    check(abs_diff(C[1], A[1]) < 100, "matmul [0,1]");
    check(abs_diff(C[2], A[2]) < 100, "matmul [1,0]");
    check(abs_diff(C[3], A[3]) < 100, "matmul [1,1]");

    // ===== PART 5: Softmax =====

    fixed32 logits[4] = {one, two, one, 0};
    fixed32 probs[4];

    // Find max
    fixed32 max_val = logits[0];
    for (int i = 1; i < 4; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // Exp and sum
    fixed32 sum = 0;
    for (int i = 0; i < 4; i++) {
        probs[i] = exp_fixed(logits[i] - max_val);
        sum = add_fixed(sum, probs[i]);
    }

    // Normalize
    for (int i = 0; i < 4; i++) {
        probs[i] = div_fixed(probs[i], sum);
    }

    // Sum of probs should be ~1
    fixed32 prob_sum = 0;
    for (int i = 0; i < 4; i++) {
        prob_sum = add_fixed(prob_sum, probs[i]);
    }
    check(abs_diff(prob_sum, one) < 5000, "softmax sum=1");

    // Largest logit should have largest prob
    check(probs[1] > probs[0], "softmax argmax");

    // ===== PART 6: RMSNorm =====

    fixed32 x[4] = {one, two, INT_TO_FIXED(3), four};
    fixed32 gamma[4] = {one, one, one, one};

    // Compute mean of squares
    fixed32 mean_sq = 0;
    for (int i = 0; i < 4; i++) {
        mean_sq = add_fixed(mean_sq, mul_fixed(x[i], x[i]));
    }
    mean_sq = mean_sq >> 2;  // divide by 4

    // RMS = sqrt(mean_sq + eps)
    fixed32 eps = 1;  // Small epsilon
    fixed32 rms = sqrt_fixed(add_fixed(mean_sq, eps));

    // Normalize
    fixed32 norm[4];
    for (int i = 0; i < 4; i++) {
        norm[i] = mul_fixed(div_fixed(x[i], rms), gamma[i]);
    }

    // After RMSNorm, mean of squares should be ~1
    fixed32 norm_sq = 0;
    for (int i = 0; i < 4; i++) {
        norm_sq = add_fixed(norm_sq, mul_fixed(norm[i], norm[i]));
    }
    norm_sq = norm_sq >> 2;
    check(abs_diff(norm_sq, one) < 10000, "rmsnorm variance");

    // ===== PART 7: SGD Update =====

    fixed32 weights[4] = {one, one, one, one};
    fixed32 grads[4] = {half, half, half, half};
    fixed32 lr = one >> 4;  // lr = 0.0625

    // w = w - lr * grad
    for (int i = 0; i < 4; i++) {
        weights[i] = sub_fixed(weights[i], mul_fixed(lr, grads[i]));
    }

    // Should be 1 - 0.0625 * 0.5 = 0.96875
    fixed32 expected_w = one - (one >> 5);  // 1 - 0.03125
    check(abs_diff(weights[0], expected_w) < 1000, "sgd update");

    // ===== SUMMARY =====

    // Return 0 if all tests passed, otherwise return number of failures
    if (tests_passed == tests_run) {
        return 0;  // SUCCESS
    } else {
        return tests_run - tests_passed;  // Number of failures
    }
}
