// qemu_mymath_test.cpp - Comprehensive test for mymath.h in QEMU

#include "mymath.h"

// Helper to check if values are close (within tolerance)
inline int is_close(fixed32 a, fixed32 b, int tolerance) {
    int diff = a - b;
    if (diff < 0) diff = -diff;
    return diff <= tolerance;
}

int main() {
    // ========================================
    // Test 1: Basic fixed-point conversion
    // ========================================
    fixed32 two = INT_TO_FIXED(2);
    if (two != 0x20000) return 1;  // 2 * 65536 = 131072

    fixed32 three = INT_TO_FIXED(3);
    if (three != 0x30000) return 2;

    // ========================================
    // Test 2: Addition
    // ========================================
    fixed32 five = add_fixed(two, three);
    if (five != INT_TO_FIXED(5)) return 3;

    // ========================================
    // Test 3: Subtraction
    // ========================================
    fixed32 one = sub_fixed(three, two);
    if (one != FIXED_ONE) return 4;

    // ========================================
    // Test 4: Multiplication
    // ========================================
    fixed32 six = mul_fixed(two, three);  // 2 * 3 = 6
    if (six != INT_TO_FIXED(6)) return 5;

    // 1.5 * 2.0 = 3.0
    fixed32 one_half = 0x18000;  // 1.5 in Q16.16
    fixed32 result = mul_fixed(one_half, two);
    if (result != INT_TO_FIXED(3)) return 6;

    // ========================================
    // Test 5: Float to Fixed conversion
    // ========================================
    // 1.5 = 0x3FC00000
    fixed32 f1 = float_to_fixed(0x3FC00000);
    if (f1 != 0x18000) return 7;

    // 2.0 = 0x40000000
    fixed32 f2 = float_to_fixed(0x40000000);
    if (f2 != 0x20000) return 8;

    // 0.5 = 0x3F000000
    fixed32 f3 = float_to_fixed(0x3F000000);
    if (f3 != 0x8000) return 9;

    // -1.0 = 0xBF800000
    fixed32 f4 = float_to_fixed(0xBF800000);
    if (f4 != -65536) return 10;

    // ========================================
    // Test 6: Division
    // ========================================
    // 6 / 2 = 3
    fixed32 div_result = div_fixed(six, two);
    if (!is_close(div_result, INT_TO_FIXED(3), 10)) return 11;

    // 1 / 2 = 0.5
    div_result = div_fixed(FIXED_ONE, two);
    if (!is_close(div_result, 0x8000, 10)) return 12;

    // ========================================
    // Test 7: Square root
    // ========================================
    // sqrt(4) = 2
    fixed32 sqrt_result = sqrt_fixed(INT_TO_FIXED(4));
    if (!is_close(sqrt_result, INT_TO_FIXED(2), 100)) return 13;

    // sqrt(1) = 1
    sqrt_result = sqrt_fixed(FIXED_ONE);
    if (!is_close(sqrt_result, FIXED_ONE, 100)) return 14;

    // ========================================
    // Test 8: Exponential (approximate)
    // ========================================
    // exp(0) = 1
    fixed32 exp_result = exp_fixed(0);
    if (!is_close(exp_result, FIXED_ONE, 100)) return 15;

    // exp(1) ≈ 2.718 => ~178145 in Q16.16
    exp_result = exp_fixed(FIXED_ONE);
    if (!is_close(exp_result, 178145, 5000)) return 16;

    // ========================================
    // Test 9: Sigmoid
    // ========================================
    // sigmoid(0) = 0.5 => 32768 in Q16.16
    fixed32 sig_result = sigmoid_fixed(0);
    if (!is_close(sig_result, 0x8000, 1000)) return 17;

    // sigmoid(large) ≈ 1
    sig_result = sigmoid_fixed(INT_TO_FIXED(10));
    if (!is_close(sig_result, FIXED_ONE, 1000)) return 18;

    // sigmoid(-large) ≈ 0
    sig_result = sigmoid_fixed(INT_TO_FIXED(-10));
    if (!is_close(sig_result, 0, 1000)) return 19;

    // ========================================
    // Test 10: ReLU
    // ========================================
    if (relu_fixed(INT_TO_FIXED(5)) != INT_TO_FIXED(5)) return 20;
    if (relu_fixed(INT_TO_FIXED(-5)) != 0) return 21;

    // ========================================
    // Test 11: SiLU
    // ========================================
    // silu(0) = 0 * sigmoid(0) = 0
    if (silu_fixed(0) != 0) return 22;

    // silu(x) for large x ≈ x
    fixed32 silu_result = silu_fixed(INT_TO_FIXED(5));
    if (!is_close(silu_result, INT_TO_FIXED(5), 5000)) return 23;

    // ========================================
    // Test 12: Fixed to Float conversion
    // ========================================
    unsigned int back_to_float = fixed_to_float(0x20000);  // 2.0
    // 2.0 = 0x40000000
    if (back_to_float != 0x40000000) return 24;

    back_to_float = fixed_to_float(FIXED_ONE);  // 1.0
    // 1.0 = 0x3F800000
    if (back_to_float != 0x3F800000) return 25;

    // ========================================
    // All tests passed!
    // ========================================
    return 0;
}
