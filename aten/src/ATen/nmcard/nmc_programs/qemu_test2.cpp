// qemu_test2.cpp - Debugging test for QEMU emulator
// Each test returns a different error code to identify failures

#include "mymath.h"

int main() {
    // Test 1: Basic fixed-point conversion
    fixed32 a = INT_TO_FIXED(2);  // 2.0 in Q16.16
    if (a != 0x20000) return 1;   // Error 1: INT_TO_FIXED failed

    // Test 2: Fixed-point addition
    fixed32 b = INT_TO_FIXED(3);
    fixed32 c = add_fixed(a, b);
    if (c != INT_TO_FIXED(5)) return 2;  // Error 2: add_fixed failed

    // Test 3: Fixed-point multiplication
    // 2.0 * 3.0 = 6.0
    fixed32 d = mul_fixed(a, b);
    if (d != INT_TO_FIXED(6)) return 3;  // Error 3: mul_fixed failed

    // Test 4: Float to fixed conversion
    // 1.5 in IEEE 754 = 0x3FC00000
    unsigned int f_1_5 = 0x3FC00000;
    fixed32 f = float_to_fixed(f_1_5);
    // 1.5 in Q16.16 = 1.5 * 65536 = 98304 = 0x18000
    if (f != 0x18000) return 4;  // Error 4: float_to_fixed failed

    // Test 5: Fixed-point subtraction
    fixed32 e = sub_fixed(b, a);  // 3 - 2 = 1
    if (e != FIXED_ONE) return 5;  // Error 5: sub_fixed failed

    // All tests passed
    return 0;
}
