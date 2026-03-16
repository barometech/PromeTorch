// qemu_test.cpp - Test for QEMU emulator
// This test performs calculations and exits

#include "mymath.h"

// Test results stored in gr7 (return value)
int main() {
    int errors = 0;

    // Test 1: Basic fixed-point conversion
    fixed32 a = INT_TO_FIXED(2);  // 2.0 in Q16.16
    if (a != 0x20000) errors++;   // 2 * 65536 = 131072 = 0x20000

    // Test 2: Fixed-point addition
    fixed32 b = INT_TO_FIXED(3);
    fixed32 c = add_fixed(a, b);
    if (c != INT_TO_FIXED(5)) errors++;  // 2 + 3 = 5

    // Test 3: Fixed-point multiplication
    // 2.0 * 3.0 = 6.0
    fixed32 d = mul_fixed(a, b);
    if (d != INT_TO_FIXED(6)) errors++;

    // Test 4: Float to fixed conversion
    // 1.5 in IEEE 754 = 0x3FC00000
    unsigned int f_1_5 = 0x3FC00000;
    fixed32 f = float_to_fixed(f_1_5);
    // 1.5 in Q16.16 = 1.5 * 65536 = 98304 = 0x18000
    if (f != 0x18000) errors++;

    // Test 5: Fixed-point subtraction
    fixed32 e = sub_fixed(b, a);  // 3 - 2 = 1
    if (e != FIXED_ONE) errors++;

    // Return error count (0 = all tests passed)
    return errors;
}
