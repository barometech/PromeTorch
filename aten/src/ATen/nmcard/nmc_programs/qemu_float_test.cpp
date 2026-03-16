// qemu_float_test.cpp - Debug float_to_fixed

// Q16.16 fixed-point
typedef int fixed32;
#define FIXED_SHIFT 16
#define FIXED_ONE (1 << FIXED_SHIFT)  // 65536

// Custom shifts (to avoid libgcc)
inline unsigned int my_lshift(unsigned int x, int n) {
    if (n <= 0) return x;
    if (n >= 32) return 0;
    if (n & 16) x = x << 16;
    if (n & 8)  x = x << 8;
    if (n & 4)  x = x << 4;
    if (n & 2)  x = x << 2;
    if (n & 1)  x = x << 1;
    return x;
}

inline unsigned int my_rshift(unsigned int x, int n) {
    if (n <= 0) return x;
    if (n >= 32) return 0;
    if (n & 16) x = x >> 16;
    if (n & 8)  x = x >> 8;
    if (n & 4)  x = x >> 4;
    if (n & 2)  x = x >> 2;
    if (n & 1)  x = x >> 1;
    return x;
}

// FIXED float_to_fixed
// Float: sign(1) | exp(8) | mantissa(23)
// value = (-1)^sign * 2^(exp-127) * 1.mantissa
// Fixed Q16.16: integer(16) | fraction(16)
// fixed = value * 2^16
inline fixed32 float_to_fixed_v2(unsigned int f) {
    if (f == 0) return 0;

    int sign = (f >> 31) ? -1 : 1;
    int exp = ((f >> 23) & 0xFF) - 127;  // unbias exponent
    unsigned int mantissa = (f & 0x7FFFFF) | 0x800000;  // add implicit 1

    // mantissa = 1.xxx * 2^23 (implicit 1 at bit 23)
    // value = mantissa / 2^23 * 2^exp = mantissa * 2^(exp - 23)
    // fixed = value * 2^16 = mantissa * 2^(exp - 23 + 16) = mantissa * 2^(exp - 7)
    //
    // So: shift = exp - 7
    // If shift > 0: result = mantissa << shift
    // If shift < 0: result = mantissa >> (-shift)

    int shift = exp - 7;

    unsigned int result;
    if (shift >= 0) {
        if (shift > 15) shift = 15;  // prevent overflow
        result = my_lshift(mantissa, shift);
    } else {
        shift = -shift;
        if (shift > 23) return 0;  // underflow
        result = my_rshift(mantissa, shift);
    }

    return sign < 0 ? -(fixed32)result : (fixed32)result;
}

int main() {
    // Test 1: 1.5 = 0x3FC00000 in IEEE 754
    // Expected: 1.5 * 65536 = 98304 = 0x18000
    unsigned int f_1_5 = 0x3FC00000;
    fixed32 result = float_to_fixed_v2(f_1_5);
    if (result != 0x18000) return 1;

    // Test 2: 2.0 = 0x40000000
    // Expected: 2.0 * 65536 = 131072 = 0x20000
    unsigned int f_2_0 = 0x40000000;
    result = float_to_fixed_v2(f_2_0);
    if (result != 0x20000) return 2;

    // Test 3: 0.5 = 0x3F000000
    // Expected: 0.5 * 65536 = 32768 = 0x8000
    unsigned int f_0_5 = 0x3F000000;
    result = float_to_fixed_v2(f_0_5);
    if (result != 0x8000) return 3;

    // Test 4: -1.0 = 0xBF800000
    // Expected: -1.0 * 65536 = -65536
    unsigned int f_neg1 = 0xBF800000;
    result = float_to_fixed_v2(f_neg1);
    if (result != -65536) return 4;

    // Test 5: 0.0 = 0x00000000
    result = float_to_fixed_v2(0);
    if (result != 0) return 5;

    return 0;  // All passed
}
