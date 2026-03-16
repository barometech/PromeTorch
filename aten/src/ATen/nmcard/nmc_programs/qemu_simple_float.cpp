// qemu_simple_float.cpp - Simplest float test without inline

typedef int fixed32;
#define FIXED_SHIFT 16

// NOT inline - regular function
fixed32 float_to_fixed_simple(unsigned int f) {
    if (f == 0) return 0;

    int sign = (f >> 31) ? -1 : 1;
    int exp = ((f >> 23) & 0xFF) - 127;
    unsigned int mantissa = (f & 0x7FFFFF) | 0x800000;

    // shift = exp - 7
    int shift = exp - 7;

    unsigned int result;
    if (shift >= 0) {
        // Left shift with constant shifts only
        if (shift >= 16) {
            result = mantissa << 16;
            shift -= 16;
        } else {
            result = mantissa;
        }
        if (shift >= 8) { result = result << 8; shift -= 8; }
        if (shift >= 4) { result = result << 4; shift -= 4; }
        if (shift >= 2) { result = result << 2; shift -= 2; }
        if (shift >= 1) { result = result << 1; }
    } else {
        shift = -shift;
        // Right shift
        if (shift >= 16) {
            result = mantissa >> 16;
            shift -= 16;
        } else {
            result = mantissa;
        }
        if (shift >= 8) { result = result >> 8; shift -= 8; }
        if (shift >= 4) { result = result >> 4; shift -= 4; }
        if (shift >= 2) { result = result >> 2; shift -= 2; }
        if (shift >= 1) { result = result >> 1; }
    }

    return sign < 0 ? -(fixed32)result : (fixed32)result;
}

int main() {
    // Test: 1.5 = 0x3FC00000
    // Expected: 98304 = 0x18000
    unsigned int f = 0x3FC00000;
    fixed32 result = float_to_fixed_simple(f);

    // Return 0 if correct, 1 if wrong
    if (result == 0x18000) {
        return 0;
    } else {
        return 1;
    }
}
