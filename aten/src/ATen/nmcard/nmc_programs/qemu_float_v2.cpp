// qemu_float_v2.cpp - Step by step float test

typedef int fixed32;

int main() {
    unsigned int f = 0x3FC00000;  // 1.5 in IEEE 754

    // Step 1: Extract sign
    int sign_bit = f >> 31;  // Should be 0

    // Step 2: Extract exponent
    int exp_raw = (f >> 23) & 0xFF;  // Should be 127 (0x7F)
    int exp = exp_raw - 127;          // Should be 0

    // Step 3: Extract mantissa with implicit 1
    unsigned int mantissa = (f & 0x7FFFFF) | 0x800000;  // Should be 0xC00000

    // Step 4: Calculate shift
    int shift = exp - 7;  // Should be -7

    // Step 5: Apply shift (right shift by 7)
    unsigned int result;
    if (shift >= 0) {
        // Won't happen for 1.5
        result = mantissa << shift;  // variable shift - risky
    } else {
        int neg_shift = -shift;  // Should be 7
        // Use constant shifts to avoid libgcc
        result = mantissa;
        if (neg_shift >= 4) { result = result >> 4; neg_shift -= 4; }
        if (neg_shift >= 2) { result = result >> 2; neg_shift -= 2; }
        if (neg_shift >= 1) { result = result >> 1; }
    }
    // result should be 0xC00000 >> 7 = 0x18000 = 98304

    // Return test result
    if (result == 0x18000) {
        return 0;  // SUCCESS
    } else {
        // Return which step failed
        if (sign_bit != 0) return 1;
        if (exp != 0) return 2;
        if (mantissa != 0xC00000) return 3;
        if (shift != -7) return 4;
        return 5;  // result wrong
    }
}
