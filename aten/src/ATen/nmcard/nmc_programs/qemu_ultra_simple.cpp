// qemu_ultra_simple.cpp - Ultra simple test

typedef int fixed32;

int main() {
    // Test: just return a computed value
    // 0x3FC00000 = 1.5 in IEEE 754

    unsigned int f = 0x3FC00000;

    // Extract parts manually
    int exp_raw = (f >> 23) & 0xFF;  // Should be 127
    int exp = exp_raw - 127;          // Should be 0

    // Return exp (should be 0)
    return exp;
}
