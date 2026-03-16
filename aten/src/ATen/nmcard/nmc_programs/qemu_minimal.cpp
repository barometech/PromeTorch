// qemu_minimal.cpp - Minimal test for QEMU

// Q16.16 fixed-point
typedef int fixed32;
#define FIXED_SHIFT 16
#define FIXED_ONE (1 << FIXED_SHIFT)  // 65536
#define INT_TO_FIXED(x) ((x) << FIXED_SHIFT)

// Simple add - just inline, no function call
inline fixed32 simple_add(fixed32 a, fixed32 b) {
    return a + b;
}

// Simple mul - manual implementation
inline fixed32 simple_mul(fixed32 a, fixed32 b) {
    // Avoid library calls - do manual long multiplication
    int sign = 1;
    unsigned int ua = (unsigned int)a;
    unsigned int ub = (unsigned int)b;

    if (a < 0) { sign = -sign; ua = (unsigned int)(-a); }
    if (b < 0) { sign = -sign; ub = (unsigned int)(-b); }

    // Split into 16-bit halves
    unsigned int a_hi = ua >> 16;
    unsigned int a_lo = ua & 0xFFFF;
    unsigned int b_hi = ub >> 16;
    unsigned int b_lo = ub & 0xFFFF;

    // Compute partial products
    unsigned int lo_lo = a_lo * b_lo;  // 16x16 = 32 bit
    unsigned int hi_lo = a_hi * b_lo;
    unsigned int lo_hi = a_lo * b_hi;
    unsigned int hi_hi = a_hi * b_hi;

    // Combine with shift
    // result = (hi_hi << 32) + ((hi_lo + lo_hi) << 16) + lo_lo
    // but we need >> 16 for fixed point
    // so: (hi_hi << 16) + (hi_lo + lo_hi) + (lo_lo >> 16)

    unsigned int result = (hi_hi << 16) + hi_lo + lo_hi + (lo_lo >> 16);

    return sign > 0 ? (fixed32)result : -(fixed32)result;
}

int main() {
    // Test 1: INT_TO_FIXED
    fixed32 a = INT_TO_FIXED(2);  // Should be 131072 = 0x20000
    if (a != 0x20000) return 1;

    // Test 2: Simple addition
    fixed32 b = INT_TO_FIXED(3);  // 196608 = 0x30000
    fixed32 c = simple_add(a, b);  // Should be 327680 = 0x50000
    if (c != INT_TO_FIXED(5)) return 2;

    // Test 3: Simple multiplication
    // 2.0 * 3.0 = 6.0 = 393216 = 0x60000
    fixed32 d = simple_mul(a, b);
    if (d != INT_TO_FIXED(6)) return 3;

    // All passed
    return 0;
}
