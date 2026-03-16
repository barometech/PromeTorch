// MatMul with custom everything - no library calls at all
// Uses fixed-point Q16.16 for float-like precision

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (1 = matmul)
// [1] = M
// [2] = K
// [3] = N
// [4] = addr_A
// [5] = addr_B
// [6] = addr_C
// [7] = status
// [8] = debug iterations

// ============================================================
// Custom shift functions - avoid LShift32/RShift32
// ============================================================

// Left shift by variable amount
unsigned int my_lshift(unsigned int x, int n) {
    // Unroll to avoid variable shift which calls LShift32
    if (n <= 0) return x;
    if (n >= 32) return 0;

    // Manual unrolled shifts
    if (n & 16) x = x << 16;
    if (n & 8)  x = x << 8;
    if (n & 4)  x = x << 4;
    if (n & 2)  x = x << 2;
    if (n & 1)  x = x << 1;
    return x;
}

// Right shift by variable amount
unsigned int my_rshift(unsigned int x, int n) {
    if (n <= 0) return x;
    if (n >= 32) return 0;

    if (n & 16) x = x >> 16;
    if (n & 8)  x = x >> 8;
    if (n & 4)  x = x >> 4;
    if (n & 2)  x = x >> 2;
    if (n & 1)  x = x >> 1;
    return x;
}

// ============================================================
// Custom multiply - no Mul32
// ============================================================

unsigned int mul_u32(unsigned int a, unsigned int b) {
    unsigned int result = 0;
    while (b != 0) {
        if (b & 1) {
            result = result + a;
        }
        a = a << 1;  // constant shift = OK
        b = b >> 1;  // constant shift = OK
    }
    return result;
}

// Fixed-point Q16.16 multiply
int mul_fixed(int a, int b) {
    int sign = 1;
    if (a < 0) { a = -a; sign = -sign; }
    if (b < 0) { b = -b; sign = -sign; }

    unsigned int a_hi = ((unsigned int)a) >> 16;  // constant shift
    unsigned int a_lo = ((unsigned int)a) & 0xFFFF;
    unsigned int b_hi = ((unsigned int)b) >> 16;
    unsigned int b_lo = ((unsigned int)b) & 0xFFFF;

    unsigned int hi_hi = mul_u32(a_hi, b_hi) << 16;
    unsigned int hi_lo = mul_u32(a_hi, b_lo);
    unsigned int lo_hi = mul_u32(a_lo, b_hi);
    unsigned int lo_lo = mul_u32(a_lo, b_lo) >> 16;

    unsigned int result = hi_hi + hi_lo + lo_hi + lo_lo;
    return sign < 0 ? -(int)result : (int)result;
}

// ============================================================
// Float conversion - using custom shifts
// ============================================================

int float_to_fixed(unsigned int f) {
    if (f == 0 || f == 0x80000000) return 0;

    int sign = (f >> 31) ? -1 : 1;
    int exp = ((f >> 23) & 0xFF) - 127;
    unsigned int mantissa = (f & 0x7FFFFF) | 0x800000;

    int shift = exp - 7;

    unsigned int result;
    if (shift >= 0) {
        if (shift > 15) shift = 15;
        result = my_lshift(mantissa, shift);
    } else {
        shift = -shift;
        if (shift > 23) return 0;
        result = my_rshift(mantissa, shift);
    }

    return sign < 0 ? -(int)result : (int)result;
}

unsigned int fixed_to_float(int x) {
    if (x == 0) return 0;

    unsigned int sign = 0;
    if (x < 0) {
        sign = 0x80000000;
        x = -x;
    }

    int exp = 0;
    unsigned int tmp = (unsigned int)x;

    // Normalize - find position of highest bit
    while (tmp >= 0x1000000) { tmp = tmp >> 1; exp++; }
    while (tmp < 0x800000 && tmp != 0) { tmp = tmp << 1; exp--; }

    if (tmp == 0) return 0;

    exp = exp + 127 + 7;
    unsigned int mantissa = tmp & 0x7FFFFF;

    return sign | (((unsigned int)exp) << 23) | mantissa;
}

// ============================================================
// MatMul - all custom, no library calls
// ============================================================

void matmul(unsigned int* A, unsigned int* B, unsigned int* C,
            unsigned int M, unsigned int K, unsigned int N) {

    for (unsigned int i = 0; i < M; i++) {
        for (unsigned int j = 0; j < N; j++) {
            int sum = 0;

            for (unsigned int k = 0; k < K; k++) {
                // Calculate indices: a_idx = i*K + k, b_idx = k*N + j
                unsigned int a_idx = mul_u32(i, K) + k;
                unsigned int b_idx = mul_u32(k, N) + j;

                int a_fixed = float_to_fixed(A[a_idx]);
                int b_fixed = float_to_fixed(B[b_idx]);

                sum = sum + mul_fixed(a_fixed, b_fixed);
            }

            unsigned int c_idx = mul_u32(i, N) + j;
            C[c_idx] = fixed_to_float(sum);
        }
    }
}

// ============================================================
// Main
// ============================================================

int main() {
    mem[7] = 0;
    mem[8] = 0;

    while (1) {
        // OP_EXIT = 255 - безопасный выход!
        if (mem[0] == 255) {
            mem[7] = 1;  // status = done
            break;
        }
        if (mem[0] == 1) {
            mem[7] = 0;

            unsigned int M = mem[1];
            unsigned int K = mem[2];
            unsigned int N = mem[3];

            unsigned int* A = (unsigned int*)mem[4];
            unsigned int* B = (unsigned int*)mem[5];
            unsigned int* C = (unsigned int*)mem[6];

            matmul(A, B, C, M, K, N);

            mem[8] = mul_u32(M, N);
            mem[7] = 1;
            mem[0] = 0;
        }
    }

    return 0;
}
