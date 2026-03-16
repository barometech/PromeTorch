// Elementwise operations for NMC4
// Add, Mul, MulAdd (for FFN gate), etc.
// All custom, no library calls - uses fixed-point Q16.16

#include "mymath.h"

#define DDR_BASE 0x00340000

volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

// Memory layout:
// [0] = cmd (op code)
// [1] = count (number of elements)
// [2] = addr_a
// [3] = addr_b (or scalar for some ops)
// [4] = addr_output
// [5] = reserved
// [6] = reserved
// [7] = status (0 = busy, 1 = done)

// Op codes:
// 1 = add: out = a + b
// 2 = mul: out = a * b
// 3 = sub: out = a - b
// 4 = mul_add: out = a * b + c (addr_c in [5])
// 5 = add_scalar: out = a + scalar (scalar in [3] as float)
// 6 = mul_scalar: out = a * scalar
// 7 = gate_mul: out = a * silu(b) (for Llama FFN gate)

void elem_add(unsigned int* a, unsigned int* b, unsigned int* out, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        fixed32 vb = float_to_fixed(b[i]);
        out[i] = fixed_to_float(add_fixed(va, vb));
    }
}

void elem_mul(unsigned int* a, unsigned int* b, unsigned int* out, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        fixed32 vb = float_to_fixed(b[i]);
        out[i] = fixed_to_float(mul_fixed(va, vb));
    }
}

void elem_sub(unsigned int* a, unsigned int* b, unsigned int* out, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        fixed32 vb = float_to_fixed(b[i]);
        out[i] = fixed_to_float(sub_fixed(va, vb));
    }
}

void elem_add_scalar(unsigned int* a, fixed32 scalar, unsigned int* out, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        out[i] = fixed_to_float(add_fixed(va, scalar));
    }
}

void elem_mul_scalar(unsigned int* a, fixed32 scalar, unsigned int* out, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        out[i] = fixed_to_float(mul_fixed(va, scalar));
    }
}

// For Llama FFN: gate = silu(gate_proj(x)) * up_proj(x)
// This computes: out = a * silu(b)
void gate_mul(unsigned int* a, unsigned int* b, unsigned int* out, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        fixed32 va = float_to_fixed(a[i]);
        fixed32 vb = float_to_fixed(b[i]);
        fixed32 silu_b = silu_fixed(vb);
        out[i] = fixed_to_float(mul_fixed(va, silu_b));
    }
}

int main() {
    mem[7] = 0;

    while (1) {
        unsigned int cmd = mem[0];
        // OP_EXIT = 255 - безопасный выход!
        if (cmd == 255) { mem[7] = 1; break; }
        if (cmd == 0) continue;

        mem[7] = 0;

        unsigned int count = mem[1];
        unsigned int* a = (unsigned int*)mem[2];
        unsigned int* b = (unsigned int*)mem[3];
        unsigned int* out = (unsigned int*)mem[4];

        switch (cmd) {
            case 1:  // add
                elem_add(a, b, out, count);
                break;

            case 2:  // mul
                elem_mul(a, b, out, count);
                break;

            case 3:  // sub
                elem_sub(a, b, out, count);
                break;

            case 5:  // add_scalar
                elem_add_scalar(a, float_to_fixed(mem[3]), out, count);
                break;

            case 6:  // mul_scalar
                elem_mul_scalar(a, float_to_fixed(mem[3]), out, count);
                break;

            case 7:  // gate_mul (for Llama FFN)
                gate_mul(a, b, out, count);
                break;

            default:
                break;
        }

        mem[7] = 1;
        mem[0] = 0;
    }

    return 0;
}
