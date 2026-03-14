#pragma once
// ============================================================================
// NMCardMath.h - Q16.16 Fixed-Point Math for NMCard Emulator
// ============================================================================
// Port of nm_card_mini_as_TRAINER/nmc_programs/mymath.h to x86
// On NMC4: no hardware MUL, no variable shifts, custom everything
// On x86: native multiply, native shifts — much simpler implementations
// Same numerical results (Q16.16 precision) for emulator fidelity

#include <cstdint>
#include <cmath>

namespace at {
namespace nmcard {

// ============================================================================
// Q16.16 Fixed-Point Type
// ============================================================================
// 16-bit integer part + 16-bit fractional part
// Range: [-32768.0, 32767.99998] with precision ~1.5e-5

using fixed32 = int32_t;

static constexpr int FIXED_SHIFT = 16;
static constexpr fixed32 FIXED_ONE = (1 << FIXED_SHIFT);       // 65536
static constexpr fixed32 FIXED_HALF = (1 << (FIXED_SHIFT - 1)); // 32768
static constexpr fixed32 FIXED_PI = 205887;      // pi in Q16.16
static constexpr fixed32 FIXED_TWO_PI = 411775;  // 2*pi
static constexpr fixed32 FIXED_PI_HALF = 102944;  // pi/2

// ============================================================================
// Conversions
// ============================================================================

inline fixed32 float_to_fixed(float f) {
    return static_cast<fixed32>(f * FIXED_ONE);
}

inline float fixed_to_float(fixed32 x) {
    return static_cast<float>(x) / FIXED_ONE;
}

inline fixed32 int_to_fixed(int i) {
    return static_cast<fixed32>(i << FIXED_SHIFT);
}

inline int fixed_to_int(fixed32 x) {
    return x >> FIXED_SHIFT;
}

// ============================================================================
// Basic Arithmetic
// ============================================================================

inline fixed32 add_fixed(fixed32 a, fixed32 b) {
    return a + b;
}

inline fixed32 sub_fixed(fixed32 a, fixed32 b) {
    return a - b;
}

// On x86: native 64-bit multiply, then shift
// On NMC4: mul_u32 via shift-add loop (no hardware MUL)
// Same Q16.16 result
inline fixed32 mul_fixed(fixed32 a, fixed32 b) {
    return static_cast<fixed32>((static_cast<int64_t>(a) * b) >> FIXED_SHIFT);
}

// Division: a / b in Q16.16
inline fixed32 div_fixed(fixed32 a, fixed32 b) {
    if (b == 0) return 0x7FFFFFFF; // max on div by zero
    return static_cast<fixed32>((static_cast<int64_t>(a) << FIXED_SHIFT) / b);
}

// ============================================================================
// Square Root (Newton-Raphson, same as NMC4 version)
// ============================================================================

inline fixed32 sqrt_fixed(fixed32 a) {
    if (a <= 0) return 0;

    // Initial approximation
    fixed32 x = a;
    if (x >= FIXED_ONE) {
        x = x >> 1;
        while (x > (FIXED_ONE << 7)) x = x >> 1;
    } else {
        x = FIXED_ONE;
    }

    // 8 iterations Newton-Raphson
    for (int i = 0; i < 8; i++) {
        if (x == 0) break;
        fixed32 x_new = (x + div_fixed(a, x)) >> 1;
        if (x_new == x) break;
        x = x_new;
    }

    return x;
}

// ============================================================================
// Exponential (LUT + linear interpolation, identical to NMC4)
// ============================================================================

// exp(-4.0) to exp(4.0) in steps of 0.5
static const fixed32 exp_lut[17] = {
    1202,       // exp(-4.0) = 0.0183
    1976,       // exp(-3.5) = 0.0302
    3248,       // exp(-3.0) = 0.0498
    5340,       // exp(-2.5) = 0.0821
    8784,       // exp(-2.0) = 0.1353
    14441,      // exp(-1.5) = 0.2231
    23730,      // exp(-1.0) = 0.3679
    39015,      // exp(-0.5) = 0.6065
    65536,      // exp(0.0)  = 1.0
    107837,     // exp(0.5)  = 1.6487
    177308,     // exp(1.0)  = 2.7183
    291433,     // exp(1.5)  = 4.4817
    479198,     // exp(2.0)  = 7.3891
    787935,     // exp(2.5)  = 12.182
    1295356,    // exp(3.0)  = 20.086
    2130162,    // exp(3.5)  = 33.115
    3502898     // exp(4.0)  = 54.598
};

inline fixed32 exp_fixed_lut(fixed32 x) {
    const fixed32 MAX_X = 4 << FIXED_SHIFT;
    const fixed32 MIN_X = -(4 << FIXED_SHIFT);

    if (x >= MAX_X) return exp_lut[16];
    if (x <= MIN_X) return exp_lut[0];

    int idx_fixed = x + MAX_X; // [0, 8*65536]
    int idx = idx_fixed >> 15; // /32768 = /0.5 in Q16
    if (idx < 0) idx = 0;
    if (idx > 15) idx = 15;

    // Linear interpolation
    fixed32 frac = idx_fixed - (idx << 15);
    fixed32 v0 = exp_lut[idx];
    fixed32 v1 = exp_lut[idx + 1];

    fixed32 diff = v1 - v0;
    fixed32 interp = mul_fixed(diff << 1, frac);

    return v0 + (interp >> 1);
}

// ============================================================================
// Activation Functions
// ============================================================================

inline fixed32 sigmoid_fixed(fixed32 x) {
    const fixed32 MAX_X = 4 << FIXED_SHIFT;

    if (x >= MAX_X) return FIXED_ONE;
    if (x <= -MAX_X) return 0;

    fixed32 exp_neg_x = exp_fixed_lut(-x);
    fixed32 denom = add_fixed(FIXED_ONE, exp_neg_x);
    return div_fixed(FIXED_ONE, denom);
}

inline fixed32 silu_fixed(fixed32 x) {
    return mul_fixed(x, sigmoid_fixed(x));
}

inline fixed32 gelu_fixed(fixed32 x) {
    // GELU ~ x * sigmoid(1.702 * x)
    const fixed32 GELU_COEF = 111543; // 1.702 in Q16.16
    fixed32 scaled = mul_fixed(x, GELU_COEF);
    return mul_fixed(x, sigmoid_fixed(scaled));
}

inline fixed32 relu_fixed(fixed32 x) {
    return x > 0 ? x : 0;
}

inline fixed32 tanh_fixed(fixed32 x) {
    fixed32 sig = sigmoid_fixed(x << 1); // sigmoid(2x)
    return sub_fixed(sig << 1, FIXED_ONE); // 2*sig - 1
}

// ============================================================================
// Trigonometric (for RoPE)
// ============================================================================

inline fixed32 sin_fixed(fixed32 x) {
    // Range reduction to [-pi, pi]
    while (x > FIXED_PI) x = sub_fixed(x, FIXED_TWO_PI);
    while (x < -FIXED_PI) x = add_fixed(x, FIXED_TWO_PI);

    // Taylor: sin(x) = x - x^3/6 + x^5/120
    fixed32 x2 = mul_fixed(x, x);
    fixed32 x3 = mul_fixed(x2, x);
    fixed32 x5 = mul_fixed(x3, x2);

    fixed32 term3 = div_fixed(x3, int_to_fixed(6));
    fixed32 term5 = div_fixed(x5, int_to_fixed(120));

    return sub_fixed(add_fixed(x, term5), term3);
}

inline fixed32 cos_fixed(fixed32 x) {
    return sin_fixed(add_fixed(x, FIXED_PI_HALF));
}

// ============================================================================
// Backward Functions (from mymath_backward.h)
// ============================================================================

inline fixed32 relu_backward(fixed32 x, fixed32 grad_output) {
    return x > 0 ? grad_output : 0;
}

inline fixed32 sigmoid_backward(fixed32 x, fixed32 grad_output) {
    fixed32 sig = sigmoid_fixed(x);
    fixed32 local_grad = mul_fixed(sig, sub_fixed(FIXED_ONE, sig));
    return mul_fixed(grad_output, local_grad);
}

inline fixed32 silu_backward(fixed32 x, fixed32 grad_output) {
    fixed32 sig = sigmoid_fixed(x);
    fixed32 one_minus_sig = sub_fixed(FIXED_ONE, sig);
    fixed32 x_term = mul_fixed(x, one_minus_sig);
    fixed32 local_grad = mul_fixed(sig, add_fixed(FIXED_ONE, x_term));
    return mul_fixed(grad_output, local_grad);
}

inline fixed32 gelu_backward(fixed32 x, fixed32 grad_output) {
    const fixed32 GELU_COEF = 111543;
    fixed32 scaled = mul_fixed(x, GELU_COEF);
    fixed32 sig = sigmoid_fixed(scaled);
    fixed32 one_minus_sig = sub_fixed(FIXED_ONE, sig);
    fixed32 term2 = mul_fixed(scaled, mul_fixed(sig, one_minus_sig));
    fixed32 local_grad = add_fixed(sig, term2);
    return mul_fixed(grad_output, local_grad);
}

inline fixed32 tanh_backward(fixed32 x, fixed32 grad_output) {
    fixed32 t = tanh_fixed(x);
    fixed32 local_grad = sub_fixed(FIXED_ONE, mul_fixed(t, t));
    return mul_fixed(grad_output, local_grad);
}

} // namespace nmcard
} // namespace at
