#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <complex>
#include <cmath>
#include "c10/macros/Macros.h"
#include "c10/util/Exception.h"

namespace c10 {

// ============================================================================
// c10::complex<T> — header-only complex number type.
//
// We avoid std::complex in deep template instantiation paths because LCC
// (Elbrus compiler) is occasionally fragile with std::complex<float>'s
// template specializations and operator overload resolution. This type is
// trivially-copyable, layout-compatible with { T re; T im; }, and has all
// arithmetic the dispatched scalar_t code paths need.
// ============================================================================

template <typename T>
struct alignas(sizeof(T) * 2) complex {
    T re;
    T im;

    complex() = default;
    PT_HOST_DEVICE complex(T r) : re(r), im(T(0)) {}
    PT_HOST_DEVICE complex(T r, T i) : re(r), im(i) {}
    // Cross-precision construct (Complex64 <-> Complex128, etc.)
    template <typename U>
    PT_HOST_DEVICE explicit complex(const complex<U>& o)
        : re(static_cast<T>(o.re)), im(static_cast<T>(o.im)) {}
    // Integer / double convenience ctors so templated code that writes
    // `scalar_t x = 0;` or `scalar_t s = 1;` compiles.
    PT_HOST_DEVICE complex(int v)         : re(static_cast<T>(v)), im(T(0)) {}
    PT_HOST_DEVICE complex(long v)        : re(static_cast<T>(v)), im(T(0)) {}
    PT_HOST_DEVICE complex(long long v)   : re(static_cast<T>(v)), im(T(0)) {}
    // The double overload would collide with `complex(T r)` when T==double, so
    // gate it on T != double via SFINAE.
    template <typename U = T,
              typename = typename std::enable_if<!std::is_same<U, double>::value>::type>
    PT_HOST_DEVICE complex(double v)      : re(static_cast<T>(v)), im(T(0)) {}

    PT_HOST_DEVICE T real() const { return re; }
    PT_HOST_DEVICE T imag() const { return im; }

    PT_HOST_DEVICE complex& operator+=(const complex& o) { re += o.re; im += o.im; return *this; }
    PT_HOST_DEVICE complex& operator-=(const complex& o) { re -= o.re; im -= o.im; return *this; }
    PT_HOST_DEVICE complex& operator*=(const complex& o) {
        T nr = re * o.re - im * o.im;
        T ni = re * o.im + im * o.re;
        re = nr; im = ni; return *this;
    }
    PT_HOST_DEVICE complex& operator/=(const complex& o) {
        T denom = o.re * o.re + o.im * o.im;
        T nr = (re * o.re + im * o.im) / denom;
        T ni = (im * o.re - re * o.im) / denom;
        re = nr; im = ni; return *this;
    }
};

template <typename T>
inline complex<T> operator+(complex<T> a, complex<T> b) { return complex<T>(a.re + b.re, a.im + b.im); }
template <typename T>
inline complex<T> operator-(complex<T> a, complex<T> b) { return complex<T>(a.re - b.re, a.im - b.im); }
template <typename T>
inline complex<T> operator*(complex<T> a, complex<T> b) {
    return complex<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
template <typename T>
inline complex<T> operator/(complex<T> a, complex<T> b) {
    T denom = b.re * b.re + b.im * b.im;
    return complex<T>((a.re * b.re + a.im * b.im) / denom,
                      (a.im * b.re - a.re * b.im) / denom);
}
template <typename T>
inline complex<T> operator-(complex<T> a) { return complex<T>(-a.re, -a.im); }
template <typename T>
inline bool operator==(complex<T> a, complex<T> b) { return a.re == b.re && a.im == b.im; }
template <typename T>
inline bool operator!=(complex<T> a, complex<T> b) { return !(a == b); }

// Complex op real-scalar — used by templated kernels that mix scalar literals
// (e.g. `scalar_t s = 0; s += a[i] * b[i]` where literals participate).
template <typename T>
inline complex<T> operator*(complex<T> a, T s) { return complex<T>(a.re * s, a.im * s); }
template <typename T>
inline complex<T> operator*(T s, complex<T> a) { return complex<T>(a.re * s, a.im * s); }
template <typename T>
inline complex<T> operator+(complex<T> a, T s) { return complex<T>(a.re + s, a.im); }
template <typename T>
inline complex<T> operator+(T s, complex<T> a) { return complex<T>(a.re + s, a.im); }

// Magnitude (sqrt(re^2 + im^2)) — used by abs(complex tensor).
template <typename T>
inline T abs(complex<T> z) { return std::sqrt(z.re * z.re + z.im * z.im); }

template <typename T>
inline complex<T> conj(complex<T> z) { return complex<T>(z.re, -z.im); }

using Complex64  = complex<float>;
using Complex128 = complex<double>;

// ============================================================================
// Half Precision Float (FP16)
// ============================================================================

struct alignas(2) Half {
    uint16_t x;

    Half() = default;

    PT_HOST_DEVICE Half(float f) : x(float_to_half_bits(f)) {}
    PT_HOST_DEVICE Half(double d) : x(float_to_half_bits(static_cast<float>(d))) {}
    // Integer ctors needed for templated numeric code (`scalar_t x = 0;`,
    // `(bool) ? 1 : -1`, etc.).
    PT_HOST_DEVICE Half(int i) : x(float_to_half_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Half(long i) : x(float_to_half_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Half(long long i) : x(float_to_half_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Half(unsigned int i) : x(float_to_half_bits(static_cast<float>(i))) {}

    PT_HOST_DEVICE Half& operator=(float f) {
        x = float_to_half_bits(f);
        return *this;
    }

    PT_HOST_DEVICE operator float() const {
        return half_bits_to_float(x);
    }

private:
    static uint16_t float_to_half_bits(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(f));

        uint16_t sign = (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFF;

        if (exp <= 0) {
            if (exp < -10) {
                return sign;
            }
            mantissa = (mantissa | 0x800000) >> (1 - exp);
            return sign | (mantissa >> 13);
        } else if (exp >= 31) {
            return sign | 0x7C00;  // Infinity
        }

        return sign | (exp << 10) | (mantissa >> 13);
    }

    static float half_bits_to_float(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;

        if (exp == 0) {
            if (mantissa == 0) {
                uint32_t result = sign;
                float f;
                std::memcpy(&f, &result, sizeof(f));
                return f;
            }
            // Denormalized
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            exp++;
            mantissa &= ~0x400;
        } else if (exp == 31) {
            uint32_t result = sign | 0x7F800000 | (mantissa << 13);
            float f;
            std::memcpy(&f, &result, sizeof(f));
            return f;
        }

        exp = exp + 127 - 15;
        uint32_t result = sign | (exp << 23) | (mantissa << 13);
        float f;
        std::memcpy(&f, &result, sizeof(f));
        return f;
    }
};

// Half precision arithmetic operators
inline Half operator+(Half a, Half b) { return Half(float(a) + float(b)); }
inline Half operator-(Half a, Half b) { return Half(float(a) - float(b)); }
inline Half operator*(Half a, Half b) { return Half(float(a) * float(b)); }
inline Half operator/(Half a, Half b) { return Half(float(a) / float(b)); }
inline Half operator-(Half a) { return Half(-float(a)); }
// Compound assignment — Half is an output type of PT_DISPATCH_FLOATING_TYPES.
inline Half& operator+=(Half& a, Half b) { a = Half(float(a) + float(b)); return a; }
inline Half& operator-=(Half& a, Half b) { a = Half(float(a) - float(b)); return a; }
inline Half& operator*=(Half& a, Half b) { a = Half(float(a) * float(b)); return a; }
inline Half& operator/=(Half& a, Half b) { a = Half(float(a) / float(b)); return a; }
inline bool operator==(Half a, Half b) { return float(a) == float(b); }
inline bool operator!=(Half a, Half b) { return float(a) != float(b); }
inline bool operator<(Half a, Half b) { return float(a) < float(b); }
inline bool operator>(Half a, Half b) { return float(a) > float(b); }
inline bool operator<=(Half a, Half b) { return float(a) <= float(b); }
inline bool operator>=(Half a, Half b) { return float(a) >= float(b); }
// Comparisons with int / float / double on both sides, and mixed arithmetic.
// Needed for templated linear-algebra code that writes `v[0] >= 0`, `2 * dot_val`,
// `sum / diag`, `1e-15`, etc. when scalar_t instantiates as Half.
inline bool operator==(Half a, int b) { return float(a) == float(b); }
inline bool operator!=(Half a, int b) { return float(a) != float(b); }
inline bool operator<(Half a, int b)  { return float(a) <  float(b); }
inline bool operator>(Half a, int b)  { return float(a) >  float(b); }
inline bool operator<=(Half a, int b) { return float(a) <= float(b); }
inline bool operator>=(Half a, int b) { return float(a) >= float(b); }
inline bool operator==(int a, Half b) { return float(a) == float(b); }
inline bool operator!=(int a, Half b) { return float(a) != float(b); }
inline bool operator<(int a, Half b)  { return float(a) <  float(b); }
inline bool operator>(int a, Half b)  { return float(a) >  float(b); }
inline bool operator<=(int a, Half b) { return float(a) <= float(b); }
inline bool operator>=(int a, Half b) { return float(a) >= float(b); }
inline bool operator<(Half a, double b)  { return float(a) <  (float)b; }
inline bool operator>(Half a, double b)  { return float(a) >  (float)b; }
inline bool operator<=(Half a, double b) { return float(a) <= (float)b; }
inline bool operator>=(Half a, double b) { return float(a) >= (float)b; }
inline bool operator<(double a, Half b)  { return (float)a <  float(b); }
inline bool operator>(double a, Half b)  { return (float)a >  float(b); }
// Mixed arithmetic (produce Half).
inline Half operator+(Half a, int b) { return Half(float(a) + float(b)); }
inline Half operator-(Half a, int b) { return Half(float(a) - float(b)); }
inline Half operator*(Half a, int b) { return Half(float(a) * float(b)); }
inline Half operator/(Half a, int b) { return Half(float(a) / float(b)); }
inline Half operator+(int a, Half b) { return Half(float(a) + float(b)); }
inline Half operator-(int a, Half b) { return Half(float(a) - float(b)); }
inline Half operator*(int a, Half b) { return Half(float(a) * float(b)); }
inline Half operator/(int a, Half b) { return Half(float(a) / float(b)); }
inline Half operator+(Half a, double b) { return Half(float(a) + (float)b); }
inline Half operator-(Half a, double b) { return Half(float(a) - (float)b); }
inline Half operator*(Half a, double b) { return Half(float(a) * (float)b); }
inline Half operator/(Half a, double b) { return Half(float(a) / (float)b); }
inline Half operator+(double a, Half b) { return Half((float)a + float(b)); }
inline Half operator-(double a, Half b) { return Half((float)a - float(b)); }
inline Half operator*(double a, Half b) { return Half((float)a * float(b)); }
inline Half operator/(double a, Half b) { return Half((float)a / float(b)); }

// ============================================================================
// BFloat16 (Brain Floating Point)
// ============================================================================

struct alignas(2) BFloat16 {
    uint16_t x;

    BFloat16() = default;

    PT_HOST_DEVICE BFloat16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(f));
        // Simply truncate lower 16 bits (round towards zero)
        // For better accuracy, could add rounding
        x = static_cast<uint16_t>(bits >> 16);
    }
    PT_HOST_DEVICE BFloat16(double d) { *this = BFloat16(static_cast<float>(d)); }
    PT_HOST_DEVICE BFloat16(int i) { *this = BFloat16(static_cast<float>(i)); }
    PT_HOST_DEVICE BFloat16(long i) { *this = BFloat16(static_cast<float>(i)); }
    PT_HOST_DEVICE BFloat16(long long i) { *this = BFloat16(static_cast<float>(i)); }
    PT_HOST_DEVICE BFloat16(unsigned int i) { *this = BFloat16(static_cast<float>(i)); }

    PT_HOST_DEVICE BFloat16& operator=(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(f));
        x = static_cast<uint16_t>(bits >> 16);
        return *this;
    }

    PT_HOST_DEVICE operator float() const {
        uint32_t bits = static_cast<uint32_t>(x) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

// BFloat16 arithmetic operators
inline BFloat16 operator+(BFloat16 a, BFloat16 b) { return BFloat16(float(a) + float(b)); }
inline BFloat16 operator-(BFloat16 a, BFloat16 b) { return BFloat16(float(a) - float(b)); }
inline BFloat16 operator*(BFloat16 a, BFloat16 b) { return BFloat16(float(a) * float(b)); }
inline BFloat16 operator/(BFloat16 a, BFloat16 b) { return BFloat16(float(a) / float(b)); }
inline BFloat16 operator-(BFloat16 a) { return BFloat16(-float(a)); }
// Compound assignment.
inline BFloat16& operator+=(BFloat16& a, BFloat16 b) { a = BFloat16(float(a) + float(b)); return a; }
inline BFloat16& operator-=(BFloat16& a, BFloat16 b) { a = BFloat16(float(a) - float(b)); return a; }
inline BFloat16& operator*=(BFloat16& a, BFloat16 b) { a = BFloat16(float(a) * float(b)); return a; }
inline BFloat16& operator/=(BFloat16& a, BFloat16 b) { a = BFloat16(float(a) / float(b)); return a; }
// Comparisons with int / double + mixed arithmetic — same purpose as Half above.
inline bool operator==(BFloat16 a, int b) { return float(a) == float(b); }
inline bool operator!=(BFloat16 a, int b) { return float(a) != float(b); }
inline bool operator<(BFloat16 a, int b)  { return float(a) <  float(b); }
inline bool operator>(BFloat16 a, int b)  { return float(a) >  float(b); }
inline bool operator<=(BFloat16 a, int b) { return float(a) <= float(b); }
inline bool operator>=(BFloat16 a, int b) { return float(a) >= float(b); }
inline bool operator==(int a, BFloat16 b) { return float(a) == float(b); }
inline bool operator!=(int a, BFloat16 b) { return float(a) != float(b); }
inline bool operator<(int a, BFloat16 b)  { return float(a) <  float(b); }
inline bool operator>(int a, BFloat16 b)  { return float(a) >  float(b); }
inline bool operator<=(int a, BFloat16 b) { return float(a) <= float(b); }
inline bool operator>=(int a, BFloat16 b) { return float(a) >= float(b); }
inline bool operator<(BFloat16 a, double b)  { return float(a) <  (float)b; }
inline bool operator>(BFloat16 a, double b)  { return float(a) >  (float)b; }
inline bool operator<=(BFloat16 a, double b) { return float(a) <= (float)b; }
inline bool operator>=(BFloat16 a, double b) { return float(a) >= (float)b; }
inline BFloat16 operator+(BFloat16 a, int b) { return BFloat16(float(a) + float(b)); }
inline BFloat16 operator-(BFloat16 a, int b) { return BFloat16(float(a) - float(b)); }
inline BFloat16 operator*(BFloat16 a, int b) { return BFloat16(float(a) * float(b)); }
inline BFloat16 operator/(BFloat16 a, int b) { return BFloat16(float(a) / float(b)); }
inline BFloat16 operator+(int a, BFloat16 b) { return BFloat16(float(a) + float(b)); }
inline BFloat16 operator-(int a, BFloat16 b) { return BFloat16(float(a) - float(b)); }
inline BFloat16 operator*(int a, BFloat16 b) { return BFloat16(float(a) * float(b)); }
inline BFloat16 operator/(int a, BFloat16 b) { return BFloat16(float(a) / float(b)); }
inline BFloat16 operator+(BFloat16 a, double b) { return BFloat16(float(a) + (float)b); }
inline BFloat16 operator-(BFloat16 a, double b) { return BFloat16(float(a) - (float)b); }
inline BFloat16 operator*(BFloat16 a, double b) { return BFloat16(float(a) * (float)b); }
inline BFloat16 operator/(BFloat16 a, double b) { return BFloat16(float(a) / (float)b); }
inline BFloat16 operator+(double a, BFloat16 b) { return BFloat16((float)a + float(b)); }
inline BFloat16 operator-(double a, BFloat16 b) { return BFloat16((float)a - float(b)); }
inline BFloat16 operator*(double a, BFloat16 b) { return BFloat16((float)a * float(b)); }
inline BFloat16 operator/(double a, BFloat16 b) { return BFloat16((float)a / float(b)); }
inline bool operator==(BFloat16 a, BFloat16 b) { return float(a) == float(b); }
inline bool operator!=(BFloat16 a, BFloat16 b) { return float(a) != float(b); }
inline bool operator<(BFloat16 a, BFloat16 b) { return float(a) < float(b); }
inline bool operator>(BFloat16 a, BFloat16 b) { return float(a) > float(b); }

// ============================================================================
// FP8 — 8-bit floating point (NVIDIA FP8 paper: arxiv.org/abs/2209.05433)
//
// Two formats are supported:
//   - Float8_e4m3fn : sign=1, exp=4 (bias=7), mantissa=3.
//                     No infinities. NaN = 0x7F (positive) and 0xFF (negative).
//                     Range ~ +/- 448, finer mantissa resolution.
//                     Suffix "fn" = "finite, NaN-only" (FN encoding).
//   - Float8_e5m2   : sign=1, exp=5 (bias=15), mantissa=2.
//                     Has +/-Inf (0x7C / 0xFC) and NaN (0x7D-0x7F, 0xFD-0xFF).
//                     Range ~ +/- 57344, wider exponent range, less precision.
//
// Conversion strategy: round-to-nearest-even from float, with subnormal
// handling. Arithmetic goes via float roundtrip (standard pattern for
// emulated FP8 on CPU — same as Half / BFloat16 above).
// ============================================================================

struct alignas(1) Float8_e4m3fn {
    uint8_t x;

    Float8_e4m3fn() = default;
    PT_HOST_DEVICE Float8_e4m3fn(float f) : x(float_to_e4m3fn_bits(f)) {}
    PT_HOST_DEVICE Float8_e4m3fn(double d) : x(float_to_e4m3fn_bits(static_cast<float>(d))) {}
    PT_HOST_DEVICE Float8_e4m3fn(int i) : x(float_to_e4m3fn_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Float8_e4m3fn(long i) : x(float_to_e4m3fn_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Float8_e4m3fn(long long i) : x(float_to_e4m3fn_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Float8_e4m3fn(unsigned int i) : x(float_to_e4m3fn_bits(static_cast<float>(i))) {}

    PT_HOST_DEVICE Float8_e4m3fn& operator=(float f) {
        x = float_to_e4m3fn_bits(f);
        return *this;
    }

    PT_HOST_DEVICE operator float() const {
        return e4m3fn_bits_to_float(x);
    }

private:
    static uint8_t float_to_e4m3fn_bits(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(f));
        uint8_t sign = (bits >> 24) & 0x80;
        uint32_t fabs = bits & 0x7FFFFFFF;
        // NaN → canonical NaN with sign preserved.
        if (fabs > 0x7F800000) return sign | 0x7F;
        // Saturate to max-finite (+/-448) for Inf and overflow (e4m3fn has no Inf).
        // Max finite = 0x7E (S=0, E=1111, M=110) = 1.75 * 2^8 = 448.
        if (fabs >= 0x43E00000) return sign | 0x7E;  // 448.0f threshold
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 7;
        uint32_t mantissa = bits & 0x7FFFFF;
        if (exp <= 0) {
            // Subnormal e4m3fn: shift mantissa with implicit 1, round-to-nearest-even.
            if (exp < -3) return sign;  // underflow to zero (smallest subnormal exp ~ -9 in float terms)
            mantissa |= 0x800000;
            int shift = 21 - exp;  // bring 23-bit mantissa down to 3-bit subnormal field
            uint32_t round_mask = (1u << shift) - 1;
            uint32_t halfway = 1u << (shift - 1);
            uint32_t round_bits = mantissa & round_mask;
            uint32_t result = mantissa >> shift;
            // Round-to-nearest-even.
            if (round_bits > halfway || (round_bits == halfway && (result & 1))) result++;
            return sign | static_cast<uint8_t>(result);
        }
        // Normal: 4-bit exp, 3-bit mantissa with RNE rounding of low 20 bits.
        uint32_t round_bits = mantissa & 0xFFFFF;       // bottom 20 bits
        uint32_t halfway    = 0x80000;                  // 1 << 19
        uint32_t mant3      = mantissa >> 20;
        if (round_bits > halfway || (round_bits == halfway && (mant3 & 1))) {
            mant3++;
            if (mant3 == 8) { mant3 = 0; exp++; }       // mantissa overflow
        }
        if (exp >= 15) return sign | 0x7E;              // saturate to max-finite
        return sign | static_cast<uint8_t>((exp << 3) | mant3);
    }

    static float e4m3fn_bits_to_float(uint8_t v) {
        uint32_t sign = static_cast<uint32_t>(v & 0x80) << 24;
        uint32_t exp  = (v >> 3) & 0x0F;
        uint32_t mant = v & 0x07;
        uint32_t out;
        if (exp == 0 && mant == 0) {
            out = sign;                                  // +/- 0
        } else if ((v & 0x7F) == 0x7F) {
            out = sign | 0x7FC00000;                     // canonical NaN
        } else if (exp == 0) {
            // Subnormal: normalize mantissa.
            int e = -6;                                  // bias-shifted subnormal exponent
            while ((mant & 0x08) == 0) { mant <<= 1; e--; }
            mant &= 0x07;
            out = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 20);
        } else {
            out = sign | (static_cast<uint32_t>(exp - 7 + 127) << 23) | (mant << 20);
        }
        float f;
        std::memcpy(&f, &out, sizeof(f));
        return f;
    }
};

struct alignas(1) Float8_e5m2 {
    uint8_t x;

    Float8_e5m2() = default;
    PT_HOST_DEVICE Float8_e5m2(float f) : x(float_to_e5m2_bits(f)) {}
    PT_HOST_DEVICE Float8_e5m2(double d) : x(float_to_e5m2_bits(static_cast<float>(d))) {}
    PT_HOST_DEVICE Float8_e5m2(int i) : x(float_to_e5m2_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Float8_e5m2(long i) : x(float_to_e5m2_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Float8_e5m2(long long i) : x(float_to_e5m2_bits(static_cast<float>(i))) {}
    PT_HOST_DEVICE Float8_e5m2(unsigned int i) : x(float_to_e5m2_bits(static_cast<float>(i))) {}

    PT_HOST_DEVICE Float8_e5m2& operator=(float f) {
        x = float_to_e5m2_bits(f);
        return *this;
    }

    PT_HOST_DEVICE operator float() const {
        return e5m2_bits_to_float(x);
    }

private:
    static uint8_t float_to_e5m2_bits(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(f));
        uint8_t sign = (bits >> 24) & 0x80;
        uint32_t fabs = bits & 0x7FFFFFFF;
        if (fabs > 0x7F800000) return sign | 0x7F;       // NaN
        if (fabs == 0x7F800000) return sign | 0x7C;      // +/- Inf
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFF;
        if (exp <= 0) {
            if (exp < -2) return sign;                   // underflow → zero
            mantissa |= 0x800000;
            int shift = 22 - exp;
            uint32_t round_mask = (1u << shift) - 1;
            uint32_t halfway = 1u << (shift - 1);
            uint32_t round_bits = mantissa & round_mask;
            uint32_t result = mantissa >> shift;
            if (round_bits > halfway || (round_bits == halfway && (result & 1))) result++;
            return sign | static_cast<uint8_t>(result);
        }
        uint32_t round_bits = mantissa & 0x1FFFFF;       // bottom 21 bits
        uint32_t halfway    = 0x100000;                  // 1 << 20
        uint32_t mant2      = mantissa >> 21;
        if (round_bits > halfway || (round_bits == halfway && (mant2 & 1))) {
            mant2++;
            if (mant2 == 4) { mant2 = 0; exp++; }
        }
        if (exp >= 31) return sign | 0x7C;               // overflow → Inf
        return sign | static_cast<uint8_t>((exp << 2) | mant2);
    }

    static float e5m2_bits_to_float(uint8_t v) {
        uint32_t sign = static_cast<uint32_t>(v & 0x80) << 24;
        uint32_t exp  = (v >> 2) & 0x1F;
        uint32_t mant = v & 0x03;
        uint32_t out;
        if (exp == 0 && mant == 0) {
            out = sign;
        } else if (exp == 0x1F) {
            out = sign | 0x7F800000 | (mant << 21);      // Inf or NaN (NaN has mant!=0)
        } else if (exp == 0) {
            int e = -14;
            while ((mant & 0x04) == 0) { mant <<= 1; e--; }
            mant &= 0x03;
            out = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 21);
        } else {
            out = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 21);
        }
        float f;
        std::memcpy(&f, &out, sizeof(f));
        return f;
    }
};

// FP8 arithmetic — via float roundtrip. Same template as Half / BFloat16.
#define PT_DEFINE_FP8_OPS(T) \
    inline T operator+(T a, T b) { return T(float(a) + float(b)); } \
    inline T operator-(T a, T b) { return T(float(a) - float(b)); } \
    inline T operator*(T a, T b) { return T(float(a) * float(b)); } \
    inline T operator/(T a, T b) { return T(float(a) / float(b)); } \
    inline T operator-(T a) { return T(-float(a)); } \
    inline T& operator+=(T& a, T b) { a = T(float(a) + float(b)); return a; } \
    inline T& operator-=(T& a, T b) { a = T(float(a) - float(b)); return a; } \
    inline T& operator*=(T& a, T b) { a = T(float(a) * float(b)); return a; } \
    inline T& operator/=(T& a, T b) { a = T(float(a) / float(b)); return a; } \
    inline bool operator==(T a, T b) { return float(a) == float(b); } \
    inline bool operator!=(T a, T b) { return float(a) != float(b); } \
    inline bool operator<(T a, T b)  { return float(a) <  float(b); } \
    inline bool operator>(T a, T b)  { return float(a) >  float(b); } \
    inline bool operator<=(T a, T b) { return float(a) <= float(b); } \
    inline bool operator>=(T a, T b) { return float(a) >= float(b); } \
    inline bool operator==(T a, int b) { return float(a) == float(b); } \
    inline bool operator!=(T a, int b) { return float(a) != float(b); } \
    inline bool operator<(T a, int b)  { return float(a) <  float(b); } \
    inline bool operator>(T a, int b)  { return float(a) >  float(b); } \
    inline bool operator==(int a, T b) { return float(a) == float(b); } \
    inline bool operator!=(int a, T b) { return float(a) != float(b); } \
    inline bool operator<(int a, T b)  { return float(a) <  float(b); } \
    inline bool operator>(int a, T b)  { return float(a) >  float(b); }

PT_DEFINE_FP8_OPS(Float8_e4m3fn)
PT_DEFINE_FP8_OPS(Float8_e5m2)
#undef PT_DEFINE_FP8_OPS

// ============================================================================
// ScalarType Enumeration
// ============================================================================

enum class ScalarType : int8_t {
    Byte = 0,           // uint8_t
    Char = 1,           // int8_t
    Short = 2,          // int16_t
    Int = 3,            // int32_t
    Long = 4,           // int64_t
    Half = 5,           // Half (float16)
    Float = 6,          // float
    Double = 7,         // double
    ComplexHalf = 8,    // complex<Half>
    ComplexFloat = 9,   // complex<float>
    ComplexDouble = 10, // complex<double>
    Bool = 11,          // bool
    BFloat16 = 12,      // BFloat16

    // Quantized types
    QInt8 = 13,
    QUInt8 = 14,
    QInt32 = 15,

    // Undefined
    Undefined = 16,

    // FP8 — NVIDIA FP8 (arxiv.org/abs/2209.05433)
    Float8_e4m3fn = 17,
    Float8_e5m2   = 18,

    // Number of types
    NumOptions = 19
};

// ============================================================================
// Type traits and mappings
// ============================================================================

namespace impl {

template<typename T>
struct ScalarTypeToCPPType;

#define DEFINE_SCALAR_TYPE_MAPPING(scalar_type, cpp_type) \
    template<> \
    struct ScalarTypeToCPPType<std::integral_constant<ScalarType, ScalarType::scalar_type>> { \
        using type = cpp_type; \
    };

DEFINE_SCALAR_TYPE_MAPPING(Byte, uint8_t)
DEFINE_SCALAR_TYPE_MAPPING(Char, int8_t)
DEFINE_SCALAR_TYPE_MAPPING(Short, int16_t)
DEFINE_SCALAR_TYPE_MAPPING(Int, int32_t)
DEFINE_SCALAR_TYPE_MAPPING(Long, int64_t)
DEFINE_SCALAR_TYPE_MAPPING(Half, c10::Half)
DEFINE_SCALAR_TYPE_MAPPING(Float, float)
DEFINE_SCALAR_TYPE_MAPPING(Double, double)
DEFINE_SCALAR_TYPE_MAPPING(ComplexHalf, c10::complex<c10::Half>)
DEFINE_SCALAR_TYPE_MAPPING(ComplexFloat, c10::complex<float>)
DEFINE_SCALAR_TYPE_MAPPING(ComplexDouble, c10::complex<double>)
DEFINE_SCALAR_TYPE_MAPPING(Bool, bool)
DEFINE_SCALAR_TYPE_MAPPING(BFloat16, c10::BFloat16)
DEFINE_SCALAR_TYPE_MAPPING(Float8_e4m3fn, c10::Float8_e4m3fn)
DEFINE_SCALAR_TYPE_MAPPING(Float8_e5m2, c10::Float8_e5m2)

#undef DEFINE_SCALAR_TYPE_MAPPING

// CPP type to ScalarType
template<typename T>
struct CppTypeToScalarType;

#define DEFINE_CPP_TYPE_MAPPING(cpp_type, scalar_type) \
    template<> \
    struct CppTypeToScalarType<cpp_type> { \
        static constexpr ScalarType value = ScalarType::scalar_type; \
    };

DEFINE_CPP_TYPE_MAPPING(uint8_t, Byte)
DEFINE_CPP_TYPE_MAPPING(int8_t, Char)
DEFINE_CPP_TYPE_MAPPING(int16_t, Short)
DEFINE_CPP_TYPE_MAPPING(int32_t, Int)
DEFINE_CPP_TYPE_MAPPING(int64_t, Long)
DEFINE_CPP_TYPE_MAPPING(c10::Half, Half)
DEFINE_CPP_TYPE_MAPPING(float, Float)
DEFINE_CPP_TYPE_MAPPING(double, Double)
DEFINE_CPP_TYPE_MAPPING(c10::complex<float>, ComplexFloat)
DEFINE_CPP_TYPE_MAPPING(c10::complex<double>, ComplexDouble)
DEFINE_CPP_TYPE_MAPPING(bool, Bool)
DEFINE_CPP_TYPE_MAPPING(c10::BFloat16, BFloat16)
DEFINE_CPP_TYPE_MAPPING(c10::Float8_e4m3fn, Float8_e4m3fn)
DEFINE_CPP_TYPE_MAPPING(c10::Float8_e5m2, Float8_e5m2)

#undef DEFINE_CPP_TYPE_MAPPING

} // namespace impl

// ============================================================================
// ScalarType Properties
// ============================================================================

constexpr size_t elementSize(ScalarType type) {
    switch (type) {
        case ScalarType::Byte: return sizeof(uint8_t);
        case ScalarType::Char: return sizeof(int8_t);
        case ScalarType::Short: return sizeof(int16_t);
        case ScalarType::Int: return sizeof(int32_t);
        case ScalarType::Long: return sizeof(int64_t);
        case ScalarType::Half: return sizeof(Half);
        case ScalarType::Float: return sizeof(float);
        case ScalarType::Double: return sizeof(double);
        case ScalarType::ComplexHalf: return 2 * sizeof(Half);
        case ScalarType::ComplexFloat: return 2 * sizeof(float);
        case ScalarType::ComplexDouble: return 2 * sizeof(double);
        case ScalarType::Bool: return sizeof(bool);
        case ScalarType::BFloat16: return sizeof(BFloat16);
        case ScalarType::QInt8: return sizeof(int8_t);
        case ScalarType::QUInt8: return sizeof(uint8_t);
        case ScalarType::QInt32: return sizeof(int32_t);
        case ScalarType::Float8_e4m3fn: return 1;
        case ScalarType::Float8_e5m2: return 1;
        default: return 0;
    }
}

constexpr bool isFloatingType(ScalarType type) {
    return type == ScalarType::Half ||
           type == ScalarType::Float ||
           type == ScalarType::Double ||
           type == ScalarType::BFloat16 ||
           type == ScalarType::Float8_e4m3fn ||
           type == ScalarType::Float8_e5m2;
}

constexpr bool isComplexType(ScalarType type) {
    return type == ScalarType::ComplexHalf ||
           type == ScalarType::ComplexFloat ||
           type == ScalarType::ComplexDouble;
}

constexpr bool isIntegralType(ScalarType type, bool include_bool = false) {
    return (type == ScalarType::Byte ||
            type == ScalarType::Char ||
            type == ScalarType::Short ||
            type == ScalarType::Int ||
            type == ScalarType::Long ||
            (include_bool && type == ScalarType::Bool));
}

constexpr bool isQIntType(ScalarType type) {
    return type == ScalarType::QInt8 ||
           type == ScalarType::QUInt8 ||
           type == ScalarType::QInt32;
}

constexpr bool isSignedType(ScalarType type) {
    return type == ScalarType::Char ||
           type == ScalarType::Short ||
           type == ScalarType::Int ||
           type == ScalarType::Long ||
           type == ScalarType::Half ||
           type == ScalarType::Float ||
           type == ScalarType::Double ||
           type == ScalarType::BFloat16 ||
           type == ScalarType::Float8_e4m3fn ||
           type == ScalarType::Float8_e5m2 ||
           isComplexType(type);
}

// ============================================================================
// ScalarType to String
// ============================================================================

inline const char* toString(ScalarType type) {
    switch (type) {
        case ScalarType::Byte: return "Byte";
        case ScalarType::Char: return "Char";
        case ScalarType::Short: return "Short";
        case ScalarType::Int: return "Int";
        case ScalarType::Long: return "Long";
        case ScalarType::Half: return "Half";
        case ScalarType::Float: return "Float";
        case ScalarType::Double: return "Double";
        case ScalarType::ComplexHalf: return "ComplexHalf";
        case ScalarType::ComplexFloat: return "ComplexFloat";
        case ScalarType::ComplexDouble: return "ComplexDouble";
        case ScalarType::Bool: return "Bool";
        case ScalarType::BFloat16: return "BFloat16";
        case ScalarType::QInt8: return "QInt8";
        case ScalarType::QUInt8: return "QUInt8";
        case ScalarType::QInt32: return "QInt32";
        case ScalarType::Float8_e4m3fn: return "Float8_e4m3fn";
        case ScalarType::Float8_e5m2: return "Float8_e5m2";
        case ScalarType::Undefined: return "Undefined";
        default: return "Unknown";
    }
}

// ============================================================================
// Type Promotion
// ============================================================================

// Result type for binary operations between two scalar types
inline ScalarType promoteTypes(ScalarType a, ScalarType b) {
    // Same type
    if (a == b) return a;

    // Handle undefined
    if (a == ScalarType::Undefined) return b;
    if (b == ScalarType::Undefined) return a;

    // Complex types dominate
    if (isComplexType(a) && isComplexType(b)) {
        if (a == ScalarType::ComplexDouble || b == ScalarType::ComplexDouble)
            return ScalarType::ComplexDouble;
        if (a == ScalarType::ComplexFloat || b == ScalarType::ComplexFloat)
            return ScalarType::ComplexFloat;
        return ScalarType::ComplexHalf;
    }
    if (isComplexType(a)) {
        if (b == ScalarType::Double) return ScalarType::ComplexDouble;
        if (b == ScalarType::Float || b == ScalarType::Half || b == ScalarType::BFloat16)
            return ScalarType::ComplexFloat;
        return a;
    }
    if (isComplexType(b)) {
        if (a == ScalarType::Double) return ScalarType::ComplexDouble;
        if (a == ScalarType::Float || a == ScalarType::Half || a == ScalarType::BFloat16)
            return ScalarType::ComplexFloat;
        return b;
    }

    // Float types
    if (isFloatingType(a) && isFloatingType(b)) {
        if (a == ScalarType::Double || b == ScalarType::Double)
            return ScalarType::Double;
        if (a == ScalarType::Float || b == ScalarType::Float)
            return ScalarType::Float;
        if (a == ScalarType::BFloat16 || b == ScalarType::BFloat16)
            return ScalarType::BFloat16;
        return ScalarType::Half;
    }

    // Float dominates integral
    if (isFloatingType(a)) return a;
    if (isFloatingType(b)) return b;

    // Both integral - promote to larger type
    // Size-based promotion
    if (elementSize(a) > elementSize(b)) return a;
    if (elementSize(b) > elementSize(a)) return b;

    // Same size - prefer signed
    if (isSignedType(a)) return a;
    if (isSignedType(b)) return b;

    return a;
}

// ============================================================================
// Dispatch Macro (for type-based dispatch in operations)
// ============================================================================

#define PT_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
    [&] { \
        switch (TYPE) { \
            case ::c10::ScalarType::Byte: { \
                using scalar_t = uint8_t; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Char: { \
                using scalar_t = int8_t; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Short: { \
                using scalar_t = int16_t; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Int: { \
                using scalar_t = int32_t; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Long: { \
                using scalar_t = int64_t; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Float: { \
                using scalar_t = float; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Double: { \
                using scalar_t = double; \
                return __VA_ARGS__(); \
            } \
            default: \
                PT_ERROR("Unsupported dtype: ", ::c10::toString(TYPE)); \
        } \
    }()

// ALL_TYPES_HALF — opt-in extension that additionally covers Half / BFloat16.
// Use in kernels that stick to operators explicitly defined on Half/BFloat16.
#define PT_DISPATCH_ALL_TYPES_HALF(TYPE, NAME, ...) \
    [&] { \
        switch (TYPE) { \
            case ::c10::ScalarType::Byte:     { using scalar_t = uint8_t; return __VA_ARGS__(); } \
            case ::c10::ScalarType::Char:     { using scalar_t = int8_t;  return __VA_ARGS__(); } \
            case ::c10::ScalarType::Short:    { using scalar_t = int16_t; return __VA_ARGS__(); } \
            case ::c10::ScalarType::Int:      { using scalar_t = int32_t; return __VA_ARGS__(); } \
            case ::c10::ScalarType::Long:     { using scalar_t = int64_t; return __VA_ARGS__(); } \
            case ::c10::ScalarType::Half:     { using scalar_t = ::c10::Half;     return __VA_ARGS__(); } \
            case ::c10::ScalarType::BFloat16: { using scalar_t = ::c10::BFloat16; return __VA_ARGS__(); } \
            case ::c10::ScalarType::Float:    { using scalar_t = float;   return __VA_ARGS__(); } \
            case ::c10::ScalarType::Double:   { using scalar_t = double;  return __VA_ARGS__(); } \
            default: PT_ERROR("Unsupported dtype: ", ::c10::toString(TYPE)); \
        } \
    }()

// FLOATING_TYPES — Float/Double only, for templated numeric code that assumes
// full IEEE arithmetic with ternary / implicit int promotion etc.
#define PT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    [&] { \
        switch (TYPE) { \
            case ::c10::ScalarType::Float: { \
                using scalar_t = float; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Double: { \
                using scalar_t = double; \
                return __VA_ARGS__(); \
            } \
            default: \
                PT_ERROR("Expected floating type (Float or Double), got: ", ::c10::toString(TYPE)); \
        } \
    }()

// FLOATING_TYPES_HALF — opt-in extension that also covers Half / BFloat16.
// Use only in kernels where arithmetic is limited to operators explicitly
// defined on Half/BFloat16 (add/sub/mul/div, compound assignment, comparisons
// vs int/float/double). Avoid in linear-algebra code that writes e.g.
// `(x >= 0) ? 1 : -1` via templates — that still needs Float/Double.
#define PT_DISPATCH_FLOATING_TYPES_HALF(TYPE, NAME, ...) \
    [&] { \
        switch (TYPE) { \
            case ::c10::ScalarType::Half: { \
                using scalar_t = ::c10::Half; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::BFloat16: { \
                using scalar_t = ::c10::BFloat16; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Float8_e4m3fn: { \
                using scalar_t = ::c10::Float8_e4m3fn; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Float8_e5m2: { \
                using scalar_t = ::c10::Float8_e5m2; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Float: { \
                using scalar_t = float; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Double: { \
                using scalar_t = double; \
                return __VA_ARGS__(); \
            } \
            default: \
                PT_ERROR("Expected float/double/half/bfloat16/fp8, got: ", ::c10::toString(TYPE)); \
        } \
    }()

// COMPLEX_TYPES — only ComplexFloat / ComplexDouble. Use this in operations
// that are *only* meaningful for complex inputs (conj, real, imag, complex
// abs, complex matmul). Calling on a real dtype yields a clean error.
#define PT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
    [&] { \
        switch (TYPE) { \
            case ::c10::ScalarType::ComplexFloat: { \
                using scalar_t = ::c10::complex<float>; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::ComplexDouble: { \
                using scalar_t = ::c10::complex<double>; \
                return __VA_ARGS__(); \
            } \
            default: \
                PT_ERROR("Expected complex type (ComplexFloat or ComplexDouble), got: ", ::c10::toString(TYPE)); \
        } \
    }()

// FLOATING_AND_COMPLEX_TYPES — Float/Double + ComplexFloat/ComplexDouble.
// Use in kernels that should accept both real and complex (add, sub, mul,
// div, neg, matmul). Arithmetic must be limited to +, -, *, /, unary -, ==,
// compound assignment — all defined on c10::complex<T>.
#define PT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...) \
    [&] { \
        switch (TYPE) { \
            case ::c10::ScalarType::Float: { \
                using scalar_t = float; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::Double: { \
                using scalar_t = double; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::ComplexFloat: { \
                using scalar_t = ::c10::complex<float>; \
                return __VA_ARGS__(); \
            } \
            case ::c10::ScalarType::ComplexDouble: { \
                using scalar_t = ::c10::complex<double>; \
                return __VA_ARGS__(); \
            } \
            default: \
                PT_ERROR("Expected float/double/complex, got: ", ::c10::toString(TYPE)); \
        } \
    }()

} // namespace c10
