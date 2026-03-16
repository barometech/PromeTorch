// mymath.h - собственные математические функции для NMC4
// Библиотека libgcc сломана, пишем всё сами
// ВАЖНО: Избегаем переменных сдвигов! Компилятор генерирует LShift32/RShift32

#ifndef MYMATH_H
#define MYMATH_H

// ============================================================
// Кастомные сдвиги - избегаем LShift32/RShift32
// Используем unrolled константные сдвиги
// ============================================================

// Left shift by variable amount (unsigned)
inline unsigned int my_lshift(unsigned int x, int n) {
    if (n <= 0) return x;
    if (n >= 32) return 0;
    // Unrolled: используем только константные сдвиги
    if (n & 16) x = x << 16;
    if (n & 8)  x = x << 8;
    if (n & 4)  x = x << 4;
    if (n & 2)  x = x << 2;
    if (n & 1)  x = x << 1;
    return x;
}

// Right shift by variable amount (unsigned/logical)
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

// Arithmetic right shift (signed) - preserves sign bit
inline int my_arshift(int x, int n) {
    if (n <= 0) return x;
    if (n >= 32) return (x < 0) ? -1 : 0;
    // Для arithmetic shift нужно сохранять знак
    // Делаем через unsigned + sign extension
    int sign = x < 0 ? -1 : 0;
    unsigned int ux = (unsigned int)x;
    ux = my_rshift(ux, n);
    // Заполняем верхние биты знаком
    if (sign < 0) {
        unsigned int mask = my_lshift(0xFFFFFFFF, 32 - n);
        ux = ux | mask;
    }
    return (int)ux;
}

// ============================================================
// Умножение 32-bit
// ============================================================

// Умножение unsigned через сдвиги и сложения
inline unsigned int mul_u32(unsigned int a, unsigned int b) {
    unsigned int result = 0;
    while (b != 0) {
        if (b & 1) {
            result = result + a;
        }
        a = a << 1;  // константный сдвиг OK
        b = b >> 1;  // константный сдвиг OK
    }
    return result;
}

// Умножение 32-bit signed
inline int mul_i32(int a, int b) {
    int sign = 1;
    if (a < 0) { a = -a; sign = -sign; }
    if (b < 0) { b = -b; sign = -sign; }
    unsigned int result = mul_u32((unsigned int)a, (unsigned int)b);
    return sign < 0 ? -(int)result : (int)result;
}

// Fixed-point Q16.16 format для float-подобных операций
// Целая часть: 16 бит, дробная: 16 бит
typedef int fixed32;

#define FIXED_SHIFT 16
#define FIXED_ONE (1 << FIXED_SHIFT)
#define FLOAT_TO_FIXED(f) ((fixed32)((f) * FIXED_ONE))
#define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_ONE)
#define INT_TO_FIXED(i) ((fixed32)((i) << FIXED_SHIFT))
#define FIXED_TO_INT(x) ((x) >> FIXED_SHIFT)

// Умножение fixed-point: (a * b) >> 16
inline fixed32 mul_fixed(fixed32 a, fixed32 b) {
    // Разбиваем на части чтобы избежать переполнения
    int a_hi = a >> 16;
    int a_lo = a & 0xFFFF;
    int b_hi = b >> 16;
    int b_lo = b & 0xFFFF;

    // a * b = (a_hi*2^16 + a_lo) * (b_hi*2^16 + b_lo)
    //       = a_hi*b_hi*2^32 + (a_hi*b_lo + a_lo*b_hi)*2^16 + a_lo*b_lo
    // Нам нужен результат >> 16, поэтому:
    //       = a_hi*b_hi*2^16 + (a_hi*b_lo + a_lo*b_hi) + (a_lo*b_lo >> 16)

    int hi_hi = (int)my_lshift((unsigned int)mul_i32(a_hi, b_hi), 16);
    int hi_lo = mul_i32(a_hi, b_lo);
    int lo_hi = mul_i32(a_lo, b_hi);
    int lo_lo = (int)my_rshift(mul_u32((unsigned)a_lo, (unsigned)b_lo), 16);

    return hi_hi + hi_lo + lo_hi + lo_lo;
}

// Сложение fixed-point (тривиально)
inline fixed32 add_fixed(fixed32 a, fixed32 b) {
    return a + b;
}

// Вычитание fixed-point
inline fixed32 sub_fixed(fixed32 a, fixed32 b) {
    return a - b;
}

// Преобразование IEEE 754 float в fixed32
// float: sign(1) | exp(8) | mantissa(23)
// value = (-1)^sign * 2^(exp-127) * 1.mantissa
// fixed Q16.16 = value * 2^16
inline fixed32 float_to_fixed(unsigned int f) {
    if (f == 0) return 0;

    int sign = (f >> 31) ? -1 : 1;
    int exp = ((f >> 23) & 0xFF) - 127;  // unbias exponent
    unsigned int mantissa = (f & 0x7FFFFF) | 0x800000;  // add implicit 1 at bit 23

    // mantissa = 1.xxx * 2^23 (implicit 1 at bit 23)
    // value = mantissa / 2^23 * 2^exp = mantissa * 2^(exp - 23)
    // fixed = value * 2^16 = mantissa * 2^(exp - 23 + 16) = mantissa * 2^(exp - 7)
    //
    // FIXED: shift = exp - 7 (was incorrectly 7 + exp)
    int shift = exp - 7;

    unsigned int result;
    if (shift >= 0) {
        if (shift > 15) shift = 15;  // prevent overflow
        result = my_lshift(mantissa, shift);
    } else {
        shift = -shift;
        if (shift > 23) return 0;  // underflow to zero
        result = my_rshift(mantissa, shift);
    }

    return sign < 0 ? -(fixed32)result : (fixed32)result;
}

// Преобразование fixed32 в IEEE 754 float
inline unsigned int fixed_to_float(fixed32 x) {
    if (x == 0) return 0;

    unsigned int sign = 0;
    if (x < 0) {
        sign = 0x80000000;
        x = -x;
    }

    // Найти старший бит - без переменных сдвигов!
    // (tmp >>= 1 и tmp <<= 1 вызывают LShift32/RShift32 которых нет)
    int exp = 0;
    unsigned int tmp = (unsigned int)x;

    // Нормализация вверх: пока tmp >= 0x1000000 (bit 24+), сдвиг вправо
    while (tmp >= 0x1000000) { tmp = my_rshift(tmp, 1); exp++; }
    // Нормализация вниз: пока tmp < 0x800000 (bit 23 не установлен), сдвиг влево
    while (tmp < 0x800000) { tmp = my_lshift(tmp, 1); exp--; }

    // exp относительно позиции 23 (мантисса)
    // но наш fixed имеет точку после 16 бит
    exp = exp + 127 - 16 + 23;

    unsigned int mantissa = tmp & 0x7FFFFF;

    return sign | (my_lshift((unsigned int)exp, 23)) | mantissa;
}

// ============================================================
// Деление fixed-point: a / b
// Алгоритм без переменных сдвигов
// ============================================================
inline fixed32 div_fixed(fixed32 a, fixed32 b) {
    if (b == 0) return 0x7FFFFFFF;  // max value on div by zero

    int sign = 1;
    if (a < 0) { a = -a; sign = -sign; }
    if (b < 0) { b = -b; sign = -sign; }

    unsigned int ua = (unsigned int)a;
    unsigned int ub = (unsigned int)b;

    // Результат = (ua * 2^16) / ub
    // Делаем через shift-subtract без переменных сдвигов
    unsigned int quotient = 0;
    unsigned int remainder = 0;

    // Используем маску которую сдвигаем константно
    unsigned int bit = 0x80000000;  // MSB

    // Обрабатываем 32 бита числителя
    for (int i = 0; i < 32; i++) {
        remainder = remainder << 1;  // константный сдвиг
        if (ua & bit) remainder |= 1;
        bit = bit >> 1;  // константный сдвиг

        quotient = quotient << 1;  // константный сдвиг
        if (remainder >= ub) {
            remainder -= ub;
            quotient |= 1;
        }
    }

    // Ещё 16 бит дробной части
    for (int i = 0; i < 16; i++) {
        remainder = remainder << 1;  // константный сдвиг

        quotient = quotient << 1;  // константный сдвиг
        if (remainder >= ub) {
            remainder -= ub;
            quotient |= 1;
        }
    }

    return sign < 0 ? -(fixed32)quotient : (fixed32)quotient;
}

// ============================================================
// Квадратный корень fixed-point
// Newton-Raphson: x_{n+1} = (x_n + a/x_n) / 2
// ============================================================
inline fixed32 sqrt_fixed(fixed32 a) {
    if (a <= 0) return 0;

    // Начальное приближение: сдвигаем вправо на половину битов
    fixed32 x = a;
    // Грубое начальное приближение
    if (x >= FIXED_ONE) {
        x = x >> 1;
        while (x > (FIXED_ONE << 7)) x = x >> 1;
    } else {
        x = FIXED_ONE;
    }

    // 8 итераций Newton-Raphson
    for (int i = 0; i < 8; i++) {
        if (x == 0) break;
        fixed32 x_new = (x + div_fixed(a, x)) >> 1;
        if (x_new == x) break;  // сошлось
        x = x_new;
    }

    return x;
}

// ============================================================
// Экспонента fixed-point
// Taylor series: exp(x) = 1 + x + x^2/2! + x^3/3! + ...
// Для малых x работает хорошо
// ============================================================
inline fixed32 exp_fixed(fixed32 x) {
    // Ограничиваем диапазон чтобы не переполниться
    // exp(4) ≈ 54.6, exp(-4) ≈ 0.018
    const fixed32 MAX_EXP = 4 << FIXED_SHIFT;
    const fixed32 MIN_EXP = -(4 << FIXED_SHIFT);

    if (x > MAX_EXP) x = MAX_EXP;
    if (x < MIN_EXP) return 0;  // exp(-4) ≈ 0

    // Taylor: 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720
    fixed32 result = FIXED_ONE;     // 1
    fixed32 term = FIXED_ONE;       // текущий член

    // term_n = term_{n-1} * x / n
    // x^1/1!
    term = mul_fixed(term, x);
    result = add_fixed(result, term);

    // x^2/2!
    term = mul_fixed(term, x);
    term = term >> 1;  // /2
    result = add_fixed(result, term);

    // x^3/3!
    term = mul_fixed(term, x);
    term = div_fixed(term, 3 << FIXED_SHIFT);
    result = add_fixed(result, term);

    // x^4/4!
    term = mul_fixed(term, x);
    term = term >> 2;  // /4
    result = add_fixed(result, term);

    // x^5/5!
    term = mul_fixed(term, x);
    term = div_fixed(term, 5 << FIXED_SHIFT);
    result = add_fixed(result, term);

    // x^6/6!
    term = mul_fixed(term, x);
    term = div_fixed(term, 6 << FIXED_SHIFT);
    result = add_fixed(result, term);

    if (result < 0) result = 0;  // underflow protection

    return result;
}

// ============================================================
// Более точная экспонента через range reduction
// exp(x) = exp(k) * exp(r) где x = k + r, |r| < 0.5
// Используем lookup table для exp(k)
// ============================================================

// Lookup table: exp(-4) to exp(4) в шагах 0.5
// exp_lut[i] = exp((i-8) * 0.5) в Q16.16
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
    // Ограничиваем
    const fixed32 MAX_X = 4 << FIXED_SHIFT;
    const fixed32 MIN_X = -(4 << FIXED_SHIFT);

    if (x >= MAX_X) return exp_lut[16];
    if (x <= MIN_X) return exp_lut[0];

    // x в диапазоне [-4, 4]
    // Находим индекс: idx = (x + 4) / 0.5 = (x + 4*65536) / 32768
    int idx_fixed = x + MAX_X;  // теперь [0, 8*65536]
    int idx = idx_fixed >> 15;  // /32768 = /0.5 в Q16
    if (idx < 0) idx = 0;
    if (idx > 15) idx = 15;

    // Линейная интерполяция между lut[idx] и lut[idx+1]
    fixed32 frac = idx_fixed - (idx << 15);  // дробная часть в Q15
    fixed32 v0 = exp_lut[idx];
    fixed32 v1 = exp_lut[idx + 1];

    // result = v0 + (v1 - v0) * frac / 32768
    fixed32 diff = v1 - v0;
    fixed32 interp = mul_fixed(diff << 1, frac);  // frac в Q15, нужен Q16

    return v0 + (interp >> 1);
}

// ============================================================
// Sigmoid: 1 / (1 + exp(-x))
// ============================================================
inline fixed32 sigmoid_fixed(fixed32 x) {
    // sigmoid(-x) = 1 - sigmoid(x), use symmetry for stability
    // For large positive x, sigmoid ≈ 1
    // For large negative x, sigmoid ≈ 0

    const fixed32 MAX_X = 4 << FIXED_SHIFT;

    if (x >= MAX_X) return FIXED_ONE;      // sigmoid(4) ≈ 0.982
    if (x <= -MAX_X) return 0;             // sigmoid(-4) ≈ 0.018

    // sigmoid(x) = 1 / (1 + exp(-x))
    fixed32 exp_neg_x = exp_fixed_lut(-x);
    fixed32 denom = add_fixed(FIXED_ONE, exp_neg_x);

    return div_fixed(FIXED_ONE, denom);
}

// ============================================================
// SiLU (Swish): x * sigmoid(x) - used in Llama
// ============================================================
inline fixed32 silu_fixed(fixed32 x) {
    fixed32 sig = sigmoid_fixed(x);
    return mul_fixed(x, sig);
}

// ============================================================
// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Simplified: use sigmoid approximation GELU ≈ x * sigmoid(1.702 * x)
// ============================================================
inline fixed32 gelu_fixed(fixed32 x) {
    // GELU ≈ x * sigmoid(1.702 * x)
    // 1.702 in Q16.16 = 111543
    const fixed32 GELU_COEF = 111543;
    fixed32 scaled = mul_fixed(x, GELU_COEF);
    fixed32 sig = sigmoid_fixed(scaled);
    return mul_fixed(x, sig);
}

// ============================================================
// ReLU: max(0, x)
// ============================================================
inline fixed32 relu_fixed(fixed32 x) {
    return x > 0 ? x : 0;
}

// ============================================================
// Tanh approximation using sigmoid: tanh(x) = 2*sigmoid(2x) - 1
// ============================================================
inline fixed32 tanh_fixed(fixed32 x) {
    fixed32 sig = sigmoid_fixed(x << 1);  // sigmoid(2x)
    return sub_fixed(sig << 1, FIXED_ONE);  // 2*sig - 1
}

// ============================================================
// Sin/Cos for RoPE (Rotary Position Embedding)
// Taylor series approximation
// ============================================================

// sin(x) ≈ x - x^3/6 + x^5/120 for small x
// Use range reduction: sin(x) for x in [-pi, pi]
inline fixed32 sin_fixed(fixed32 x) {
    // Range reduction to [-pi, pi]
    // pi in Q16.16 ≈ 205887
    const fixed32 PI = 205887;
    const fixed32 TWO_PI = 411775;

    // Reduce to [-pi, pi]
    while (x > PI) x = sub_fixed(x, TWO_PI);
    while (x < -PI) x = add_fixed(x, TWO_PI);

    // Taylor: sin(x) = x - x^3/6 + x^5/120 - x^7/5040
    fixed32 x2 = mul_fixed(x, x);
    fixed32 x3 = mul_fixed(x2, x);
    fixed32 x5 = mul_fixed(x3, x2);

    // x^3/6: 6 in Q16 = 393216
    fixed32 term3 = div_fixed(x3, 393216);

    // x^5/120: 120 in Q16 = 7864320
    fixed32 term5 = div_fixed(x5, 7864320);

    return sub_fixed(add_fixed(x, term5), term3);
}

// cos(x) = sin(x + pi/2)
inline fixed32 cos_fixed(fixed32 x) {
    const fixed32 PI_HALF = 102944;  // pi/2 in Q16.16
    return sin_fixed(add_fixed(x, PI_HALF));
}

#endif // MYMATH_H
