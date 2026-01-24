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
// Half Precision Float (FP16)
// ============================================================================

struct alignas(2) Half {
    uint16_t x;

    Half() = default;

    PT_HOST_DEVICE Half(float f) : x(float_to_half_bits(f)) {}

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
inline bool operator==(Half a, Half b) { return float(a) == float(b); }
inline bool operator!=(Half a, Half b) { return float(a) != float(b); }
inline bool operator<(Half a, Half b) { return float(a) < float(b); }
inline bool operator>(Half a, Half b) { return float(a) > float(b); }
inline bool operator<=(Half a, Half b) { return float(a) <= float(b); }
inline bool operator>=(Half a, Half b) { return float(a) >= float(b); }

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
inline bool operator==(BFloat16 a, BFloat16 b) { return float(a) == float(b); }
inline bool operator!=(BFloat16 a, BFloat16 b) { return float(a) != float(b); }
inline bool operator<(BFloat16 a, BFloat16 b) { return float(a) < float(b); }
inline bool operator>(BFloat16 a, BFloat16 b) { return float(a) > float(b); }

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

    // Number of types
    NumOptions = 17
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
DEFINE_SCALAR_TYPE_MAPPING(ComplexHalf, std::complex<c10::Half>)
DEFINE_SCALAR_TYPE_MAPPING(ComplexFloat, std::complex<float>)
DEFINE_SCALAR_TYPE_MAPPING(ComplexDouble, std::complex<double>)
DEFINE_SCALAR_TYPE_MAPPING(Bool, bool)
DEFINE_SCALAR_TYPE_MAPPING(BFloat16, c10::BFloat16)

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
DEFINE_CPP_TYPE_MAPPING(std::complex<float>, ComplexFloat)
DEFINE_CPP_TYPE_MAPPING(std::complex<double>, ComplexDouble)
DEFINE_CPP_TYPE_MAPPING(bool, Bool)
DEFINE_CPP_TYPE_MAPPING(c10::BFloat16, BFloat16)

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
        default: return 0;
    }
}

constexpr bool isFloatingType(ScalarType type) {
    return type == ScalarType::Half ||
           type == ScalarType::Float ||
           type == ScalarType::Double ||
           type == ScalarType::BFloat16;
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

} // namespace c10
