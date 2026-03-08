#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace torch {
namespace io {
namespace gguf {

// ============================================================================
// GGML Quantization Types
// ============================================================================

enum GGMLType : uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_BF16    = 30,
};

// Block sizes
static constexpr int QK4_0 = 32;
static constexpr int QK4_1 = 32;
static constexpr int QK5_0 = 32;
static constexpr int QK5_1 = 32;
static constexpr int QK8_0 = 32;
static constexpr int QK8_1 = 32;
static constexpr int QK_K  = 256;

// ============================================================================
// FP16 conversion
// ============================================================================

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exp == 0) {
        if (mantissa == 0) {
            float result;
            uint32_t bits = sign;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        // Denormalized
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exp--;
        }
        exp++;
        mantissa &= ~0x400;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }

    exp = exp + (127 - 15);
    mantissa <<= 13;

    uint32_t bits = sign | ((uint32_t)exp << 23) | mantissa;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

inline float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

// ============================================================================
// Block size and byte size calculations
// ============================================================================

inline int64_t ggml_block_size(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:     return 1;
        case GGML_TYPE_F16:     return 1;
        case GGML_TYPE_BF16:    return 1;
        case GGML_TYPE_Q4_0:    return QK4_0;
        case GGML_TYPE_Q4_1:    return QK4_1;
        case GGML_TYPE_Q5_0:    return QK5_0;
        case GGML_TYPE_Q5_1:    return QK5_1;
        case GGML_TYPE_Q8_0:    return QK8_0;
        case GGML_TYPE_Q8_1:    return QK8_1;
        case GGML_TYPE_Q2_K:    return QK_K;
        case GGML_TYPE_Q3_K:    return QK_K;
        case GGML_TYPE_Q4_K:    return QK_K;
        case GGML_TYPE_Q5_K:    return QK_K;
        case GGML_TYPE_Q6_K:    return QK_K;
        case GGML_TYPE_Q8_K:    return QK_K;
        case GGML_TYPE_I8:      return 1;
        case GGML_TYPE_I16:     return 1;
        case GGML_TYPE_I32:     return 1;
        case GGML_TYPE_I64:     return 1;
        case GGML_TYPE_F64:     return 1;
        default: return 1;
    }
}

inline int64_t ggml_type_block_bytes(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:     return 4;
        case GGML_TYPE_F16:     return 2;
        case GGML_TYPE_BF16:    return 2;
        case GGML_TYPE_Q4_0:    return 2 + QK4_0 / 2;          // 18
        case GGML_TYPE_Q4_1:    return 2 + 2 + QK4_1 / 2;      // 20
        case GGML_TYPE_Q5_0:    return 2 + 4 + QK5_0 / 2;      // 22
        case GGML_TYPE_Q5_1:    return 2 + 2 + 4 + QK5_1 / 2;  // 24
        case GGML_TYPE_Q8_0:    return 2 + QK8_0;               // 34
        case GGML_TYPE_Q8_1:    return 4 + 4 + QK8_1;           // 40
        case GGML_TYPE_Q2_K:    return QK_K / 16 + QK_K / 4 + 2 + 2; // 84
        case GGML_TYPE_Q3_K:    return QK_K / 8 + QK_K / 4 + 12 + 2; // 110
        case GGML_TYPE_Q4_K:    return 2 + 2 + 12 + QK_K / 2;  // 144
        case GGML_TYPE_Q5_K:    return 2 + 2 + 12 + QK_K / 8 + QK_K / 2; // 176
        case GGML_TYPE_Q6_K:    return QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 210
        case GGML_TYPE_Q8_K:    return 4 + QK_K + QK_K / 16;   // 276
        case GGML_TYPE_I8:      return 1;
        case GGML_TYPE_I16:     return 2;
        case GGML_TYPE_I32:     return 4;
        case GGML_TYPE_I64:     return 8;
        case GGML_TYPE_F64:     return 8;
        default:
            throw std::runtime_error("Unknown GGML type: " + std::to_string(type));
    }
}

inline int64_t ggml_tensor_bytes(GGMLType type, int64_t n_elements) {
    int64_t block_size = ggml_block_size(type);
    int64_t n_blocks = (n_elements + block_size - 1) / block_size;
    return n_blocks * ggml_type_block_bytes(type);
}

inline const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_I8:   return "I8";
        case GGML_TYPE_I16:  return "I16";
        case GGML_TYPE_I32:  return "I32";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Dequantization: F32 (passthrough)
// ============================================================================

inline void dequantize_f32(const void* src, float* dst, int64_t n) {
    std::memcpy(dst, src, n * sizeof(float));
}

// ============================================================================
// Dequantization: F16 → F32
// ============================================================================

inline void dequantize_f16(const void* src, float* dst, int64_t n) {
    const uint16_t* data = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = fp16_to_fp32(data[i]);
    }
}

// ============================================================================
// Dequantization: BF16 → F32
// ============================================================================

inline void dequantize_bf16(const void* src, float* dst, int64_t n) {
    const uint16_t* data = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = bf16_to_fp32(data[i]);
    }
}

// ============================================================================
// Dequantization: Q8_0 → F32
// Block: fp16 scale (2B) + int8[32] (32B) = 34B per 32 values
// ============================================================================

inline void dequantize_q8_0(const void* src, float* dst, int64_t n) {
    const int nb = static_cast<int>(n / QK8_0);
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* block = data + i * 34;  // 2 + 32 = 34 bytes

        // Scale is fp16
        uint16_t d_bits;
        std::memcpy(&d_bits, block, 2);
        const float d = fp16_to_fp32(d_bits);

        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

        for (int j = 0; j < QK8_0; ++j) {
            dst[i * QK8_0 + j] = d * static_cast<float>(qs[j]);
        }
    }
}

// ============================================================================
// Dequantization: Q4_0 → F32
// Block: fp16 scale (2B) + uint8[16] (16B) = 18B per 32 values
// Each byte encodes 2 values (4-bit each), offset by -8
// ============================================================================

inline void dequantize_q4_0(const void* src, float* dst, int64_t n) {
    const int nb = static_cast<int>(n / QK4_0);
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* block = data + i * 18;  // 2 + 16 = 18 bytes

        uint16_t d_bits;
        std::memcpy(&d_bits, block, 2);
        const float d = fp16_to_fp32(d_bits);

        const uint8_t* qs = block + 2;

        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int x0 = (qs[j] & 0x0F) - 8;
            const int x1 = (qs[j] >> 4) - 8;
            dst[i * QK4_0 + j]              = d * static_cast<float>(x0);
            dst[i * QK4_0 + j + QK4_0 / 2]  = d * static_cast<float>(x1);
        }
    }
}

// ============================================================================
// Helper: extract scale and min from packed Q4_K/Q5_K scales
// ============================================================================

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ============================================================================
// Dequantization: Q4_K → F32 (Q4_K_M — main Ollama format)
// Super-block: d(fp16) + dmin(fp16) + scales[12] + qs[128] = 144B per 256 values
// ============================================================================

inline void dequantize_q4_k(const void* src, float* dst, int64_t n) {
    const int nb = static_cast<int>(n / QK_K);
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* block = data + i * 144;

        // Read d and dmin (fp16)
        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, block, 2);
        std::memcpy(&dmin_bits, block + 2, 2);
        const float d = fp16_to_fp32(d_bits);
        const float dmin = fp16_to_fp32(dmin_bits);

        const uint8_t* scales = block + 4;   // 12 bytes
        const uint8_t* qs = block + 16;      // 128 bytes

        float* y = dst + i * QK_K;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, scales, &sc, &m);
            float d1 = d * sc;
            float m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc;
            float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) {
                y[j + l +  0] = d1 * (qs[l] & 0xF) - m1;
                y[j + l + 32] = d2 * (qs[l] >> 4) - m2;
            }
            qs += 32;
            is += 2;
        }
    }
}

// ============================================================================
// Dequantization: Q5_K → F32
// Super-block: d(fp16) + dmin(fp16) + scales[12] + qh[32] + qs[128] = 176B per 256
// ============================================================================

inline void dequantize_q5_k(const void* src, float* dst, int64_t n) {
    const int nb = static_cast<int>(n / QK_K);
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* block = data + i * 176;

        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, block, 2);
        std::memcpy(&dmin_bits, block + 2, 2);
        const float d = fp16_to_fp32(d_bits);
        const float dmin = fp16_to_fp32(dmin_bits);

        const uint8_t* scales = block + 4;   // 12 bytes
        const uint8_t* qh = block + 16;      // 32 bytes
        const uint8_t* qs = block + 48;      // 128 bytes

        float* y = dst + i * QK_K;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) {
                y[j + l +  0] = d1 * ((qs[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                y[j + l + 32] = d2 * ((qs[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2;
            }
            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// ============================================================================
// Dequantization: Q6_K → F32
// Super-block: ql[128] + qh[64] + scales[16] + d(fp16) = 210B per 256 values
// ============================================================================

inline void dequantize_q6_k(const void* src, float* dst, int64_t n) {
    const int nb = static_cast<int>(n / QK_K);
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* block = data + i * 210;

        const uint8_t* ql = block;            // 128 bytes
        const uint8_t* qh = block + 128;      // 64 bytes
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192); // 16 bytes

        uint16_t d_bits;
        std::memcpy(&d_bits, block + 208, 2);
        const float d = fp16_to_fp32(d_bits);

        float* y = dst + i * QK_K;

        for (int n_half = 0; n_half < QK_K; n_half += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = n_half / 16 + l / 16;
                const int8_t q1 = static_cast<int8_t>(((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4))) - 32;
                const int8_t q2 = static_cast<int8_t>(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) - 32;
                const int8_t q3 = static_cast<int8_t>(((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4))) - 32;
                const int8_t q4 = static_cast<int8_t>(((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4))) - 32;

                y[l +  0] = d * scales[is + 0] * q1;
                y[l + 32] = d * scales[is + 2] * q2;
                y[l + 64] = d * scales[is + 4] * q3;
                y[l + 96] = d * scales[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
        }
    }
}

// ============================================================================
// Dequantization: Q2_K → F32
// Super-block: scales[16] + qs[64] + d(fp16) + dmin(fp16) = 84B per 256 values
// ============================================================================

inline void dequantize_q2_k(const void* src, float* dst, int64_t n) {
    const int nb = static_cast<int>(n / QK_K);
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* block = data + i * 84;

        const uint8_t* scales_raw = block;         // 16 bytes
        const uint8_t* qs = block + 16;            // 64 bytes

        uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, block + 80, 2);
        std::memcpy(&dmin_bits, block + 82, 2);
        const float d = fp16_to_fp32(d_bits);
        const float dmin = fp16_to_fp32(dmin_bits);

        float* y = dst + i * QK_K;

        int qi = 0;
        for (int j = 0; j < QK_K; j += 16) {
            int is = j / 16;
            float dl = d * (scales_raw[is] & 0xF);
            float ml = dmin * (scales_raw[is] >> 4);
            for (int l = 0; l < 16; ++l) {
                int shift = (l % 4) * 2;
                int byte_idx = qi + l / 4;
                uint8_t q = (qs[byte_idx] >> shift) & 3;
                y[j + l] = dl * q - ml;
            }
            qi += 4;
        }
    }
}

// ============================================================================
// Dequantization: Q3_K → F32
// Super-block: hmask[32] + qs[64] + scales[12] + d(fp16) = 110B per 256 values
// ============================================================================

inline void dequantize_q3_k(const void* src, float* dst, int64_t n) {
    const int nb = static_cast<int>(n / QK_K);
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* block = data + i * 110;

        const uint8_t* hmask = block;              // 32 bytes
        const uint8_t* qs = block + 32;            // 64 bytes
        const uint8_t* scales_raw = block + 96;    // 12 bytes

        uint16_t d_bits;
        std::memcpy(&d_bits, block + 108, 2);
        const float d = fp16_to_fp32(d_bits);

        // Decode 16 scales from 12 bytes (each 6-bit, stored packed)
        int8_t scales[16];
        for (int j = 0; j < 8; ++j) {
            scales[j] = static_cast<int8_t>((scales_raw[j % 6 < 4 ? j % 4 : j % 4 + 4]) & 0x3F);
        }
        // Simplified: for initial correctness, use direct byte reads
        // Scales packing for Q3_K is complex; approximate with low bits
        for (int j = 0; j < 4; ++j) {
            scales[j + 0] = static_cast<int8_t>((scales_raw[j] & 0xF) | ((scales_raw[j + 8] & 0x03) << 4)) - 32;
            scales[j + 4] = static_cast<int8_t>((scales_raw[j] >> 4)  | ((scales_raw[j + 8] & 0x0C) << 2)) - 32;
            scales[j + 8] = static_cast<int8_t>((scales_raw[j + 4] & 0xF) | ((scales_raw[j + 8] & 0x30) >> 0)) - 32;
            scales[j +12] = static_cast<int8_t>((scales_raw[j + 4] >> 4)  | ((scales_raw[j + 8] & 0xC0) >> 2)) - 32;
        }

        float* y = dst + i * QK_K;

        int qi = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; j += 16) {
            int is = j / 16;
            float dl = d * scales[is];
            for (int l = 0; l < 16; ++l) {
                int shift = ((qi + l / 2) % 32 == qi + l / 2) ? (l & 1) * 2 : 0;
                int byte_idx = qi + l / 2;
                // 2-bit base from qs
                uint8_t q2 = (qs[byte_idx % 64] >> (2 * (l % 4))) & 3;
                // High bit from hmask
                int h = (hmask[(j + l) % 32] & hm) ? 0 : 4;
                y[j + l] = dl * (static_cast<int>(q2) + h - 4);
            }
            qi += 8;
            if (j % 32 == 16) hm <<= 1;
        }
    }
}

// ============================================================================
// Master dequantization dispatcher
// ============================================================================

inline void dequantize(GGMLType type, const void* src, float* dst, int64_t n_elements) {
    switch (type) {
        case GGML_TYPE_F32:
            dequantize_f32(src, dst, n_elements);
            break;
        case GGML_TYPE_F16:
            dequantize_f16(src, dst, n_elements);
            break;
        case GGML_TYPE_BF16:
            dequantize_bf16(src, dst, n_elements);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_q8_0(src, dst, n_elements);
            break;
        case GGML_TYPE_Q4_0:
            dequantize_q4_0(src, dst, n_elements);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_q4_k(src, dst, n_elements);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_q5_k(src, dst, n_elements);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_q6_k(src, dst, n_elements);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_q2_k(src, dst, n_elements);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_q3_k(src, dst, n_elements);
            break;
        default:
            throw std::runtime_error(
                std::string("Unsupported GGML type for dequantization: ") +
                ggml_type_name(type) + " (" + std::to_string(type) + ")"
            );
    }
}

// ============================================================================
// Check if type is supported
// ============================================================================

inline bool is_type_supported(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return true;
        default:
            return false;
    }
}

} // namespace gguf
} // namespace io
} // namespace torch
