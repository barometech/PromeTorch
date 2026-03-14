#pragma once
// ============================================================================
// NMCardOps.h - NMCard Operation Declarations
// ============================================================================
// Thin wrappers that dispatch to hardware (if available) or emulator.
// Hardware-accelerated: matmul, rmsnorm, softmax, silu, rope,
//                       elem_add, elem_mul, gate_mul
// CPU fallback: abs, sqrt, exp, log, sin, cos, tanh, sigmoid, comparisons,
//               reductions, backward ops, optimizers

#include "aten/src/ATen/nmcard/NMCardEmulator.h"
#include "aten/src/ATen/nmcard/NMCardHardware.h"
#include <cstdint>

namespace at {
namespace nmcard_ops {

using nmcard::NMCardEmulator;
using nmcard::NMCardHardware;

// ============================================================================
// Element-wise Unary Operations
// ============================================================================

// neg, relu — emulator only (no hardware opcode)
inline void launch_neg(const float* input, float* output, int64_t n) {
    NMCardEmulator::get().neg(input, output, n);
}

inline void launch_relu(const float* input, float* output, int64_t n) {
    NMCardEmulator::get().relu(input, output, n);
}

// silu — hardware OP_SILU(4)
inline void launch_silu(const float* input, float* output, int64_t n) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().silu(input, output, n);
    } else {
        NMCardEmulator::get().silu(input, output, n);
    }
}

// gelu — no hardware opcode in dispatcher, emulator only
inline void launch_gelu(const float* input, float* output, int64_t n) {
    NMCardEmulator::get().gelu(input, output, n);
}

// Unary ops that fall back to CPU-like float (not natively on NMC4)
inline void launch_abs(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = input[i] < 0 ? -input[i] : input[i];
}

inline void launch_sqrt(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::sqrt(input[i]);
}

inline void launch_exp(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::exp(input[i]);
}

inline void launch_log(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::log(input[i]);
}

inline void launch_sin(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::sin(input[i]);
}

inline void launch_cos(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::cos(input[i]);
}

inline void launch_tanh(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::tanh(input[i]);
}

inline void launch_sigmoid(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = 1.0f / (1.0f + std::exp(-input[i]));
}

inline void launch_square(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = input[i] * input[i];
}

inline void launch_rsqrt(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = 1.0f / std::sqrt(input[i]);
}

inline void launch_reciprocal(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = 1.0f / input[i];
}

inline void launch_sign(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = (input[i] > 0) ? 1.0f : ((input[i] < 0) ? -1.0f : 0.0f);
}

inline void launch_ceil(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::ceil(input[i]);
}

inline void launch_floor(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::floor(input[i]);
}

inline void launch_round(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::round(input[i]);
}

inline void launch_log2(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::log2(input[i]);
}

inline void launch_log10(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::log10(input[i]);
}

inline void launch_tan(const float* input, float* output, int64_t n) {
    for (int64_t i = 0; i < n; i++) output[i] = std::tan(input[i]);
}

inline void launch_leaky_relu(const float* input, float* output, float alpha, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
}

// ============================================================================
// Element-wise Binary Operations
// ============================================================================

// elem_add — hardware OP_ELEM_ADD(10)
inline void launch_add(const float* a, const float* b, float* out, int64_t n) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().elem_add(a, b, out, n);
    } else {
        NMCardEmulator::get().elem_add(a, b, out, n);
    }
}

// elem_sub — no hardware opcode (OP_ELEM_SUB=12 declared but not in dispatcher)
inline void launch_sub(const float* a, const float* b, float* out, int64_t n) {
    NMCardEmulator::get().elem_sub(a, b, out, n);
}

// elem_mul — hardware OP_ELEM_MUL(11)
inline void launch_mul(const float* a, const float* b, float* out, int64_t n) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().elem_mul(a, b, out, n);
    } else {
        NMCardEmulator::get().elem_mul(a, b, out, n);
    }
}

inline void launch_add_scalar(const float* a, float scalar, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + scalar;
}

inline void launch_mul_scalar(const float* a, float scalar, float* out, int64_t n) {
    NMCardEmulator::get().mul_scalar(a, scalar, out, n);
}

inline void launch_div(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] / b[i];
}

inline void launch_div_scalar(const float* a, float scalar, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] / scalar;
}

inline void launch_pow(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = std::pow(a[i], b[i]);
}

inline void launch_pow_scalar(const float* a, float exp_val, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = std::pow(a[i], exp_val);
}

inline void launch_maximum(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] > b[i] ? a[i] : b[i];
}

inline void launch_minimum(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] < b[i] ? a[i] : b[i];
}

// Broadcast operations
inline void launch_mul_broadcast_row(const float* a, const float* b, float* out,
                                      int64_t outer_size, int64_t inner_size) {
    for (int64_t i = 0; i < outer_size; i++) {
        for (int64_t j = 0; j < inner_size; j++) {
            out[i * inner_size + j] = a[i * inner_size + j] * b[j];
        }
    }
}

inline void launch_mul_broadcast_col(const float* a, const float* b, float* out,
                                      int64_t outer_size, int64_t inner_size) {
    for (int64_t i = 0; i < outer_size; i++) {
        for (int64_t j = 0; j < inner_size; j++) {
            out[i * inner_size + j] = a[i * inner_size + j] * b[i];
        }
    }
}

inline void launch_add_broadcast_col(const float* a, const float* b, float* out,
                                      int64_t outer_size, int64_t inner_size) {
    for (int64_t i = 0; i < outer_size; i++) {
        for (int64_t j = 0; j < inner_size; j++) {
            out[i * inner_size + j] = a[i * inner_size + j] + b[i];
        }
    }
}

// ============================================================================
// Fill and Copy
// ============================================================================

inline void launch_fill(float* data, float value, int64_t n) {
    for (int64_t i = 0; i < n; i++) data[i] = value;
}

inline void launch_copy(const float* src, float* dst, int64_t n) {
    std::memcpy(dst, src, n * sizeof(float));
}

// ============================================================================
// MatMul — hardware OP_MATMUL(1)
// ============================================================================

inline void launch_matmul(const float* A, const float* B, float* C,
                           int64_t M, int64_t K, int64_t N) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().matmul(A, B, C, M, K, N);
    } else {
        NMCardEmulator::get().matmul(A, B, C, M, K, N);
    }
}

// ============================================================================
// Reduce Operations (no hardware opcodes — CPU fallback)
// ============================================================================

inline void launch_sum(const float* input, float* output, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) sum += input[i];
    *output = sum;
}

inline void launch_sum_dim(const float* input, float* output,
                            int64_t outer, int64_t reduce, int64_t inner) {
    for (int64_t i = 0; i < outer; i++) {
        for (int64_t k = 0; k < inner; k++) {
            float sum = 0.0f;
            for (int64_t j = 0; j < reduce; j++) {
                sum += input[i * reduce * inner + j * inner + k];
            }
            output[i * inner + k] = sum;
        }
    }
}

inline void launch_max(const float* input, float* output, int64_t n) {
    float max_val = input[0];
    for (int64_t i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    *output = max_val;
}

// ============================================================================
// Comparison Operations (no hardware opcodes — CPU fallback)
// ============================================================================

inline void launch_eq(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
}

inline void launch_ne(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = (a[i] != b[i]) ? 1.0f : 0.0f;
}

inline void launch_gt(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}

inline void launch_lt(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}

inline void launch_ge(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
}

inline void launch_le(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
}

inline void launch_clamp(const float* input, float* output, float min_val, float max_val, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float x = input[i];
        if (x < min_val) x = min_val;
        if (x > max_val) x = max_val;
        output[i] = x;
    }
}

// ============================================================================
// NMCard-native Operations (hardware accelerated where opcodes exist)
// ============================================================================

// softmax — hardware OP_SOFTMAX(3)
inline void launch_softmax(const float* input, float* output,
                            int64_t batch, int64_t dim) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().softmax(input, output, batch, dim);
    } else {
        NMCardEmulator::get().softmax(input, output, batch, dim);
    }
}

// rmsnorm — hardware OP_RMSNORM(2)
inline void launch_rmsnorm(const float* input, float* output, const float* gamma,
                            int64_t batch, int64_t hidden) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().rmsnorm(input, output, gamma, batch, hidden);
    } else {
        NMCardEmulator::get().rmsnorm(input, output, gamma, batch, hidden);
    }
}

// rope — hardware OP_ROPE(5)
inline void launch_rope(const float* input, float* output, const float* freqs,
                         int64_t seq_len, int64_t head_dim, int64_t pos_offset) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().rope(input, output, freqs, seq_len, head_dim, pos_offset);
    } else {
        NMCardEmulator::get().rope(input, output, freqs, seq_len, head_dim, pos_offset);
    }
}

// gate_mul — hardware OP_GATE_MUL(13)
inline void launch_gate_mul(const float* a, const float* b, float* out, int64_t count) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().gate_mul(a, b, out, count);
    } else {
        NMCardEmulator::get().gate_mul(a, b, out, count);
    }
}

// layernorm — no hardware opcode (OP_LAYERNORM=16 declared but not in dispatcher)
inline void launch_layernorm(const float* input, float* output,
                              const float* gamma, const float* beta,
                              int64_t batch, int64_t hidden) {
    NMCardEmulator::get().layernorm(input, output, gamma, beta, batch, hidden);
}

// ============================================================================
// Training Operations (no hardware opcodes — emulator/CPU only)
// ============================================================================

inline void launch_matmul_backward(const float* A, const float* B, const float* grad_C,
                                    float* grad_A, float* grad_B,
                                    int64_t M, int64_t K, int64_t N) {
    NMCardEmulator::get().matmul_backward(A, B, grad_C, grad_A, grad_B, M, K, N);
}

inline void launch_silu_backward(const float* input, const float* grad_output,
                                  float* grad_input, int64_t count) {
    NMCardEmulator::get().silu_backward_op(input, grad_output, grad_input, count);
}

inline void launch_gelu_backward(const float* input, const float* grad_output,
                                  float* grad_input, int64_t count) {
    NMCardEmulator::get().gelu_backward_op(input, grad_output, grad_input, count);
}

inline void launch_relu_backward(const float* input, const float* grad_output,
                                  float* grad_input, int64_t count) {
    NMCardEmulator::get().relu_backward_op(input, grad_output, grad_input, count);
}

inline void launch_sgd_step(float* weights, const float* grads, float lr, int64_t size) {
    NMCardEmulator::get().sgd_step(weights, grads, lr, size);
}

inline void launch_adam_step(float* weights, const float* grads,
                              float* m, float* v,
                              float lr, float beta1, float beta2, float eps,
                              int64_t size) {
    NMCardEmulator::get().adam_step(weights, grads, m, v, lr, beta1, beta2, eps, size);
}

} // namespace nmcard_ops
} // namespace at
