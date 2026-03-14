#pragma once
// ============================================================================
// NMCardEmulator.h - Software Emulator for NM Card Mini (16 NMC4 Cores)
// ============================================================================
// Reproduces dispatcher.cpp operations on host CPU
// Two modes:
//   fixed_point=true:  float → Q16.16 → operation → float (NMC4 precision)
//   fixed_point=false: native float32 (for debugging)
//
// Emulates 16 virtual NMC4 cores with independent execution

#include "aten/src/ATen/nmcard/NMCardMath.h"
#include "c10/macros/Macros.h"
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <mutex>

#ifndef ATEN_NMCARD_API
#if defined(PT_PLATFORM_WINDOWS) || defined(_MSC_VER)
    #if defined(ATEN_NMCARD_EXPORTS)
        #define ATEN_NMCARD_API __declspec(dllexport)
    #else
        #define ATEN_NMCARD_API __declspec(dllimport)
    #endif
#else
    #define ATEN_NMCARD_API __attribute__((visibility("default")))
#endif
#endif

namespace at {
namespace nmcard {

// ============================================================================
// Operation codes (matching dispatcher.cpp)
// ============================================================================

enum class NMCardOp : int {
    NOP          = 0,
    MATMUL       = 1,
    RMSNORM      = 2,
    SOFTMAX      = 3,
    SILU         = 4,
    ROPE         = 5,
    ATTENTION    = 6,
    ELEM_ADD     = 10,
    ELEM_MUL     = 11,
    ELEM_SUB     = 12,
    GATE_MUL     = 13,
    MUL_SCALAR   = 14,
    GELU         = 15,
    LAYERNORM    = 16,

    // Training ops (from mymath_backward.h)
    MATMUL_BACKWARD = 30,
    SILU_BACKWARD   = 31,
    GELU_BACKWARD   = 32,
    SOFTMAX_BACKWARD = 33,
    RMSNORM_BACKWARD = 34,
    ROPE_BACKWARD    = 35,
    CROSS_ENTROPY    = 40,
    CROSS_ENTROPY_BACKWARD = 41,
    SGD_STEP    = 50,
    ADAM_STEP   = 51,

    EXIT = 255
};

// ============================================================================
// NMCard Emulator
// ============================================================================

class ATEN_NMCARD_API NMCardEmulator {
public:
    // Singleton accessor - implemented in NMCardEmulator.cpp
    static NMCardEmulator& get();

    // Configuration
    void set_fixed_point(bool enable) { use_fixed_point_ = enable; }
    bool is_fixed_point() const { return use_fixed_point_; }

    void set_num_cores(int n) { num_cores_ = (n > 0 && n <= 16) ? n : 16; }
    int num_cores() const { return num_cores_; }

    // ========================================================================
    // Forward Operations
    // ========================================================================

    // MatMul: C[M,N] = A[M,K] @ B[K,N]
    void matmul(const float* A, const float* B, float* C,
                int64_t M, int64_t K, int64_t N) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < M; i++) {
                for (int64_t j = 0; j < N; j++) {
                    fixed32 sum = 0;
                    for (int64_t k = 0; k < K; k++) {
                        fixed32 a = float_to_fixed(A[i * K + k]);
                        fixed32 b = float_to_fixed(B[k * N + j]);
                        sum = add_fixed(sum, mul_fixed(a, b));
                    }
                    C[i * N + j] = fixed_to_float(sum);
                }
            }
        } else {
            for (int64_t i = 0; i < M; i++) {
                for (int64_t j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
    }

    // RMSNorm: y = x * gamma / sqrt(mean(x^2) + eps)
    void rmsnorm(const float* input, float* output, const float* gamma,
                 int64_t batch, int64_t hidden) {
        float eps = 1e-5f;
        if (use_fixed_point_) {
            fixed32 eps_q = 1; // ~1.5e-5 in Q16.16
            for (int64_t b = 0; b < batch; b++) {
                const float* x = input + b * hidden;
                float* y = output + b * hidden;

                fixed32 sum_sq = 0;
                for (int64_t i = 0; i < hidden; i++) {
                    fixed32 xi = float_to_fixed(x[i]);
                    sum_sq = add_fixed(sum_sq, mul_fixed(xi, xi));
                }
                fixed32 rms = sqrt_fixed(add_fixed(
                    div_fixed(sum_sq, int_to_fixed(static_cast<int>(hidden))), eps_q));
                fixed32 inv_rms = div_fixed(FIXED_ONE, rms);

                for (int64_t i = 0; i < hidden; i++) {
                    fixed32 xi = float_to_fixed(x[i]);
                    fixed32 g = float_to_fixed(gamma[i]);
                    y[i] = fixed_to_float(mul_fixed(mul_fixed(xi, inv_rms), g));
                }
            }
        } else {
            for (int64_t b = 0; b < batch; b++) {
                const float* x = input + b * hidden;
                float* y = output + b * hidden;

                float sum_sq = 0.0f;
                for (int64_t i = 0; i < hidden; i++) {
                    sum_sq += x[i] * x[i];
                }
                float rms = std::sqrt(sum_sq / hidden + eps);
                float inv_rms = 1.0f / rms;

                for (int64_t i = 0; i < hidden; i++) {
                    y[i] = x[i] * inv_rms * gamma[i];
                }
            }
        }
    }

    // Softmax over last dimension
    void softmax(const float* input, float* output,
                 int64_t batch, int64_t dim) {
        if (use_fixed_point_) {
            for (int64_t b = 0; b < batch; b++) {
                const float* x = input + b * dim;
                float* y = output + b * dim;

                fixed32 max_val = float_to_fixed(x[0]);
                for (int64_t i = 1; i < dim; i++) {
                    fixed32 xi = float_to_fixed(x[i]);
                    if (xi > max_val) max_val = xi;
                }

                fixed32 exp_sum = 0;
                std::vector<fixed32> exp_vals(dim);
                for (int64_t i = 0; i < dim; i++) {
                    fixed32 xi = float_to_fixed(x[i]);
                    exp_vals[i] = exp_fixed_lut(sub_fixed(xi, max_val));
                    exp_sum = add_fixed(exp_sum, exp_vals[i]);
                }

                if (exp_sum == 0) exp_sum = 1;
                for (int64_t i = 0; i < dim; i++) {
                    y[i] = fixed_to_float(div_fixed(exp_vals[i], exp_sum));
                }
            }
        } else {
            for (int64_t b = 0; b < batch; b++) {
                const float* x = input + b * dim;
                float* y = output + b * dim;

                float max_val = x[0];
                for (int64_t i = 1; i < dim; i++) {
                    if (x[i] > max_val) max_val = x[i];
                }

                float exp_sum = 0.0f;
                for (int64_t i = 0; i < dim; i++) {
                    y[i] = std::exp(x[i] - max_val);
                    exp_sum += y[i];
                }

                for (int64_t i = 0; i < dim; i++) {
                    y[i] /= exp_sum;
                }
            }
        }
    }

    // SiLU: y = x * sigmoid(x)
    void silu(const float* input, float* output, int64_t count) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < count; i++) {
                fixed32 x = float_to_fixed(input[i]);
                output[i] = fixed_to_float(silu_fixed(x));
            }
        } else {
            for (int64_t i = 0; i < count; i++) {
                float x = input[i];
                output[i] = x / (1.0f + std::exp(-x));
            }
        }
    }

    // GELU
    void gelu(const float* input, float* output, int64_t count) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < count; i++) {
                fixed32 x = float_to_fixed(input[i]);
                output[i] = fixed_to_float(gelu_fixed(x));
            }
        } else {
            for (int64_t i = 0; i < count; i++) {
                float x = input[i];
                output[i] = x * 0.5f * (1.0f + std::tanh(
                    std::sqrt(2.0f / 3.14159265f) * (x + 0.044715f * x * x * x)));
            }
        }
    }

    // RoPE: Rotary Position Embedding
    void rope(const float* input, float* output, const float* freqs,
              int64_t seq_len, int64_t head_dim, int64_t pos_offset) {
        int64_t half_dim = head_dim / 2;
        if (use_fixed_point_) {
            for (int64_t pos = 0; pos < seq_len; pos++) {
                int64_t m = pos + pos_offset;
                int64_t row = pos * head_dim;
                for (int64_t i = 0; i < half_dim; i++) {
                    fixed32 inv_freq = float_to_fixed(freqs[i]);
                    fixed32 angle = mul_fixed(int_to_fixed(static_cast<int>(m)), inv_freq);
                    fixed32 cos_v = cos_fixed(angle);
                    fixed32 sin_v = sin_fixed(angle);

                    int64_t i0 = i * 2;
                    int64_t i1 = i0 + 1;
                    fixed32 x0 = float_to_fixed(input[row + i0]);
                    fixed32 x1 = float_to_fixed(input[row + i1]);

                    output[row + i0] = fixed_to_float(
                        sub_fixed(mul_fixed(x0, cos_v), mul_fixed(x1, sin_v)));
                    output[row + i1] = fixed_to_float(
                        add_fixed(mul_fixed(x0, sin_v), mul_fixed(x1, cos_v)));
                }
            }
        } else {
            for (int64_t pos = 0; pos < seq_len; pos++) {
                float m = static_cast<float>(pos + pos_offset);
                int64_t row = pos * head_dim;
                for (int64_t i = 0; i < half_dim; i++) {
                    float angle = m * freqs[i];
                    float cos_v = std::cos(angle);
                    float sin_v = std::sin(angle);

                    int64_t i0 = i * 2;
                    int64_t i1 = i0 + 1;
                    float x0 = input[row + i0];
                    float x1 = input[row + i1];

                    output[row + i0] = x0 * cos_v - x1 * sin_v;
                    output[row + i1] = x0 * sin_v + x1 * cos_v;
                }
            }
        }
    }

    // Elementwise operations
    void elem_add(const float* a, const float* b, float* out, int64_t count) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < count; i++) {
                fixed32 va = float_to_fixed(a[i]);
                fixed32 vb = float_to_fixed(b[i]);
                out[i] = fixed_to_float(add_fixed(va, vb));
            }
        } else {
            for (int64_t i = 0; i < count; i++) out[i] = a[i] + b[i];
        }
    }

    void elem_sub(const float* a, const float* b, float* out, int64_t count) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < count; i++) {
                fixed32 va = float_to_fixed(a[i]);
                fixed32 vb = float_to_fixed(b[i]);
                out[i] = fixed_to_float(sub_fixed(va, vb));
            }
        } else {
            for (int64_t i = 0; i < count; i++) out[i] = a[i] - b[i];
        }
    }

    void elem_mul(const float* a, const float* b, float* out, int64_t count) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < count; i++) {
                fixed32 va = float_to_fixed(a[i]);
                fixed32 vb = float_to_fixed(b[i]);
                out[i] = fixed_to_float(mul_fixed(va, vb));
            }
        } else {
            for (int64_t i = 0; i < count; i++) out[i] = a[i] * b[i];
        }
    }

    // Gate Mul: out = a * silu(b) — for Llama FFN
    void gate_mul(const float* a, const float* b, float* out, int64_t count) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < count; i++) {
                fixed32 va = float_to_fixed(a[i]);
                fixed32 vb = float_to_fixed(b[i]);
                out[i] = fixed_to_float(mul_fixed(va, silu_fixed(vb)));
            }
        } else {
            for (int64_t i = 0; i < count; i++) {
                float sig_b = b[i] / (1.0f + std::exp(-b[i]));
                out[i] = a[i] * sig_b;
            }
        }
    }

    // Scalar multiply
    void mul_scalar(const float* input, float scalar, float* output, int64_t count) {
        if (use_fixed_point_) {
            fixed32 s = float_to_fixed(scalar);
            for (int64_t i = 0; i < count; i++) {
                fixed32 x = float_to_fixed(input[i]);
                output[i] = fixed_to_float(mul_fixed(x, s));
            }
        } else {
            for (int64_t i = 0; i < count; i++) output[i] = input[i] * scalar;
        }
    }

    // LayerNorm
    void layernorm(const float* input, float* output, const float* gamma,
                   const float* beta, int64_t batch, int64_t hidden) {
        float eps = 1e-5f;
        for (int64_t b = 0; b < batch; b++) {
            const float* x = input + b * hidden;
            float* y = output + b * hidden;

            // Mean
            float mean = 0.0f;
            for (int64_t i = 0; i < hidden; i++) mean += x[i];
            mean /= hidden;

            // Variance
            float var = 0.0f;
            for (int64_t i = 0; i < hidden; i++) {
                float d = x[i] - mean;
                var += d * d;
            }
            var /= hidden;

            float inv_std = 1.0f / std::sqrt(var + eps);
            for (int64_t i = 0; i < hidden; i++) {
                y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        }
    }

    // Neg
    void neg(const float* input, float* output, int64_t count) {
        for (int64_t i = 0; i < count; i++) output[i] = -input[i];
    }

    // ReLU
    void relu(const float* input, float* output, int64_t count) {
        for (int64_t i = 0; i < count; i++) {
            output[i] = input[i] > 0.0f ? input[i] : 0.0f;
        }
    }

    // ========================================================================
    // Backward Operations
    // ========================================================================

    // MatMul backward: dA = dC @ B^T, dB = A^T @ dC
    void matmul_backward(const float* A, const float* B, const float* grad_C,
                         float* grad_A, float* grad_B,
                         int64_t M, int64_t K, int64_t N) {
        // grad_A[M,K] = grad_C[M,N] @ B^T[N,K]
        if (grad_A) {
            matmul_ABt(grad_C, B, grad_A, M, N, K);
        }
        // grad_B[K,N] = A^T[K,M] @ grad_C[M,N]
        if (grad_B) {
            matmul_AtB(A, grad_C, grad_B, M, K, N);
        }
    }

    // SiLU backward
    void silu_backward_op(const float* input, const float* grad_output,
                          float* grad_input, int64_t count) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < count; i++) {
                fixed32 x = float_to_fixed(input[i]);
                fixed32 go = float_to_fixed(grad_output[i]);
                grad_input[i] = fixed_to_float(at::nmcard::silu_backward(x, go));
            }
        } else {
            for (int64_t i = 0; i < count; i++) {
                float x = input[i];
                float sig = 1.0f / (1.0f + std::exp(-x));
                float grad = sig * (1.0f + x * (1.0f - sig));
                grad_input[i] = grad_output[i] * grad;
            }
        }
    }

    // GELU backward
    void gelu_backward_op(const float* input, const float* grad_output,
                          float* grad_input, int64_t count) {
        if (use_fixed_point_) {
            for (int64_t i = 0; i < count; i++) {
                fixed32 x = float_to_fixed(input[i]);
                fixed32 go = float_to_fixed(grad_output[i]);
                grad_input[i] = fixed_to_float(at::nmcard::gelu_backward(x, go));
            }
        } else {
            for (int64_t i = 0; i < count; i++) {
                float x = input[i];
                float sig = 1.0f / (1.0f + std::exp(-1.702f * x));
                float grad = sig + 1.702f * x * sig * (1.0f - sig);
                grad_input[i] = grad_output[i] * grad;
            }
        }
    }

    // ReLU backward
    void relu_backward_op(const float* input, const float* grad_output,
                          float* grad_input, int64_t count) {
        for (int64_t i = 0; i < count; i++) {
            grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
        }
    }

    // Softmax backward: grad_input_i = y_i * (grad_output_i - sum_j(y_j * grad_output_j))
    void softmax_backward_op(const float* output, const float* grad_output,
                             float* grad_input, int64_t batch, int64_t dim) {
        for (int64_t b = 0; b < batch; b++) {
            const float* y = output + b * dim;
            const float* go = grad_output + b * dim;
            float* gi = grad_input + b * dim;

            float dot = 0.0f;
            for (int64_t i = 0; i < dim; i++) {
                dot += y[i] * go[i];
            }

            for (int64_t i = 0; i < dim; i++) {
                gi[i] = y[i] * (go[i] - dot);
            }
        }
    }

    // RMSNorm backward (simplified)
    void rmsnorm_backward_op(const float* input, const float* gamma,
                             const float* grad_output,
                             float* grad_input, float* grad_gamma,
                             int64_t batch, int64_t hidden) {
        float eps = 1e-5f;
        for (int64_t b = 0; b < batch; b++) {
            const float* x = input + b * hidden;
            const float* go = grad_output + b * hidden;
            float* gi = grad_input + b * hidden;

            float sum_sq = 0.0f;
            for (int64_t i = 0; i < hidden; i++) {
                sum_sq += x[i] * x[i];
            }
            float rms = std::sqrt(sum_sq / hidden + eps);
            float inv_rms = 1.0f / rms;

            for (int64_t i = 0; i < hidden; i++) {
                float x_norm = x[i] * inv_rms;
                if (grad_gamma) {
                    grad_gamma[i] += go[i] * x_norm;
                }
                gi[i] = go[i] * gamma[i] * inv_rms;
            }
        }
    }

    // Cross-entropy loss
    float cross_entropy(const float* pred, int target_class, int vocab_size) {
        float p = pred[target_class];
        if (p < 1e-7f) p = 1e-7f;
        return -std::log(p);
    }

    // Cross-entropy backward: grad = pred - one_hot(target)
    void cross_entropy_backward_op(const float* pred, int target_class,
                                   float* grad_output, int vocab_size) {
        for (int i = 0; i < vocab_size; i++) {
            grad_output[i] = pred[i] - (i == target_class ? 1.0f : 0.0f);
        }
    }

    // ========================================================================
    // Optimizers
    // ========================================================================

    // SGD: weight -= lr * grad
    void sgd_step(float* weights, const float* grads, float lr, int64_t size) {
        if (use_fixed_point_) {
            fixed32 lr_q = float_to_fixed(lr);
            for (int64_t i = 0; i < size; i++) {
                fixed32 w = float_to_fixed(weights[i]);
                fixed32 g = float_to_fixed(grads[i]);
                weights[i] = fixed_to_float(sub_fixed(w, mul_fixed(lr_q, g)));
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                weights[i] -= lr * grads[i];
            }
        }
    }

    // Adam optimizer
    void adam_step(float* weights, const float* grads,
                   float* m, float* v,
                   float lr, float beta1, float beta2, float eps,
                   int64_t size) {
        if (use_fixed_point_) {
            fixed32 lr_q = float_to_fixed(lr);
            fixed32 b1 = float_to_fixed(beta1);
            fixed32 b2 = float_to_fixed(beta2);
            fixed32 eps_q = 1; // ~1.5e-5
            fixed32 one_minus_b1 = sub_fixed(FIXED_ONE, b1);
            fixed32 one_minus_b2 = sub_fixed(FIXED_ONE, b2);

            for (int64_t i = 0; i < size; i++) {
                fixed32 w = float_to_fixed(weights[i]);
                fixed32 g = float_to_fixed(grads[i]);
                fixed32 mi = float_to_fixed(m[i]);
                fixed32 vi = float_to_fixed(v[i]);

                mi = add_fixed(mul_fixed(b1, mi), mul_fixed(one_minus_b1, g));
                fixed32 g_sq = mul_fixed(g, g);
                vi = add_fixed(mul_fixed(b2, vi), mul_fixed(one_minus_b2, g_sq));

                fixed32 denom = add_fixed(sqrt_fixed(vi), eps_q);
                fixed32 update = mul_fixed(lr_q, div_fixed(mi, denom));
                w = sub_fixed(w, update);

                weights[i] = fixed_to_float(w);
                m[i] = fixed_to_float(mi);
                v[i] = fixed_to_float(vi);
            }
        } else {
            for (int64_t i = 0; i < size; i++) {
                m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
                v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
                weights[i] -= lr * m[i] / (std::sqrt(v[i]) + eps);
            }
        }
    }

    NMCardEmulator() = default;

private:
    bool use_fixed_point_ = false; // default: float32 for easier debugging
    int num_cores_ = 16;

    // Helper: C[M,K] = A[M,N] @ B^T[K,N] (B transposed)
    void matmul_ABt(const float* A, const float* B, float* C,
                    int64_t M, int64_t N, int64_t K) {
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < K; j++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[j * N + k];
                }
                C[i * K + j] = sum;
            }
        }
    }

    // Helper: C[K,N] = A^T[K,M] @ B[M,N] (A transposed)
    void matmul_AtB(const float* A, const float* B, float* C,
                    int64_t M, int64_t K, int64_t N) {
        // Zero output
        std::memset(C, 0, K * N * sizeof(float));
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < K; j++) {
                float a_val = A[i * K + j];
                for (int64_t k = 0; k < N; k++) {
                    C[j * N + k] += a_val * B[i * N + k];
                }
            }
        }
    }
};

} // namespace nmcard
} // namespace at
