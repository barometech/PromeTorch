#pragma once
// ============================================================================
// LinQEmulator.h — Software emulator for LinQ H1M tensor accelerator
// ============================================================================
// Emulates the LinQ H1M NPU operations in software:
//   - INT8 GEMM with INT32 accumulation (96 TOPS peak on hardware)
//   - FP32 GEMM for training compatibility
//   - Activations: ReLU, SiLU, GELU, Softmax
//   - Normalization: LayerNorm, RMSNorm
//   - Element-wise: add, mul, sub
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

#ifdef _MSC_VER
    #ifdef BUILDING_ATEN_LINQ
        #define ATEN_LINQ_API __declspec(dllexport)
    #else
        #define ATEN_LINQ_API __declspec(dllimport)
    #endif
#else
    #define ATEN_LINQ_API
#endif

namespace at {
namespace linq {

class ATEN_LINQ_API LinQEmulator {
public:
    static LinQEmulator& get() {
        static LinQEmulator instance;
        return instance;
    }

    // Configuration
    void set_int8_mode(bool enable) { use_int8_ = enable; }
    bool is_int8_mode() const { return use_int8_; }
    void set_num_cores(int n) { num_cores_ = std::clamp(n, 1, 32); }
    int num_cores() const { return num_cores_; }

    // ================================================================
    // Matrix operations
    // ================================================================

    // FP32 GEMM: C = A @ B
    void matmul_fp32(const float* A, const float* B, float* C,
                     int64_t M, int64_t K, int64_t N) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    // INT8 GEMM with INT32 accumulation (quantized inference)
    void matmul_int8(const int8_t* A, const int8_t* B, int32_t* C,
                     int64_t M, int64_t K, int64_t N) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                int32_t sum = 0;
                for (int64_t k = 0; k < K; ++k) {
                    sum += static_cast<int32_t>(A[i * K + k]) *
                           static_cast<int32_t>(B[k * N + j]);
                }
                C[i * N + j] = sum;
            }
        }
    }

    // ================================================================
    // Activation functions
    // ================================================================

    void relu(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

    void silu(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            float x = input[i];
            output[i] = x / (1.0f + std::exp(-x));
        }
    }

    void gelu(const float* input, float* output, int64_t n) {
        constexpr float sqrt2pi = 0.7978845608028654f;
        constexpr float c = 0.044715f;
        for (int64_t i = 0; i < n; ++i) {
            float x = input[i];
            float inner = sqrt2pi * (x + c * x * x * x);
            output[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }

    void sigmoid(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }

    void tanh_op(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            output[i] = std::tanh(input[i]);
    }

    // ================================================================
    // Normalization
    // ================================================================

    void softmax(const float* input, float* output, int64_t n) {
        float max_val = input[0];
        for (int64_t i = 1; i < n; ++i)
            if (input[i] > max_val) max_val = input[i];

        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        float inv_sum = 1.0f / sum;
        for (int64_t i = 0; i < n; ++i)
            output[i] *= inv_sum;
    }

    void layernorm(const float* input, const float* weight, const float* bias,
                   float* output, int64_t batch, int64_t hidden, float eps) {
        for (int64_t b = 0; b < batch; ++b) {
            const float* row = input + b * hidden;
            float* out_row = output + b * hidden;

            // Mean
            float mean = 0.0f;
            for (int64_t i = 0; i < hidden; ++i) mean += row[i];
            mean /= static_cast<float>(hidden);

            // Variance
            float var = 0.0f;
            for (int64_t i = 0; i < hidden; ++i) {
                float d = row[i] - mean;
                var += d * d;
            }
            var /= static_cast<float>(hidden);

            float inv_std = 1.0f / std::sqrt(var + eps);
            for (int64_t i = 0; i < hidden; ++i) {
                float norm = (row[i] - mean) * inv_std;
                out_row[i] = weight ? (norm * weight[i] + (bias ? bias[i] : 0.0f)) : norm;
            }
        }
    }

    void rmsnorm(const float* input, const float* weight,
                 float* output, int64_t numel, float eps) {
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < numel; ++i) sum_sq += input[i] * input[i];
        float rms = std::sqrt(sum_sq / static_cast<float>(numel) + eps);
        float inv_rms = 1.0f / rms;
        for (int64_t i = 0; i < numel; ++i)
            output[i] = input[i] * inv_rms * (weight ? weight[i] : 1.0f);
    }

    // ================================================================
    // Element-wise operations
    // ================================================================

    void elem_add(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }

    void elem_sub(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
    }

    void elem_mul(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
    }

    void elem_div(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
    }

    void neg(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i) output[i] = -input[i];
    }

    void abs_op(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i) output[i] = std::fabs(input[i]);
    }

    void sqrt_op(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i) output[i] = std::sqrt(input[i]);
    }

    void exp_op(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i) output[i] = std::exp(input[i]);
    }

    void log_op(const float* input, float* output, int64_t n) {
        for (int64_t i = 0; i < n; ++i) output[i] = std::log(input[i]);
    }

    // ================================================================
    // Additional unary operations
    // ================================================================

    void sin_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::sin(in[i]);
    }
    void cos_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::cos(in[i]);
    }
    void tan_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::tan(in[i]);
    }
    void log2_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::log2(in[i]);
    }
    void log10_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::log10(in[i]);
    }
    void rsqrt_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = 1.0f / std::sqrt(in[i]);
    }
    void square_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = in[i] * in[i];
    }
    void reciprocal_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = 1.0f / in[i];
    }
    void ceil_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::ceil(in[i]);
    }
    void floor_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::floor(in[i]);
    }
    void round_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::round(in[i]);
    }
    void sign_op(const float* in, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = (in[i] > 0.0f) ? 1.0f : ((in[i] < 0.0f) ? -1.0f : 0.0f);
    }
    void leaky_relu(const float* in, float* out, int64_t n, float alpha) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = in[i] > 0.0f ? in[i] : alpha * in[i];
    }

    // ================================================================
    // Clamp / max / min
    // ================================================================

    void clamp_op(const float* in, float* out, int64_t n, float lo, float hi) {
        for (int64_t i = 0; i < n; ++i)
            out[i] = std::max(lo, std::min(hi, in[i]));
    }
    void maximum_op(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::max(a[i], b[i]);
    }
    void minimum_op(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::min(a[i], b[i]);
    }

    // ================================================================
    // Scalar binary ops
    // ================================================================

    void add_scalar(const float* in, float* out, int64_t n, float s) {
        for (int64_t i = 0; i < n; ++i) out[i] = in[i] + s;
    }
    void mul_scalar(const float* in, float* out, int64_t n, float s) {
        for (int64_t i = 0; i < n; ++i) out[i] = in[i] * s;
    }
    void pow_scalar(const float* in, float* out, int64_t n, float p) {
        for (int64_t i = 0; i < n; ++i) out[i] = std::pow(in[i], p);
    }

    // ================================================================
    // Comparison ops (return 0.0f / 1.0f)
    // ================================================================

    void eq_op(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
    }
    void ne_op(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] != b[i]) ? 1.0f : 0.0f;
    }
    void lt_op(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
    }
    void le_op(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
    }
    void gt_op(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }
    void ge_op(const float* a, const float* b, float* out, int64_t n) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
    }
    void eq_scalar(const float* a, float* out, int64_t n, float s) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] == s) ? 1.0f : 0.0f;
    }
    void ne_scalar(const float* a, float* out, int64_t n, float s) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] != s) ? 1.0f : 0.0f;
    }
    void lt_scalar(const float* a, float* out, int64_t n, float s) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] < s) ? 1.0f : 0.0f;
    }
    void le_scalar(const float* a, float* out, int64_t n, float s) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] <= s) ? 1.0f : 0.0f;
    }
    void gt_scalar(const float* a, float* out, int64_t n, float s) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] > s) ? 1.0f : 0.0f;
    }
    void ge_scalar(const float* a, float* out, int64_t n, float s) {
        for (int64_t i = 0; i < n; ++i) out[i] = (a[i] >= s) ? 1.0f : 0.0f;
    }

    // ================================================================
    // Reduction ops
    // ================================================================

    float sum(const float* in, int64_t n) {
        float s = 0.0f;
        for (int64_t i = 0; i < n; ++i) s += in[i];
        return s;
    }
    float max_val(const float* in, int64_t n) {
        float m = in[0];
        for (int64_t i = 1; i < n; ++i) if (in[i] > m) m = in[i];
        return m;
    }
    float min_val(const float* in, int64_t n) {
        float m = in[0];
        for (int64_t i = 1; i < n; ++i) if (in[i] < m) m = in[i];
        return m;
    }
    int64_t argmax(const float* in, int64_t n) {
        int64_t idx = 0;
        for (int64_t i = 1; i < n; ++i) if (in[i] > in[idx]) idx = i;
        return idx;
    }
    int64_t argmin(const float* in, int64_t n) {
        int64_t idx = 0;
        for (int64_t i = 1; i < n; ++i) if (in[i] < in[idx]) idx = i;
        return idx;
    }

    // ================================================================
    // Memory ops
    // ================================================================

    void fill(float* data, int64_t n, float val) {
        for (int64_t i = 0; i < n; ++i) data[i] = val;
    }
    void copy(const float* src, float* dst, int64_t n) {
        std::memcpy(dst, src, n * sizeof(float));
    }

    // ================================================================
    // Fused ops
    // ================================================================

    void addcmul(const float* self, const float* t1, const float* t2,
                 float* out, int64_t n, float value) {
        for (int64_t i = 0; i < n; ++i) out[i] = self[i] + value * t1[i] * t2[i];
    }
    void addcdiv(const float* self, const float* t1, const float* t2,
                 float* out, int64_t n, float value) {
        for (int64_t i = 0; i < n; ++i) out[i] = self[i] + value * t1[i] / t2[i];
    }

    // ================================================================
    // Matrix-vector multiply: y = A @ x
    // ================================================================

    void matvec(const float* A, const float* x, float* y, int64_t M, int64_t N) {
        for (int64_t i = 0; i < M; ++i) {
            float s = 0.0f;
            for (int64_t j = 0; j < N; ++j) s += A[i * N + j] * x[j];
            y[i] = s;
        }
    }

    // Dot product
    float dot(const float* a, const float* b, int64_t n) {
        float s = 0.0f;
        for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    }

    // ================================================================
    // Quantization helpers
    // ================================================================

    void quantize_fp32_to_int8(const float* input, int8_t* output,
                               float* scale, int64_t n) {
        float max_abs = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float v = std::fabs(input[i]);
            if (v > max_abs) max_abs = v;
        }
        *scale = max_abs / 127.0f;
        float inv_scale = (*scale > 0.0f) ? (127.0f / max_abs) : 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float v = input[i] * inv_scale;
            v = std::max(-127.0f, std::min(127.0f, v));
            output[i] = static_cast<int8_t>(std::round(v));
        }
    }

    void dequantize_int32_to_fp32(const int32_t* input, float* output,
                                  float scale_a, float scale_b, int64_t n) {
        float scale = scale_a * scale_b;
        for (int64_t i = 0; i < n; ++i)
            output[i] = static_cast<float>(input[i]) * scale;
    }

private:
    bool use_int8_ = false;
    int num_cores_ = 32;
};

} // namespace linq
} // namespace at
