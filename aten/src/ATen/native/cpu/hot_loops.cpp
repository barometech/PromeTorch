// ============================================================================
// hot_loops.cpp — Compiled inner loops for LTO on Elbrus E2K
// ============================================================================
// This file is compiled ONCE into libaten.a with full optimization (-O3).
// On Elbrus, LCC can schedule VLIW bundles across the entire function body.
// On x86, this eliminates duplicate copies of inline functions from headers.
//
// All functions use TUDA VecF for portable SIMD (AVX2/NEON/E2K/Scalar).
// ============================================================================

#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/tuda/TudaMath.h"
#include "aten/src/ATen/native/cpu/tuda/TudaBLAS.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {
namespace native {
namespace hot {

using tuda::VecF;

// ============================================================================
// BLAS wrappers
// ============================================================================

void sgemm(int64_t M, int64_t K, int64_t N,
           float alpha, const float* A, int64_t lda,
           const float* B, int64_t ldb,
           float beta, float* C, int64_t ldc) {
    tuda::blas::sgemm(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

void sgemm_nt(int64_t M, int64_t K, int64_t N,
              float alpha, const float* A, int64_t lda,
              const float* B, int64_t ldb,
              float beta, float* C, int64_t ldc) {
    tuda::blas::sgemm_nt(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
}

void sgemv(int64_t M, int64_t N,
           float alpha, const float* A, int64_t lda,
           const float* x,
           float beta, float* y) {
    tuda::blas::sgemv(M, N, alpha, A, lda, x, beta, y);
}

float sdot(int64_t n, const float* a, const float* b) {
    return tuda::blas::sdot(n, a, b);
}

// ============================================================================
// Element-wise binary loops
// ============================================================================

void add_loop(const float* a, const float* b, float* out, int64_t n, float alpha) {
    constexpr int W = VecF::width;
    int64_t nsimd = n - (n % (4*W));
    if (alpha == 1.0f) {
        _Pragma("omp parallel for schedule(static) if(n > 4096)")
        for (int64_t i = 0; i < nsimd; i += 4*W) {
            (VecF::load(a+i)     + VecF::load(b+i)).store(out+i);
            (VecF::load(a+i+W)   + VecF::load(b+i+W)).store(out+i+W);
            (VecF::load(a+i+2*W) + VecF::load(b+i+2*W)).store(out+i+2*W);
            (VecF::load(a+i+3*W) + VecF::load(b+i+3*W)).store(out+i+3*W);
        }
        for (int64_t i = nsimd; i + W <= n; i += W)
            (VecF::load(a+i) + VecF::load(b+i)).store(out+i);
        for (int64_t i = n - (n % W); i < n; ++i) out[i] = a[i] + b[i];
    } else {
        VecF valpha = VecF::broadcast(alpha);
        _Pragma("omp parallel for schedule(static) if(n > 4096)")
        for (int64_t i = 0; i < nsimd; i += 4*W) {
            VecF::fmadd(valpha, VecF::load(b+i),     VecF::load(a+i)).store(out+i);
            VecF::fmadd(valpha, VecF::load(b+i+W),   VecF::load(a+i+W)).store(out+i+W);
            VecF::fmadd(valpha, VecF::load(b+i+2*W), VecF::load(a+i+2*W)).store(out+i+2*W);
            VecF::fmadd(valpha, VecF::load(b+i+3*W), VecF::load(a+i+3*W)).store(out+i+3*W);
        }
        for (int64_t i = nsimd; i + W <= n; i += W)
            VecF::fmadd(valpha, VecF::load(b+i), VecF::load(a+i)).store(out+i);
        for (int64_t i = n - (n % W); i < n; ++i) out[i] = a[i] + alpha * b[i];
    }
}

void sub_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    int64_t nsimd = n - (n % (4*W));
    _Pragma("omp parallel for schedule(static) if(n > 4096)")
    for (int64_t i = 0; i < nsimd; i += 4*W) {
        (VecF::load(a+i)     - VecF::load(b+i)).store(out+i);
        (VecF::load(a+i+W)   - VecF::load(b+i+W)).store(out+i+W);
        (VecF::load(a+i+2*W) - VecF::load(b+i+2*W)).store(out+i+2*W);
        (VecF::load(a+i+3*W) - VecF::load(b+i+3*W)).store(out+i+3*W);
    }
    for (int64_t i = nsimd; i + W <= n; i += W)
        (VecF::load(a+i) - VecF::load(b+i)).store(out+i);
    for (int64_t i = n - (n % W); i < n; ++i) out[i] = a[i] - b[i];
}

void mul_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    int64_t nsimd = n - (n % (4*W));
    _Pragma("omp parallel for schedule(static) if(n > 4096)")
    for (int64_t i = 0; i < nsimd; i += 4*W) {
        (VecF::load(a+i)     * VecF::load(b+i)).store(out+i);
        (VecF::load(a+i+W)   * VecF::load(b+i+W)).store(out+i+W);
        (VecF::load(a+i+2*W) * VecF::load(b+i+2*W)).store(out+i+2*W);
        (VecF::load(a+i+3*W) * VecF::load(b+i+3*W)).store(out+i+3*W);
    }
    for (int64_t i = nsimd; i + W <= n; i += W)
        (VecF::load(a+i) * VecF::load(b+i)).store(out+i);
    for (int64_t i = n - (n % W); i < n; ++i) out[i] = a[i] * b[i];
}

void div_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    int64_t nsimd = n - (n % (4*W));
    _Pragma("omp parallel for schedule(static) if(n > 4096)")
    for (int64_t i = 0; i < nsimd; i += 4*W) {
        (VecF::load(a+i)     / VecF::load(b+i)).store(out+i);
        (VecF::load(a+i+W)   / VecF::load(b+i+W)).store(out+i+W);
        (VecF::load(a+i+2*W) / VecF::load(b+i+2*W)).store(out+i+2*W);
        (VecF::load(a+i+3*W) / VecF::load(b+i+3*W)).store(out+i+3*W);
    }
    for (int64_t i = nsimd; i + W <= n; i += W)
        (VecF::load(a+i) / VecF::load(b+i)).store(out+i);
    for (int64_t i = n - (n % W); i < n; ++i) out[i] = a[i] / b[i];
}

void add_broadcast_loop(const float* a, const float* b, float* out,
                        int64_t outer, int64_t inner, float alpha) {
    constexpr int W = VecF::width;
    if (alpha == 1.0f) {
        _Pragma("omp parallel for schedule(static) if(outer > 16)")
        for (int64_t o = 0; o < outer; ++o) {
            const float* row_a = a + o * inner;
            float* row_out = out + o * inner;
            int64_t j = 0;
            for (; j + W <= inner; j += W)
                (VecF::load(row_a+j) + VecF::load(b+j)).store(row_out+j);
            for (; j < inner; ++j) row_out[j] = row_a[j] + b[j];
        }
    } else {
        VecF valpha = VecF::broadcast(alpha);
        _Pragma("omp parallel for schedule(static) if(outer > 16)")
        for (int64_t o = 0; o < outer; ++o) {
            const float* row_a = a + o * inner;
            float* row_out = out + o * inner;
            int64_t j = 0;
            for (; j + W <= inner; j += W)
                VecF::fmadd(valpha, VecF::load(b+j), VecF::load(row_a+j)).store(row_out+j);
            for (; j < inner; ++j) row_out[j] = row_a[j] + alpha * b[j];
        }
    }
}

// ============================================================================
// Unary loops — use TUDA vectorized math
// ============================================================================

#define DEFINE_HOT_UNARY(name, vec_expr, scalar_expr)                     \
void name##_loop(const float* in, float* out, int64_t n) {               \
    constexpr int W = VecF::width;                                        \
    int64_t nsimd = n - (n % (4*W));                                      \
    _Pragma("omp parallel for schedule(static) if(n > 4096)")             \
    for (int64_t i = 0; i < nsimd; i += 4*W) {                           \
        { VecF v = VecF::load(in+i);       (vec_expr).store(out+i); }    \
        { VecF v = VecF::load(in+i+W);     (vec_expr).store(out+i+W); } \
        { VecF v = VecF::load(in+i+2*W);   (vec_expr).store(out+i+2*W); } \
        { VecF v = VecF::load(in+i+3*W);   (vec_expr).store(out+i+3*W); } \
    }                                                                     \
    for (int64_t i = nsimd; i + W <= n; i += W) {                         \
        VecF v = VecF::load(in+i);                                        \
        (vec_expr).store(out+i);                                          \
    }                                                                     \
    for (int64_t i = n - (n % W); i < n; ++i) {                           \
        float x = in[i];                                                  \
        out[i] = (scalar_expr);                                           \
    }                                                                     \
}

DEFINE_HOT_UNARY(neg,        tuda::neg_vec(v),        -x)
DEFINE_HOT_UNARY(abs,        tuda::abs_vec(v),        std::abs(x))
DEFINE_HOT_UNARY(sqrt,       tuda::sqrt_vec(v),       std::sqrt(x))
DEFINE_HOT_UNARY(rsqrt,      tuda::rsqrt_vec(v),      1.0f / std::sqrt(x))
DEFINE_HOT_UNARY(square,     tuda::square_vec(v),     x * x)
DEFINE_HOT_UNARY(exp,        tuda::exp_vec(v),        std::exp(x))
DEFINE_HOT_UNARY(log,        tuda::log_vec(v),        std::log(x))
DEFINE_HOT_UNARY(log2,       tuda::log2_vec(v),       std::log2(x))
DEFINE_HOT_UNARY(log10,      tuda::log10_vec(v),      std::log10(x))
DEFINE_HOT_UNARY(sin,        tuda::sin_vec(v),        std::sin(x))
DEFINE_HOT_UNARY(cos,        tuda::cos_vec(v),        std::cos(x))
DEFINE_HOT_UNARY(tan,        tuda::tan_vec(v),        std::tan(x))
DEFINE_HOT_UNARY(tanh,       tuda::tanh_vec(v),       std::tanh(x))
DEFINE_HOT_UNARY(sigmoid,    tuda::sigmoid_vec(v),    1.0f / (1.0f + std::exp(-x)))
DEFINE_HOT_UNARY(relu,       tuda::relu_vec(v),       (x > 0.0f ? x : 0.0f))
DEFINE_HOT_UNARY(reciprocal, tuda::reciprocal_vec(v), 1.0f / x)
DEFINE_HOT_UNARY(ceil,       tuda::ceil_vec(v),       std::ceil(x))
DEFINE_HOT_UNARY(floor,      tuda::floor_vec(v),      std::floor(x))
DEFINE_HOT_UNARY(round,      tuda::round_vec(v),      std::round(x))
DEFINE_HOT_UNARY(sign,       tuda::sign_vec(v),       ((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)))

#undef DEFINE_HOT_UNARY

// ============================================================================
// Reduction loops
// ============================================================================

float sum_loop(const float* data, int64_t n) {
    return tuda::vec_sum(data, n);
}

void sum_dim_loop(const float* in, float* out,
                  int64_t outer_size, int64_t reduce_size, int64_t inner_size) {
    constexpr int W = VecF::width;

    if (inner_size == 1) {
        // Reducing last dim: contiguous sum of reduce_size elements per outer
        _Pragma("omp parallel for schedule(static) if(outer_size > 16)")
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            out[outer] = tuda::vec_sum(in + outer * reduce_size, reduce_size);
        }
    } else {
        // Accumulate reduce_size rows, each of length inner_size
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            float* out_row = out + outer * inner_size;
            std::memset(out_row, 0, inner_size * sizeof(float));
            for (int64_t r = 0; r < reduce_size; ++r) {
                const float* in_row = in + (outer * reduce_size + r) * inner_size;
                int64_t j = 0;
                for (; j + 4*W <= inner_size; j += 4*W) {
                    (VecF::load(out_row+j)     + VecF::load(in_row+j)).store(out_row+j);
                    (VecF::load(out_row+j+W)   + VecF::load(in_row+j+W)).store(out_row+j+W);
                    (VecF::load(out_row+j+2*W) + VecF::load(in_row+j+2*W)).store(out_row+j+2*W);
                    (VecF::load(out_row+j+3*W) + VecF::load(in_row+j+3*W)).store(out_row+j+3*W);
                }
                for (; j + W <= inner_size; j += W)
                    (VecF::load(out_row+j) + VecF::load(in_row+j)).store(out_row+j);
                for (; j < inner_size; ++j)
                    out_row[j] += in_row[j];
            }
        }
    }
}

// ============================================================================
// In-place scalar loops (optimizer hot paths)
// ============================================================================

void mul_scalar_inplace(float* data, float scalar, int64_t n) {
    constexpr int W = VecF::width;
    VecF vs = VecF::broadcast(scalar);
    int64_t nsimd = n - (n % (4*W));
    _Pragma("omp parallel for schedule(static) if(n > 4096)")
    for (int64_t i = 0; i < nsimd; i += 4*W) {
        (VecF::load(data+i)     * vs).store(data+i);
        (VecF::load(data+i+W)   * vs).store(data+i+W);
        (VecF::load(data+i+2*W) * vs).store(data+i+2*W);
        (VecF::load(data+i+3*W) * vs).store(data+i+3*W);
    }
    for (int64_t i = nsimd; i + W <= n; i += W)
        (VecF::load(data+i) * vs).store(data+i);
    for (int64_t i = n - (n % W); i < n; ++i) data[i] *= scalar;
}

void axpy_inplace(float* data, float scalar, const float* other, int64_t n) {
    constexpr int W = VecF::width;
    VecF vs = VecF::broadcast(scalar);
    int64_t nsimd = n - (n % (4*W));
    _Pragma("omp parallel for schedule(static) if(n > 4096)")
    for (int64_t i = 0; i < nsimd; i += 4*W) {
        VecF::fmadd(vs, VecF::load(other+i),     VecF::load(data+i)).store(data+i);
        VecF::fmadd(vs, VecF::load(other+i+W),   VecF::load(data+i+W)).store(data+i+W);
        VecF::fmadd(vs, VecF::load(other+i+2*W), VecF::load(data+i+2*W)).store(data+i+2*W);
        VecF::fmadd(vs, VecF::load(other+i+3*W), VecF::load(data+i+3*W)).store(data+i+3*W);
    }
    for (int64_t i = nsimd; i + W <= n; i += W)
        VecF::fmadd(vs, VecF::load(other+i), VecF::load(data+i)).store(data+i);
    for (int64_t i = n - (n % W); i < n; ++i) data[i] += scalar * other[i];
}

void adam_step_loop(float* param, const float* grad,
                    float* exp_avg, float* exp_avg_sq,
                    int64_t n, float lr, float beta1, float beta2,
                    float eps, float weight_decay,
                    float bias_correction1, float bias_correction2,
                    bool amsgrad, float* max_exp_avg_sq) {
    constexpr int W = VecF::width;
    VecF vbeta1 = VecF::broadcast(beta1);
    VecF vbeta2 = VecF::broadcast(beta2);
    VecF v1mb1  = VecF::broadcast(1.0f - beta1);
    VecF v1mb2  = VecF::broadcast(1.0f - beta2);
    VecF veps   = VecF::broadcast(eps);
    float step_size = lr / bias_correction1;
    VecF vstep  = VecF::broadcast(step_size);
    float bc2_sqrt = std::sqrt(bias_correction2);
    VecF vbc2   = VecF::broadcast(1.0f / bc2_sqrt);
    VecF vwd    = VecF::broadcast(weight_decay);

    int64_t nsimd = n - (n % W);
    _Pragma("omp parallel for schedule(static) if(n > 4096)")
    for (int64_t i = 0; i < nsimd; i += W) {
        VecF g = VecF::load(grad + i);
        VecF p = VecF::load(param + i);

        // Weight decay (decoupled, AdamW style)
        if (weight_decay != 0.0f) {
            g = g + vwd * p;
        }

        // Update biased first moment: m = beta1 * m + (1-beta1) * g
        VecF m = VecF::fmadd(vbeta1, VecF::load(exp_avg + i), v1mb1 * g);
        m.store(exp_avg + i);

        // Update biased second moment: v = beta2 * v + (1-beta2) * g^2
        VecF v = VecF::fmadd(vbeta2, VecF::load(exp_avg_sq + i), v1mb2 * g * g);
        v.store(exp_avg_sq + i);

        VecF denom;
        if (amsgrad && max_exp_avg_sq) {
            VecF mv = VecF::load(max_exp_avg_sq + i);
            float* mv_ptr = max_exp_avg_sq + i;
            float* v_ptr = exp_avg_sq + i;
            for (int j = 0; j < W; ++j) {
                if (v_ptr[j] > mv_ptr[j]) mv_ptr[j] = v_ptr[j];
            }
            mv = VecF::load(max_exp_avg_sq + i);
            denom = tuda::sqrt_vec(mv * vbc2 * vbc2) + veps;
        } else {
            denom = tuda::sqrt_vec(v * vbc2 * vbc2) + veps;
        }

        // param -= step_size * m / denom
        p = p - vstep * m / denom;
        p.store(param + i);
    }

    // Scalar tail
    for (int64_t i = nsimd; i < n; ++i) {
        float g = grad[i];
        float p = param[i];

        if (weight_decay != 0.0f) g += weight_decay * p;

        exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;

        float denom_val;
        if (amsgrad && max_exp_avg_sq) {
            max_exp_avg_sq[i] = std::max(max_exp_avg_sq[i], exp_avg_sq[i]);
            denom_val = std::sqrt(max_exp_avg_sq[i] / bias_correction2) + eps;
        } else {
            denom_val = std::sqrt(exp_avg_sq[i] / bias_correction2) + eps;
        }

        param[i] = p - step_size * exp_avg[i] / denom_val;
    }
}

void sgd_step_loop(float* param, const float* grad, float* momentum_buf,
                   int64_t n, float lr, float momentum, float dampening,
                   float weight_decay, bool nesterov) {
    constexpr int W = VecF::width;
    VecF vlr = VecF::broadcast(lr);
    VecF vwd = VecF::broadcast(weight_decay);
    VecF vmom = VecF::broadcast(momentum);
    VecF vdamp = VecF::broadcast(1.0f - dampening);

    if (momentum == 0.0f) {
        // No momentum: simple SGD — fully parallelizable
        int64_t nsimd = n - (n % W);
        _Pragma("omp parallel for schedule(static) if(n > 4096)")
        for (int64_t i = 0; i < nsimd; i += W) {
            VecF g = VecF::load(grad + i);
            VecF p = VecF::load(param + i);
            if (weight_decay != 0.0f) g = g + vwd * p;
            p = p - vlr * g;
            p.store(param + i);
        }
        for (int64_t i = nsimd; i < n; ++i) {
            float g = grad[i];
            if (weight_decay != 0.0f) g += weight_decay * param[i];
            param[i] -= lr * g;
        }
    } else {
        // SGD with momentum — sequential (momentum_buf[i] depends on previous iteration only within same i)
        // Actually each i is independent, so we CAN parallelize!
        int64_t nsimd = n - (n % W);
        _Pragma("omp parallel for schedule(static) if(n > 4096)")
        for (int64_t i = 0; i < nsimd; i += W) {
            VecF g = VecF::load(grad + i);
            VecF p = VecF::load(param + i);
            if (weight_decay != 0.0f) g = g + vwd * p;

            VecF buf = VecF::fmadd(vmom, VecF::load(momentum_buf + i), vdamp * g);
            buf.store(momentum_buf + i);

            if (nesterov) {
                p = p - vlr * (g + vmom * buf);
            } else {
                p = p - vlr * buf;
            }
            p.store(param + i);
        }
        for (int64_t i = nsimd; i < n; ++i) {
            float g = grad[i];
            if (weight_decay != 0.0f) g += weight_decay * param[i];

            momentum_buf[i] = momentum * momentum_buf[i] + (1.0f - dampening) * g;

            if (nesterov) {
                param[i] -= lr * (g + momentum * momentum_buf[i]);
            } else {
                param[i] -= lr * momentum_buf[i];
            }
        }
    }
}

} // namespace hot
} // namespace native
} // namespace at
