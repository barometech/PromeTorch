// ============================================================================
// hot_loops.cpp — Compiled inner loops for LTO on Elbrus E2K
// ============================================================================
// THREADING: c10::ThreadPool (persistent threads) replaces OpenMP fork/join.
// On Elbrus, OpenMP fork/join costs ~10ms per region = 93s overhead/epoch.
// ============================================================================

#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/tuda/TudaMath.h"
#include "aten/src/ATen/native/cpu/tuda/TudaBLAS.h"
#include "c10/util/ThreadPool.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <atomic>

namespace at {
namespace native {
namespace hot {

using tuda::VecF;

// BLAS wrappers
void sgemm(int64_t M, int64_t K, int64_t N, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) { tuda::blas::sgemm(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc); }
void sgemm_nt(int64_t M, int64_t K, int64_t N, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) { tuda::blas::sgemm_nt(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc); }
void sgemv(int64_t M, int64_t N, float alpha, const float* A, int64_t lda, const float* x, float beta, float* y) { tuda::blas::sgemv(M, N, alpha, A, lda, x, beta, y); }
float sdot(int64_t n, const float* a, const float* b) { return tuda::blas::sdot(n, a, b); }

// ============================================================================
// Element-wise binary loops
// ============================================================================

void add_loop(const float* a, const float* b, float* out, int64_t n, float alpha) {
    constexpr int W = VecF::width;
    if (alpha == 1.0f) {
        c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
            int64_t i = start;
            for (; i < end && (i % W != 0); ++i) out[i] = a[i] + b[i];
            int64_t se = end - ((end - i) % (4*W));
            for (; i < se; i += 4*W) {
                (VecF::load(a+i) + VecF::load(b+i)).store(out+i);
                (VecF::load(a+i+W) + VecF::load(b+i+W)).store(out+i+W);
                (VecF::load(a+i+2*W) + VecF::load(b+i+2*W)).store(out+i+2*W);
                (VecF::load(a+i+3*W) + VecF::load(b+i+3*W)).store(out+i+3*W);
            }
            for (; i + W <= end; i += W) (VecF::load(a+i) + VecF::load(b+i)).store(out+i);
            for (; i < end; ++i) out[i] = a[i] + b[i];
        });
    } else {
        VecF va = VecF::broadcast(alpha);
        c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
            int64_t i = start;
            for (; i < end && (i % W != 0); ++i) out[i] = a[i] + alpha * b[i];
            int64_t se = end - ((end - i) % (4*W));
            for (; i < se; i += 4*W) {
                VecF::fmadd(va, VecF::load(b+i), VecF::load(a+i)).store(out+i);
                VecF::fmadd(va, VecF::load(b+i+W), VecF::load(a+i+W)).store(out+i+W);
                VecF::fmadd(va, VecF::load(b+i+2*W), VecF::load(a+i+2*W)).store(out+i+2*W);
                VecF::fmadd(va, VecF::load(b+i+3*W), VecF::load(a+i+3*W)).store(out+i+3*W);
            }
            for (; i + W <= end; i += W) VecF::fmadd(va, VecF::load(b+i), VecF::load(a+i)).store(out+i);
            for (; i < end; ++i) out[i] = a[i] + alpha * b[i];
        });
    }
}

void sub_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) out[i] = a[i] - b[i];
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            (VecF::load(a+i) - VecF::load(b+i)).store(out+i);
            (VecF::load(a+i+W) - VecF::load(b+i+W)).store(out+i+W);
            (VecF::load(a+i+2*W) - VecF::load(b+i+2*W)).store(out+i+2*W);
            (VecF::load(a+i+3*W) - VecF::load(b+i+3*W)).store(out+i+3*W);
        }
        for (; i + W <= end; i += W) (VecF::load(a+i) - VecF::load(b+i)).store(out+i);
        for (; i < end; ++i) out[i] = a[i] - b[i];
    });
}

void mul_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) out[i] = a[i] * b[i];
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            (VecF::load(a+i) * VecF::load(b+i)).store(out+i);
            (VecF::load(a+i+W) * VecF::load(b+i+W)).store(out+i+W);
            (VecF::load(a+i+2*W) * VecF::load(b+i+2*W)).store(out+i+2*W);
            (VecF::load(a+i+3*W) * VecF::load(b+i+3*W)).store(out+i+3*W);
        }
        for (; i + W <= end; i += W) (VecF::load(a+i) * VecF::load(b+i)).store(out+i);
        for (; i < end; ++i) out[i] = a[i] * b[i];
    });
}

void div_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) out[i] = a[i] / b[i];
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            (VecF::load(a+i) / VecF::load(b+i)).store(out+i);
            (VecF::load(a+i+W) / VecF::load(b+i+W)).store(out+i+W);
            (VecF::load(a+i+2*W) / VecF::load(b+i+2*W)).store(out+i+2*W);
            (VecF::load(a+i+3*W) / VecF::load(b+i+3*W)).store(out+i+3*W);
        }
        for (; i + W <= end; i += W) (VecF::load(a+i) / VecF::load(b+i)).store(out+i);
        for (; i < end; ++i) out[i] = a[i] / b[i];
    });
}

void add_broadcast_loop(const float* a, const float* b, float* out, int64_t outer, int64_t inner, float alpha) {
    constexpr int W = VecF::width;
    if (alpha == 1.0f) {
        c10::parallel_for_1d(outer, [&](int64_t start, int64_t end) {
            for (int64_t o = start; o < end; ++o) {
                const float* ra = a + o * inner; float* ro = out + o * inner;
                int64_t j = 0;
                for (; j + W <= inner; j += W) (VecF::load(ra+j) + VecF::load(b+j)).store(ro+j);
                for (; j < inner; ++j) ro[j] = ra[j] + b[j];
            }
        }, 16);
    } else {
        VecF va = VecF::broadcast(alpha);
        c10::parallel_for_1d(outer, [&](int64_t start, int64_t end) {
            for (int64_t o = start; o < end; ++o) {
                const float* ra = a + o * inner; float* ro = out + o * inner;
                int64_t j = 0;
                for (; j + W <= inner; j += W) VecF::fmadd(va, VecF::load(b+j), VecF::load(ra+j)).store(ro+j);
                for (; j < inner; ++j) ro[j] = ra[j] + alpha * b[j];
            }
        }, 16);
    }
}

// ============================================================================
// Unary loops
// ============================================================================

#define DEFINE_HOT_UNARY(name, vec_expr, scalar_expr) \
void name##_loop(const float* in, float* out, int64_t n) { \
    constexpr int W = VecF::width; \
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) { \
        int64_t i = start; \
        for (; i < end && (i % W != 0); ++i) { float x = in[i]; out[i] = (scalar_expr); } \
        int64_t se = end - ((end - i) % (4*W)); \
        for (; i < se; i += 4*W) { \
            { VecF v = VecF::load(in+i);     (vec_expr).store(out+i); } \
            { VecF v = VecF::load(in+i+W);   (vec_expr).store(out+i+W); } \
            { VecF v = VecF::load(in+i+2*W); (vec_expr).store(out+i+2*W); } \
            { VecF v = VecF::load(in+i+3*W); (vec_expr).store(out+i+3*W); } \
        } \
        for (; i + W <= end; i += W) { VecF v = VecF::load(in+i); (vec_expr).store(out+i); } \
        for (; i < end; ++i) { float x = in[i]; out[i] = (scalar_expr); } \
    }); \
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
    if (n >= 65536) {
        auto& pool = c10::get_thread_pool();
        int nt = pool.num_threads();
        std::vector<float> partial(nt, 0.0f);
        std::atomic<int> cid{0};
        pool.parallel_for(n, [&](int64_t s, int64_t e) {
            float ls = tuda::vec_sum(data + s, e - s);
            int id = cid.fetch_add(1, std::memory_order_relaxed);
            partial[id] = ls;
        });
        float t = 0; for (int i = 0; i < nt; ++i) t += partial[i];
        return t;
    }
    return tuda::vec_sum(data, n);
}

void sum_dim_loop(const float* in, float* out, int64_t outer_size, int64_t reduce_size, int64_t inner_size) {
    constexpr int W = VecF::width;
    if (inner_size == 1) {
        c10::parallel_for_1d(outer_size, [&](int64_t start, int64_t end) {
            for (int64_t o = start; o < end; ++o)
                out[o] = tuda::vec_sum(in + o * reduce_size, reduce_size);
        }, 16);
    } else {
        c10::parallel_for_1d(outer_size, [&](int64_t start, int64_t end) {
            for (int64_t o = start; o < end; ++o) {
                float* orow = out + o * inner_size;
                std::memset(orow, 0, inner_size * sizeof(float));
                for (int64_t r = 0; r < reduce_size; ++r) {
                    const float* irow = in + (o * reduce_size + r) * inner_size;
                    int64_t j = 0;
                    for (; j + 4*W <= inner_size; j += 4*W) {
                        (VecF::load(orow+j) + VecF::load(irow+j)).store(orow+j);
                        (VecF::load(orow+j+W) + VecF::load(irow+j+W)).store(orow+j+W);
                        (VecF::load(orow+j+2*W) + VecF::load(irow+j+2*W)).store(orow+j+2*W);
                        (VecF::load(orow+j+3*W) + VecF::load(irow+j+3*W)).store(orow+j+3*W);
                    }
                    for (; j + W <= inner_size; j += W) (VecF::load(orow+j) + VecF::load(irow+j)).store(orow+j);
                    for (; j < inner_size; ++j) orow[j] += irow[j];
                }
            }
        }, 16);
    }
}

// ============================================================================
// In-place scalar loops (optimizer hot paths)
// ============================================================================

void mul_scalar_inplace(float* data, float scalar, int64_t n) {
    constexpr int W = VecF::width;
    VecF vs = VecF::broadcast(scalar);
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) data[i] *= scalar;
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            (VecF::load(data+i) * vs).store(data+i);
            (VecF::load(data+i+W) * vs).store(data+i+W);
            (VecF::load(data+i+2*W) * vs).store(data+i+2*W);
            (VecF::load(data+i+3*W) * vs).store(data+i+3*W);
        }
        for (; i + W <= end; i += W) (VecF::load(data+i) * vs).store(data+i);
        for (; i < end; ++i) data[i] *= scalar;
    });
}

void axpy_inplace(float* data, float scalar, const float* other, int64_t n) {
    constexpr int W = VecF::width;
    VecF vs = VecF::broadcast(scalar);
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) data[i] += scalar * other[i];
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            VecF::fmadd(vs, VecF::load(other+i), VecF::load(data+i)).store(data+i);
            VecF::fmadd(vs, VecF::load(other+i+W), VecF::load(data+i+W)).store(data+i+W);
            VecF::fmadd(vs, VecF::load(other+i+2*W), VecF::load(data+i+2*W)).store(data+i+2*W);
            VecF::fmadd(vs, VecF::load(other+i+3*W), VecF::load(data+i+3*W)).store(data+i+3*W);
        }
        for (; i + W <= end; i += W) VecF::fmadd(vs, VecF::load(other+i), VecF::load(data+i)).store(data+i);
        for (; i < end; ++i) data[i] += scalar * other[i];
    });
}

void adam_step_loop(float* param, const float* grad, float* exp_avg, float* exp_avg_sq,
                    int64_t n, float lr, float beta1, float beta2, float eps, float weight_decay,
                    float bias_correction1, float bias_correction2, bool amsgrad, float* max_exp_avg_sq) {
    constexpr int W = VecF::width;
    VecF vb1 = VecF::broadcast(beta1), vb2 = VecF::broadcast(beta2);
    VecF v1b1 = VecF::broadcast(1.0f - beta1), v1b2 = VecF::broadcast(1.0f - beta2);
    VecF ve = VecF::broadcast(eps);
    float ss = lr / bias_correction1;
    VecF vss = VecF::broadcast(ss);
    VecF vbc2 = VecF::broadcast(1.0f / std::sqrt(bias_correction2));
    VecF vwd = VecF::broadcast(weight_decay);

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) {
            float g = grad[i], p = param[i];
            if (weight_decay != 0.0f) g += weight_decay * p;
            exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
            exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
            float dv;
            if (amsgrad && max_exp_avg_sq) { max_exp_avg_sq[i] = std::max(max_exp_avg_sq[i], exp_avg_sq[i]); dv = std::sqrt(max_exp_avg_sq[i] / bias_correction2) + eps; }
            else dv = std::sqrt(exp_avg_sq[i] / bias_correction2) + eps;
            param[i] = p - ss * exp_avg[i] / dv;
        }
        int64_t se = end - ((end - i) % W);
        for (; i < se; i += W) {
            VecF g = VecF::load(grad+i), p = VecF::load(param+i);
            if (weight_decay != 0.0f) g = g + vwd * p;
            VecF m = VecF::fmadd(vb1, VecF::load(exp_avg+i), v1b1 * g); m.store(exp_avg+i);
            VecF v = VecF::fmadd(vb2, VecF::load(exp_avg_sq+i), v1b2 * g * g); v.store(exp_avg_sq+i);
            VecF dn;
            if (amsgrad && max_exp_avg_sq) {
                float *mp = max_exp_avg_sq+i, *vp = exp_avg_sq+i;
                for (int j = 0; j < W; ++j) if (vp[j] > mp[j]) mp[j] = vp[j];
                dn = tuda::sqrt_vec(VecF::load(max_exp_avg_sq+i) * vbc2 * vbc2) + ve;
            } else dn = tuda::sqrt_vec(v * vbc2 * vbc2) + ve;
            (p - vss * m / dn).store(param+i);
        }
        for (; i < end; ++i) {
            float g = grad[i], p = param[i];
            if (weight_decay != 0.0f) g += weight_decay * p;
            exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
            exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
            float dv;
            if (amsgrad && max_exp_avg_sq) { max_exp_avg_sq[i] = std::max(max_exp_avg_sq[i], exp_avg_sq[i]); dv = std::sqrt(max_exp_avg_sq[i] / bias_correction2) + eps; }
            else dv = std::sqrt(exp_avg_sq[i] / bias_correction2) + eps;
            param[i] = p - ss * exp_avg[i] / dv;
        }
    });
}

void sgd_step_loop(float* param, const float* grad, float* momentum_buf,
                   int64_t n, float lr, float momentum, float dampening,
                   float weight_decay, bool nesterov) {
    constexpr int W = VecF::width;
    VecF vlr = VecF::broadcast(lr), vwd = VecF::broadcast(weight_decay);
    VecF vmom = VecF::broadcast(momentum), vdamp = VecF::broadcast(1.0f - dampening);

    if (momentum == 0.0f) {
        c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
            int64_t i = start;
            for (; i < end && (i % W != 0); ++i) { float g = grad[i]; if (weight_decay != 0.0f) g += weight_decay * param[i]; param[i] -= lr * g; }
            int64_t se = end - ((end - i) % W);
            for (; i < se; i += W) { VecF g = VecF::load(grad+i), p = VecF::load(param+i); if (weight_decay != 0.0f) g = g + vwd * p; (p - vlr * g).store(param+i); }
            for (; i < end; ++i) { float g = grad[i]; if (weight_decay != 0.0f) g += weight_decay * param[i]; param[i] -= lr * g; }
        });
    } else {
        c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
            int64_t i = start;
            for (; i < end && (i % W != 0); ++i) {
                float g = grad[i]; if (weight_decay != 0.0f) g += weight_decay * param[i];
                momentum_buf[i] = momentum * momentum_buf[i] + (1.0f - dampening) * g;
                param[i] -= nesterov ? lr * (g + momentum * momentum_buf[i]) : lr * momentum_buf[i];
            }
            int64_t se = end - ((end - i) % W);
            for (; i < se; i += W) {
                VecF g = VecF::load(grad+i), p = VecF::load(param+i);
                if (weight_decay != 0.0f) g = g + vwd * p;
                VecF buf = VecF::fmadd(vmom, VecF::load(momentum_buf+i), vdamp * g); buf.store(momentum_buf+i);
                if (nesterov) p = p - vlr * (g + vmom * buf); else p = p - vlr * buf;
                p.store(param+i);
            }
            for (; i < end; ++i) {
                float g = grad[i]; if (weight_decay != 0.0f) g += weight_decay * param[i];
                momentum_buf[i] = momentum * momentum_buf[i] + (1.0f - dampening) * g;
                param[i] -= nesterov ? lr * (g + momentum * momentum_buf[i]) : lr * momentum_buf[i];
            }
        });
    }
}

} // namespace hot
} // namespace native
} // namespace at
