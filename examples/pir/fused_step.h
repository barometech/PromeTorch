// ============================================================================
// fused_step.h — Fused forward+backward+Adam for PIR on Elbrus
// ============================================================================
// ZERO autograd. All ops on raw float*. Pre-allocated buffers.
// Expected: 3-5x speedup over autograd path.
// ============================================================================
#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declarations — these call into hot_loops.cpp NUMA pool
namespace at { namespace native { namespace hot {
void sgemm(int64_t M, int64_t K, int64_t N, float alpha,
           const float* A, int64_t lda, const float* B, int64_t ldb,
           float beta, float* C, int64_t ldc);
void sgemm_nt(int64_t M, int64_t K, int64_t N, float alpha,
              const float* A, int64_t lda, const float* B, int64_t ldb,
              float beta, float* C, int64_t ldc);
void sgemm_tn(int64_t M, int64_t K, int64_t N, float alpha,
              const float* A, int64_t lda, const float* B, int64_t ldb,
              float beta, float* C, int64_t ldc);
}}}

namespace fused {

// ============================================================================
// Element-wise ops (OMP parallelized)
// ============================================================================

static void sigmoid_fwd(const float* __restrict x, float* __restrict out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        out[i]   = 1.0f / (1.0f + std::exp(-x[i]));
        out[i+1] = 1.0f / (1.0f + std::exp(-x[i+1]));
        out[i+2] = 1.0f / (1.0f + std::exp(-x[i+2]));
        out[i+3] = 1.0f / (1.0f + std::exp(-x[i+3]));
        out[i+4] = 1.0f / (1.0f + std::exp(-x[i+4]));
        out[i+5] = 1.0f / (1.0f + std::exp(-x[i+5]));
    }
}

static void sigmoid_bwd(const float* __restrict grad, const float* __restrict sig_out, float* __restrict dx, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        dx[i]   = grad[i]   * sig_out[i]   * (1.0f - sig_out[i]);
        dx[i+1] = grad[i+1] * sig_out[i+1] * (1.0f - sig_out[i+1]);
        dx[i+2] = grad[i+2] * sig_out[i+2] * (1.0f - sig_out[i+2]);
        dx[i+3] = grad[i+3] * sig_out[i+3] * (1.0f - sig_out[i+3]);
        dx[i+4] = grad[i+4] * sig_out[i+4] * (1.0f - sig_out[i+4]);
        dx[i+5] = grad[i+5] * sig_out[i+5] * (1.0f - sig_out[i+5]);
    }
}

static void silu_fwd(const float* __restrict x, float* __restrict out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        float s0 = 1.0f/(1.0f+std::exp(-x[i]));   out[i]   = x[i]*s0;
        float s1 = 1.0f/(1.0f+std::exp(-x[i+1])); out[i+1] = x[i+1]*s1;
        float s2 = 1.0f/(1.0f+std::exp(-x[i+2])); out[i+2] = x[i+2]*s2;
        float s3 = 1.0f/(1.0f+std::exp(-x[i+3])); out[i+3] = x[i+3]*s3;
        float s4 = 1.0f/(1.0f+std::exp(-x[i+4])); out[i+4] = x[i+4]*s4;
        float s5 = 1.0f/(1.0f+std::exp(-x[i+5])); out[i+5] = x[i+5]*s5;
    }
}

static void silu_bwd(const float* __restrict grad, const float* __restrict x, float* __restrict dx, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        for (int64_t j = 0; j < 6; j++) {
            float s = 1.0f / (1.0f + std::exp(-x[i+j]));
            dx[i+j] = grad[i+j] * (s + x[i+j] * s * (1.0f - s));
        }
    }
}

static void mul_fwd(const float* __restrict a, const float* __restrict b, float* __restrict out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        out[i]=a[i]*b[i]; out[i+1]=a[i+1]*b[i+1]; out[i+2]=a[i+2]*b[i+2];
        out[i+3]=a[i+3]*b[i+3]; out[i+4]=a[i+4]*b[i+4]; out[i+5]=a[i+5]*b[i+5];
    }
}

static void mul_bwd(const float* __restrict grad, const float* __restrict a, const float* __restrict b,
                    float* __restrict da, float* __restrict db, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        da[i]=grad[i]*b[i]; db[i]=grad[i]*a[i];
        da[i+1]=grad[i+1]*b[i+1]; db[i+1]=grad[i+1]*a[i+1];
        da[i+2]=grad[i+2]*b[i+2]; db[i+2]=grad[i+2]*a[i+2];
        da[i+3]=grad[i+3]*b[i+3]; db[i+3]=grad[i+3]*a[i+3];
        da[i+4]=grad[i+4]*b[i+4]; db[i+4]=grad[i+4]*a[i+4];
        da[i+5]=grad[i+5]*b[i+5]; db[i+5]=grad[i+5]*a[i+5];
    }
}

static void add_fwd(const float* __restrict a, const float* __restrict b, float* __restrict out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        out[i]=a[i]+b[i]; out[i+1]=a[i+1]+b[i+1]; out[i+2]=a[i+2]+b[i+2];
        out[i+3]=a[i+3]+b[i+3]; out[i+4]=a[i+4]+b[i+4]; out[i+5]=a[i+5]+b[i+5];
    }
}

// RMSNorm: out = x * rsqrt(mean(x²) + eps) * weight
static void rmsnorm_fwd(const float* __restrict x, const float* __restrict weight, float* __restrict out,
                        float* __restrict rms_cache,
                        int64_t BT, int64_t D, float eps = 1e-6f) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < BT; i++) {
        const float* __restrict xi = x + i * D;
        float* __restrict oi = out + i * D;
        // 6-wide accumulation for VLIW (6 FPU channels)
        float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0;
        #pragma loop count(768)
        for (int64_t d = 0; d < D; d += 6) {
            s0 += xi[d]*xi[d]; s1 += xi[d+1]*xi[d+1]; s2 += xi[d+2]*xi[d+2];
            s3 += xi[d+3]*xi[d+3]; s4 += xi[d+4]*xi[d+4]; s5 += xi[d+5]*xi[d+5];
        }
        float inv_rms = 1.0f / std::sqrt((s0+s1+s2+s3+s4+s5) / D + eps);
        rms_cache[i] = inv_rms;
        #pragma loop count(768)
        for (int64_t d = 0; d < D; d += 6) {
            oi[d]=xi[d]*inv_rms*weight[d]; oi[d+1]=xi[d+1]*inv_rms*weight[d+1];
            oi[d+2]=xi[d+2]*inv_rms*weight[d+2]; oi[d+3]=xi[d+3]*inv_rms*weight[d+3];
            oi[d+4]=xi[d+4]*inv_rms*weight[d+4]; oi[d+5]=xi[d+5]*inv_rms*weight[d+5];
        }
    }
}

static void rmsnorm_bwd(const float* grad, const float* x, const float* weight,
                        const float* rms_cache, float* dx, float* dweight,
                        int64_t BT, int64_t D) {
    // dweight accumulation
    memset(dweight, 0, D * sizeof(float));

    #pragma omp parallel
    {
        // Thread-local dweight accumulator
        std::vector<float> dw_local(D, 0.0f);

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < BT; i++) {
            float inv_rms = rms_cache[i];
            const float* xi = x + i * D;
            const float* gi = grad + i * D;
            float* dxi = dx + i * D;

            // d(x * inv_rms * w) / dx = inv_rms * w + x * d(inv_rms)/dx * w
            // Simplified: dx = inv_rms * (grad * w - x * (dot(grad*w, x) / (D * rms²)))
            float dot = 0.0f;
            for (int64_t d = 0; d < D; d++)
                dot += gi[d] * weight[d] * xi[d];
            dot *= inv_rms * inv_rms / D;

            for (int64_t d = 0; d < D; d++) {
                dxi[d] = inv_rms * (gi[d] * weight[d] - xi[d] * dot);
                dw_local[d] += gi[d] * xi[d] * inv_rms;
            }
        }

        // Reduce thread-local dweight
        #pragma omp critical
        for (int64_t d = 0; d < D; d++)
            dweight[d] += dw_local[d];
    }
}

// Parallel scan: h[t] = gate[t]*h[t-1] + x[t]
static void parallel_scan_fwd(const float* gates, const float* x, float* out,
                               int64_t B, int64_t T, int64_t D) {
    int64_t BD = B * D;
    #pragma omp parallel for schedule(static)
    for (int64_t bd = 0; bd < BD; bd++) {
        int64_t b = bd / D, d = bd % D;
        float h = 0.0f;
        for (int64_t t = 0; t < T; t++) {
            int64_t idx = b * T * D + t * D + d;
            h = gates[idx] * h + x[idx];
            out[idx] = h;
        }
    }
}

static void parallel_scan_bwd(const float* grad, const float* gates,
                               const float* x, const float* out,
                               float* dx, float* dgates,
                               int64_t B, int64_t T, int64_t D) {
    int64_t BD = B * D;
    #pragma omp parallel for schedule(static)
    for (int64_t bd = 0; bd < BD; bd++) {
        int64_t b = bd / D, d = bd % D;
        float dh = 0.0f;
        for (int64_t t = T - 1; t >= 0; t--) {
            int64_t idx = b * T * D + t * D + d;
            dh += grad[idx];
            dx[idx] = dh;
            float h_prev = (t > 0) ? out[b * T * D + (t-1) * D + d] : 0.0f;
            dgates[idx] = dh * h_prev;
            dh *= gates[idx];
        }
    }
}

// ============================================================================
// Linear forward/backward on raw pointers (calls NUMA GEMM)
// ============================================================================

// Y = X @ W^T  (X:[M,K], W:[N,K] → Y:[M,N])
static void linear_fwd(const float* X, const float* W, float* Y,
                        int64_t M, int64_t K, int64_t N) {
    at::native::hot::sgemm_nt(M, K, N, 1.0f, X, K, W, K, 0.0f, Y, N);
}

// dX = dY @ W  (dY:[M,N], W:[N,K] → dX:[M,K])
static void linear_bwd_input(const float* dY, const float* W, float* dX,
                              int64_t M, int64_t K, int64_t N) {
    at::native::hot::sgemm(M, N, K, 1.0f, dY, N, W, K, 0.0f, dX, K);
}

// dW = dY^T @ X  (dY:[M,N], X:[M,K] → dW:[N,K])
static void linear_bwd_weight(const float* dY, const float* X, float* dW,
                               int64_t M, int64_t K, int64_t N) {
    at::native::hot::sgemm_tn(M, N, K, 1.0f, dY, N, X, K, 0.0f, dW, K);
}

// Accumulate: dst += src
static void accum(float* __restrict dst, const float* __restrict src, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        dst[i]+=src[i]; dst[i+1]+=src[i+1]; dst[i+2]+=src[i+2];
        dst[i+3]+=src[i+3]; dst[i+4]+=src[i+4]; dst[i+5]+=src[i+5];
    }
}

// ============================================================================
// Adam update on raw buffers
// ============================================================================
static void adam_update(float* param, const float* grad,
                        float* m, float* v,
                        int64_t n, int64_t step,
                        float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                        float eps = 1e-8f, float wd = 0.01f) {
    float bc1 = 1.0f - std::pow(beta1, step);
    float bc2 = 1.0f - std::pow(beta2, step);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        // Weight decay (AdamW style)
        param[i] -= lr * wd * param[i];

        // Adam
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// ============================================================================
// Softmax cross-entropy loss + backward (fused)
// ============================================================================
static float cross_entropy_fwd_bwd(const float* logits, const float* targets,
                                    float* dlogits,
                                    int64_t BT, int64_t V) {
    double total_loss = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:total_loss)
    for (int64_t i = 0; i < BT; i++) {
        const float* li = logits + i * V;
        float* di = dlogits + i * V;
        int target = (int)targets[i];

        // Stable softmax
        float max_val = li[0];
        for (int64_t v = 1; v < V; v++)
            if (li[v] > max_val) max_val = li[v];

        float sum_exp = 0.0f;
        for (int64_t v = 0; v < V; v++) {
            di[v] = std::exp(li[v] - max_val);
            sum_exp += di[v];
        }

        for (int64_t v = 0; v < V; v++)
            di[v] /= sum_exp;

        // FIX 4.1: log-sum-exp trick — more numerically stable than -log(softmax+eps)
        total_loss += (max_val - li[target] + std::log(sum_exp));

        // Gradient: softmax - one_hot
        di[target] -= 1.0f;

        // Normalize by batch
        float scale = 1.0f / BT;
        for (int64_t v = 0; v < V; v++)
            di[v] *= scale;
    }

    return (float)(total_loss / BT);
}

} // namespace fused
