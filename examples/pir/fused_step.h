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

static void sigmoid_fwd(const float* x, float* out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++)
        out[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

static void sigmoid_bwd(const float* grad, const float* sig_out, float* dx, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++)
        dx[i] = grad[i] * sig_out[i] * (1.0f - sig_out[i]);
}

static void silu_fwd(const float* x, float* out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + std::exp(-x[i]));
        out[i] = x[i] * s;
    }
}

static void silu_bwd(const float* grad, const float* x, float* dx, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + std::exp(-x[i]));
        dx[i] = grad[i] * (s + x[i] * s * (1.0f - s));
    }
}

static void mul_fwd(const float* a, const float* b, float* out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++)
        out[i] = a[i] * b[i];
}

static void mul_bwd(const float* grad, const float* a, const float* b,
                    float* da, float* db, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        da[i] = grad[i] * b[i];
        db[i] = grad[i] * a[i];
    }
}

static void add_fwd(const float* a, const float* b, float* out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

// RMSNorm: out = x * rsqrt(mean(x²) + eps) * weight
static void rmsnorm_fwd(const float* x, const float* weight, float* out,
                        float* rms_cache, // [B*T] — save 1/rms for backward
                        int64_t BT, int64_t D, float eps = 1e-6f) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < BT; i++) {
        float sum_sq = 0.0f;
        const float* xi = x + i * D;
        for (int64_t d = 0; d < D; d++)
            sum_sq += xi[d] * xi[d];
        float inv_rms = 1.0f / std::sqrt(sum_sq / D + eps);
        rms_cache[i] = inv_rms;
        float* oi = out + i * D;
        for (int64_t d = 0; d < D; d++)
            oi[d] = xi[d] * inv_rms * weight[d];
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
static void accum(float* dst, const float* src, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++)
        dst[i] += src[i];
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

        total_loss += -std::log(di[target] + 1e-10f);

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
