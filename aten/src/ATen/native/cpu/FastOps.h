#pragma once
// ============================================================================
// FastOps.h -- Zero-overhead float32 fast paths for Elbrus E8C2
// ============================================================================
// These functions skip ALL runtime dispatch:
//   - No PT_DISPATCH_ALL_TYPES / PT_DISPATCH_FLOATING_TYPES switch
//   - No dtype check (caller guarantees float32)
//   - No contiguous check (caller guarantees contiguous)
//   - No device check (caller guarantees CPU)
//
// On Elbrus VLIW, eliminating dispatch overhead matters because:
//   1. Branch mispredictions stall the wide VLIW pipeline
//   2. Switch statements prevent LCC from scheduling VLIW bundles
//   3. For small tensors (batch=32, hidden=128), dispatch = 50% of op time
//
// All functions are inline so LCC can fold them into the caller's VLIW schedule.
// On x86, the compiler can similarly inline and eliminate call overhead.
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"

namespace at {
namespace native {
namespace fast {

// ============================================================================
// Matrix multiply: C = A[M,K] @ B[K,N]
// Caller guarantees: float32, contiguous, 2D, compatible shapes.
// ============================================================================
inline Tensor mm_f32(const Tensor& a, const Tensor& b) {
    const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    Tensor c = at::empty({M, N});
    hot::sgemm(M, K, N, 1.0f,
               a.data_ptr<float>(), K,
               b.data_ptr<float>(), N,
               0.0f, c.mutable_data_ptr<float>(), N);
    return c;
}

// ============================================================================
// Matrix multiply with B transposed: C = A[M,K] @ B^T, B stored as [N,K]
// This is the Linear layer pattern: x[M,K] @ W^T where W is [N,K].
// ============================================================================
inline Tensor mm_nt_f32(const Tensor& a, const Tensor& b) {
    const int64_t M = a.size(0), K = a.size(1), N = b.size(0);
    Tensor c = at::empty({M, N});
    hot::sgemm_nt(M, K, N, 1.0f,
                  a.data_ptr<float>(), K,
                  b.data_ptr<float>(), K,
                  0.0f, c.mutable_data_ptr<float>(), N);
    return c;
}

// ============================================================================
// Bias addition: out[i,j] += bias[j] for all rows
// Operates in-place on the output buffer.
// ============================================================================
inline void add_bias_f32(float* out, const float* bias, int64_t M, int64_t N) {
    hot::add_broadcast_loop(out, bias, out, M, N, 1.0f);
}

// ============================================================================
// ReLU in-place: data[i] = max(data[i], 0)
// ============================================================================
inline void relu_inplace_f32(float* data, int64_t n) {
    for (int64_t i = 0; i < n; ++i)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

// ============================================================================
// Fused linear + bias: out = x[M,K] @ W^T[K,N] + bias[N]
// ONE tensor allocation, ONE sgemm call, ONE bias pass.
// ============================================================================
inline Tensor fused_linear_f32(const Tensor& x, const Tensor& W, const Tensor& bias) {
    const int64_t M = x.size(0), K = x.size(1), N = W.size(0);
    Tensor out = at::empty({M, N});
    float* out_data = out.mutable_data_ptr<float>();

    hot::sgemm_nt(M, K, N, 1.0f,
                  x.data_ptr<float>(), K,
                  W.data_ptr<float>(), K,
                  0.0f, out_data, N);

    const float* b = bias.data_ptr<float>();
    hot::add_broadcast_loop(out_data, b, out_data, M, N, 1.0f);
    return out;
}

// ============================================================================
// Fused linear + bias + relu: out = relu(x[M,K] @ W^T[K,N] + bias[N])
// ONE tensor allocation, ZERO intermediate tensors.
// Bias and relu fused into a single pass over the output.
// ============================================================================
inline Tensor fused_linear_relu_f32(const Tensor& x, const Tensor& W, const Tensor& bias) {
    const int64_t M = x.size(0), K = x.size(1), N = W.size(0);
    Tensor out = at::empty({M, N});
    float* out_data = out.mutable_data_ptr<float>();

    hot::sgemm_nt(M, K, N, 1.0f,
                  x.data_ptr<float>(), K,
                  W.data_ptr<float>(), K,
                  0.0f, out_data, N);

    // Fused bias + relu in single pass (better cache locality than two passes)
    const float* b = bias.data_ptr<float>();
    for (int64_t i = 0; i < M; ++i) {
        float* row = out_data + i * N;
        for (int64_t j = 0; j < N; ++j) {
            float v = row[j] + b[j];
            row[j] = v > 0.0f ? v : 0.0f;
        }
    }
    return out;
}

// ============================================================================
// Fused linear (no bias): out = x[M,K] @ W^T[K,N]
// For layers without bias — skip the bias loop entirely.
// ============================================================================
inline Tensor fused_linear_nobias_f32(const Tensor& x, const Tensor& W) {
    return mm_nt_f32(x, W);
}

// ============================================================================
// Fused linear (no bias) + relu: out = relu(x[M,K] @ W^T[K,N])
// ============================================================================
inline Tensor fused_linear_relu_nobias_f32(const Tensor& x, const Tensor& W) {
    const int64_t M = x.size(0), K = x.size(1), N = W.size(0);
    Tensor out = at::empty({M, N});
    float* out_data = out.mutable_data_ptr<float>();

    hot::sgemm_nt(M, K, N, 1.0f,
                  x.data_ptr<float>(), K,
                  W.data_ptr<float>(), K,
                  0.0f, out_data, N);

    relu_inplace_f32(out_data, M * N);
    return out;
}

// ============================================================================
// Fast element-wise ops (skip contiguous check + dtype dispatch)
// ============================================================================

// out = a + b (same shape, contiguous, float32)
inline Tensor add_f32(const Tensor& a, const Tensor& b) {
    Tensor out = at::empty(a.sizes());
    hot::add_loop(a.data_ptr<float>(), b.data_ptr<float>(),
                  out.mutable_data_ptr<float>(), a.numel(), 1.0f);
    return out;
}

// out = a - b
inline Tensor sub_f32(const Tensor& a, const Tensor& b) {
    Tensor out = at::empty(a.sizes());
    hot::sub_loop(a.data_ptr<float>(), b.data_ptr<float>(),
                  out.mutable_data_ptr<float>(), a.numel());
    return out;
}

// out = a * b (element-wise)
inline Tensor mul_f32(const Tensor& a, const Tensor& b) {
    Tensor out = at::empty(a.sizes());
    hot::mul_loop(a.data_ptr<float>(), b.data_ptr<float>(),
                  out.mutable_data_ptr<float>(), a.numel());
    return out;
}

// out = relu(a)
inline Tensor relu_f32(const Tensor& a) {
    Tensor out = at::empty(a.sizes());
    hot::relu_loop(a.data_ptr<float>(), out.mutable_data_ptr<float>(), a.numel());
    return out;
}

// out = neg(a)
inline Tensor neg_f32(const Tensor& a) {
    Tensor out = at::empty(a.sizes());
    hot::neg_loop(a.data_ptr<float>(), out.mutable_data_ptr<float>(), a.numel());
    return out;
}

// out = exp(a)
inline Tensor exp_f32(const Tensor& a) {
    Tensor out = at::empty(a.sizes());
    hot::exp_loop(a.data_ptr<float>(), out.mutable_data_ptr<float>(), a.numel());
    return out;
}

// out = log(a)
inline Tensor log_f32(const Tensor& a) {
    Tensor out = at::empty(a.sizes());
    hot::log_loop(a.data_ptr<float>(), out.mutable_data_ptr<float>(), a.numel());
    return out;
}

// out = tanh(a)
inline Tensor tanh_f32(const Tensor& a) {
    Tensor out = at::empty(a.sizes());
    hot::tanh_loop(a.data_ptr<float>(), out.mutable_data_ptr<float>(), a.numel());
    return out;
}

// out = sigmoid(a)
inline Tensor sigmoid_f32(const Tensor& a) {
    Tensor out = at::empty(a.sizes());
    hot::sigmoid_loop(a.data_ptr<float>(), out.mutable_data_ptr<float>(), a.numel());
    return out;
}

// scalar = sum(a)
inline float sum_f32(const Tensor& a) {
    return hot::sum_loop(a.data_ptr<float>(), a.numel());
}

} // namespace fast
} // namespace native
} // namespace at
