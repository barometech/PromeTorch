#pragma once
// ============================================================================
// hot_loops.h — Non-inline inner loop declarations for LTO on Elbrus
// ============================================================================
// On E2K (Elbrus VLIW), inline functions in headers prevent LCC from
// performing cross-function VLIW scheduling. By moving hot inner loops
// into a .cpp file compiled once with -O3, LCC can:
//   1. Schedule VLIW bundles with full pipeline visibility
//   2. Share compiled code across all translation units (no duplication)
//   3. Apply LTO across the entire aten library
//
// On x86 (AVX2), these are still beneficial: reduces binary size and
// allows the linker to deduplicate previously-inline function copies.
// ============================================================================

#include <cstdint>

namespace at {
namespace native {
namespace hot {

// ============================================================================
// NUMA-aware GEMM for multi-chip Elbrus (4-chip E8C2: 1840 GFLOPS)
// ============================================================================
// On 4-chip E8C2, default EML gives 330 GFLOPS (cross-NUMA penalty).
// These functions split M rows across NUMA nodes, each computing locally.
// Enabled only when PT_USE_NUMA and PT_USE_EML_BLAS are both defined.
// Automatically selected by sgemm/sgemm_nt/sgemm_tn when matrix >= 256x256.

#ifdef PT_USE_NUMA
void sgemm_numa(int64_t M, int64_t K, int64_t N, float alpha,
                const float* A, int64_t lda, const float* B, int64_t ldb,
                float beta, float* C, int64_t ldc);
void sgemm_nt_numa(int64_t M, int64_t K, int64_t N, float alpha,
                   const float* A, int64_t lda, const float* B, int64_t ldb,
                   float beta, float* C, int64_t ldc);
#endif

// ============================================================================
// BLAS wrappers (delegate to TudaBLAS sgemm/sgemv/sdot)
// ============================================================================

// C = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
void sgemm(int64_t M, int64_t K, int64_t N,
           float alpha, const float* A, int64_t lda,
           const float* B, int64_t ldb,
           float beta, float* C, int64_t ldc);

// C = alpha * A[M,K] @ B^T[N,K] + beta * C[M,N]  (B stored row-major [N,K])
void sgemm_nt(int64_t M, int64_t K, int64_t N,
              float alpha, const float* A, int64_t lda,
              const float* B, int64_t ldb,
              float beta, float* C, int64_t ldc);

// C[K,N] = alpha * A^T[K,M] @ B[M,N] + beta * C[K,N]
// A is stored row-major [M,K], transposed logically.
// Used in backward: grad_weight = grad^T @ input
void sgemm_tn(int64_t M, int64_t K, int64_t N,
              float alpha, const float* A, int64_t lda,
              const float* B, int64_t ldb,
              float beta, float* C, int64_t ldc);

// y = alpha * A[M,N] @ x + beta * y
void sgemv(int64_t M, int64_t N,
           float alpha, const float* A, int64_t lda,
           const float* x,
           float beta, float* y);

// result = dot(a, b)
float sdot(int64_t n, const float* a, const float* b);

// ============================================================================
// Element-wise loops (float32 SIMD)
// ============================================================================

// out[i] = a[i] + alpha * b[i],  i in [0, n)
void add_loop(const float* a, const float* b, float* out, int64_t n, float alpha);

// out[i] = a[i] - b[i],  i in [0, n)
void sub_loop(const float* a, const float* b, float* out, int64_t n);

// out[i] = a[i] * b[i],  i in [0, n)
void mul_loop(const float* a, const float* b, float* out, int64_t n);

// out[i] = a[i] / b[i],  i in [0, n)
void div_loop(const float* a, const float* b, float* out, int64_t n);

// out[i] = a[i] + alpha * b[i],  broadcast b[j] over rows  (bias add pattern)
// a is [outer, inner], b is [inner], out is [outer, inner]
void add_broadcast_loop(const float* a, const float* b, float* out,
                        int64_t outer, int64_t inner, float alpha);

// ============================================================================
// Unary loops (float32 SIMD)
// ============================================================================

void neg_loop(const float* in, float* out, int64_t n);
void abs_loop(const float* in, float* out, int64_t n);
void sqrt_loop(const float* in, float* out, int64_t n);
void rsqrt_loop(const float* in, float* out, int64_t n);
void square_loop(const float* in, float* out, int64_t n);
void exp_loop(const float* in, float* out, int64_t n);
void log_loop(const float* in, float* out, int64_t n);
void log2_loop(const float* in, float* out, int64_t n);
void log10_loop(const float* in, float* out, int64_t n);
void sin_loop(const float* in, float* out, int64_t n);
void cos_loop(const float* in, float* out, int64_t n);
void tan_loop(const float* in, float* out, int64_t n);
void tanh_loop(const float* in, float* out, int64_t n);
void sigmoid_loop(const float* in, float* out, int64_t n);
void relu_loop(const float* in, float* out, int64_t n);
void reciprocal_loop(const float* in, float* out, int64_t n);
void ceil_loop(const float* in, float* out, int64_t n);
void floor_loop(const float* in, float* out, int64_t n);
void round_loop(const float* in, float* out, int64_t n);
void sign_loop(const float* in, float* out, int64_t n);

// ============================================================================
// Reduction loops (float32 SIMD)
// ============================================================================

// Sum all elements
float sum_loop(const float* data, int64_t n);

// Sum along a dimension: accumulate reduce_size rows of inner_size elements
void sum_dim_loop(const float* in, float* out,
                  int64_t outer_size, int64_t reduce_size, int64_t inner_size);

// ============================================================================
// In-place scalar loops (optimizer hot paths)
// ============================================================================

// data[i] *= scalar
void mul_scalar_inplace(float* data, float scalar, int64_t n);

// data[i] += scalar * other[i]
void axpy_inplace(float* data, float scalar, const float* other, int64_t n);

// Adam/AdamW inner loop: single pass over all parameter elements
// p -= lr * m_hat / (sqrt(v_hat) + eps)
// where m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
void adam_step_loop(float* param, const float* grad,
                    float* exp_avg, float* exp_avg_sq,
                    int64_t n, float lr, float beta1, float beta2,
                    float eps, float weight_decay,
                    float bias_correction1, float bias_correction2,
                    bool amsgrad, float* max_exp_avg_sq);

// SGD inner loop with momentum
void sgd_step_loop(float* param, const float* grad, float* momentum_buf,
                   int64_t n, float lr, float momentum, float dampening,
                   float weight_decay, bool nesterov);

// ============================================================================
// Fused multi-parameter optimizer steps (E2K cache-friendly)
// ============================================================================
// On Elbrus E2K, per-parameter function calls cause instruction cache misses
// because parameters are scattered in memory. These fused versions collect ALL
// parameters into a single array of packs and process them in ONE function call.
// Benefits:
//   1. Single function entry — no per-parameter call overhead
//   2. Bias correction computed once, not per-parameter
//   3. LCC can schedule the entire outer+inner loop as one VLIW region
//   4. Better branch predictor utilization (one loop structure, not N)

struct AdamParamPack {
    float* param;
    const float* grad;
    float* exp_avg;
    float* exp_avg_sq;
    int64_t numel;
};

// Fused multi-parameter Adam: process ALL parameters in one function call
// Reduces per-parameter function call overhead and improves instruction cache
void fused_adam_multi(AdamParamPack* params, int num_params,
                     float lr, float beta1, float beta2, float eps,
                     float weight_decay, int step,
                     bool amsgrad = false, float** max_exp_avg_sq = nullptr);

struct SGDParamPack {
    float* param;
    const float* grad;
    float* momentum_buf;  // nullptr if no momentum
    int64_t numel;
};

// Fused multi-parameter SGD: process ALL parameters in one function call
void fused_sgd_multi(SGDParamPack* params, int num_params,
                    float lr, float momentum, float dampening,
                    float weight_decay, bool nesterov);

// ============================================================================
// Fused kernels (E2K-optimized: large loop bodies, no branches)
// ============================================================================

// ============================================================================
// Fused backward helpers (zero-intermediate backward passes)
// ============================================================================

// Apply relu mask in-place: out[i] = (mask[i] > 0) ? grad[i] : 0
// mask is the post-relu output (positive = active)
void relu_mask_mul(const float* grad, const float* mask, float* out, int64_t n);

// Column sum: out[j] = sum_i(data[i*cols + j]) for j in [0, cols)
// Used for grad_bias = grad.sum(dim=0)
void col_sum(const float* data, float* out, int64_t rows, int64_t cols);

// In-place add: dst[i] += src[i]  (no allocation)
void add_inplace(float* dst, const float* src, int64_t n);

// Fused bias_add + relu: out[i*N+j] = max(0, out[i*N+j] + bias[j])
// Single pass avoids extra memory traffic (write-back + re-read)
void bias_relu_fused(float* out, const float* bias, int64_t M, int64_t N);

// Fused bias_add + gelu: out[i*N+j] = gelu(out[i*N+j] + bias[j])
void bias_gelu_fused(float* out, const float* bias, int64_t M, int64_t N);

// Fused cross-entropy: softmax + log + nll in one pass
// Avoids 3 separate passes over logits (max, exp+sum, log+gather)
// loss = -log(softmax(logits)[target]) averaged over batch
// grad[i,j] = softmax[i,j] - (j == target[i]) / batch
void cross_entropy_fused(const float* logits, const int64_t* targets,
                         float* loss, float* grad,
                         int64_t batch, int64_t classes);

// Fused softmax: max + exp + sum + normalize in one pass per row
// Avoids 3 separate passes over data
void softmax_fused(const float* in, float* out, int64_t rows, int64_t cols);

// Fused residual + layer_norm: out = LayerNorm(x + residual)
// Avoids materializing the intermediate x + residual tensor
void residual_layernorm_fused(const float* x, const float* residual,
                              const float* gamma, const float* beta_param,
                              float* out, int64_t rows, int64_t cols, float eps);

// ============================================================================
// Fused element-wise AVX2 kernels (beat PyTorch on x86)
// ============================================================================
// These fuse multiple ops into single passes, eliminating tensor allocation
// overhead that costs 1.5x vs PyTorch on element-wise ops.

// Fused activation+scale:

// out[i*features+j] = scale * relu(x[i*features+j] + bias[j])
// Fuses bias_add + relu + scale into one pass per row
void fused_bias_relu_scale(const float* x, const float* bias, float scale,
                           float* out, int64_t batch, int64_t features);

// out[i] = alpha * sigmoid(x[i]) * x[i]  (SiLU/Swish with scale)
// Fuses sigmoid + mul + scale into one pass
void fused_silu_scale(const float* x, float alpha, float* out, int64_t n);

// x[i] = x[i] * (1 - mask[i]) * scale  (dropout, in-place)
// mask[i] = 1 means DROP. Fuses mask + mul + scale into one pass.
void fused_dropout_scale(float* x, const uint8_t* mask, float scale, int64_t n);

// Fused backward chains:

// out[i] = (input[i] > 0 ? grad[i] : 0) * scale
// Combines relu_backward + scale into one pass
void fused_relu_backward_scale(const float* grad, const float* input,
                                float scale, float* out, int64_t n);

// Fused optimizer steps (in-place, AVX2, single-pass):

// SGD with momentum: single pass updates buf and param together
//   buf = momentum * buf + (1 - dampening) * grad
//   param -= lr * buf
void fused_sgd_momentum_avx2(float* param, const float* grad, float* buf,
                              int64_t n, float lr, float momentum, float dampening);

// Adam: single pass update (m, v, param all updated in one loop)
//   m = beta1 * m + (1 - beta1) * grad
//   v = beta2 * v + (1 - beta2) * grad^2
//   param -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
// bc1 = 1 - beta1^t, bc2 = 1 - beta2^t (precomputed by caller)
void fused_adam_avx2(float* param, const float* grad,
                     float* m, float* v, int64_t n,
                     float lr, float beta1, float beta2, float eps,
                     float weight_decay, float bc1, float bc2);

// Fused GELU: out[i] = x[i] * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Single pass, no intermediate tensors
void fused_gelu(const float* x, float* out, int64_t n);

// Fused bias + GELU: out[i*N+j] = gelu(data[i*N+j] + bias[j])
// Fills the declared-but-unimplemented bias_gelu_fused from above
// (implementation shared, declaration already present)

// ============================================================================
// INFERENCE FUSIONS — CPU decode hot path (forward_decode_cpu)
// ============================================================================

// Fusion 1: GEMV + residual add (accumulate into x_next instead of h_buf + add)
// Eliminates: intermediate h_buf write + separate x_next = x + h AVX2 pass
// y[i] = x_residual[i] + dot(W[i,:], src)  for i in [0, N)
// Instead of:  gemv(W, src, h_buf);  x_next = x + h_buf
// We do:       gemv_residual_add(W, src, x, x_next)
void gemv_residual_add(const float* W, int64_t N, int64_t K,
                       const float* src, const float* x_residual,
                       float* x_out);

// Fusion 2: Final RMSNorm + output GEMV + argmax (skip full logits store)
// Eliminates: logits_buf write + separate argmax pass over vocab_size elements
// Returns argmax token id and writes logits only if logits_out != nullptr
// x is [H], norm_w is [H], output_weight is [V, H]
// This avoids writing V floats and re-reading them for argmax
int32_t fused_rmsnorm_gemv_argmax(const float* x, const float* norm_w,
                                   float eps, bool add_one, int64_t H,
                                   const float* output_weight, int64_t V,
                                   float* logits_out);

// Fusion 3: Softmax for attention already done inline in forward_decode_cpu
// (QK dot + scale + softmax + V weighted sum is one function per head)
// softmax_fused and residual_layernorm_fused declared above — implementations added to .cpp

// ============================================================================
// TRAINING FUSIONS — eliminate redundant passes in training loop
// ============================================================================

// Fusion 4: AdamW single-pass step (decoupled weight decay + Adam in one loop)
// Current code: param *= (1 - lr*wd); then adam_step_loop (two passes over param)
// Fused: single pass does both weight decay and Adam update per element
void fused_adamw_step(float* param, const float* grad,
                      float* exp_avg, float* exp_avg_sq,
                      int64_t n, float lr, float beta1, float beta2,
                      float eps, float weight_decay,
                      float bc1, float bc2);

// Fusion 5: Fused multi-parameter AdamW (like fused_adam_multi but with
// decoupled weight decay built in — no separate mul_ pass)
struct AdamWParamPack {
    float* param;
    const float* grad;
    float* exp_avg;
    float* exp_avg_sq;
    int64_t numel;
};

void fused_adamw_multi(AdamWParamPack* params, int num_params,
                       float lr, float beta1, float beta2, float eps,
                       float weight_decay, int step);

// Fusion 6: zero_grad + set_to_none — memset all grad buffers in one call
// Eliminates per-parameter function call overhead and improves cache locality
// for the common pattern: optimizer.zero_grad() before backward()
struct GradBufPack {
    float* grad_data;
    int64_t numel;
};
void fused_zero_grad_multi(GradBufPack* bufs, int num_bufs);

// Fusion 7: Fused RoPE (precomputed sin/cos table, AVX2)
// Current code computes pow() + sin() + cos() per dimension per token
// Precompute table once, then apply with FMA
//
// LLaMA convention (NORM): pairs (2d, 2d+1) — used by llama / mistral arches.
// NeoX convention (NEOX):  pairs (d, d + head_dim/2) — used by qwen2 / qwen3 /
// gemma / phi3 / stablelm / falcon. Picked per architecture by ModelConfig.
void rope_apply_fused(float* q, float* k,
                      const float* cos_table, const float* sin_table,
                      int64_t n_heads, int64_t n_kv_heads, int64_t head_dim);

void rope_apply_fused_neox(float* q, float* k,
                           const float* cos_table, const float* sin_table,
                           int64_t n_heads, int64_t n_kv_heads, int64_t head_dim);

// Precompute RoPE table for a given position. `scale` defaults to 1.0
// (no scaling). Pass >1 for linear-scaled RoPE, e.g. Gemma3 has scale=8.0
// from `<arch>.rope.scaling.factor` in GGUF metadata.
//
// `rope_factors` is an optional per-frequency divisor (Phi-3 LongRoPE):
// `inv_freq[d] = (1/base^(2d/dim)) / factor[d]`. nullptr → no factor.
//
// `attn_factor` multiplies cos и sin (yarn `mscale`) — Phi-3.5 mini имеет 1.19
// (sqrt(1+log(scale)/log(orig_ctx))). Default 1.0 = без эффекта.
void rope_precompute(float* cos_out, float* sin_out,
                     int64_t pos, int64_t head_dim, float freq_base,
                     float scale = 1.0f,
                     const float* rope_factors = nullptr,
                     float attn_factor = 1.0f);

// YaRN RoPE precompute (DeepSeek-V2/V3, GigaChat3 deepseek2).
// See implementation in hot_loops.cpp for full formula and parameter docs.
void rope_precompute_yarn(float* cos_out, float* sin_out,
                          int64_t pos, int64_t head_dim,
                          float base, float factor,
                          float beta_fast, float beta_slow,
                          int64_t orig_ctx,
                          float log_multiplier);

} // namespace hot
} // namespace native
} // namespace at
