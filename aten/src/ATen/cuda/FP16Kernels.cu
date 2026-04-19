// ============================================================================
// FP16 CUDA Kernels for PromeTorch (AMP support)
// ============================================================================
// Element-wise, activation, softmax, norm kernels operating on __half storage
// with FP32 accumulation for numerical stability.
//
// Design notes:
//   - Reads/writes are __half to save bandwidth & Tensor-Core-friendly layout.
//   - Arithmetic is done in float (via __half2float / __float2half) to avoid
//     underflow/overflow, and because not every SM has cheap fp16 intrinsics
//     for transcendentals.
//   - Reductions (softmax, norms) accumulate in float — this is mandatory for
//     correctness: summing 4096 Half values in Half saturates fast.
//   - Also provides a device-side inf/nan check (launch_check_inf_nan) used by
//     torch::amp::GradScaler on CUDA tensors.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

#include "aten/src/ATen/cuda/CUDAOps.h"

namespace at {
namespace cuda {

// ============================================================================
// Kernel Configuration (kept consistent with CUDAKernels.cu)
// ============================================================================

namespace {
constexpr int FP16_BLOCK_SIZE = 256;
constexpr int FP16_MAX_GRID   = 65535;

inline int fp16_num_blocks(int64_t n) {
    int64_t blk = (n + FP16_BLOCK_SIZE - 1) / FP16_BLOCK_SIZE;
    if (blk > FP16_MAX_GRID) blk = FP16_MAX_GRID;
    if (blk < 1) blk = 1;
    return static_cast<int>(blk);
}
} // namespace

// ============================================================================
// Element-wise Binary (FP16)
// ============================================================================

__global__ void add_fp16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float av = __half2float(a[idx]);
        float bv = __half2float(b[idx]);
        out[idx] = __float2half(av + bv);
    }
}

__global__ void sub_fp16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float av = __half2float(a[idx]);
        float bv = __half2float(b[idx]);
        out[idx] = __float2half(av - bv);
    }
}

__global__ void mul_fp16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float av = __half2float(a[idx]);
        float bv = __half2float(b[idx]);
        out[idx] = __float2half(av * bv);
    }
}

__global__ void div_fp16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float av = __half2float(a[idx]);
        float bv = __half2float(b[idx]);
        out[idx] = __float2half(av / bv);
    }
}

// Broadcast add: [outer, inner] + [inner]  (typical bias addition)
__global__ void add_broadcast_fp16_kernel(
    const __half* a, const __half* b, __half* out,
    int64_t outer_size, int64_t inner_size)
{
    int64_t total = outer_size * inner_size;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t inner_idx = idx % inner_size;
        float av = __half2float(a[idx]);
        float bv = __half2float(b[inner_idx]);
        out[idx] = __float2half(av + bv);
    }
}

void launch_add_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t stream) {
    add_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}
void launch_sub_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t stream) {
    sub_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}
void launch_mul_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t stream) {
    mul_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}
void launch_div_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t stream) {
    div_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}
void launch_add_broadcast_fp16(
    const __half* a, const __half* b, __half* out,
    int64_t outer_size, int64_t inner_size, cudaStream_t stream)
{
    int64_t total = outer_size * inner_size;
    add_broadcast_fp16_kernel<<<fp16_num_blocks(total), FP16_BLOCK_SIZE, 0, stream>>>(
        a, b, out, outer_size, inner_size);
}

// ============================================================================
// Activations (FP16)
// ============================================================================

__global__ void relu_fp16_kernel(const __half* input, __half* output, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(input[idx]);
        output[idx] = __float2half(v > 0.0f ? v : 0.0f);
    }
}

__global__ void sigmoid_fp16_kernel(const __half* input, __half* output, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(input[idx]);
        output[idx] = __float2half(1.0f / (1.0f + expf(-v)));
    }
}

__global__ void tanh_fp16_kernel(const __half* input, __half* output, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(input[idx]);
        output[idx] = __float2half(tanhf(v));
    }
}

// GELU (erf approximation): 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_fp16_kernel(const __half* input, __half* output, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(input[idx]);
        float out = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        output[idx] = __float2half(out);
    }
}

// SiLU (swish): x * sigmoid(x)
__global__ void silu_fp16_kernel(const __half* input, __half* output, int64_t n) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(input[idx]);
        float s = 1.0f / (1.0f + expf(-v));
        output[idx] = __float2half(v * s);
    }
}

void launch_relu_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream) {
    relu_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(input, output, n);
}
void launch_sigmoid_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream) {
    sigmoid_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(input, output, n);
}
void launch_tanh_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream) {
    tanh_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(input, output, n);
}
void launch_gelu_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream) {
    gelu_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(input, output, n);
}
void launch_silu_fp16(const __half* input, __half* output, int64_t n, cudaStream_t stream) {
    silu_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

// ============================================================================
// Softmax (FP16 storage, FP32 accumulation — fused max+exp+sum+div)
// ============================================================================
// One block per outer_idx. Each thread handles a slice of inner.
// When inner_size == 1 (the usual softmax-over-last-dim case), the block
// reduces across dim_size in shared memory. When inner_size > 1 we fall back
// to per-(outer,inner) serial reduction.

__global__ void softmax_fp16_kernel(
    const __half* input, __half* output,
    int64_t outer_size, int64_t dim_size, int64_t inner_size)
{
    int64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    // Each thread takes a subset of inner indices
    for (int64_t inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
        // pass 1: max
        float max_val = -FLT_MAX;
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
            float v = __half2float(input[idx]);
            if (v > max_val) max_val = v;
        }
        // pass 2: exp + sum
        float sum = 0.0f;
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
            float v = __half2float(input[idx]);
            float e = expf(v - max_val);
            // stash FP32 exp into output (as fp16 — acceptable precision loss here;
            // the subsequent normalize reads it back anyway). We can recompute
            // to preserve precision: prefer storing & re-reading half, since the
            // intermediate tensor is FP16-typed.
            output[idx] = __float2half(e);
            sum += e;
        }
        // pass 3: normalize
        float inv_sum = 1.0f / sum;
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
            float e = __half2float(output[idx]);
            output[idx] = __float2half(e * inv_sum);
        }
    }
}

void launch_softmax_fp16(
    const __half* input, __half* output,
    int64_t outer_size, int64_t dim_size, int64_t inner_size,
    cudaStream_t stream)
{
    int threads = FP16_BLOCK_SIZE;
    if ((int64_t)threads > inner_size && inner_size > 0) {
        // shrink block if inner is small; still need at least 1
        threads = 1;
        while ((int64_t)threads < inner_size && threads < FP16_BLOCK_SIZE) threads <<= 1;
    }
    int64_t blocks = outer_size;
    if (blocks > FP16_MAX_GRID) blocks = FP16_MAX_GRID;
    if (blocks < 1) blocks = 1;
    softmax_fp16_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, output, outer_size, dim_size, inner_size);
}

// ============================================================================
// LayerNorm (FP16 storage, FP32 accumulation)
// ============================================================================
// Input shape: [rows, hidden]. Each block handles one row.

__global__ void layernorm_fp16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ gamma,  // may be nullptr
    const __half* __restrict__ beta,   // may be nullptr
    __half* __restrict__ output,
    int hidden, float eps)
{
    int row = blockIdx.x;
    const __half* x = input + (int64_t)row * hidden;
    __half* y       = output + (int64_t)row * hidden;

    extern __shared__ float smem_ln[];  // 2 * blockDim.x floats: sum, sum_sq

    // pass 1: mean & variance
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __half2float(x[i]);
        local_sum   += v;
        local_sqsum += v * v;
    }
    smem_ln[threadIdx.x]              = local_sum;
    smem_ln[blockDim.x + threadIdx.x] = local_sqsum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem_ln[threadIdx.x]              += smem_ln[threadIdx.x + s];
            smem_ln[blockDim.x + threadIdx.x] += smem_ln[blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean     = smem_ln[0] / (float)hidden;
    float mean_sq  = smem_ln[blockDim.x] / (float)hidden;
    float var      = mean_sq - mean * mean;
    float inv_std  = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v  = __half2float(x[i]);
        float g  = gamma ? __half2float(gamma[i]) : 1.0f;
        float bb = beta  ? __half2float(beta[i])  : 0.0f;
        float r  = (v - mean) * inv_std * g + bb;
        y[i] = __float2half(r);
    }
}

void launch_layernorm_fp16(
    const __half* input,
    const __half* gamma,  // may be nullptr
    const __half* beta,   // may be nullptr
    __half* output,
    int rows, int hidden, float eps,
    cudaStream_t stream)
{
    int block_size = 256;
    if (hidden > 512)  block_size = 512;
    if (hidden > 1024) block_size = 1024;
    if (block_size > hidden) block_size = hidden;
    // round up to pow2 for reduction
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;
    if (block_size > 1024) block_size = 1024;

    int shared_mem = 2 * block_size * sizeof(float);
    layernorm_fp16_kernel<<<rows, block_size, shared_mem, stream>>>(
        input, gamma, beta, output, hidden, eps);
}

// ============================================================================
// RMSNorm (FP16 storage, FP32 accumulation)
// ============================================================================

__global__ void rmsnorm_fp16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,  // may be nullptr
    __half* __restrict__ output,
    int hidden, float eps, bool add_one)
{
    int row = blockIdx.x;
    const __half* x = input + (int64_t)row * hidden;
    __half* y       = output + (int64_t)row * hidden;

    extern __shared__ float smem_rms[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __half2float(x[i]);
        local_sum += v * v;
    }
    smem_rms[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem_rms[threadIdx.x] += smem_rms[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(smem_rms[0] / (float)hidden + eps);

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __half2float(x[i]);
        float w = 1.0f;
        if (weight) {
            float wv = __half2float(weight[i]);
            w = add_one ? (1.0f + wv) : wv;
        }
        y[i] = __float2half(v * rms * w);
    }
}

void launch_rmsnorm_fp16(
    const __half* input,
    const __half* weight,  // may be nullptr
    __half* output,
    int rows, int hidden, float eps, bool add_one,
    cudaStream_t stream)
{
    int block_size = 256;
    if (hidden > 512)  block_size = 512;
    if (hidden > 1024) block_size = 1024;
    if (block_size > hidden) block_size = hidden;
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;
    if (block_size > 1024) block_size = 1024;

    int shared_mem = block_size * sizeof(float);
    rmsnorm_fp16_kernel<<<rows, block_size, shared_mem, stream>>>(
        input, weight, output, hidden, eps, add_one);
}

// ============================================================================
// inf/nan check (works for float AND half) — used by GradScaler on CUDA
// ============================================================================
// Writes 1 into *found when any element is inf or nan, else leaves it at 0.
// Caller is responsible for zeroing *found before the launch.

__global__ void check_inf_nan_fp32_kernel(const float* data, int64_t n, int* found) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float v = data[idx];
        if (isinf(v) || isnan(v)) {
            atomicExch(found, 1);
            return;
        }
    }
}

__global__ void check_inf_nan_fp16_kernel(const __half* data, int64_t n, int* found) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        float v = __half2float(data[idx]);
        if (isinf(v) || isnan(v)) {
            atomicExch(found, 1);
            return;
        }
    }
}

void launch_check_inf_nan_fp32(const float* data, int64_t n, int* found_device, cudaStream_t stream) {
    check_inf_nan_fp32_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(data, n, found_device);
}

void launch_check_inf_nan_fp16(const __half* data, int64_t n, int* found_device, cudaStream_t stream) {
    check_inf_nan_fp16_kernel<<<fp16_num_blocks(n), FP16_BLOCK_SIZE, 0, stream>>>(data, n, found_device);
}

} // namespace cuda
} // namespace at
