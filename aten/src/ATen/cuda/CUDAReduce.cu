// ============================================================================
// CUDA Reduction Kernels for PromeTorch
// ============================================================================
// Efficient parallel reductions using shared memory

#include <cuda_runtime.h>
#include <cfloat>

// Include the header to get ATEN_CUDA_API macro for proper DLL export
#include "aten/src/ATen/cuda/CUDAOps.h"

namespace at {
namespace cuda {

constexpr int REDUCE_BLOCK_SIZE = 256;
#ifdef __HIP_PLATFORM_AMD__
constexpr int WARP_SIZE = 64;
constexpr unsigned WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;
#else
constexpr int WARP_SIZE = 32;
constexpr unsigned WARP_MASK = 0xFFFFFFFF;
#endif

// ============================================================================
// Warp-level reduction utilities
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(WARP_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(WARP_MASK, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(WARP_MASK, val, offset));
    }
    return val;
}

// ============================================================================
// Block-level reduction
// ============================================================================

__device__ float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float shared[WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -FLT_MAX;

    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

__device__ float block_reduce_min(float val) {
    __shared__ float shared[WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_min(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : FLT_MAX;

    if (wid == 0) {
        val = warp_reduce_min(val);
    }

    return val;
}

// ============================================================================
// Global Sum Reduction
// ============================================================================

__global__ void sum_kernel(const float* input, float* output, int64_t n) {
    float sum = 0.0f;

    // Grid-stride loop
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// Sum along dimension kernel
__global__ void sum_dim_kernel(
    const float* input,
    float* output,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.x;
    int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        for (int64_t r = 0; r < reduce_size; ++r) {
            int64_t idx = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
            sum += input[idx];
        }
        output[outer_idx * inner_size + inner_idx] = sum;
    }
}

// ============================================================================
// Global Mean Reduction
// ============================================================================

__global__ void mean_kernel(const float* input, float* output, int64_t n) {
    float sum = 0.0f;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum / n);
    }
}

// Mean along dimension
__global__ void mean_dim_kernel(
    const float* input,
    float* output,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.x;
    int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        for (int64_t r = 0; r < reduce_size; ++r) {
            int64_t idx = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
            sum += input[idx];
        }
        output[outer_idx * inner_size + inner_idx] = sum / reduce_size;
    }
}

// ============================================================================
// Global Max Reduction
// ============================================================================

__global__ void max_kernel(const float* input, float* output, int64_t n) {
    float max_val = -FLT_MAX;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        max_val = max(max_val, input[i]);
    }

    max_val = block_reduce_max(max_val);

    if (threadIdx.x == 0) {
        // Atomic max for floats using atomicCAS
        unsigned int* address_as_ui = (unsigned int*)output;
        unsigned int old = *address_as_ui, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ui, assumed,
                __float_as_uint(max(__uint_as_float(assumed), max_val)));
        } while (assumed != old);
    }
}

// Max along dimension with indices
__global__ void max_dim_kernel(
    const float* input,
    float* output,
    int64_t* indices,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.x;
    int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float max_val = -FLT_MAX;
        int64_t max_idx = 0;

        for (int64_t r = 0; r < reduce_size; ++r) {
            int64_t idx = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
            if (input[idx] > max_val) {
                max_val = input[idx];
                max_idx = r;
            }
        }

        int64_t out_idx = outer_idx * inner_size + inner_idx;
        output[out_idx] = max_val;
        if (indices != nullptr) {
            indices[out_idx] = max_idx;
        }
    }
}

// ============================================================================
// Global Min Reduction
// ============================================================================

__global__ void min_kernel(const float* input, float* output, int64_t n) {
    float min_val = FLT_MAX;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        min_val = min(min_val, input[i]);
    }

    min_val = block_reduce_min(min_val);

    if (threadIdx.x == 0) {
        unsigned int* address_as_ui = (unsigned int*)output;
        unsigned int old = *address_as_ui, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ui, assumed,
                __float_as_uint(min(__uint_as_float(assumed), min_val)));
        } while (assumed != old);
    }
}

// Min along dimension with indices
__global__ void min_dim_kernel(
    const float* input,
    float* output,
    int64_t* indices,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.x;
    int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float min_val = FLT_MAX;
        int64_t min_idx = 0;

        for (int64_t r = 0; r < reduce_size; ++r) {
            int64_t idx = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
            if (input[idx] < min_val) {
                min_val = input[idx];
                min_idx = r;
            }
        }

        int64_t out_idx = outer_idx * inner_size + inner_idx;
        output[out_idx] = min_val;
        if (indices != nullptr) {
            indices[out_idx] = min_idx;
        }
    }
}

// ============================================================================
// Product Reduction
// ============================================================================

__global__ void prod_kernel(const float* input, float* output, int64_t n) {
    float prod = 1.0f;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        prod *= input[i];
    }

    // Block reduce product (using log/exp for numerical stability would be better)
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        prod *= __shfl_down_sync(WARP_MASK, prod, offset);
    }

    if (lane == 0) shared[wid] = prod;
    __syncthreads();

    prod = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 1.0f;

    if (wid == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            prod *= __shfl_down_sync(WARP_MASK, prod, offset);
        }
    }

    // Can't atomically multiply floats easily, so this is approximate for multiple blocks
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = prod;
    }
}

// ============================================================================
// Variance and Standard Deviation
// ============================================================================

// Two-pass: first compute mean, then compute variance
__global__ void variance_kernel(
    const float* input,
    float* output,
    float mean,
    int64_t n,
    bool unbiased
) {
    float sum_sq = 0.0f;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float diff = input[i] - mean;
        sum_sq += diff * diff;
    }

    sum_sq = block_reduce_sum(sum_sq);

    if (threadIdx.x == 0) {
        float divisor = unbiased ? (n - 1) : n;
        atomicAdd(output, sum_sq / divisor);
    }
}

// Variance along dimension
__global__ void variance_dim_kernel(
    const float* input,
    float* output,
    const float* mean,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size,
    bool unbiased
) {
    int64_t outer_idx = blockIdx.x;
    int64_t inner_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float m = mean[outer_idx * inner_size + inner_idx];
        float sum_sq = 0.0f;

        for (int64_t r = 0; r < reduce_size; ++r) {
            int64_t idx = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
            float diff = input[idx] - m;
            sum_sq += diff * diff;
        }

        float divisor = unbiased ? (reduce_size - 1) : reduce_size;
        output[outer_idx * inner_size + inner_idx] = sum_sq / divisor;
    }
}

// ============================================================================
// Norm Kernels
// ============================================================================

__global__ void l1_norm_kernel(const float* input, float* output, int64_t n) {
    float sum = 0.0f;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += fabsf(input[i]);
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

__global__ void l2_norm_kernel(const float* input, float* output, int64_t n) {
    float sum_sq = 0.0f;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = input[i];
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum_sq);
    }
}

// ============================================================================
// ArgMax / ArgMin
// ============================================================================

__global__ void argmax_kernel(const float* input, int64_t* output, int64_t n) {
    __shared__ float shared_val[REDUCE_BLOCK_SIZE];
    __shared__ int64_t shared_idx[REDUCE_BLOCK_SIZE];

    int tid = threadIdx.x;
    float max_val = -FLT_MAX;
    int64_t max_idx = 0;

    for (int64_t i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }

    shared_val[tid] = max_val;
    shared_idx[tid] = max_idx;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_val[tid + s] > shared_val[tid]) {
            shared_val[tid] = shared_val[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && blockIdx.x == 0) {
        *output = shared_idx[0];
    }
}

__global__ void argmin_kernel(const float* input, int64_t* output, int64_t n) {
    __shared__ float shared_val[REDUCE_BLOCK_SIZE];
    __shared__ int64_t shared_idx[REDUCE_BLOCK_SIZE];

    int tid = threadIdx.x;
    float min_val = FLT_MAX;
    int64_t min_idx = 0;

    for (int64_t i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        if (input[i] < min_val) {
            min_val = input[i];
            min_idx = i;
        }
    }

    shared_val[tid] = min_val;
    shared_idx[tid] = min_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_val[tid + s] < shared_val[tid]) {
            shared_val[tid] = shared_val[tid + s];
            shared_idx[tid] = shared_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && blockIdx.x == 0) {
        *output = shared_idx[0];
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void launch_sum(const float* input, float* output, int64_t n, cudaStream_t stream) {
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int blocks = (n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    blocks = min(blocks, 1024);
    sum_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_sum_dim(const float* input, float* output,
                    int64_t outer_size, int64_t reduce_size, int64_t inner_size,
                    cudaStream_t stream) {
    dim3 blocks(outer_size, (inner_size + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE);
    sum_dim_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(
        input, output, outer_size, reduce_size, inner_size);
}

void launch_mean(const float* input, float* output, int64_t n, cudaStream_t stream) {
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int blocks = (n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    blocks = min(blocks, 1024);
    mean_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_mean_dim(const float* input, float* output,
                     int64_t outer_size, int64_t reduce_size, int64_t inner_size,
                     cudaStream_t stream) {
    dim3 blocks(outer_size, (inner_size + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE);
    mean_dim_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(
        input, output, outer_size, reduce_size, inner_size);
}

void launch_max(const float* input, float* output, int64_t n, cudaStream_t stream) {
    float init_val = -FLT_MAX;
    cudaMemcpyAsync(output, &init_val, sizeof(float), cudaMemcpyHostToDevice, stream);
    int blocks = (n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    blocks = min(blocks, 1024);
    max_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_max_dim(const float* input, float* output, int64_t* indices,
                    int64_t outer_size, int64_t reduce_size, int64_t inner_size,
                    cudaStream_t stream) {
    dim3 blocks(outer_size, (inner_size + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE);
    max_dim_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(
        input, output, indices, outer_size, reduce_size, inner_size);
}

void launch_min(const float* input, float* output, int64_t n, cudaStream_t stream) {
    float init_val = FLT_MAX;
    cudaMemcpyAsync(output, &init_val, sizeof(float), cudaMemcpyHostToDevice, stream);
    int blocks = (n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    blocks = min(blocks, 1024);
    min_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_min_dim(const float* input, float* output, int64_t* indices,
                    int64_t outer_size, int64_t reduce_size, int64_t inner_size,
                    cudaStream_t stream) {
    dim3 blocks(outer_size, (inner_size + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE);
    min_dim_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(
        input, output, indices, outer_size, reduce_size, inner_size);
}

void launch_l1_norm(const float* input, float* output, int64_t n, cudaStream_t stream) {
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int blocks = (n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    blocks = min(blocks, 1024);
    l1_norm_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_l2_norm(const float* input, float* output, int64_t n, cudaStream_t stream) {
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int blocks = (n + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    blocks = min(blocks, 1024);
    l2_norm_kernel<<<blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_argmax(const float* input, int64_t* output, int64_t n, cudaStream_t stream) {
    argmax_kernel<<<1, REDUCE_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_argmin(const float* input, int64_t* output, int64_t n, cudaStream_t stream) {
    argmin_kernel<<<1, REDUCE_BLOCK_SIZE, 0, stream>>>(input, output, n);
}

// ============================================================================
// GPU Temperature Top-K Sampling Kernel
// ============================================================================
// Applies temperature, finds top-K values+indices on GPU, copies only K floats
// to host for CPU softmax+sampling. Reduces D2H from vocab×4 bytes to K×8 bytes.
//
// Algorithm: each thread scans N/blockDim elements, maintains local top-K heap.
// Then threads reduce via shared memory. K must be small (≤ 64).

__global__ void topk_sample_kernel(
    const float* __restrict__ logits,
    float* __restrict__ out_vals,     // [K] top values (with temperature applied)
    int32_t* __restrict__ out_indices, // [K] top indices
    int64_t N,
    float inv_temperature)            // 1.0f / temperature
{
    // Phase 1: Each thread finds its local max element
    // We use a multi-pass approach: first find global max (for stability),
    // then find top-K using a threshold-based approach.

    __shared__ float s_max;
    __shared__ float s_topk_vals[64];
    __shared__ int32_t s_topk_indices[64];
    __shared__ int s_count;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Step 1: Find global max (for numerical stability)
    float local_max = -FLT_MAX;
    for (int64_t i = tid; i < N; i += block_size) {
        float v = logits[i] * inv_temperature;
        if (v > local_max) local_max = v;
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(WARP_MASK, local_max, offset));

    __shared__ float warp_max[8];
    if ((tid & (WARP_SIZE - 1)) == 0) warp_max[tid / WARP_SIZE] = local_max;
    __syncthreads();

    if (tid == 0) {
        float m = warp_max[0];
        for (int i = 1; i < (block_size + WARP_SIZE - 1) / WARP_SIZE; i++)
            m = fmaxf(m, warp_max[i]);
        s_max = m;
        s_count = 0;
    }
    __syncthreads();

    float global_max = s_max;

    // Step 2: Find top-K using threshold. Start with threshold = max - 5.0,
    // then lower if we don't have enough elements (adaptive).
    // For top-40 out of 151936: typically max - 3.0 captures enough.

    int K = 40;  // top-K (hardcoded for simplicity, matches common default)
    float threshold = global_max - 10.0f;  // Start wide, narrow if too many

    // Count elements above threshold
    int local_count = 0;
    for (int64_t i = tid; i < N; i += block_size) {
        float v = logits[i] * inv_temperature;
        if (v >= threshold) local_count++;
    }

    // Sum local counts
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        local_count += __shfl_down_sync(WARP_MASK, local_count, offset);

    __shared__ int warp_counts[8];
    if ((tid & (WARP_SIZE - 1)) == 0) warp_counts[tid / WARP_SIZE] = local_count;
    __syncthreads();

    int total = 0;
    if (tid == 0) {
        for (int i = 0; i < (block_size + WARP_SIZE - 1) / WARP_SIZE; i++)
            total += warp_counts[i];
        // If too many elements, raise threshold
        if (total > 64) threshold = global_max - 3.0f;
        s_count = 0;
    }
    __syncthreads();

    // Step 3: Collect top elements atomically
    for (int64_t i = tid; i < N; i += block_size) {
        float v = logits[i] * inv_temperature;
        if (v >= threshold) {
            int pos = atomicAdd(&s_count, 1);
            if (pos < 64) {
                s_topk_vals[pos] = v;
                s_topk_indices[pos] = static_cast<int32_t>(i);
            }
        }
    }
    __syncthreads();

    // Step 4: Thread 0 sorts by value (insertion sort on small array)
    if (tid == 0) {
        int count = min(s_count, 64);
        // Simple selection sort for small array
        for (int i = 0; i < min(count, K) && i < count; i++) {
            int max_j = i;
            for (int j = i + 1; j < count; j++) {
                if (s_topk_vals[j] > s_topk_vals[max_j]) max_j = j;
            }
            if (max_j != i) {
                float tv = s_topk_vals[i]; s_topk_vals[i] = s_topk_vals[max_j]; s_topk_vals[max_j] = tv;
                int32_t ti = s_topk_indices[i]; s_topk_indices[i] = s_topk_indices[max_j]; s_topk_indices[max_j] = ti;
            }
        }
        // Write top-K to output
        int out_count = min(count, K);
        for (int i = 0; i < out_count; i++) {
            out_vals[i] = s_topk_vals[i];
            out_indices[i] = s_topk_indices[i];
        }
        // Mark end
        if (out_count < K) {
            out_vals[out_count] = -FLT_MAX;
            out_indices[out_count] = -1;
        }
    }
}

ATEN_CUDA_API void launch_topk_sample(
    const float* logits,
    float* d_topk_vals,        // device: [K] output values
    int32_t* d_topk_indices,   // device: [K] output indices
    int64_t vocab_size,
    float temperature,
    cudaStream_t stream)
{
    float inv_temp = (temperature > 1e-6f) ? 1.0f / temperature : 1.0f;
    topk_sample_kernel<<<1, 256, 0, stream>>>(
        logits, d_topk_vals, d_topk_indices, vocab_size, inv_temp);
}

// ============================================================================
// Cross Entropy Loss Kernel
// ============================================================================
// Computes: -log(softmax(logits)[target])
// Each block handles one sample in the batch

__global__ void cross_entropy_loss_kernel(
    const float* logits,    // (batch_size, num_classes)
    const float* targets,   // (batch_size,) - class indices as float
    float* output,          // (batch_size,) for None, or scalar for Mean/Sum
    int batch_size,
    int num_classes,
    int reduction           // 0=None, 1=Mean, 2=Sum
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        const float* row = logits + idx * num_classes;
        int target_class = static_cast<int>(targets[idx]);

        // Compute log-softmax at target class using logsumexp trick
        // 1. Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int c = 0; c < num_classes; ++c) {
            max_val = max(max_val, row[c]);
        }

        // 2. Compute sum(exp(x - max))
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += expf(row[c] - max_val);
        }

        // 3. log_softmax[target] = logit[target] - max - log(sum_exp)
        float log_prob = row[target_class] - max_val - logf(sum_exp);

        // 4. loss = -log_prob
        float loss = -log_prob;

        if (reduction == 0) {
            // None: output per-sample loss
            output[idx] = loss;
        } else {
            // Mean or Sum: accumulate
            atomicAdd(output, loss);
        }
    }
}

// Kernel for dividing by batch_size (for Mean reduction)
__global__ void divide_scalar_kernel(float* data, float divisor) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *data /= divisor;
    }
}

void launch_cross_entropy_loss(
    const float* logits,
    const float* targets,
    float* output,
    int batch_size,
    int num_classes,
    int reduction,
    cudaStream_t stream
) {
    // Initialize output
    if (reduction != 0) {
        // Mean or Sum: output is scalar, initialize to 0
        cudaMemsetAsync(output, 0, sizeof(float), stream);
    }

    // Launch kernel
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cross_entropy_loss_kernel<<<blocks, threads, 0, stream>>>(
        logits, targets, output, batch_size, num_classes, reduction
    );

    // If Mean reduction, divide by batch_size
    if (reduction == 1) {
        divide_scalar_kernel<<<1, 1, 0, stream>>>(output, static_cast<float>(batch_size));
    }
}

// ============================================================================
// NLL Loss Kernel
// ============================================================================
// Input is already log probabilities

__global__ void nll_loss_kernel(
    const float* log_probs,  // (batch_size, num_classes)
    const float* targets,    // (batch_size,) - class indices as float
    float* output,           // (batch_size,) for None, or scalar for Mean/Sum
    int batch_size,
    int num_classes,
    int reduction            // 0=None, 1=Mean, 2=Sum
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        int target_class = static_cast<int>(targets[idx]);
        float log_prob = log_probs[idx * num_classes + target_class];
        float loss = -log_prob;

        if (reduction == 0) {
            output[idx] = loss;
        } else {
            atomicAdd(output, loss);
        }
    }
}

void launch_nll_loss(
    const float* log_probs,
    const float* targets,
    float* output,
    int batch_size,
    int num_classes,
    int reduction,
    cudaStream_t stream
) {
    if (reduction != 0) {
        cudaMemsetAsync(output, 0, sizeof(float), stream);
    }

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    nll_loss_kernel<<<blocks, threads, 0, stream>>>(
        log_probs, targets, output, batch_size, num_classes, reduction
    );

    if (reduction == 1) {
        divide_scalar_kernel<<<1, 1, 0, stream>>>(output, static_cast<float>(batch_size));
    }
}

} // namespace cuda
} // namespace at
