// ============================================================================
// CUDA Kernels for PromeTorch
// ============================================================================
// Basic element-wise operations optimized for GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

// Include the header to get ATEN_CUDA_API macro for proper DLL export
#include "aten/src/ATen/cuda/CUDAOps.h"

namespace at {
namespace cuda {

// ============================================================================
// Kernel Configuration
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_GRID_SIZE = 65535;

inline int get_num_blocks(int64_t n) {
    return static_cast<int>(std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (int64_t)MAX_GRID_SIZE));
}

// ============================================================================
// Element-wise Unary Kernels
// ============================================================================

template<typename T>
__global__ void neg_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = -input[idx];
    }
}

template<typename T>
__global__ void abs_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] >= 0 ? input[idx] : -input[idx];
    }
}

template<typename T>
__global__ void sqrt_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);
    }
}

template<typename T>
__global__ void rsqrt_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = rsqrtf(input[idx]);
    }
}

template<typename T>
__global__ void square_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = input[idx];
        output[idx] = val * val;
    }
}

template<typename T>
__global__ void exp_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx]);
    }
}

template<typename T>
__global__ void log_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = logf(input[idx]);
    }
}

template<typename T>
__global__ void sin_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sinf(input[idx]);
    }
}

template<typename T>
__global__ void cos_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = cosf(input[idx]);
    }
}

template<typename T>
__global__ void tanh_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

template<typename T>
__global__ void relu_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

template<typename T>
__global__ void leaky_relu_kernel(const T* input, T* output, T alpha, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = input[idx];
        output[idx] = val > 0 ? val : alpha * val;
    }
}

template<typename T>
__global__ void silu_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = input[idx];
        output[idx] = val / (1.0f + expf(-val));
    }
}

template<typename T>
__global__ void gelu_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T x = input[idx];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr T kSqrt2OverPi = 0.7978845608f;
        constexpr T kCoeff = 0.044715f;
        T x3 = x * x * x;
        output[idx] = 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + kCoeff * x3)));
    }
}

// ============================================================================
// Additional Unary Kernels
// ============================================================================

template<typename T>
__global__ void log2_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { output[idx] = log2f(input[idx]); }
}

template<typename T>
__global__ void log10_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { output[idx] = log10f(input[idx]); }
}

template<typename T>
__global__ void tan_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { output[idx] = tanf(input[idx]); }
}

template<typename T>
__global__ void ceil_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { output[idx] = ceilf(input[idx]); }
}

template<typename T>
__global__ void floor_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { output[idx] = floorf(input[idx]); }
}

template<typename T>
__global__ void round_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { output[idx] = roundf(input[idx]); }
}

template<typename T>
__global__ void sign_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { output[idx] = (input[idx] > T(0)) - (input[idx] < T(0)); }
}

template<typename T>
__global__ void reciprocal_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { output[idx] = T(1) / input[idx]; }
}

// ============================================================================
// Element-wise Binary Kernels
// ============================================================================

template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void add_scalar_kernel(const T* a, T scalar, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

template<typename T>
__global__ void sub_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

template<typename T>
__global__ void mul_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

template<typename T>
__global__ void mul_scalar_kernel(const T* a, T scalar, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

template<typename T>
__global__ void div_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

template<typename T>
__global__ void div_scalar_kernel(const T* a, T scalar, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

template<typename T>
__global__ void pow_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void pow_scalar_kernel(const T* a, T exp, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(a[idx], exp);
    }
}

template<typename T>
__global__ void maximum_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] > b[idx] ? a[idx] : b[idx];
    }
}

template<typename T>
__global__ void minimum_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] < b[idx] ? a[idx] : b[idx];
    }
}

// ============================================================================
// Fill and Copy Kernels
// ============================================================================

template<typename T>
__global__ void fill_kernel(T* data, T value, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

template<typename T>
__global__ void copy_kernel(const T* src, T* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Comparison Kernels
// ============================================================================

template<typename T>
__global__ void eq_kernel(const T* a, const T* b, bool* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] == b[idx];
    }
}

template<typename T>
__global__ void ne_kernel(const T* a, const T* b, bool* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] != b[idx];
    }
}

template<typename T>
__global__ void lt_kernel(const T* a, const T* b, bool* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] < b[idx];
    }
}

template<typename T>
__global__ void le_kernel(const T* a, const T* b, bool* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] <= b[idx];
    }
}

template<typename T>
__global__ void gt_kernel(const T* a, const T* b, bool* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] > b[idx];
    }
}

template<typename T>
__global__ void ge_kernel(const T* a, const T* b, bool* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] >= b[idx];
    }
}

// ============================================================================
// Float-returning Comparison Kernels (for CUDA dispatch)
// ============================================================================

__global__ void eq_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f; }
}

__global__ void ne_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f; }
}

__global__ void lt_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f; }
}

__global__ void le_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f; }
}

__global__ void gt_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f; }
}

__global__ void ge_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f; }
}

// Scalar comparison kernels
__global__ void eq_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] == val) ? 1.0f : 0.0f; }
}

__global__ void ne_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] != val) ? 1.0f : 0.0f; }
}

__global__ void lt_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] < val) ? 1.0f : 0.0f; }
}

__global__ void le_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] <= val) ? 1.0f : 0.0f; }
}

__global__ void gt_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] > val) ? 1.0f : 0.0f; }
}

__global__ void ge_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (a[idx] >= val) ? 1.0f : 0.0f; }
}

// ============================================================================
// Fused Operations Kernels
// ============================================================================

// addcmul: out = self + value * t1 * t2
__global__ void addcmul_kernel(const float* self, const float* t1, const float* t2, float value, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = self[idx] + value * t1[idx] * t2[idx]; }
}

// addcdiv: out = self + value * t1 / t2
__global__ void addcdiv_kernel(const float* self, const float* t1, const float* t2, float value, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = self[idx] + value * t1[idx] / t2[idx]; }
}

// ============================================================================
// Clamp Kernel
// ============================================================================

template<typename T>
__global__ void clamp_kernel(const T* input, T* output, T min_val, T max_val, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = input[idx];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        output[idx] = val;
    }
}

// ============================================================================
// Where Kernel (conditional selection)
// ============================================================================

template<typename T>
__global__ void where_kernel(const bool* cond, const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cond[idx] ? a[idx] : b[idx];
    }
}

// ============================================================================
// Masked Fill Kernel
// ============================================================================

template<typename T>
__global__ void masked_fill_kernel(T* data, const bool* mask, T value, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx]) {
            data[idx] = value;
        }
    }
}

// ============================================================================
// Broadcasting Mul Kernels
// ============================================================================

// [outer, inner] * [outer, 1] -> broadcast second operand across inner dimension
template<typename T>
__global__ void mul_broadcast_row_kernel(const T* a, const T* b, T* out, int64_t outer_size, int64_t inner_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (idx < total) {
        int64_t outer_idx = idx / inner_size;
        out[idx] = a[idx] * b[outer_idx];
    }
}

// [outer, inner] * [inner] -> broadcast second operand across outer dimension
template<typename T>
__global__ void mul_broadcast_col_kernel(const T* a, const T* b, T* out, int64_t outer_size, int64_t inner_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (idx < total) {
        int64_t inner_idx = idx % inner_size;
        out[idx] = a[idx] * b[inner_idx];
    }
}

// [outer, inner] + [inner] -> broadcast second operand across outer dimension (for bias)
template<typename T>
__global__ void add_broadcast_col_kernel(const T* a, const T* b, T* out, int64_t outer_size, int64_t inner_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    if (idx < total) {
        int64_t inner_idx = idx % inner_size;
        out[idx] = a[idx] + b[inner_idx];
    }
}

// ============================================================================
// Softmax Kernel
// ============================================================================

template<typename T>
__global__ void softmax_kernel(const T* input, T* output, int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    int64_t outer_idx = blockIdx.x;
    int64_t inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        // Find max for numerical stability
        T max_val = -FLT_MAX;
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
            max_val = max(max_val, input[idx]);
        }

        // Compute exp and sum
        T sum = 0;
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
            T exp_val = expf(input[idx] - max_val);
            output[idx] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t idx = outer_idx * dim_size * inner_size + i * inner_size + inner_idx;
            output[idx] /= sum;
        }
    }
}

// ============================================================================
// Launch Wrapper Functions
// ============================================================================

// Unary operations
void launch_neg(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    neg_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_abs(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    abs_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_sqrt(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    sqrt_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_rsqrt(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    rsqrt_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_square(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    square_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_exp(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    exp_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_log(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    log_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_sin(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    sin_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_cos(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    cos_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_tanh(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    tanh_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_sigmoid(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    sigmoid_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_relu(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    relu_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_leaky_relu(const float* input, float* output, float alpha, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    leaky_relu_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, alpha, n);
}

void launch_silu(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    silu_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_gelu(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    gelu_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

// Additional unary operations
void launch_log2(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    log2_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_log10(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    log10_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_tan(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    tan_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_ceil(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    ceil_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_floor(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    floor_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_round(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    round_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_sign(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    sign_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void launch_reciprocal(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    reciprocal_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

// Float-returning comparison operations (tensor vs tensor)
void launch_eq(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    eq_float_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_ne(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    ne_float_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_lt(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    lt_float_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_le(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    le_float_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_gt(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    gt_float_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_ge(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    ge_float_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

// Scalar comparison operations
void launch_eq_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    eq_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, val, out, n);
}

void launch_ne_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    ne_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, val, out, n);
}

void launch_lt_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    lt_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, val, out, n);
}

void launch_le_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    le_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, val, out, n);
}

void launch_gt_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    gt_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, val, out, n);
}

void launch_ge_scalar(const float* a, float val, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    ge_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, val, out, n);
}

// Fused operations
void launch_addcmul(const float* self, const float* t1, const float* t2, float value, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    addcmul_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(self, t1, t2, value, out, n);
}

void launch_addcdiv(const float* self, const float* t1, const float* t2, float value, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    addcdiv_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(self, t1, t2, value, out, n);
}

// Binary operations
void launch_add(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    add_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_add_scalar(const float* a, float scalar, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    add_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

void launch_sub(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    sub_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_mul(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    mul_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_mul_scalar(const float* a, float scalar, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    mul_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

void launch_div(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    div_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_div_scalar(const float* a, float scalar, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    div_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
}

void launch_pow(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    pow_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_pow_scalar(const float* a, float exp, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    pow_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, exp, out, n);
}

void launch_maximum(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    maximum_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void launch_minimum(const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    minimum_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

// Fill and copy
void launch_fill(float* data, float value, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    fill_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data, value, n);
}

void launch_copy(const float* src, float* dst, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(src, dst, n);
}

// Clamp
void launch_clamp(const float* input, float* output, float min_val, float max_val, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    clamp_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, output, min_val, max_val, n);
}

// Where
void launch_where(const bool* cond, const float* a, const float* b, float* out, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    where_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(cond, a, b, out, n);
}

// Masked fill
void launch_masked_fill(float* data, const bool* mask, float value, int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    masked_fill_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data, mask, value, n);
}

// Softmax
void launch_softmax(const float* input, float* output, int64_t outer_size, int64_t dim_size, int64_t inner_size, cudaStream_t stream) {
    dim3 blocks(outer_size);
    dim3 threads(inner_size);
    softmax_kernel<<<blocks, threads, 0, stream>>>(input, output, outer_size, dim_size, inner_size);
}

// Broadcasting mul: [outer, inner] * [outer, 1]
void launch_mul_broadcast_row(const float* a, const float* b, float* out, int64_t outer_size, int64_t inner_size, cudaStream_t stream) {
    int64_t total = outer_size * inner_size;
    int blocks = get_num_blocks(total);
    mul_broadcast_row_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, outer_size, inner_size);
}

// Broadcasting mul: [outer, inner] * [inner]
void launch_mul_broadcast_col(const float* a, const float* b, float* out, int64_t outer_size, int64_t inner_size, cudaStream_t stream) {
    int64_t total = outer_size * inner_size;
    int blocks = get_num_blocks(total);
    mul_broadcast_col_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, outer_size, inner_size);
}

// Broadcasting add: [outer, inner] + [inner] (for bias addition)
void launch_add_broadcast_col(const float* a, const float* b, float* out, int64_t outer_size, int64_t inner_size, cudaStream_t stream) {
    int64_t total = outer_size * inner_size;
    int blocks = get_num_blocks(total);
    add_broadcast_col_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, outer_size, inner_size);
}

// ============================================================================
// Parallel Scan Kernel for PIR (Recurrent Scan)
// ============================================================================
// h[t] = gate[t] * h[t-1] + x[t]
// Each (batch, dim) pair is processed by one thread sequentially over time.
// This parallelizes over B*D dimensions.

template<typename T>
__global__ void parallel_scan_kernel(
    const T* __restrict__ x,           // [B, T, D] input
    const T* __restrict__ gate_logits, // [B, T, D] gate logits
    const T* __restrict__ base_decay,  // [D] base decay values
    T* __restrict__ output,            // [B, T, D] output
    T* __restrict__ gates,             // [B, T, D] computed gates (for backward)
    int64_t B, int64_t T_len, int64_t D
) {
    // Each thread handles one (batch, dim) pair
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_bd = B * D;

    if (idx >= total_bd) return;

    int64_t b = idx / D;  // batch index
    int64_t d = idx % D;  // dim index

    T base = base_decay[d];
    T h = 0;  // hidden state

    // Sequential scan over time dimension
    for (int64_t t = 0; t < T_len; ++t) {
        int64_t offset = (b * T_len + t) * D + d;

        T gate_logit = gate_logits[offset];
        T modulation = tanhf(gate_logit) * 0.1f;
        T gate = base * (1.0f + modulation);

        // Clamp gate to [0.5, 0.999]
        gate = gate < 0.5f ? 0.5f : (gate > 0.999f ? 0.999f : gate);

        gates[offset] = gate;
        h = gate * h + x[offset];
        output[offset] = h;
    }
}

// Rotary embedding kernel - applies rotation to pairs
template<typename T>
__global__ void rotary_embedding_kernel(
    const T* __restrict__ x,         // [B, T, D] input
    const T* __restrict__ cos_cache, // [max_seq, D] precomputed cos
    const T* __restrict__ sin_cache, // [max_seq, D] precomputed sin
    T* __restrict__ output,          // [B, T, D] output
    int64_t B, int64_t T_len, int64_t D, int64_t cache_dim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = B * T_len * (D / 2);

    if (idx >= total) return;

    int64_t half_d = D / 2;
    int64_t bt_idx = idx / half_d;
    int64_t i = idx % half_d;  // dimension pair index

    int64_t b = bt_idx / T_len;
    int64_t t = bt_idx % T_len;

    int64_t base_offset = (b * T_len + t) * D;
    int64_t cache_offset = t * cache_dim;

    T x1 = x[base_offset + i];
    T x2 = x[base_offset + half_d + i];
    T cos_val = cos_cache[cache_offset + i];
    T sin_val = sin_cache[cache_offset + i];

    output[base_offset + i] = x1 * cos_val - x2 * sin_val;
    output[base_offset + half_d + i] = x1 * sin_val + x2 * cos_val;
}

// Launch wrapper for parallel scan
void launch_parallel_scan(
    const float* x, const float* gate_logits, const float* base_decay,
    float* output, float* gates,
    int64_t B, int64_t T, int64_t D,
    cudaStream_t stream
) {
    int64_t total_bd = B * D;
    int blocks = static_cast<int>((total_bd + BLOCK_SIZE - 1) / BLOCK_SIZE);
    blocks = blocks > MAX_GRID_SIZE ? MAX_GRID_SIZE : blocks;
    parallel_scan_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        x, gate_logits, base_decay, output, gates, B, T, D
    );
}

// Launch wrapper for rotary embedding
void launch_rotary_embedding(
    const float* x, const float* cos_cache, const float* sin_cache, float* output,
    int64_t B, int64_t T, int64_t D, int64_t cache_dim,
    cudaStream_t stream
) {
    int64_t total = B * T * (D / 2);
    int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    blocks = blocks > MAX_GRID_SIZE ? MAX_GRID_SIZE : blocks;
    rotary_embedding_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        x, cos_cache, sin_cache, output, B, T, D, cache_dim
    );
}

} // namespace cuda
} // namespace at
