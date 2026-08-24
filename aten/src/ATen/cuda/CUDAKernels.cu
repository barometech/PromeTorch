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
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = -input[idx];
    }
}

template<typename T>
__global__ void abs_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = input[idx] >= 0 ? input[idx] : -input[idx];
    }
}

template<typename T>
__global__ void sqrt_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = sqrtf(input[idx]);
    }
}

template<typename T>
__global__ void rsqrt_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = rsqrtf(input[idx]);
    }
}

template<typename T>
__global__ void square_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        T val = input[idx];
        output[idx] = val * val;
    }
}

template<typename T>
__global__ void exp_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = expf(input[idx]);
    }
}

template<typename T>
__global__ void log_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = logf(input[idx]);
    }
}

template<typename T>
__global__ void sin_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = sinf(input[idx]);
    }
}

template<typename T>
__global__ void cos_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = cosf(input[idx]);
    }
}

template<typename T>
__global__ void tanh_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = tanhf(input[idx]);
    }
}

template<typename T>
__global__ void sigmoid_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

template<typename T>
__global__ void relu_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

template<typename T>
__global__ void leaky_relu_kernel(const T* input, T* output, T alpha, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        T val = input[idx];
        output[idx] = val > 0 ? val : alpha * val;
    }
}

template<typename T>
__global__ void silu_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        T val = input[idx];
        output[idx] = val / (1.0f + expf(-val));
    }
}

template<typename T>
__global__ void gelu_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
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
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = log2f(input[idx]);
    }
}

template<typename T>
__global__ void log10_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = log10f(input[idx]);
    }
}

template<typename T>
__global__ void tan_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = tanf(input[idx]);
    }
}

template<typename T>
__global__ void ceil_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = ceilf(input[idx]);
    }
}

template<typename T>
__global__ void floor_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = floorf(input[idx]);
    }
}

template<typename T>
__global__ void round_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = roundf(input[idx]);
    }
}

template<typename T>
__global__ void sign_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = (input[idx] > T(0)) - (input[idx] < T(0));
    }
}

template<typename T>
__global__ void reciprocal_kernel(const T* input, T* output, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        output[idx] = T(1) / input[idx];
    }
}

// ============================================================================
// Element-wise Binary Kernels
// ============================================================================

template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void add_scalar_kernel(const T* a, T scalar, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] + scalar;
    }
}

template<typename T>
__global__ void sub_kernel(const T* a, const T* b, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] - b[idx];
    }
}

template<typename T>
__global__ void mul_kernel(const T* a, const T* b, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] * b[idx];
    }
}

template<typename T>
__global__ void mul_scalar_kernel(const T* a, T scalar, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] * scalar;
    }
}

template<typename T>
__global__ void div_kernel(const T* a, const T* b, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] / b[idx];
    }
}

template<typename T>
__global__ void div_scalar_kernel(const T* a, T scalar, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] / scalar;
    }
}

template<typename T>
__global__ void pow_kernel(const T* a, const T* b, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = powf(a[idx], b[idx]);
    }
}

template<typename T>
__global__ void pow_scalar_kernel(const T* a, T exp, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = powf(a[idx], exp);
    }
}

template<typename T>
__global__ void maximum_kernel(const T* a, const T* b, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] > b[idx] ? a[idx] : b[idx];
    }
}

template<typename T>
__global__ void minimum_kernel(const T* a, const T* b, T* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] < b[idx] ? a[idx] : b[idx];
    }
}

// ============================================================================
// Fill and Copy Kernels
// ============================================================================

template<typename T>
__global__ void fill_kernel(T* data, T value, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        data[idx] = value;
    }
}

template<typename T>
__global__ void copy_kernel(const T* src, T* dst, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Comparison Kernels
// ============================================================================

template<typename T>
__global__ void eq_kernel(const T* a, const T* b, bool* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] == b[idx];
    }
}

template<typename T>
__global__ void ne_kernel(const T* a, const T* b, bool* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] != b[idx];
    }
}

template<typename T>
__global__ void lt_kernel(const T* a, const T* b, bool* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] < b[idx];
    }
}

template<typename T>
__global__ void le_kernel(const T* a, const T* b, bool* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] <= b[idx];
    }
}

template<typename T>
__global__ void gt_kernel(const T* a, const T* b, bool* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] > b[idx];
    }
}

template<typename T>
__global__ void ge_kernel(const T* a, const T* b, bool* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = a[idx] >= b[idx];
    }
}

// ============================================================================
// Float-returning Comparison Kernels (for CUDA dispatch)
// ============================================================================

__global__ void eq_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void ne_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void lt_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void le_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void gt_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void ge_float_kernel(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
    }
}

// Scalar comparison kernels
__global__ void eq_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] == val) ? 1.0f : 0.0f;
    }
}

__global__ void ne_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] != val) ? 1.0f : 0.0f;
    }
}

__global__ void lt_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] < val) ? 1.0f : 0.0f;
    }
}

__global__ void le_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] <= val) ? 1.0f : 0.0f;
    }
}

__global__ void gt_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] > val) ? 1.0f : 0.0f;
    }
}

__global__ void ge_scalar_kernel(const float* a, float val, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = (a[idx] >= val) ? 1.0f : 0.0f;
    }
}

// ============================================================================
// Fused Operations Kernels
// ============================================================================

// addcmul: out = self + value * t1 * t2
__global__ void addcmul_kernel(const float* self, const float* t1, const float* t2, float value, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = self[idx] + value * t1[idx] * t2[idx];
    }
}

// addcdiv: out = self + value * t1 / t2
__global__ void addcdiv_kernel(const float* self, const float* t1, const float* t2, float value, float* out, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = self[idx] + value * t1[idx] / t2[idx];
    }
}

// ============================================================================
// Clamp Kernel
// ============================================================================

template<typename T>
__global__ void clamp_kernel(const T* input, T* output, T min_val, T max_val, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
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
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        out[idx] = cond[idx] ? a[idx] : b[idx];
    }
}

// ============================================================================
// Masked Fill Kernel
// ============================================================================

template<typename T>
__global__ void masked_fill_kernel(T* data, const bool* mask, T value, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
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
    int64_t total = outer_size * inner_size;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t outer_idx = idx / inner_size;
        out[idx] = a[idx] * b[outer_idx];
    }
}

// [outer, inner] * [inner] -> broadcast second operand across outer dimension
template<typename T>
__global__ void mul_broadcast_col_kernel(const T* a, const T* b, T* out, int64_t outer_size, int64_t inner_size) {
    int64_t total = outer_size * inner_size;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t inner_idx = idx % inner_size;
        out[idx] = a[idx] * b[inner_idx];
    }
}

// [outer, inner] + [inner] -> broadcast second operand across outer dimension (for bias)
template<typename T>
__global__ void add_broadcast_col_kernel(const T* a, const T* b, T* out, int64_t outer_size, int64_t inner_size) {
    int64_t total = outer_size * inner_size;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t inner_idx = idx % inner_size;
        out[idx] = a[idx] + b[inner_idx];
    }
}

// ============================================================================
// Softmax Kernel (handles inner_size > 1024 via loop)
// ============================================================================

template<typename T>
__global__ void softmax_kernel(const T* input, T* output, int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    int64_t outer_idx = blockIdx.x;

    if (outer_idx < outer_size) {
        // Each thread handles multiple inner indices if inner_size > blockDim.x
        for (int64_t inner_idx = threadIdx.x; inner_idx < inner_size; inner_idx += blockDim.x) {
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

// Softmax - cap threads to 1024, kernel loops over inner_size internally
void launch_softmax(const float* input, float* output, int64_t outer_size, int64_t dim_size, int64_t inner_size, cudaStream_t stream) {
    int threads_per_block = std::min((int)inner_size, 1024);
    dim3 blocks(outer_size);
    dim3 threads(threads_per_block);
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
    // Each thread handles one (batch, dim) pair, with grid-stride loop
    int64_t total_bd = B * D;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_bd; idx += (int64_t)blockDim.x * gridDim.x) {
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
            gate = gate < 0.5f ? 0.5f : (gate > 0.9999f ? 0.9999f : gate);

            gates[offset] = gate;
            h = gate * h + x[offset];
            output[offset] = h;
        }
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
    int64_t total = B * T_len * (D / 2);

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
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
}

// Launch wrapper for parallel scan
// Chunked-scan kernels are defined later in this file; declare them here.
#define SCAN_CHUNK 32
__global__ void scan_fwd_chunk_kernel(
    const float*, const float*, const float*, float*, float*, float*, float*, float*,
    int64_t, int64_t, int64_t, int64_t, int, int, const float*);
__global__ void scan_carry_kernel(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
__global__ void scan_fixup_kernel(
    float*, const float*, const float*, int64_t, int64_t, int64_t, int64_t);
__global__ void scan_bwd_chunk_kernel(
    const float*, const float*, float*, float*, float*, float*,
    int64_t, int64_t, int64_t, int64_t);
__global__ void scan_bwd_carry_kernel(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
__global__ void scan_bwd_fixup_kernel(
    float*, const float*, const float*, const float*, const float*, const float*,
    float*, const float*, const float*, int64_t, int64_t, int64_t, int64_t, int, int);

// Simple grow-only device scratch (single training thread; sizes are stable
// across a run, so this allocates once and reuses).
static float* scan_scratch(int slot, size_t bytes) {
    static float* bufs[8] = {nullptr};
    static size_t sizes[8] = {0};
    if (sizes[slot] < bytes) {
        if (bufs[slot]) cudaFree(bufs[slot]);
        cudaMalloc(&bufs[slot], bytes);
        sizes[slot] = bytes;
    }
    return bufs[slot];
}

static int scan_vp_mode() {
    static int v = -1;
    if (v < 0) v = getenv("PT_SCAN_VP") ? 1 : 0;   // default: v1-математика
    return v;
}

static bool scan_use_sequential() {
    static int v = -1;
    if (v < 0) v = (getenv("PT_SCAN_SEQ") != nullptr) ? 1 : 0;
    return v == 1;
}

// Internal: chunked forward scan; fuse=1 treats x as raw values (v*sigmoid(gl)).
static void scan_forward_chunked(
    const float* x, const float* gate_logits, const float* base_decay,
    float* output, float* gates,
    int64_t B, int64_t T, int64_t D, int fuse, cudaStream_t stream,
    const float* resets = nullptr
) {
    int64_t nC = (T + SCAN_CHUNK - 1) / SCAN_CHUNK;
    float* prefix_g = scan_scratch(0, sizeof(float) * B * T * D);
    float* P  = scan_scratch(1, sizeof(float) * B * nC * D);
    float* E  = scan_scratch(2, sizeof(float) * B * nC * D);
    float* CI = scan_scratch(3, sizeof(float) * B * nC * D);
    int64_t work1 = B * nC * D;
    int b1 = static_cast<int>((work1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    b1 = b1 > MAX_GRID_SIZE ? MAX_GRID_SIZE : b1;
    scan_fwd_chunk_kernel<<<b1, BLOCK_SIZE, 0, stream>>>(
        x, gate_logits, base_decay, output, gates, prefix_g, P, E, B, T, D, nC, fuse,
        scan_vp_mode(), resets);
    int64_t work2 = B * D;
    int b2 = static_cast<int>((work2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    b2 = b2 > MAX_GRID_SIZE ? MAX_GRID_SIZE : b2;
    scan_carry_kernel<<<b2, BLOCK_SIZE, 0, stream>>>(P, E, CI, B, D, nC);
    int64_t work3 = B * T * D;
    int b3 = static_cast<int>((work3 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    b3 = b3 > MAX_GRID_SIZE ? MAX_GRID_SIZE : b3;
    scan_fixup_kernel<<<b3, BLOCK_SIZE, 0, stream>>>(
        output, prefix_g, CI, B, T, D, nC);
}

void launch_parallel_scan(
    const float* x, const float* gate_logits, const float* base_decay,
    float* output, float* gates,
    int64_t B, int64_t T, int64_t D,
    cudaStream_t stream
) {
    if (scan_use_sequential() || T <= SCAN_CHUNK) {
        int64_t total_bd = B * D;
        int blocks = static_cast<int>((total_bd + BLOCK_SIZE - 1) / BLOCK_SIZE);
        blocks = blocks > MAX_GRID_SIZE ? MAX_GRID_SIZE : blocks;
        parallel_scan_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            x, gate_logits, base_decay, output, gates, B, T, D
        );
        return;
    }
    scan_forward_chunked(x, gate_logits, base_decay, output, gates, B, T, D, 0, stream);
}

// Fused variant: x is RAW values; kernel scans v*sigmoid(gl) (chunked only).
void launch_parallel_scan_fused(
    const float* values, const float* gate_logits, const float* base_decay,
    float* output, float* gates,
    int64_t B, int64_t T, int64_t D,
    cudaStream_t stream,
    const float* resets
) {
    scan_forward_chunked(values, gate_logits, base_decay, output, gates, B, T, D, 1, stream,
                         resets);
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

// ============================================================================
// Backward kernels for PIR training hot path (device-side; replaces the CPU
// round-trip paths in MathBackward.h that copied tensors over PCIe each step)
// ============================================================================

// --- parallel scan backward: mirrors ParallelScanBackward CPU loop 1:1 ---
__global__ void parallel_scan_backward_kernel(
    const float* __restrict__ grad_out,     // [B, T, D]
    const float* __restrict__ gates,        // [B, T, D] post-clamp gates
    const float* __restrict__ gate_logits,  // [B, T, D]
    const float* __restrict__ base_decay,   // [D]
    const float* __restrict__ hidden,       // [B, T, D] scan output h
    float* __restrict__ grad_x,             // [B, T, D]
    float* __restrict__ grad_gate_logits,   // [B, T, D]
    int64_t B, int64_t T_len, int64_t D
) {
    int64_t total_bd = B * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_bd;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t b = idx / D;
        int64_t d = idx % D;
        float base = base_decay[d];
        float grad_h = 0.0f;
        for (int64_t t = T_len - 1; t >= 0; --t) {
            int64_t offset = (b * T_len + t) * D + d;
            grad_h += grad_out[offset];
            grad_x[offset] = grad_h;
            if (t > 0) {
                float h_prev = hidden[offset - D];
                float tanh_val = tanhf(gate_logits[offset]);
                float d_gate_d_logit = base * 0.1f * (1.0f - tanh_val * tanh_val);
                grad_gate_logits[offset] = grad_h * h_prev * d_gate_d_logit;
                grad_h *= gates[offset];
            } else {
                grad_gate_logits[offset] = 0.0f;
            }
        }
    }
}

// Internal: chunked backward; fuse=1 additionally produces d_values in grad_x
// and adds the sigmoid-mul term to grad_gate_logits (values = raw v tensor).
static void scan_backward_chunked(
    const float* grad_out, const float* gates, const float* gate_logits,
    const float* base_decay, const float* hidden, const float* values,
    float* grad_x, float* grad_gate_logits,
    int64_t B, int64_t T, int64_t D, int fuse, cudaStream_t stream
) {
    int64_t nC = (T + SCAN_CHUNK - 1) / SCAN_CHUNK;
    float* suffix_s = scan_scratch(4, sizeof(float) * B * T * D);
    float* Pb = scan_scratch(5, sizeof(float) * B * nC * D);
    float* Eb = scan_scratch(6, sizeof(float) * B * nC * D);
    float* CR = scan_scratch(7, sizeof(float) * B * nC * D);
    int64_t work1 = B * nC * D;
    int b1 = static_cast<int>((work1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    b1 = b1 > MAX_GRID_SIZE ? MAX_GRID_SIZE : b1;
    scan_bwd_chunk_kernel<<<b1, BLOCK_SIZE, 0, stream>>>(
        grad_out, gates, grad_x, suffix_s, Pb, Eb, B, T, D, nC);
    int64_t work2 = B * D;
    int b2 = static_cast<int>((work2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    b2 = b2 > MAX_GRID_SIZE ? MAX_GRID_SIZE : b2;
    scan_bwd_carry_kernel<<<b2, BLOCK_SIZE, 0, stream>>>(Pb, Eb, CR, B, D, nC);
    int64_t work3 = B * T * D;
    int b3 = static_cast<int>((work3 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    b3 = b3 > MAX_GRID_SIZE ? MAX_GRID_SIZE : b3;
    scan_bwd_fixup_kernel<<<b3, BLOCK_SIZE, 0, stream>>>(
        grad_x, suffix_s, CR, gate_logits, base_decay, hidden,
        grad_gate_logits, values, gates, B, T, D, nC, fuse, scan_vp_mode());
}

void launch_parallel_scan_backward(
    const float* grad_out, const float* gates, const float* gate_logits,
    const float* base_decay, const float* hidden,
    float* grad_x, float* grad_gate_logits,
    int64_t B, int64_t T, int64_t D, cudaStream_t stream
) {
    if (scan_use_sequential() || T <= SCAN_CHUNK) {
        int64_t total = B * D;
        int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        blocks = blocks > MAX_GRID_SIZE ? MAX_GRID_SIZE : blocks;
        if (blocks < 1) blocks = 1;
        parallel_scan_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_out, gates, gate_logits, base_decay, hidden,
            grad_x, grad_gate_logits, B, T, D);
        return;
    }
    scan_backward_chunked(grad_out, gates, gate_logits, base_decay, hidden,
                          nullptr, grad_x, grad_gate_logits, B, T, D, 0, stream);
}

// Fused variant: also emits d_values (in grad_x) and full d_gate_logits.
void launch_parallel_scan_backward_fused(
    const float* grad_out, const float* gates, const float* gate_logits,
    const float* base_decay, const float* hidden, const float* values,
    float* grad_values, float* grad_gate_logits,
    int64_t B, int64_t T, int64_t D, cudaStream_t stream
) {
    scan_backward_chunked(grad_out, gates, gate_logits, base_decay, hidden,
                          values, grad_values, grad_gate_logits, B, T, D, 1, stream);
}

// --- SiLU backward: dx = dout * sig * (1 + x * (1 - sig)) ---
__global__ void silu_backward_kernel(
    const float* __restrict__ grad_out, const float* __restrict__ input,
    const float* __restrict__ sigmoid_saved, float* __restrict__ grad_in, int64_t n
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        float sig = sigmoid_saved[i];
        grad_in[i] = grad_out[i] * sig * (1.0f + input[i] * (1.0f - sig));
    }
}

void launch_silu_backward(const float* grad_out, const float* input,
                          const float* sigmoid_saved, float* grad_in,
                          int64_t n, cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    silu_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        grad_out, input, sigmoid_saved, grad_in, n);
}

// --- RMSNorm backward: grad_input (one block per outer position) ---
__global__ void rmsnorm_backward_dx_kernel(
    const float* __restrict__ grad_out,  // [outer, D]
    const float* __restrict__ input,     // [outer, D]
    const float* __restrict__ weight,    // [D]
    const float* __restrict__ inv_rms,   // [outer]
    float* __restrict__ grad_in,         // [outer, D]
    int64_t outer, int64_t D
) {
    __shared__ float sdata[BLOCK_SIZE];
    for (int64_t i = blockIdx.x; i < outer; i += gridDim.x) {
        int64_t offset = i * D;
        float ir = inv_rms[i];
        // block-reduce sum(x * w * gout) over D
        float local = 0.0f;
        for (int64_t j = threadIdx.x; j < D; j += blockDim.x) {
            local += input[offset + j] * weight[j] * grad_out[offset + j];
        }
        sdata[threadIdx.x] = local;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        float mean_xwg = sdata[0] / static_cast<float>(D);
        __syncthreads();
        float ir3 = ir * ir * ir;
        for (int64_t j = threadIdx.x; j < D; j += blockDim.x) {
            float term1 = ir * weight[j] * grad_out[offset + j];
            float term2 = ir3 * input[offset + j] * mean_xwg;
            grad_in[offset + j] = term1 - term2;
        }
        __syncthreads();
    }
}

// --- RMSNorm backward: grad_weight accumulation (+=), one thread per j ---
__global__ void rmsnorm_backward_dw_kernel(
    const float* __restrict__ grad_out, const float* __restrict__ input,
    const float* __restrict__ inv_rms, float* __restrict__ grad_w,
    int64_t outer, int64_t D
) {
    for (int64_t j = blockIdx.x * blockDim.x + threadIdx.x; j < D;
         j += (int64_t)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (int64_t i = 0; i < outer; ++i) {
            acc += grad_out[i * D + j] * input[i * D + j] * inv_rms[i];
        }
        grad_w[j] += acc;   // accumulate into existing gradient
    }
}

void launch_rmsnorm_backward(
    const float* grad_out, const float* input, const float* weight,
    const float* inv_rms, float* grad_in, float* grad_w,
    int64_t outer, int64_t D, cudaStream_t stream
) {
    int blocks = static_cast<int>(outer > MAX_GRID_SIZE ? MAX_GRID_SIZE : outer);
    if (blocks < 1) blocks = 1;
    rmsnorm_backward_dx_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        grad_out, input, weight, inv_rms, grad_in, outer, D);
    int dwblocks = static_cast<int>((D + BLOCK_SIZE - 1) / BLOCK_SIZE);
    rmsnorm_backward_dw_kernel<<<dwblocks, BLOCK_SIZE, 0, stream>>>(
        grad_out, input, inv_rms, grad_w, outer, D);
}

// --- Rotary embedding backward: inverse rotation (batch_first layout) ---
__global__ void rotary_backward_kernel(
    const float* __restrict__ grad_out,   // [B, S, D]
    const float* __restrict__ cos_cache,  // [max_seq, cache_dim]
    const float* __restrict__ sin_cache,
    float* __restrict__ grad_in,          // [B, S, D]
    int64_t B, int64_t S, int64_t D, int64_t cache_dim
) {
    int64_t half_d = D / 2;
    int64_t total = B * S * half_d;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t i = idx % half_d;
        int64_t bs = idx / half_d;
        int64_t s = bs % S;
        int64_t offset = bs * D;
        float g1 = grad_out[offset + i];
        float g2 = grad_out[offset + half_d + i];
        float c = cos_cache[s * cache_dim + i];
        float sn = sin_cache[s * cache_dim + i];
        grad_in[offset + i] = g1 * c + g2 * sn;
        grad_in[offset + half_d + i] = -g1 * sn + g2 * c;
    }
}

void launch_rotary_backward(
    const float* grad_out, const float* cos_cache, const float* sin_cache,
    float* grad_in, int64_t B, int64_t S, int64_t D, int64_t cache_dim,
    cudaStream_t stream
) {
    int64_t total = B * S * (D / 2);
    int blocks = get_num_blocks(total);
    rotary_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        grad_out, cos_cache, sin_cache, grad_in, B, S, D, cache_dim);
}

// --- Embedding backward: scatter-add grad_weight[idx[i]] += grad_out[i,:] ---
__global__ void embedding_backward_kernel(
    const float* __restrict__ grad_out,  // [N, D]
    const float* __restrict__ indices,   // [N] (float-encoded ids)
    float* __restrict__ grad_weight,     // [V, D] accumulated (+=)
    int64_t N, int64_t D, int64_t V,
    int64_t padding_idx, int has_padding
) {
    int64_t total = N * D;
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (int64_t)blockDim.x * gridDim.x) {
        int64_t i = t / D;
        int64_t j = t % D;
        int64_t idx = static_cast<int64_t>(indices[i]);
        if (has_padding && idx == padding_idx) continue;
        if (idx >= 0 && idx < V) {
            atomicAdd(&grad_weight[idx * D + j], grad_out[t]);
        }
    }
}

void launch_embedding_backward(
    const float* grad_out, const float* indices, float* grad_weight,
    int64_t N, int64_t D, int64_t V, int64_t padding_idx, int has_padding,
    cudaStream_t stream
) {
    int blocks = get_num_blocks(N * D);
    embedding_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        grad_out, indices, grad_weight, N, D, V, padding_idx, has_padding);
}

// --- Cross-entropy forward: softmax + per-position loss (one block per row) ---
__global__ void cross_entropy_forward_kernel(
    const float* __restrict__ logits,   // [N, V]
    const float* __restrict__ targets,  // [N] (float-encoded ids)
    float* __restrict__ softmax_out,    // [N, V]
    float* __restrict__ losses,         // [N] (0 for invalid targets)
    float* __restrict__ valid,          // [N] (1/0)
    int64_t N, int64_t V
) {
    __shared__ float sdata[BLOCK_SIZE];
    for (int64_t i = blockIdx.x; i < N; i += gridDim.x) {
        int64_t offset = i * V;
        // 1) max reduction
        float local_max = -3.402823466e+38f;
        for (int64_t v = threadIdx.x; v < V; v += blockDim.x) {
            float val = logits[offset + v];
            local_max = val > local_max ? val : local_max;
        }
        sdata[threadIdx.x] = local_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                sdata[threadIdx.x] = sdata[threadIdx.x] > sdata[threadIdx.x + s]
                                       ? sdata[threadIdx.x] : sdata[threadIdx.x + s];
            __syncthreads();
        }
        float max_logit = sdata[0];
        __syncthreads();
        // 2) exp + sum reduction (store un-normalized exp)
        float local_sum = 0.0f;
        for (int64_t v = threadIdx.x; v < V; v += blockDim.x) {
            float e = __expf(logits[offset + v] - max_logit);
            softmax_out[offset + v] = e;
            local_sum += e;
        }
        sdata[threadIdx.x] = local_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        float sum_exp = sdata[0];
        __syncthreads();
        // 3) normalize
        float inv_sum = 1.0f / sum_exp;
        for (int64_t v = threadIdx.x; v < V; v += blockDim.x) {
            softmax_out[offset + v] *= inv_sum;
        }
        // 4) per-row loss
        if (threadIdx.x == 0) {
            int64_t tgt = static_cast<int64_t>(targets[i]);
            if (tgt >= 0 && tgt < V) {
                losses[i] = -logf(softmax_out[offset + tgt] + 1e-10f);
                valid[i] = 1.0f;
            } else {
                losses[i] = 0.0f;
                valid[i] = 0.0f;
            }
        }
        __syncthreads();
    }
}

void launch_cross_entropy_forward(
    const float* logits, const float* targets, float* softmax_out,
    float* losses, float* valid, int64_t N, int64_t V, cudaStream_t stream
) {
    int blocks = static_cast<int>(N > MAX_GRID_SIZE ? MAX_GRID_SIZE : N);
    if (blocks < 1) blocks = 1;
    cross_entropy_forward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        logits, targets, softmax_out, losses, valid, N, V);
}

// --- Cross-entropy backward: grad = (softmax - onehot) * scale ---
__global__ void cross_entropy_backward_kernel(
    const float* __restrict__ softmax_saved,  // [N, V]
    const float* __restrict__ targets,        // [N]
    float* __restrict__ grad_in,              // [N, V]
    float scale, int64_t N, int64_t V, int64_t ignore_index
) {
    int64_t total = N * V;
    for (int64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (int64_t)blockDim.x * gridDim.x) {
        int64_t i = t / V;
        int64_t c = t % V;
        int64_t tgt = static_cast<int64_t>(targets[i]);
        if (tgt == ignore_index || tgt < 0 || tgt >= V) {
            grad_in[t] = 0.0f;
        } else {
            float g = softmax_saved[t];
            if (c == tgt) g -= 1.0f;
            grad_in[t] = g * scale;
        }
    }
}

void launch_cross_entropy_backward(
    const float* softmax_saved, const float* targets, float* grad_in,
    float scale, int64_t N, int64_t V, int64_t ignore_index, cudaStream_t stream
) {
    int blocks = get_num_blocks(N * V);
    cross_entropy_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        softmax_saved, targets, grad_in, scale, N, V, ignore_index);
}

// ============================================================================
// Chunked parallel scan (forward + backward) — replaces the sequential-over-T
// kernels for training. Parallelism B*nC*D instead of B*D: each thread scans
// one chunk of C timesteps, then a tiny sequential carry pass links chunks,
// then a fix-up pass applies the carry. ~T/C times more parallelism.
// ============================================================================

#define SCAN_CHUNK 32

// Pass 1 (fwd): per-chunk local scan with zero carry-in.
//   gates[t] = clamp(base*(1+0.1*tanh(gl[t])))            (written out)
//   local_h[t] (written into output), prefix_g[t] = prod_{k<=t within chunk} g_k
//   P_c = prefix_g at chunk end, E_c = local_h at chunk end
__global__ void scan_fwd_chunk_kernel(
    const float* __restrict__ x, const float* __restrict__ gate_logits,
    const float* __restrict__ base_decay,
    float* __restrict__ output, float* __restrict__ gates,
    float* __restrict__ prefix_g,        // [B, T, D]
    float* __restrict__ P,               // [B, nC, D]
    float* __restrict__ E,               // [B, nC, D]
    int64_t B, int64_t T_len, int64_t D, int64_t nC,
    int fuse_gate_mul,                   // if 1: x is raw values, scan v*sigmoid(gl)
    int scan_vp_enabled,                 // 1: VP-скейл sqrt(1-g^2) (v4-эксперименты); 0: v1
    const float* __restrict__ resets     // [B, T] or nullptr; 1 -> g=0 (сброс state
                                         // на границе документа; bwd корректен
                                         // автоматически: он читает сохранённые gates)
) {
    int64_t total = B * nC * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t d = idx % D;
        int64_t c = (idx / D) % nC;
        int64_t b = idx / (D * nC);
        float base = base_decay[d];
        int64_t t0 = c * SCAN_CHUNK;
        int64_t t1 = t0 + SCAN_CHUNK; if (t1 > T_len) t1 = T_len;
        float h = 0.0f, pg = 1.0f;
        for (int64_t t = t0; t < t1; ++t) {
            int64_t off = (b * T_len + t) * D + d;
            float gl = gate_logits[off];
            float m = tanhf(gl) * 0.1f;
            float g = base * (1.0f + m);
            g = g < 0.5f ? 0.5f : (g > 0.9999f ? 0.9999f : g);
            if (resets != nullptr && resets[b * T_len + t] != 0.0f) g = 0.0f;
            gates[off] = g;
            float xv = x[off];
            // v4 EMA-параметризация: h = g*h + (1-g)*v*sigmoid(gl).
            // (1-g) ограничивает норму состояния масштабом входа — без него медленные
            // каналы (g->0.9999) интегрировали вход до x220 нормы (эхо-камера, замер
            // 2026-07-31). При reset g=0 -> (1-g)=1: чистый старт с полным входом.
            if (fuse_gate_mul) {
                float in_scale = scan_vp_enabled ? sqrtf(1.0f - g * g) : 1.0f;
                xv *= in_scale / (1.0f + __expf(-gl));
            }
            h = g * h + xv;
            pg *= g;
            output[off] = h;       // local h (carry applied in pass 3)
            prefix_g[off] = pg;
        }
        int64_t pc = (b * nC + c) * D + d;
        P[pc] = pg;
        E[pc] = h;
    }
}

// Pass 2 (fwd): sequential carry across chunks, per (b,d).
//   carry_in[c] = carry; carry = P[c]*carry + E[c]
__global__ void scan_carry_kernel(
    const float* __restrict__ P, const float* __restrict__ E,
    float* __restrict__ carry_in,        // [B, nC, D]
    int64_t B, int64_t D, int64_t nC
) {
    int64_t total = B * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t b = idx / D, d = idx % D;
        float carry = 0.0f;
        for (int64_t c = 0; c < nC; ++c) {
            int64_t pc = (b * nC + c) * D + d;
            carry_in[pc] = carry;
            carry = P[pc] * carry + E[pc];
        }
    }
}

// Pass 3 (fwd): output[t] += prefix_g[t] * carry_in[chunk(t)]
__global__ void scan_fixup_kernel(
    float* __restrict__ output, const float* __restrict__ prefix_g,
    const float* __restrict__ carry_in,
    int64_t B, int64_t T_len, int64_t D, int64_t nC
) {
    int64_t total = B * T_len * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t d = idx % D;
        int64_t t = (idx / D) % T_len;
        int64_t b = idx / (D * T_len);
        int64_t c = t / SCAN_CHUNK;
        float ci = carry_in[(b * nC + c) * D + d];
        if (ci != 0.0f) output[idx] += prefix_g[idx] * ci;
    }
}

// Backward recurrence: dh[t] = go[t] + g[t+1]*dh[t+1]  (dh[T]=0)
// Pass 1 (bwd): per-chunk reverse scan with zero carry-in from the right.
//   ldh[t] (written into grad_x), suffix_s[t] = prod_{k=t+1..chunkEnd+1} g_k
//   (uses g from the NEXT chunk at the boundary; for the last chunk s ends at g[T]=none -> uses 1)
__global__ void scan_bwd_chunk_kernel(
    const float* __restrict__ grad_out, const float* __restrict__ gates,
    float* __restrict__ grad_x,          // receives ldh
    float* __restrict__ suffix_s,        // [B, T, D]
    float* __restrict__ Pb,              // [B, nC, D]: suffix_s at chunk start
    float* __restrict__ Eb,              // [B, nC, D]: ldh at chunk start
    int64_t B, int64_t T_len, int64_t D, int64_t nC
) {
    int64_t total = B * nC * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t d = idx % D;
        int64_t c = (idx / D) % nC;
        int64_t b = idx / (D * nC);
        int64_t t0 = c * SCAN_CHUNK;
        int64_t t1 = t0 + SCAN_CHUNK; if (t1 > T_len) t1 = T_len;
        float dh = 0.0f, s = 1.0f;
        // seed: multiply by gate at (t1) when stepping from t1 to t1-1
        for (int64_t t = t1 - 1; t >= t0; --t) {
            int64_t off = (b * T_len + t) * D + d;
            float g_next = (t + 1 < T_len) ? gates[off + D] : 0.0f;  // g[t+1] (unused if t+1==T)
            if (t == t1 - 1) {
                // first step in chunk: dh = go[t] (carry from right is handled via s)
                dh = grad_out[off];
                s = (t + 1 < T_len) ? g_next : 0.0f;   // path factor to reach carry position
            } else {
                dh = grad_out[off] + gates[off + D] * dh;
                s = s * gates[off + D];
            }
            grad_x[off] = dh;        // ldh
            suffix_s[off] = s;
        }
        int64_t pc = (b * nC + c) * D + d;
        Pb[pc] = s;                  // factor from chunk start to right boundary
        Eb[pc] = dh;                 // ldh at chunk start
    }
}

// Pass 2 (bwd): sequential carry right-to-left, per (b,d).
//   carry_r[c] = dh at position t1(c) (i.e., first element of chunk c+1's dh)
__global__ void scan_bwd_carry_kernel(
    const float* __restrict__ Pb, const float* __restrict__ Eb,
    float* __restrict__ carry_r,         // [B, nC, D]
    int64_t B, int64_t D, int64_t nC
) {
    int64_t total = B * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t b = idx / D, d = idx % D;
        float carry = 0.0f;   // dh beyond the last chunk = 0
        for (int64_t c = nC - 1; c >= 0; --c) {
            int64_t pc = (b * nC + c) * D + d;
            carry_r[pc] = carry;
            carry = Eb[pc] + Pb[pc] * carry;   // dh at chunk start
        }
    }
}

// Pass 3 (bwd): dh[t] = ldh[t] + suffix_s[t]*carry_r[chunk]; grad_x=dh;
//   grad_gl[t] = dh[t]*h[t-1]*base*0.1*(1-tanh^2(gl[t])) for t>0 else 0.
__global__ void scan_bwd_fixup_kernel(
    float* __restrict__ grad_x, const float* __restrict__ suffix_s,
    const float* __restrict__ carry_r,
    const float* __restrict__ gate_logits, const float* __restrict__ base_decay,
    const float* __restrict__ hidden,
    float* __restrict__ grad_gl,
    const float* __restrict__ values,    // raw values (fused mode) or nullptr
    const float* __restrict__ gates_saved,  // сохранённые fwd-гейты (resets/EMA-консистентность)
    int64_t B, int64_t T_len, int64_t D, int64_t nC,
    int fuse_gate_mul, int scan_vp
) {
    int64_t total = B * T_len * D;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t d = idx % D;
        int64_t t = (idx / D) % T_len;
        int64_t b = idx / (D * T_len);
        int64_t c = t / SCAN_CHUNK;
        float dh = grad_x[idx] + suffix_s[idx] * carry_r[(b * nC + c) * D + d];
        float gl = gate_logits[idx];
        float dgl_gate = 0.0f;
        if (t > 0) {
            float tv = tanhf(gl);
            dgl_gate = dh * hidden[idx - D] * base_decay[d] * 0.1f * (1.0f - tv * tv);
        }
        if (fuse_gate_mul) {
            // v4 VP: h = g*h_prev + s(g)*u, s(g)=sqrt(1-g^2), u = v*sigmoid(gl).
            //   d_values = dh*s*sig
            //   d_gl     = dh*(h_prev - u*g/s)*dg/dgl + dh*s*v*sig'(gl)
            // g читаем из СОХРАНЁННЫХ gates (консистентно с fwd, включая resets).
            float sig = 1.0f / (1.0f + __expf(-gl));
            float v = values[idx];
            float tv2 = tanhf(gl);
            float g = gates_saved[idx];
            float s_vp = scan_vp ? sqrtf(1.0f - g * g) : 1.0f;
            float u = v * sig;
            float hp = (t > 0) ? hidden[idx - D] : 0.0f;
            // stop-gradient по s(g) (как в LRU): аналитический член -u*g/s взрывал
            // градиент гейта x71 у длинных каналов -> клип душил сеть (стагнация 9.7).
            dgl_gate = dh * hp * base_decay[d] * 0.1f * (1.0f - tv2 * tv2);
            grad_x[idx] = dh * s_vp * sig;             // becomes d_values
            grad_gl[idx] = dgl_gate + dh * s_vp * v * sig * (1.0f - sig);
        } else {
            grad_x[idx] = dh;
            grad_gl[idx] = dgl_gate;
        }
    }
}

// --- Sum of squares (for grad-norm clipping): atomicAdd into result[0] ---
__global__ void sumsq_kernel(const float* __restrict__ x, float* __restrict__ result,
                             int64_t n) {
    __shared__ float sdata[BLOCK_SIZE];
    float local = 0.0f;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        float v = x[i];
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, sdata[0]);
}

void launch_sumsq_accumulate(const float* x, float* result, int64_t n,
                             cudaStream_t stream) {
    int blocks = get_num_blocks(n);
    sumsq_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(x, result, n);
}

} // namespace cuda
} // namespace at
