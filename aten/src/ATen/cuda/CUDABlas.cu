// ============================================================================
// CUDA BLAS Kernels for PromeTorch
// ============================================================================
// Matrix multiplication and linear algebra operations

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// Include the header to get ATEN_CUDA_API macro for proper DLL export
#include "aten/src/ATen/cuda/CUDAOps.h"
#include "aten/src/ATen/cuda/CuBLASHandle.h"

namespace at {
namespace cuda {

// ============================================================================
// Tiled Matrix Multiplication Configuration
// ============================================================================

// Tile sizes for shared memory
constexpr int TILE_SIZE = 32;
constexpr int TILE_K = 8;  // K dimension tile for register blocking

// ============================================================================
// GEMM: C = alpha * A @ B + beta * C
// ============================================================================

// Basic tiled matrix multiplication
// A: [M, K], B: [K, N], C: [M, N]
__global__ void gemm_nn_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * sum;
        }
    }
}

// A^T @ B: A: [K, M], B: [K, N], C: [M, N]
__global__ void gemm_tn_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // A^T: access A[k, m] = A[k * M + m], we want A^T[m, k]
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[(t * TILE_SIZE + tx) * M + row];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * sum;
        }
    }
}

// A @ B^T: A: [M, K], B: [N, K], C: [M, N]
__global__ void gemm_nt_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // B^T: access B[n, k] = B[n * K + k], we want B^T[k, n]
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[col * K + t * TILE_SIZE + ty];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * sum;
        }
    }
}

// A^T @ B^T: A: [K, M], B: [N, K], C: [M, N]
__global__ void gemm_tt_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[(t * TILE_SIZE + tx) * M + row];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[col * K + t * TILE_SIZE + ty];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * sum;
        }
    }
}

// ============================================================================
// Batched GEMM
// ============================================================================

__global__ void batched_gemm_nn_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch, int M, int N, int K,
    float alpha, float beta
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int b = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Offset pointers for batch
    const float* A_batch = A + b * M * K;
    const float* B_batch = B + b * K * N;
    float* C_batch = C + b * M * N;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A_batch[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B_batch[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f) {
            C_batch[row * N + col] = alpha * sum + beta * C_batch[row * N + col];
        } else {
            C_batch[row * N + col] = alpha * sum;
        }
    }
}

// ============================================================================
// Matrix-Vector Multiplication: y = A @ x
// ============================================================================

__global__ void gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int N
) {
    __shared__ float shared[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row < M) {
        float sum = 0.0f;

        // Each thread handles multiple elements
        for (int j = tid; j < N; j += blockDim.x) {
            sum += A[row * N + j] * x[j];
        }

        shared[tid] = sum;
        __syncthreads();

        // Block reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            y[row] = shared[0];
        }
    }
}

// ============================================================================
// Vector Dot Product
// ============================================================================

__global__ void dot_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    int64_t n
) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int64_t i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

// ============================================================================
// Outer Product: C = a @ b^T
// ============================================================================

__global__ void outer_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ C,
    int M, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] = a[row] * b[col];
    }
}

// ============================================================================
// Addmm: C = beta * C + alpha * A @ B
// ============================================================================

// This is essentially GEMM, aliased for clarity
void launch_addmm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b,
    cudaStream_t stream
);

// ============================================================================
// Transpose Kernel
// ============================================================================

__global__ void transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Write transposed
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

// ============================================================================
// BF16 mixed-precision GEMM path (A100+): convert inputs to bf16 on the fly,
// run cublasGemmEx on bf16 tensor cores (312 TFLOPS vs 156 TF32), accumulate
// and output in FP32 — numerically the torch-autocast recipe, so the rest of
// the framework (autograd, kernels) is untouched. PT_NO_BF16=1 reverts.
// ============================================================================

__global__ void f32_to_bf16_kernel(const float* __restrict__ in,
                                   __nv_bfloat16* __restrict__ out, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)blockDim.x * gridDim.x) {
        out[i] = __float2bfloat16(in[i]);
    }
}

static __nv_bfloat16* bf16_scratch(int slot, size_t elems) {
    static __nv_bfloat16* bufs[2] = {nullptr, nullptr};
    static size_t sizes[2] = {0, 0};
    if (sizes[slot] < elems) {
        if (bufs[slot]) cudaFree(bufs[slot]);
        cudaMalloc(&bufs[slot], elems * sizeof(__nv_bfloat16));
        sizes[slot] = elems;
    }
    return bufs[slot];
}

static bool bf16_gemm_enabled() {
    static int v = -1;
    if (v < 0) {
        if (getenv("PT_NO_BF16")) { v = 0; }
        else {
            int dev = 0; cudaGetDevice(&dev);
            cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
            v = (prop.major >= 8) ? 1 : 0;   // Ampere+ has bf16 tensor cores
        }
    }
    return v == 1;
}

// launch_gemm — primary matmul dispatch.
// Uses cuBLAS sgemm when available (tensor cores, tuned kernels). This is what the
// README's "CUDA GEMM" claim really means in this repo. The hand-written tiled
// kernels above (gemm_nn_kernel / gemm_tn_kernel / gemm_nt_kernel) are kept as a
// portable fallback and pedagogical reference; invoke them via launch_gemm_native
// below.
void launch_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b,
    cudaStream_t stream
) {
    cublasHandle_t handle = CuBLASHandle::get();
    cublasSetStream(handle, stream);

    cublasOperation_t op_a = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    // cuBLAS column-major: C(N,M) = B(N,K) @ A(K,M), giving row-major C = A @ B.
    int lda = trans_b ? K : N;
    int ldb = trans_a ? M : K;
    int ldc = N;

    // BF16 tensor-core path for compute-heavy shapes. Conversion cost is 3 memory
    // sweeps of the inputs; only worth it when the GEMM itself dominates.
    // Measured on A100 (PIR-107M): small 640x640 GEMMs are FASTER on plain TF32
    // (conversion overhead loses); bf16 wins only on large shapes (lm_head-class,
    // >= ~8 GFLOP per GEMM). Threshold keeps bf16 exactly where it pays.
    if (bf16_gemm_enabled() && (long long)M * N * K >= (4LL << 30) && K >= 64) {
        int64_t nA = (int64_t)M * K;
        int64_t nB = (int64_t)K * N;
        __nv_bfloat16* A16 = bf16_scratch(0, nA);
        __nv_bfloat16* B16 = bf16_scratch(1, nB);
        int ba = static_cast<int>((nA + 255) / 256); if (ba > 65535) ba = 65535;
        int bb = static_cast<int>((nB + 255) / 256); if (bb > 65535) bb = 65535;
        f32_to_bf16_kernel<<<ba, 256, 0, stream>>>(A, A16, nA);
        f32_to_bf16_kernel<<<bb, 256, 0, stream>>>(B, B16, nB);
        cublasStatus_t st = cublasGemmEx(
            handle, op_a, op_b,
            N, M, K,
            &alpha,
            B16, CUDA_R_16BF, lda,
            A16, CUDA_R_16BF, ldb,
            &beta,
            C, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st == CUBLAS_STATUS_SUCCESS) return;
        // fall through to FP32 sgemm on any failure
    }

    cublasSgemm(handle, op_a, op_b,
                N, M, K,
                &alpha,
                B, lda,
                A, ldb,
                &beta,
                C, ldc);
}

// launch_gemm_native — hand-written tiled CUDA kernel. No cuBLAS dependency.
// Useful on platforms without cuBLAS and for validating correctness of our kernels.
// Supported: NN, TN, NT (no TT). Performance is below cuBLAS but functional.
void launch_gemm_native(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b,
    cudaStream_t stream
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    if (!trans_a && !trans_b) {
        gemm_nn_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (trans_a && !trans_b) {
        gemm_tn_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (!trans_a && trans_b) {
        gemm_nt_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        // Both transposed: fall back to cuBLAS.
        launch_gemm(A, B, C, M, N, K, alpha, beta, trans_a, trans_b, stream);
    }
}

void launch_batched_gemm(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    // Use cuBLAS strided batched GEMM
    cublasHandle_t handle = CuBLASHandle::get();
    cublasSetStream(handle, stream);

    long long int strideA = (long long int)M * K;
    long long int strideB = (long long int)K * N;
    long long int strideC = (long long int)M * N;

    // cuBLAS col-major: C^T = B^T @ A^T
    cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N, strideB,
        A, K, strideA,
        &beta,
        C, N, strideC,
        batch);
}

void launch_gemv(
    const float* A, const float* x, float* y,
    int M, int N,
    cudaStream_t stream
) {
    gemv_kernel<<<M, 256, 0, stream>>>(A, x, y, M, N);
}

void launch_dot(
    const float* a, const float* b, float* result,
    int64_t n,
    cudaStream_t stream
) {
    cudaMemsetAsync(result, 0, sizeof(float), stream);
    int blocks = (n + 255) / 256;
    blocks = min(blocks, 1024);
    dot_kernel<<<blocks, 256, 0, stream>>>(a, b, result, n);
}

void launch_outer(
    const float* a, const float* b, float* C,
    int M, int N,
    cudaStream_t stream
) {
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    dim3 threads(16, 16);
    outer_kernel<<<blocks, threads, 0, stream>>>(a, b, C, M, N);
}

void launch_transpose(
    const float* input, float* output,
    int rows, int cols,
    cudaStream_t stream
) {
    dim3 blocks((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    transpose_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols);
}

void launch_addmm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b,
    cudaStream_t stream
) {
    launch_gemm(A, B, C, M, N, K, alpha, beta, trans_a, trans_b, stream);
}

} // namespace cuda
} // namespace at
