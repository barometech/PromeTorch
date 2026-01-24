// ============================================================================
// CUDA BLAS Kernels for PromeTorch
// ============================================================================
// Matrix multiplication and linear algebra operations

#include <cuda_runtime.h>
#include <cmath>

// Include the header to get ATEN_CUDA_API macro for proper DLL export
#include "aten/src/ATen/cuda/CUDAOps.h"

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

void launch_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    bool trans_a, bool trans_b,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);

    if (!trans_a && !trans_b) {
        gemm_nn_kernel<<<blocks, threads, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (trans_a && !trans_b) {
        gemm_tn_kernel<<<blocks, threads, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else if (!trans_a && trans_b) {
        gemm_nt_kernel<<<blocks, threads, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        gemm_tt_kernel<<<blocks, threads, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

void launch_batched_gemm(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    batched_gemm_nn_kernel<<<blocks, threads, 0, stream>>>(A, B, C, batch, M, N, K, alpha, beta);
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
