#pragma once
// ============================================================================
// MPSKernels.h — Public C++ entry points to MPS kernels.
// ============================================================================
// Implementations are in MPSKernels.mm (Obj-C++, Apple-only). The functions
// below take raw pointers into host-visible MTLBuffer contents (returned by
// MPSAllocator). They commit a command buffer to the shared MTLCommandQueue
// and return immediately; the caller is responsible for calling
// `at::mps::MPSDevice::get().synchronize()` before reading results on CPU.
//
// On non-Apple builds the .mm file is excluded from the build and these
// symbols remain undefined — CMake only adds the MPS library when
// PT_USE_MPS=ON, so nothing references them.
// ============================================================================

#include <cstddef>
#include <cstdint>

namespace at {
namespace mps {

// Element-wise: c[i] = a[i] + b[i]
void launch_add_mps(const float* a, const float* b, float* c, std::size_t numel);

// Element-wise: c[i] = a[i] * b[i]
void launch_mul_mps(const float* a, const float* b, float* c, std::size_t numel);

// Element-wise: out[i] = max(0, in[i])
void launch_relu_mps(const float* in, float* out, std::size_t numel);

// Matmul (row-major): C[M,N] = A[M,K] * B[K,N]  (MPSMatrixMultiplication)
void launch_matmul_mps(const float* A, const float* B, float* C,
                       int M, int N, int K);

} // namespace mps
} // namespace at
