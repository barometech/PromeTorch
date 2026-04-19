// ============================================================================
// PromeTorch — HIP / ROCm compatibility shim
// ============================================================================
// This header is auto-included by scripts/hipify.sh into every translated
// source file. It provides safety-net aliases for CUDA tokens that slipped
// past hipify-perl (or our sed fallback), so a single CUDA source tree can
// compile unchanged against HIP's API surface.
//
// Scope:
//   1. Runtime & memory API tokens that hipify sometimes misses in macros.
//   2. cuBLAS -> rocBLAS type/function aliases.
//   3. Warp-sync intrinsics: on HIP the `*_sync` variants do not exist;
//      the mask is implicit because AMD wavefronts are lock-step in hardware.
//   4. WARP_SIZE reconciliation (AMD = 64, NVIDIA = 32). Kernels that assume
//      warpSize=32 should include this header and switch on __HIP_PLATFORM_AMD__.
//
// This file is safe to include from both CUDA and HIP builds: every macro is
// guarded by __HIP_PLATFORM_AMD__ / __HIPCC__ so NVIDIA builds see a no-op.
// ============================================================================

#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---------------------------------------------------------------------------
// Runtime types & enums — most already aliased by hipify-perl, but these
// catch anything translated via macro expansion where perl can't see it.
// ---------------------------------------------------------------------------
#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif
#ifndef cudaEvent_t
#define cudaEvent_t  hipEvent_t
#endif
#ifndef cudaError_t
#define cudaError_t  hipError_t
#endif
#ifndef cudaSuccess
#define cudaSuccess  hipSuccess
#endif

#ifndef cudaMalloc
#define cudaMalloc              hipMalloc
#define cudaFree                hipFree
#define cudaMemcpy              hipMemcpy
#define cudaMemcpyAsync         hipMemcpyAsync
#define cudaMemset              hipMemset
#define cudaMemsetAsync         hipMemsetAsync
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamCreate        hipStreamCreate
#define cudaStreamSynchronize   hipStreamSynchronize
#define cudaStreamDestroy       hipStreamDestroy
#define cudaDeviceSynchronize   hipDeviceSynchronize
#define cudaGetLastError        hipGetLastError
#define cudaGetErrorString      hipGetErrorString
#define cudaSetDevice           hipSetDevice
#define cudaGetDevice           hipGetDevice
#define cudaGetDeviceCount      hipGetDeviceCount
#define cudaDeviceProp          hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaFuncSetAttribute    hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
        hipFuncAttributeMaxDynamicSharedMemorySize
#endif

// ---------------------------------------------------------------------------
// cuBLAS -> rocBLAS. We only alias what PromeTorch actually calls
// (CUDABlas.cu + Linear path). Extend as needed.
// ---------------------------------------------------------------------------
#include <rocblas/rocblas.h>

#ifndef cublasHandle_t
#define cublasHandle_t           rocblas_handle
#define cublasStatus_t           rocblas_status
#define CUBLAS_STATUS_SUCCESS    rocblas_status_success
#define CUBLAS_OP_N              rocblas_operation_none
#define CUBLAS_OP_T              rocblas_operation_transpose
#define cublasCreate             rocblas_create_handle
#define cublasDestroy            rocblas_destroy_handle
#define cublasSetStream          rocblas_set_stream
#define cublasSgemm              rocblas_sgemm
#define cublasSgemv              rocblas_sgemv
#define cublasSscal              rocblas_sscal
#define cublasSaxpy              rocblas_saxpy
#endif

// ---------------------------------------------------------------------------
// Warp intrinsics.
// NVIDIA (post-Volta) requires the *_sync variants with an active-lane mask.
// HIP/ROCm has no mask parameter — wavefronts execute lock-step. These
// macros collapse *_sync calls down to the mask-less HIP form so kernel code
// written against the CUDA Volta+ API compiles against both backends.
//
// Note: AMD __ballot returns a 64-bit lane mask (wavefront=64), so callers
// that store the result should prefer `uint64_t` (or use the WARP_MASK
// constant). On NVIDIA this remains 32-bit — handle via platform #ifdef at
// the call site. CUDAQuantGemv.cu already does this correctly.
// ---------------------------------------------------------------------------
#ifndef __shfl_sync
#define __shfl_sync(mask, val, src, ...)       __shfl((val), (src), ##__VA_ARGS__)
#define __shfl_up_sync(mask, val, off, ...)    __shfl_up((val), (off), ##__VA_ARGS__)
#define __shfl_down_sync(mask, val, off, ...)  __shfl_down((val), (off), ##__VA_ARGS__)
#define __shfl_xor_sync(mask, val, lane, ...)  __shfl_xor((val), (lane), ##__VA_ARGS__)
#define __ballot_sync(mask, pred)              __ballot((pred))
#define __any_sync(mask, pred)                 __any((pred))
#define __all_sync(mask, pred)                 __all((pred))
#define __activemask()                          0xFFFFFFFFFFFFFFFFULL
#endif

// warpSize fallback for device code. AMD wavefront is 64 (gfx9/gfx10/gfx11
// compute) — kernels that hard-code 32 must use this constant.
#ifndef PT_WARP_SIZE
#  if defined(__HIP_PLATFORM_AMD__)
#    define PT_WARP_SIZE 64
#  else
#    define PT_WARP_SIZE 32
#  endif
#endif

#else
// CUDA build: no-op. Kernels see the real CUDA tokens.
#ifndef PT_WARP_SIZE
#define PT_WARP_SIZE 32
#endif
#endif  // __HIPCC__ / __HIP_PLATFORM_AMD__
