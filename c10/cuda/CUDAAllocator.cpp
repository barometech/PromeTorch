// ============================================================================
// CUDAAllocator.cpp - Implementation of CUDA Caching Allocator
// ============================================================================
// CRITICAL: The singleton MUST be in a .cpp file to avoid DLL boundary issues!
// If the singleton is in a header (inline), each DLL gets its own copy,
// causing heap corruption when one DLL allocates and another frees.
//
// This is the same pattern used in TensorImpl.cpp for AutogradMetaFactory.
// ============================================================================

#include "c10/cuda/CUDAAllocator.h"
#include <iostream>

namespace c10 {
namespace cuda {

// ============================================================================
// Global Singleton Instance - ONLY ONE COPY in the entire process
// ============================================================================

// The allocator instance - lives in c10.dll (or aten_cuda.dll depending on build)
static CUDACachingAllocator g_cuda_allocator;

// ============================================================================
// Exported Singleton Accessor
// ============================================================================

ATEN_CUDA_API CUDACachingAllocator& CUDACachingAllocator::get() {
    return g_cuda_allocator;
}

// ============================================================================
// Exported Static Methods (Deleters)
// ============================================================================
// These must be in .cpp because they call get() which must resolve to the
// single global instance, not a per-DLL instance.

ATEN_CUDA_API void CUDACachingAllocator::deleter(void* /*data*/, void* ctx) {
    if (ctx == nullptr) return;
    // Check if allocator was shutdown - if so, memory is already freed
    if (g_cuda_allocator.is_shutdown()) {
        return;  // Don't try to free - allocator already cleaned up
    }
    Block* block = static_cast<Block*>(ctx);
    // CRITICAL: Use the global singleton, not a local static!
    g_cuda_allocator.free_block(block);
}

ATEN_CUDA_API void CUDACachingAllocator::Delete(void* data, void* ctx) {
    deleter(data, ctx);
}

ATEN_CUDA_API void CUDACachingAllocator::null_deleter(void*, void*) {
    // Do nothing - used for zero-size allocations
}

// ============================================================================
// Registration Helper
// ============================================================================

static bool g_cuda_allocator_registered = false;

ATEN_CUDA_API void register_cuda_allocator() {
    if (!g_cuda_allocator_registered) {
        AllocatorRegistry::get().registerAllocator(
            DeviceType::CUDA,
            &g_cuda_allocator
        );
        g_cuda_allocator_registered = true;
    }
}

// ============================================================================
// Global CUDA Shutdown
// ============================================================================

ATEN_CUDA_API void cuda_shutdown() {
    // Synchronize all GPU operations first
    cudaDeviceSynchronize();

    // Shutdown the caching allocator
    g_cuda_allocator.shutdown();

    // Note: cudaDeviceReset() is NOT called here because it can cause crashes
    // if there are any remaining CUDA resources (streams, events, etc.)
    // The driver will clean up when the process exits anyway.
}

} // namespace cuda
} // namespace c10
