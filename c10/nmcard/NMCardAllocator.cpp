// ============================================================================
// NMCardAllocator.cpp - Singleton Implementation
// ============================================================================
// CRITICAL: Singleton MUST be in .cpp to avoid DLL boundary issues!
// Same pattern as CUDAAllocator.cpp

#include "c10/nmcard/NMCardAllocator.h"

namespace c10 {
namespace nmcard {

// Global singleton
static NMCardAllocator g_nmcard_allocator;

ATEN_NMCARD_API NMCardAllocator& NMCardAllocator::get() {
    return g_nmcard_allocator;
}

ATEN_NMCARD_API void NMCardAllocator::deleter(void* /*data*/, void* ctx) {
    if (ctx == nullptr) return;
    Block* block = static_cast<Block*>(ctx);
    g_nmcard_allocator.free_block(block);
}

ATEN_NMCARD_API void NMCardAllocator::Delete(void* data, void* ctx) {
    deleter(data, ctx);
}

ATEN_NMCARD_API void NMCardAllocator::null_deleter(void*, void*) {
    // Do nothing - used for zero-size allocations
}

// Registration
static bool g_nmcard_allocator_registered = false;

ATEN_NMCARD_API void register_nmcard_allocator() {
    if (!g_nmcard_allocator_registered) {
        AllocatorRegistry::get().registerAllocator(
            DeviceType::PrivateUse1,
            &g_nmcard_allocator
        );
        g_nmcard_allocator_registered = true;
    }
}

} // namespace nmcard
} // namespace c10
