// ============================================================================
// LinQAllocator.cpp — Singleton implementation (DLL-safe)
// ============================================================================

#define BUILDING_ATEN_LINQ
#include "c10/linq/LinQAllocator.h"

namespace c10 {
namespace linq {

// Global singleton — MUST be in .cpp to avoid DLL boundary issues
static LinQAllocator g_linq_allocator;

ATEN_LINQ_API LinQAllocator& LinQAllocator::get() {
    return g_linq_allocator;
}

ATEN_LINQ_API void LinQAllocator::deleter(void* /*data*/, void* ctx) {
    if (ctx == nullptr) return;
    Block* block = static_cast<Block*>(ctx);
    g_linq_allocator.raw_deallocate(block->ptr);
}

static bool g_linq_allocator_registered = false;

ATEN_LINQ_API void register_linq_allocator() {
    if (!g_linq_allocator_registered) {
        AllocatorRegistry::get().registerAllocator(
            DeviceType::PrivateUse2, &g_linq_allocator);
        g_linq_allocator_registered = true;
    }
}

} // namespace linq
} // namespace c10
