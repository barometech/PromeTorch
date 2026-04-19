#pragma once
// ============================================================================
// MPSAllocator.h — Memory Allocator for Apple Metal Performance Shaders
// ============================================================================
// Backed by id<MTLDevice> newBufferWithLength:options:. On non-Apple
// platforms (Windows, Linux/Elbrus) the allocator is NOT compiled and any
// attempt to use the MPS device throws a clear error.
//
// Pattern mirrors CUDAAllocator and NMCardAllocator:
//   * Singleton lives in MPSAllocator.mm (.cpp-equivalent for Obj-C++)
//     to avoid DLL boundary / duplicate-instance issues on Windows.
//   * Caching: free blocks are kept in a size-ordered multimap; allocate()
//     first tries to reuse a cached block before calling out to Metal.
//
// Build: enable with -DPT_USE_MPS=ON. CMake links -framework Metal and
// -framework MetalPerformanceShadersGraph. Non-Apple platforms skip the
// whole subtree (the object file is not added to any target).
// ============================================================================

#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/macros/Macros.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>

// Export/import macro (parallel to ATEN_NMCARD_API). Only meaningful on
// Windows, but MPS is Apple-only anyway — the macro is a no-op everywhere.
#ifndef ATEN_MPS_API
#if defined(PT_PLATFORM_WINDOWS) || defined(_MSC_VER)
    #if defined(ATEN_MPS_EXPORTS)
        #define ATEN_MPS_API __declspec(dllexport)
    #else
        #define ATEN_MPS_API __declspec(dllimport)
    #endif
#else
    #define ATEN_MPS_API __attribute__((visibility("default")))
#endif
#endif

namespace c10 {
namespace mps {

// ============================================================================
// Block — caching record. `raw` is an `id<MTLBuffer>` on Apple (opaque void*
// in this header so we never pull Obj-C runtime into C++ TUs).
// ============================================================================

struct MPSBlock {
    void*  contents;   // [buffer contents] — host-visible pointer (unified memory)
    void*  buffer;     // id<MTLBuffer>, retained; null on non-Apple builds
    size_t size;
    bool   allocated;

    MPSBlock(void* c, void* b, size_t s)
        : contents(c), buffer(b), size(s), allocated(true) {}
};

// ============================================================================
// MPSAllocator
// ============================================================================

class ATEN_MPS_API MPSAllocator : public Allocator {
public:
    static MPSAllocator& get();

    DataPtr allocate(size_t nbytes) override;
    void    raw_deallocate(void* ptr) override;
    DeleterFn raw_deleter() const override;

    // Free all cached (non-live) blocks. Live allocations stay valid.
    void empty_cache();

    size_t get_allocated_memory() const;
    size_t get_cached_memory()    const;

    MPSAllocator()  = default;
    ~MPSAllocator() = default;

private:
    // Raw alloc via Metal (implemented in MPSAllocator.mm on Apple; on
    // non-Apple we refuse at the caller — see MPSDevice.h).
    void*  metal_new_buffer(size_t nbytes, void** out_contents);
    void   metal_release_buffer(void* buffer);

    MPSBlock* find_free_block(size_t size);
    void      free_block(MPSBlock* block);

    // Deleters for DataPtr — static so they have C-function linkage.
    static void deleter     (void* data, void* ctx);
    static void null_deleter(void* data, void* ctx);

    mutable std::mutex              mutex_;
    std::map<void*, MPSBlock*>      ptr_to_block_;   // keyed by contents ptr
    std::multimap<size_t, MPSBlock*> free_blocks_;   // size-ordered cache
    size_t allocated_bytes_ = 0;
    size_t cached_bytes_    = 0;
};

// DLL-exported registration. Non-Apple builds leave this as an undefined
// reference — and the CMake glue only calls it when PT_USE_MPS=ON.
ATEN_MPS_API void register_mps_allocator();

// Inline variant so code that links statically can still force registration
// in its own AllocatorRegistry copy (same trick as NMCardAllocator).
inline void register_mps_allocator_local() {
    auto& alloc = MPSAllocator::get();
    AllocatorRegistry::get().registerAllocator(DeviceType::MPS, &alloc);
}

} // namespace mps
} // namespace c10
