#pragma once
// ============================================================================
// NMCardAllocator.h - Memory Allocator for NM Card Mini Backend
// ============================================================================
// Emulator mode: uses aligned host RAM, tagged with device PrivateUse1 (nmcard)
// Pattern follows CUDAAllocator: singleton in .cpp (DLL safety!)

#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/macros/Macros.h"

#include <mutex>
#include <map>
#include <set>
#include <cstdlib>

// Export/Import macro for NMCard allocator
#ifndef ATEN_NMCARD_API
#if defined(PT_PLATFORM_WINDOWS) || defined(_MSC_VER)
    #if defined(ATEN_NMCARD_EXPORTS)
        #define ATEN_NMCARD_API __declspec(dllexport)
    #else
        #define ATEN_NMCARD_API __declspec(dllimport)
    #endif
#else
    #define ATEN_NMCARD_API __attribute__((visibility("default")))
#endif
#endif

namespace c10 {
namespace nmcard {

// ============================================================================
// Block structure for caching allocator
// ============================================================================

struct Block {
    void* ptr;
    size_t size;
    bool allocated;

    Block(void* p, size_t s) : ptr(p), size(s), allocated(true) {}
};

// ============================================================================
// NMCard Caching Allocator (Emulator Mode)
// ============================================================================
// In emulator mode: data lives in host RAM but is tagged as device=nmcard
// This mirrors the NMC4's DDR3L memory model without actual hardware

class ATEN_NMCARD_API NMCardAllocator : public Allocator {
public:
    // Singleton accessor - implemented in NMCardAllocator.cpp
    static NMCardAllocator& get();

    DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return DataPtr(nullptr, nullptr, &null_deleter, Device(DeviceType::PrivateUse1, 0));
        }

        // Round up to 64-byte alignment (cache line)
        size_t alloc_size = (nbytes + 63) & ~63;

        std::lock_guard<std::mutex> lock(mutex_);

        // Try to find a cached block
        Block* block = find_free_block(alloc_size);

        if (block == nullptr) {
            void* ptr = nullptr;
            #if defined(_MSC_VER)
                ptr = _aligned_malloc(alloc_size, 64);
            #else
                int ret = posix_memalign(&ptr, 64, alloc_size);
                if (ret != 0) ptr = nullptr;
            #endif

            if (ptr == nullptr) {
                PT_OOM_ERROR("NMCard emulator: failed to allocate ", nbytes, " bytes");
            }

            block = new Block(ptr, alloc_size);
            ptr_to_block_[ptr] = block;
            allocated_bytes_ += alloc_size;
        } else {
            block->allocated = true;
            cached_bytes_ -= block->size;
        }

        return DataPtr(
            block->ptr,
            block,
            &NMCardAllocator::deleter,
            Device(DeviceType::PrivateUse1, 0)
        );
    }

    void raw_deallocate(void* ptr) override {
        if (ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = ptr_to_block_.find(ptr);
        if (it != ptr_to_block_.end()) {
            Block* block = it->second;
            block->allocated = false;
            free_blocks_.insert(block);
            cached_bytes_ += block->size;
        }
    }

    DeleterFn raw_deleter() const override {
        return &NMCardAllocator::Delete;
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto* block : free_blocks_) {
            #if defined(_MSC_VER)
                _aligned_free(block->ptr);
            #else
                free(block->ptr);
            #endif
            ptr_to_block_.erase(block->ptr);
            allocated_bytes_ -= block->size;
            delete block;
        }
        free_blocks_.clear();
        cached_bytes_ = 0;
    }

    size_t get_allocated_memory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocated_bytes_;
    }

    size_t get_cached_memory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_bytes_;
    }

    NMCardAllocator() = default;
    ~NMCardAllocator() = default;

private:
    Block* find_free_block(size_t size) {
        Block* best = nullptr;
        for (auto* block : free_blocks_) {
            if (block->size >= size) {
                if (best == nullptr || block->size < best->size) {
                    best = block;
                }
            }
        }
        if (best != nullptr) {
            free_blocks_.erase(best);
        }
        return best;
    }

    void free_block(Block* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        block->allocated = false;
        free_blocks_.insert(block);
        cached_bytes_ += block->size;
    }

    // Static deleters - implemented in NMCardAllocator.cpp
    static void deleter(void* data, void* ctx);
    static void Delete(void* data, void* ctx);
    static void null_deleter(void* data, void* ctx);

    mutable std::mutex mutex_;
    std::map<void*, Block*> ptr_to_block_;
    std::set<Block*> free_blocks_;
    size_t allocated_bytes_ = 0;
    size_t cached_bytes_ = 0;
};

// Register NMCard allocator with AllocatorRegistry (DLL-exported version)
ATEN_NMCARD_API void register_nmcard_allocator();

// Inline registration — call this from your main() to ensure the allocator
// is registered in the CALLER's AllocatorRegistry instance.
// Needed because AllocatorRegistry::get() can have separate instances per DLL.
inline void register_nmcard_allocator_local() {
    auto& alloc = NMCardAllocator::get();
    AllocatorRegistry::get().registerAllocator(DeviceType::PrivateUse1, &alloc);
}

} // namespace nmcard
} // namespace c10
