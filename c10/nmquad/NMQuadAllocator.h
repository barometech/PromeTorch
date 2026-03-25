#pragma once
// ============================================================================
// NMQuadAllocator.h - Memory Allocator for NM QUAD Backend (4x NM6408)
// ============================================================================
// Host-side caching allocator. Tensors "on NM QUAD" live in host RAM but are
// tagged with device PrivateUse3 (nmquad). device_index 0..3 = chip ID.
// Data is transferred to DDR via PL_WriteMemBlock before kernel execution.

#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/macros/Macros.h"

#include <mutex>
#include <map>
#include <cstdlib>

namespace c10 {
namespace nmquad {

constexpr int NUM_CHIPS = 4;

struct Block {
    void* ptr;
    size_t size;
    bool allocated;
    Block(void* p, size_t s) : ptr(p), size(s), allocated(true) {}
};

// ============================================================================
// NMQuad Caching Allocator
// ============================================================================
// Data lives in host RAM, tagged as device=nmquad:N (N=chip_id 0..3)
// Before NMC kernel execution, data is DMA'd to chip DDR via PL_WriteMemBlock

class NMQuadAllocator : public Allocator {
public:
    static NMQuadAllocator& get() {
        static NMQuadAllocator instance;
        return instance;
    }

    DataPtr allocate(size_t nbytes) override {
        return allocate_on_chip(nbytes, 0);
    }

    DataPtr allocate_on_chip(size_t nbytes, int chip_id) {
        if (nbytes == 0) {
            return DataPtr(nullptr, nullptr, &null_deleter,
                          Device(DeviceType::PrivateUse3, static_cast<DeviceIndex>(chip_id)));
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Try to reuse a cached block
        auto& cache = free_blocks_[chip_id];
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->second >= nbytes) {
                void* ptr = it->first;
                size_t block_size = it->second;
                cache.erase(it);
                allocated_[ptr] = block_size;
                return DataPtr(ptr, ptr, &raw_deleter,
                              Device(DeviceType::PrivateUse3, static_cast<DeviceIndex>(chip_id)));
            }
        }

        // Allocate new aligned block
        void* ptr = nullptr;
#ifdef _MSC_VER
        ptr = _aligned_malloc(nbytes, 64);
#else
        if (posix_memalign(&ptr, 64, nbytes) != 0) ptr = nullptr;
#endif
        if (!ptr) {
            throw std::bad_alloc();
        }

        allocated_[ptr] = nbytes;
        total_allocated_ += nbytes;

        return DataPtr(ptr, ptr, &raw_deleter,
                      Device(DeviceType::PrivateUse3, static_cast<DeviceIndex>(chip_id)));
    }

    void raw_deallocate(void* ptr) override {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocated_.find(ptr);
        if (it != allocated_.end()) {
            size_t size = it->second;
            allocated_.erase(it);
            // Cache for reuse (chip 0 by default — we don't track which chip)
            free_blocks_[0][ptr] = size;
        }
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < NUM_CHIPS; ++i) {
            for (auto& [ptr, size] : free_blocks_[i]) {
#ifdef _MSC_VER
                _aligned_free(ptr);
#else
                free(ptr);
#endif
                total_allocated_ -= size;
            }
            free_blocks_[i].clear();
        }
    }

    size_t total_allocated() const { return total_allocated_; }

    DeleterFn raw_deleter() const override {
        return &raw_deleter;
    }

private:
    NMQuadAllocator() = default;
    ~NMQuadAllocator() { empty_cache(); }

    static void null_deleter(void*, void*) {}

    static void raw_deleter(void* ptr, void*) {
        NMQuadAllocator::get().raw_deallocate(ptr);
    }

    std::mutex mutex_;
    std::map<void*, size_t> allocated_;
    std::map<void*, size_t> free_blocks_[NUM_CHIPS];
    size_t total_allocated_ = 0;
};

// Register with global allocator registry
inline void register_nmquad_allocator() {
    auto& reg = c10::AllocatorRegistry::get();
    reg.registerAllocator(DeviceType::PrivateUse3, &NMQuadAllocator::get());
}

} // namespace nmquad
} // namespace c10
