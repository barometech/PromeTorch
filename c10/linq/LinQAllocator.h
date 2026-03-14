#pragma once
// ============================================================================
// LinQAllocator.h — Memory allocator for LinQ H1M tensor accelerator
// ============================================================================
// Emulator mode: uses host RAM with aligned allocation
// Hardware mode: would use PCIe DMA to device memory
// ============================================================================

#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include <mutex>
#include <map>
#include <set>

#ifdef _MSC_VER
    #ifdef BUILDING_ATEN_LINQ
        #define ATEN_LINQ_API __declspec(dllexport)
    #else
        #define ATEN_LINQ_API __declspec(dllimport)
    #endif
#else
    #define ATEN_LINQ_API
#endif

namespace c10 {
namespace linq {

class ATEN_LINQ_API LinQAllocator : public Allocator {
public:
    // Singleton — MUST be defined in .cpp to avoid DLL issues
    static LinQAllocator& get();

    DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return DataPtr(nullptr, nullptr, &null_deleter,
                          Device(DeviceType::PrivateUse2, 0));
        }

        size_t alloc_size = (nbytes + 63) & ~63; // 64-byte aligned

        std::lock_guard<std::mutex> lock(mutex_);

        // Try to find a free block
        Block* block = find_free_block(alloc_size);
        if (!block) {
            void* ptr = nullptr;
#ifdef _MSC_VER
            ptr = _aligned_malloc(alloc_size, 64);
#else
            posix_memalign(&ptr, 64, alloc_size);
#endif
            PT_CHECK_MSG(ptr != nullptr, "LinQ: allocation failed for ", alloc_size, " bytes");
            block = new Block{ptr, alloc_size, true};
            ptr_to_block_[ptr] = block;
            total_allocated_ += alloc_size;
        }

        block->allocated = true;

        return DataPtr(block->ptr, block, &LinQAllocator::deleter,
                      Device(DeviceType::PrivateUse2, 0));
    }

    DeleterFn raw_deleter() const override {
        return &LinQAllocator::deleter;
    }

    void raw_deallocate(void* ptr) override {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = ptr_to_block_.find(ptr);
        if (it != ptr_to_block_.end()) {
            it->second->allocated = false;
            free_blocks_.insert(it->second);
        }
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto* block : free_blocks_) {
#ifdef _MSC_VER
            _aligned_free(block->ptr);
#else
            free(block->ptr);
#endif
            total_allocated_ -= block->size;
            ptr_to_block_.erase(block->ptr);
            delete block;
        }
        free_blocks_.clear();
    }

    size_t get_allocated_memory() const { return total_allocated_; }
    size_t get_cached_memory() const {
        size_t cached = 0;
        for (auto* b : free_blocks_) cached += b->size;
        return cached;
    }

private:
    struct Block {
        void* ptr;
        size_t size;
        bool allocated;
    };

    static void null_deleter(void*, void*) {}
    static void deleter(void* data, void* ctx);

    Block* find_free_block(size_t size) {
        // Best-fit search
        Block* best = nullptr;
        for (auto* b : free_blocks_) {
            if (b->size >= size && (!best || b->size < best->size)) {
                best = b;
            }
        }
        if (best) {
            free_blocks_.erase(best);
        }
        return best;
    }

    std::mutex mutex_;
    std::map<void*, Block*> ptr_to_block_;
    std::set<Block*> free_blocks_;
    size_t total_allocated_ = 0;
};

ATEN_LINQ_API void register_linq_allocator();

} // namespace linq
} // namespace c10
