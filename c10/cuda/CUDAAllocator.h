#pragma once

#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "c10/macros/Macros.h"

#include <cuda_runtime.h>
#include <mutex>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <algorithm>

// Export/Import macro for CUDA allocator (part of aten_cuda library)
// ATEN_CUDA_EXPORTS is defined only when building aten_cuda.dll
#if defined(PT_PLATFORM_WINDOWS)
    #if defined(ATEN_CUDA_EXPORTS)
        #define ATEN_CUDA_API __declspec(dllexport)
    #else
        #define ATEN_CUDA_API __declspec(dllimport)
    #endif
#else
    #define ATEN_CUDA_API __attribute__((visibility("default")))
#endif

namespace c10 {
namespace cuda {

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(error) + \
                " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// ============================================================================
// CUDA Stream wrapper
// ============================================================================

class CUDAStream {
public:
    CUDAStream() : stream_(nullptr), device_(-1) {}

    explicit CUDAStream(int device) : device_(device) {
        int current_device;
        CUDA_CHECK(cudaGetDevice(&current_device));
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaSetDevice(current_device));
    }

    ~CUDAStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    // Move only
    CUDAStream(CUDAStream&& other) noexcept
        : stream_(other.stream_), device_(other.device_) {
        other.stream_ = nullptr;
        other.device_ = -1;
    }

    CUDAStream& operator=(CUDAStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            device_ = other.device_;
            other.stream_ = nullptr;
            other.device_ = -1;
        }
        return *this;
    }

    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;

    cudaStream_t stream() const { return stream_; }
    int device() const { return device_; }

    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_;
    int device_;
};

// ============================================================================
// CUDA Device Guard (RAII)
// ============================================================================

class CUDAGuard {
public:
    explicit CUDAGuard(int device) : prev_device_(-1) {
        CUDA_CHECK(cudaGetDevice(&prev_device_));
        if (prev_device_ != device) {
            CUDA_CHECK(cudaSetDevice(device));
        }
    }

    ~CUDAGuard() {
        if (prev_device_ >= 0) {
            cudaSetDevice(prev_device_);
        }
    }

    CUDAGuard(const CUDAGuard&) = delete;
    CUDAGuard& operator=(const CUDAGuard&) = delete;

private:
    int prev_device_;
};

// ============================================================================
// Block structure for caching allocator
// ============================================================================

struct Block {
    void* ptr;
    size_t size;
    bool allocated;
    int device;
    cudaStream_t stream;

    Block(void* p, size_t s, int dev, cudaStream_t str)
        : ptr(p), size(s), allocated(true), device(dev), stream(str) {}
};

// ============================================================================
// CUDA Caching Allocator
// ============================================================================
// IMPORTANT: The singleton is implemented in CUDAAllocator.cpp to avoid
// DLL boundary issues. Each DLL must use the SAME allocator instance,
// otherwise allocating in one DLL and freeing in another causes heap corruption.

class ATEN_CUDA_API CUDACachingAllocator : public Allocator {
public:
    // Singleton accessor - implemented in CUDAAllocator.cpp
    // DO NOT make this inline - it MUST return the same instance across all DLLs!
    static CUDACachingAllocator& get();

    DataPtr allocate(size_t nbytes) override {
        return allocate_impl(nbytes, 0, nullptr);
    }

    DataPtr allocate(size_t nbytes, int device, cudaStream_t stream = nullptr) {
        return allocate_impl(nbytes, device, stream);
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
        return &CUDACachingAllocator::Delete;
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);

        // Free all cached blocks
        for (auto& block : free_blocks_) {
            CUDAGuard guard(block->device);
            CUDA_CHECK(cudaFree(block->ptr));
            delete block;
        }
        free_blocks_.clear();

        // Note: allocated blocks are still in use
    }

    size_t get_allocated_memory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocated_bytes_;
    }

    size_t get_cached_memory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_bytes_;
    }

    void record_stream(void* ptr, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = ptr_to_block_.find(ptr);
        if (it != ptr_to_block_.end()) {
            it->second->stream = stream;
        }
    }

    // Constructor/Destructor - public for singleton creation in CUDAAllocator.cpp
    // Users should NOT create instances directly - use get() instead!
    CUDACachingAllocator() = default;
    ~CUDACachingAllocator() {
        // Clean up on destruction
        empty_cache();

        // Free allocated blocks (this shouldn't happen in normal use)
        for (auto& pair : ptr_to_block_) {
            cudaFree(pair.second->ptr);
            delete pair.second;
        }
    }

private:

    DataPtr allocate_impl(size_t nbytes, int device, cudaStream_t stream) {
        if (nbytes == 0) {
            return DataPtr(nullptr, nullptr, &null_deleter, Device(DeviceType::CUDA, device));
        }

        // Round up to alignment (512 bytes for coalesced access)
        constexpr size_t kMinBlockSize = 512;
        constexpr size_t kSmallSize = 1048576;  // 1MB
        constexpr size_t kSmallBuffer = 2097152;  // 2MB
        constexpr size_t kLargeBuffer = 20971520;  // 20MB

        size_t alloc_size;
        if (nbytes <= kSmallSize) {
            alloc_size = kSmallBuffer;
        } else {
            alloc_size = ((nbytes + kLargeBuffer - 1) / kLargeBuffer) * kLargeBuffer;
        }
        alloc_size = std::max(alloc_size, ((nbytes + kMinBlockSize - 1) / kMinBlockSize) * kMinBlockSize);

        std::lock_guard<std::mutex> lock(mutex_);

        // Try to find a cached block
        Block* block = find_free_block(alloc_size, device, stream);

        if (block == nullptr) {
            // Allocate new block
            void* ptr = nullptr;
            {
                CUDAGuard guard(device);
                cudaError_t err = cudaMalloc(&ptr, alloc_size);

                if (err == cudaErrorMemoryAllocation) {
                    // Try to free cached memory and retry
                    empty_cache_locked();
                    err = cudaMalloc(&ptr, alloc_size);
                }

                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        "CUDA out of memory. Tried to allocate " +
                        std::to_string(nbytes / 1048576.0) + " MB");
                }
            }

            block = new Block(ptr, alloc_size, device, stream);
            ptr_to_block_[ptr] = block;
            allocated_bytes_ += alloc_size;
        } else {
            // Reuse cached block
            block->allocated = true;
            block->stream = stream;
            cached_bytes_ -= block->size;
        }

        return DataPtr(
            block->ptr,
            block,
            &CUDACachingAllocator::deleter,
            Device(DeviceType::CUDA, device)
        );
    }

    Block* find_free_block(size_t size, int device, cudaStream_t stream) {
        // Find smallest block that fits
        Block* best = nullptr;
        auto it = free_blocks_.begin();

        while (it != free_blocks_.end()) {
            Block* block = *it;

            if (block->device == device && block->size >= size) {
                // Check if stream is compatible
                if (block->stream == stream || block->stream == nullptr) {
                    if (best == nullptr || block->size < best->size) {
                        best = block;
                    }
                }
            }
            ++it;
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

    void empty_cache_locked() {
        for (auto& block : free_blocks_) {
            CUDAGuard guard(block->device);
            cudaFree(block->ptr);
            ptr_to_block_.erase(block->ptr);
            allocated_bytes_ -= block->size;
            delete block;
        }
        free_blocks_.clear();
        cached_bytes_ = 0;
    }

    // Static deleters - implemented in CUDAAllocator.cpp
    // These MUST be in .cpp because they call get() which must resolve to
    // the global singleton, not a per-DLL instance.
    static void deleter(void* data, void* ctx);
    static void Delete(void* data, void* ctx);
    static void null_deleter(void* data, void* ctx);

    mutable std::mutex mutex_;
    std::map<void*, Block*> ptr_to_block_;
    std::set<Block*> free_blocks_;
    size_t allocated_bytes_ = 0;
    size_t cached_bytes_ = 0;
};

// ============================================================================
// CUDA Memory Functions
// ============================================================================

inline void cuda_memcpy_h2d(void* dst, const void* src, size_t nbytes, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
    }
}

inline void cuda_memcpy_d2h(void* dst, const void* src, size_t nbytes, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
    }
}

inline void cuda_memcpy_d2d(void* dst, const void* src, size_t nbytes, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
    }
}

inline void cuda_memset(void* ptr, int value, size_t nbytes, cudaStream_t stream = nullptr) {
    if (stream) {
        CUDA_CHECK(cudaMemsetAsync(ptr, value, nbytes, stream));
    } else {
        CUDA_CHECK(cudaMemset(ptr, value, nbytes));
    }
}

inline void cuda_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

inline int cuda_device_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

inline int cuda_current_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

inline void cuda_set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

// ============================================================================
// Register CUDA Allocator
// ============================================================================
// Implemented in CUDAAllocator.cpp to ensure single registration point

ATEN_CUDA_API void register_cuda_allocator();

} // namespace cuda
} // namespace c10
