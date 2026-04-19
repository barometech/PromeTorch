// ============================================================================
// MPSAllocator.mm — Apple Metal backend for the MPS allocator singleton
// ============================================================================
// Compiled only on Apple platforms with PT_USE_MPS=ON. On any other OS this
// TU is excluded from the build by CMake, so the Obj-C runtime is never
// required. Keep all Obj-C usage inside the `#if` block below.
// ============================================================================

#include "c10/mps/MPSAllocator.h"

#if defined(__APPLE__) && defined(PT_USE_MPS)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "aten/src/ATen/mps/MPSDevice.h"  // shared MTLDevice / command queue

namespace c10 {
namespace mps {

static MPSAllocator g_mps_allocator;

ATEN_MPS_API MPSAllocator& MPSAllocator::get() {
    return g_mps_allocator;
}

// ---- Metal alloc / free -----------------------------------------------------

void* MPSAllocator::metal_new_buffer(size_t nbytes, void** out_contents) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)at::mps::MPSDevice::get().mtl_device();
    if (dev == nil) {
        PT_OOM_ERROR("MPSAllocator: no MTLDevice available");
    }
    // Shared storage = CPU+GPU visible, unified memory on Apple Silicon.
    // For discrete GPUs this falls back to managed (Intel Macs with eGPU).
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    id<MTLBuffer> buf = [dev newBufferWithLength:nbytes options:opts];
    if (buf == nil) {
        PT_OOM_ERROR("MPSAllocator: newBufferWithLength failed for ", nbytes, " bytes");
    }
    // +1 retain so we own the buffer across ObjC ARC boundaries.
    void* raw = (__bridge_retained void*)buf;
    if (out_contents) *out_contents = [buf contents];
    return raw;
}

void MPSAllocator::metal_release_buffer(void* buffer) {
    if (!buffer) return;
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer;
    (void)buf;  // ARC releases on scope exit.
}

// ---- Allocate / deallocate --------------------------------------------------

DataPtr MPSAllocator::allocate(size_t nbytes) {
    if (nbytes == 0) {
        return DataPtr(nullptr, nullptr, &MPSAllocator::null_deleter,
                       Device(DeviceType::MPS, 0));
    }

    // 256-byte alignment (Metal buffer alignment on Apple Silicon).
    size_t alloc_size = (nbytes + 255) & ~size_t(255);

    std::lock_guard<std::mutex> lock(mutex_);

    MPSBlock* block = find_free_block(alloc_size);
    if (block == nullptr) {
        void* contents = nullptr;
        void* buffer   = metal_new_buffer(alloc_size, &contents);

        block = new MPSBlock(contents, buffer, alloc_size);
        ptr_to_block_[contents] = block;
        allocated_bytes_ += alloc_size;
    } else {
        block->allocated = true;
        cached_bytes_   -= block->size;
    }

    return DataPtr(
        block->contents,
        block,
        &MPSAllocator::deleter,
        Device(DeviceType::MPS, 0)
    );
}

void MPSAllocator::raw_deallocate(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_block_.find(ptr);
    if (it == ptr_to_block_.end()) return;
    free_block(it->second);
}

DeleterFn MPSAllocator::raw_deleter() const {
    return &MPSAllocator::deleter;
}

MPSBlock* MPSAllocator::find_free_block(size_t size) {
    auto it = free_blocks_.lower_bound(size);
    if (it == free_blocks_.end()) return nullptr;
    MPSBlock* blk = it->second;
    free_blocks_.erase(it);
    return blk;
}

void MPSAllocator::free_block(MPSBlock* block) {
    // Caller already holds mutex_.
    block->allocated = false;
    free_blocks_.insert({block->size, block});
    cached_bytes_ += block->size;
}

void MPSAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : free_blocks_) {
        MPSBlock* b = kv.second;
        metal_release_buffer(b->buffer);
        ptr_to_block_.erase(b->contents);
        allocated_bytes_ -= b->size;
        delete b;
    }
    free_blocks_.clear();
    cached_bytes_ = 0;
}

size_t MPSAllocator::get_allocated_memory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_bytes_;
}
size_t MPSAllocator::get_cached_memory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_bytes_;
}

void MPSAllocator::deleter(void* /*data*/, void* ctx) {
    if (!ctx) return;
    MPSBlock* block = static_cast<MPSBlock*>(ctx);
    std::lock_guard<std::mutex> lock(g_mps_allocator.mutex_);
    g_mps_allocator.free_block(block);
}

void MPSAllocator::null_deleter(void*, void*) { /* no-op */ }

// ---- Registration -----------------------------------------------------------

static bool g_mps_allocator_registered = false;

ATEN_MPS_API void register_mps_allocator() {
    if (!g_mps_allocator_registered) {
        AllocatorRegistry::get().registerAllocator(
            DeviceType::MPS, &g_mps_allocator);
        g_mps_allocator_registered = true;
    }
}

} // namespace mps
} // namespace c10

#else // not (Apple + PT_USE_MPS)

// Provide a single symbol so linkers that accidentally reference
// register_mps_allocator on a non-MPS build fail with a clear message.
#include "c10/util/Exception.h"

namespace c10 {
namespace mps {

ATEN_MPS_API MPSAllocator& MPSAllocator::get() {
    static MPSAllocator inst;
    return inst;
}
DataPtr   MPSAllocator::allocate(size_t)  {
    PT_ERROR("MPS backend only available on Apple platforms with PT_USE_MPS=ON");
}
void      MPSAllocator::raw_deallocate(void*) {}
DeleterFn MPSAllocator::raw_deleter() const { return &MPSAllocator::null_deleter; }
void      MPSAllocator::empty_cache() {}
size_t    MPSAllocator::get_allocated_memory() const { return 0; }
size_t    MPSAllocator::get_cached_memory()    const { return 0; }
void*     MPSAllocator::metal_new_buffer(size_t, void**) { return nullptr; }
void      MPSAllocator::metal_release_buffer(void*) {}
MPSBlock* MPSAllocator::find_free_block(size_t) { return nullptr; }
void      MPSAllocator::free_block(MPSBlock*) {}
void      MPSAllocator::deleter(void*, void*) {}
void      MPSAllocator::null_deleter(void*, void*) {}

ATEN_MPS_API void register_mps_allocator() {
    PT_ERROR("MPS backend only available on Apple platforms with PT_USE_MPS=ON");
}

} // namespace mps
} // namespace c10

#endif // __APPLE__ && PT_USE_MPS
