// ============================================================================
// MPSDevice.mm — Lazy singleton for default id<MTLDevice> + command queue.
// ============================================================================
// Apple-only. On any other platform this TU compiles a stub that raises an
// error when accessed — the CMake rule skips adding the .mm file on non-Apple
// builds anyway, so this fallback is belt+suspenders.
// ============================================================================

#include "aten/src/ATen/mps/MPSDevice.h"

#if defined(__APPLE__) && defined(PT_USE_MPS)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

namespace at {
namespace mps {

struct MPSDevice::Impl {
    id<MTLDevice>        device = nil;
    id<MTLCommandQueue>  queue  = nil;

    Impl() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (device == nil) return;
            queue  = [device newCommandQueue];
        }
    }
    ~Impl() {
        queue  = nil;   // ARC releases
        device = nil;
    }
};

MPSDevice::MPSDevice() : impl_(new Impl()) {}
MPSDevice::~MPSDevice() { delete impl_; impl_ = nullptr; }

MPSDevice& MPSDevice::get() {
    static MPSDevice inst;
    return inst;
}

void* MPSDevice::mtl_device() const {
    return (__bridge void*)impl_->device;
}
void* MPSDevice::mtl_command_queue() const {
    return (__bridge void*)impl_->queue;
}

bool MPSDevice::is_available() const {
    return impl_ && impl_->device != nil && impl_->queue != nil;
}

void MPSDevice::synchronize() {
    if (!is_available()) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

} // namespace mps
} // namespace at

#else // non-Apple build

namespace at {
namespace mps {

struct MPSDevice::Impl { int _unused = 0; };

MPSDevice::MPSDevice()  : impl_(nullptr) {}
MPSDevice::~MPSDevice() = default;

MPSDevice& MPSDevice::get() { static MPSDevice inst; return inst; }

void* MPSDevice::mtl_device()        const { return nullptr; }
void* MPSDevice::mtl_command_queue() const { return nullptr; }
bool  MPSDevice::is_available()      const { return false; }
void  MPSDevice::synchronize()             { /* no-op */ }

} // namespace mps
} // namespace at

#endif
