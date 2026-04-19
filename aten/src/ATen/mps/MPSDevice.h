#pragma once
// ============================================================================
// MPSDevice.h — Singleton holder for the default id<MTLDevice> and its
//               id<MTLCommandQueue>. Lazy-init on first access.
// ============================================================================
// Everything here is declared in plain C++ types (void*) so this header can
// be included from .cpp TUs that are not compiled as Obj-C++. The Obj-C
// objects are created inside MPSDevice.mm where <Metal/Metal.h> is visible.
//
// Cast-back recipe on the caller side (inside a .mm TU):
//
//     id<MTLDevice>        dev = (__bridge id<MTLDevice>)MPSDevice::get().mtl_device();
//     id<MTLCommandQueue>  q   = (__bridge id<MTLCommandQueue>)MPSDevice::get().mtl_command_queue();
//
// On non-Apple builds the implementation throws; callers get the same error
// as the allocator: "MPS backend only available on Apple platforms with
// PT_USE_MPS=ON".
// ============================================================================

#include "c10/macros/Macros.h"
#include "c10/util/Exception.h"

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

namespace at {
namespace mps {

class ATEN_MPS_API MPSDevice {
public:
    static MPSDevice& get();

    // Opaque handles — cast with `(__bridge id<MTLDevice>)` in Obj-C++.
    void* mtl_device()        const;
    void* mtl_command_queue() const;

    // Wait for all currently-enqueued command buffers to finish.
    // No-op on non-Apple builds.
    void synchronize();

    // True when we can actually run Metal work (Apple + init succeeded).
    bool is_available() const;

private:
    MPSDevice();
    ~MPSDevice();
    MPSDevice(const MPSDevice&)            = delete;
    MPSDevice& operator=(const MPSDevice&) = delete;

    // Pimpl holds retained id<MTLDevice>/id<MTLCommandQueue> on Apple,
    // stays nullptr elsewhere.
    struct Impl;
    Impl* impl_;
};

// Throw a uniform error when MPS isn't available, to use from dispatch paths.
[[noreturn]] inline void mps_not_available() {
    PT_ERROR("MPS backend only available on Apple platforms with PT_USE_MPS=ON");
}

} // namespace mps
} // namespace at
