#include "c10/core/Device.h"

namespace c10 {

// ============================================================================
// DeviceGuard Implementation
// ============================================================================

// Note: This is a simplified implementation for CPU-only builds.
// Real implementation would call CUDA/HIP device switching APIs.

namespace {

// Thread-local current device (for multi-device support)
thread_local Device current_device_tls = kCPU;

void setDevice(Device device) {
    // For CPU, nothing to do
    if (device.is_cpu()) {
        current_device_tls = device;
        return;
    }

    // For CUDA, would call cudaSetDevice
    #if PT_CUDA_ENABLED
        if (device.is_cuda()) {
            cudaSetDevice(device.index());
        }
    #endif

    current_device_tls = device;
}

Device getDevice(DeviceType type) {
    if (type == DeviceType::CPU) {
        return kCPU;
    }

    #if PT_CUDA_ENABLED
        if (type == DeviceType::CUDA) {
            int device;
            cudaGetDevice(&device);
            return Device(DeviceType::CUDA, static_cast<DeviceIndex>(device));
        }
    #endif

    return current_device_tls;
}

} // anonymous namespace

DeviceGuard::DeviceGuard(Device device)
    : original_device_(getDevice(device.type()))
    , current_device_(device)
    , active_(true)
{
    if (original_device_ != current_device_) {
        setDevice(current_device_);
    }
}

DeviceGuard::~DeviceGuard() {
    if (active_ && original_device_ != current_device_) {
        setDevice(original_device_);
    }
}

DeviceGuard::DeviceGuard(DeviceGuard&& other) noexcept
    : original_device_(other.original_device_)
    , current_device_(other.current_device_)
    , active_(other.active_)
{
    other.active_ = false;
}

DeviceGuard& DeviceGuard::operator=(DeviceGuard&& other) noexcept {
    if (this != &other) {
        if (active_ && original_device_ != current_device_) {
            setDevice(original_device_);
        }

        original_device_ = other.original_device_;
        current_device_ = other.current_device_;
        active_ = other.active_;
        other.active_ = false;
    }
    return *this;
}

void DeviceGuard::reset_device(Device device) {
    if (current_device_ != device) {
        setDevice(device);
        current_device_ = device;
    }
}

void DeviceGuard::reset() {
    if (active_ && original_device_ != current_device_) {
        setDevice(original_device_);
        current_device_ = original_device_;
    }
}

} // namespace c10
