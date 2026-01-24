#include "c10/core/Allocator.h"

namespace c10 {

// ============================================================================
// Warning System Implementation
// ============================================================================

bool Warning::enabled_ = true;

void Warning::warn(const std::string& message, WarningType type) {
    if (!enabled_) return;

    const char* type_str = "UserWarning";
    switch (type) {
        case WarningType::UserWarning:
            type_str = "UserWarning";
            break;
        case WarningType::DeprecationWarning:
            type_str = "DeprecationWarning";
            break;
        case WarningType::RuntimeWarning:
            type_str = "RuntimeWarning";
            break;
    }

    // Print to stderr
    std::fprintf(stderr, "%s: %s\n", type_str, message.c_str());
}

void Warning::set_enabled(bool enabled) {
    enabled_ = enabled;
}

bool Warning::is_enabled() {
    return enabled_;
}

// ============================================================================
// Pinned Memory Allocator Implementation (Stub for CPU-only build)
// ============================================================================

DataPtr PinnedMemoryAllocator::allocate(size_t nbytes) {
    // For now, just use regular CPU allocation
    // Real implementation would use cudaMallocHost or similar
    return CPUAllocator::get().allocate(nbytes);
}

void PinnedMemoryAllocator::raw_deallocate(void* ptr) {
    CPUAllocator::get().raw_deallocate(ptr);
}

DeleterFn PinnedMemoryAllocator::raw_deleter() const {
    return CPUAllocator::get().raw_deleter();
}

void PinnedMemoryAllocator::Delete(void* data, void* context) {
    CPUAllocator::get().raw_deallocate(data);
}

} // namespace c10
