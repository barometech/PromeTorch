// TensorImpl.cpp - Implementation of factory functions
// These MUST be in a .cpp file to avoid DLL boundary issues with static variables

#include "c10/core/TensorImpl.h"

namespace c10 {

// Global factory instance - only one copy exists in c10.dll
static AutogradMetaFactory g_autograd_meta_factory = []() -> std::unique_ptr<AutogradMeta> {
    return std::make_unique<AutogradMeta>();  // Default: create base class
};

// Get the global factory (exported from c10.dll)
PT_API AutogradMetaFactory& get_autograd_meta_factory_impl() {
    return g_autograd_meta_factory;
}

// Set the global factory (exported from c10.dll)
PT_API void set_autograd_meta_factory_impl(AutogradMetaFactory factory) {
    g_autograd_meta_factory = factory;
}

// Create autograd metadata using the registered factory
PT_API std::unique_ptr<AutogradMeta> create_autograd_meta_impl() {
    return g_autograd_meta_factory();
}

} // namespace c10
