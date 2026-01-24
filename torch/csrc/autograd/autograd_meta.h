#pragma once

#include "c10/core/TensorImpl.h"
#include "torch/csrc/autograd/edge.h"
#include <memory>
#include <vector>
#include <functional>

namespace torch {
namespace autograd {

// Forward declarations
struct Node;
struct AutogradMetaImpl;  // Forward declare for factory

// Factory function to create AutogradMetaImpl (defined after struct)
std::unique_ptr<c10::AutogradMeta> create_autograd_meta_impl();

// ============================================================================
// AutogradMetaImpl - Extended Autograd Metadata
// ============================================================================
// This inherits from c10::AutogradMeta and adds full autograd support including
// grad_fn, grad_accumulator, and hooks.
//
// IMPORTANT: Always create AutogradMetaImpl (not c10::AutogradMeta) when
// autograd support is needed, otherwise accessing grad_fn etc will crash!

struct AutogradMetaImpl : public c10::AutogradMeta {
    // The gradient function (backward node) that produces this tensor's gradient
    // This is nullptr for leaf tensors
    std::shared_ptr<Node> grad_fn;

    // Gradient accumulator for leaf tensors
    // We use weak_ptr to avoid preventing the AccumulateGrad from being freed
    std::weak_ptr<Node> grad_accumulator_;

    // Version counter for detecting in-place modifications
    // Not used when grad_fn is nullptr (leaf tensors)
    uint32_t version_counter_ = 0;

    // Post-accumulate-grad hooks
    std::vector<std::function<void(at::Tensor&)>> hooks_;

    AutogradMetaImpl() : c10::AutogradMeta() {}

    // Check if this tensor is a view of another tensor
    bool is_view() const {
        return false;  // Simplified - views not fully implemented yet
    }

    // Increment version counter (called on in-place operations)
    void bump_version() {
        ++version_counter_;
    }

    // Get the current version
    uint32_t current_version() const {
        return version_counter_;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

// Debug counter for metadata upgrades
inline std::atomic<int64_t> g_meta_upgrade_count{0};
inline std::atomic<int64_t> g_meta_create_count{0};

// Ensure tensor has AutogradMetaImpl (upgrade from c10::AutogradMeta if needed)
inline AutogradMetaImpl* ensure_autograd_meta_impl(at::Tensor& tensor) {
    auto& impl = tensor.getIntrusivePtr();
    auto* raw_meta = impl->autograd_meta();

    if (!raw_meta) {
        // No metadata - create AutogradMetaImpl
        g_meta_create_count++;
        impl->set_autograd_meta(std::make_unique<AutogradMetaImpl>());
        return static_cast<AutogradMetaImpl*>(impl->autograd_meta());
    }

    // Try to dynamic_cast to check if it's already AutogradMetaImpl
    auto* meta_impl = dynamic_cast<AutogradMetaImpl*>(raw_meta);
    if (meta_impl) {
        return meta_impl;
    }

    // It's base c10::AutogradMeta - need to upgrade
    g_meta_upgrade_count++;

    // DEBUG: Print tensor info on upgrade
    std::cout << "[META UPGRADE #" << g_meta_upgrade_count.load() << "] "
              << "shape=[";
    auto sizes = tensor.sizes();
    for (size_t i = 0; i < sizes.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << sizes[i];
    }
    std::cout << "] is_leaf=" << (raw_meta->is_leaf_ ? "true" : "false")
              << " req_grad=" << (raw_meta->requires_grad_ ? "true" : "false")
              << std::endl;

    // Copy fields from old metadata
    auto new_meta = std::make_unique<AutogradMetaImpl>();
    new_meta->grad_ = raw_meta->grad_;
    new_meta->output_nr_ = raw_meta->output_nr_;
    new_meta->requires_grad_ = raw_meta->requires_grad_;
    new_meta->retains_grad_ = raw_meta->retains_grad_;
    new_meta->is_leaf_ = raw_meta->is_leaf_;

    // Replace metadata
    impl->set_autograd_meta(std::move(new_meta));
    return static_cast<AutogradMetaImpl*>(impl->autograd_meta());
}

// Get autograd metadata for a tensor (read-only, may return nullptr)
inline AutogradMetaImpl* get_autograd_meta(const at::Tensor& tensor) {
    auto* raw_meta = tensor.autograd_meta();
    if (!raw_meta) return nullptr;
    return dynamic_cast<AutogradMetaImpl*>(raw_meta);
}

// Set grad_fn for a tensor
inline void set_grad_fn(at::Tensor& tensor, std::shared_ptr<Node> grad_fn, uint32_t output_nr = 0) {
    // Ensure we have AutogradMetaImpl (upgrades if necessary)
    auto* meta = ensure_autograd_meta_impl(tensor);
    meta->grad_fn = std::move(grad_fn);
    meta->output_nr_ = output_nr;
    meta->is_leaf_ = false;  // Tensors with grad_fn are not leaves
}

// Get grad_fn from a tensor
inline std::shared_ptr<Node> grad_fn(const at::Tensor& tensor) {
    if (!tensor.requires_grad()) {
        return nullptr;
    }
    auto* meta = get_autograd_meta(tensor);
    if (meta) {
        return meta->grad_fn;
    }
    return nullptr;
}

// Get gradient accumulator for a leaf tensor
std::shared_ptr<Node> get_grad_accumulator(const at::Tensor& tensor);

// Clear grad_fn to release autograd graph (call after backward with retain_graph=false)
inline void clear_grad_fn(at::Tensor& tensor) {
    auto* meta = get_autograd_meta(tensor);
    if (meta) {
        meta->grad_fn.reset();  // Release the backward function
        meta->is_leaf_ = true;   // Tensor becomes a leaf
    }
}

// Create an edge from a tensor's grad_fn
inline Edge gradient_edge(const at::Tensor& tensor) {
    if (!tensor.defined() || !tensor.requires_grad()) {
        return Edge();
    }

    auto* meta = get_autograd_meta(tensor);

    // If we have AutogradMetaImpl with grad_fn, use it
    if (meta && meta->grad_fn) {
        return Edge(meta->grad_fn, meta->output_nr_);
    }

    // For leaf tensors (or base AutogradMeta), return edge to gradient accumulator
    auto accumulator = get_grad_accumulator(tensor);
    return Edge(accumulator, 0);
}

// Print and reset debug counters
inline void print_meta_stats() {
    std::cout << "[META STATS] create=" << g_meta_create_count.load()
              << " upgrade=" << g_meta_upgrade_count.load()
              << std::endl;
}

// Graph cleanup functions are defined in autograd.h to avoid circular dependencies

inline void reset_meta_stats() {
    g_meta_create_count = 0;
    g_meta_upgrade_count = 0;
}

// ============================================================================
// AutogradMetaImpl Factory - creates AutogradMetaImpl directly
// ============================================================================
// This is called by c10::create_autograd_meta() when registered

inline std::unique_ptr<c10::AutogradMeta> create_autograd_meta_impl() {
    g_meta_create_count++;
    return std::make_unique<AutogradMetaImpl>();
}

// Auto-registration: register factory when this header is included
// This ensures AutogradMetaImpl is created instead of base AutogradMeta
namespace {
struct AutogradMetaFactoryRegistrar {
    AutogradMetaFactoryRegistrar() {
        c10::set_autograd_meta_factory(&create_autograd_meta_impl);
    }
};
// Static instance triggers registration at program start
static AutogradMetaFactoryRegistrar g_autograd_meta_factory_registrar;
}

} // namespace autograd
} // namespace torch
