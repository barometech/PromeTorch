#pragma once

#include "torch/csrc/autograd/edge.h"
#include "aten/src/ATen/core/Tensor.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include <vector>
#include <memory>
#include <string>
#include <atomic>
#include <functional>
#include <mutex>
#include <iostream>

namespace torch {
namespace autograd {

using at::Tensor;
using variable_list = std::vector<Tensor>;
using edge_list = std::vector<Edge>;

// DEBUG: Global counters for tracking Node lifecycle
inline std::atomic<int64_t> g_nodes_created{0};
inline std::atomic<int64_t> g_nodes_destroyed{0};
inline std::atomic<int64_t> g_nodes_released{0};
inline std::atomic<int64_t> g_accum_grad_created{0};
inline std::atomic<int64_t> g_accum_grad_destroyed{0};

inline void print_node_stats() {
    std::cout << "[NODE STATS] created=" << g_nodes_created.load()
              << " destroyed=" << g_nodes_destroyed.load()
              << " released=" << g_nodes_released.load()
              << " alive=" << (g_nodes_created.load() - g_nodes_destroyed.load())
              << " | AccumGrad: created=" << g_accum_grad_created.load()
              << " destroyed=" << g_accum_grad_destroyed.load()
              << " alive=" << (g_accum_grad_created.load() - g_accum_grad_destroyed.load())
              << std::endl;
}

// ============================================================================
// Node - Base Class for Autograd Functions
// ============================================================================
// A Node represents an operation in the computational graph. During the
// forward pass, operations create Nodes that remember how to compute
// gradients. During backward, the Engine calls each Node's apply() method
// to compute gradients with respect to its inputs.
//
// PyTorch calls this "Function" but we use "Node" to be consistent with
// the computational graph terminology.

struct Node : std::enable_shared_from_this<Node> {
protected:
    // Edges to the next functions in the backward graph
    // These are the functions that will receive our computed gradients
    edge_list next_edges_;

    // Unique sequence number for topological ordering
    uint64_t sequence_nr_;

    // Static counter for sequence numbers
    static std::atomic<uint64_t> sequence_nr_counter_;

    // Mutex for thread-safe operations
    mutable std::mutex mutex_;

public:
    Node() : sequence_nr_(sequence_nr_counter_++) {
        g_nodes_created++;
    }

    virtual ~Node() {
        g_nodes_destroyed++;
    }

    // ========================================================================
    // Core Interface
    // ========================================================================

    // Compute gradients with respect to inputs given gradients of outputs
    // This is the main method that subclasses must implement
    virtual variable_list apply(variable_list&& grads) = 0;

    // Name of this node for debugging/visualization
    virtual std::string name() const {
        return "Node";
    }

    // ========================================================================
    // Edge Management
    // ========================================================================

    // Get the number of inputs this function had
    size_t num_inputs() const {
        return next_edges_.size();
    }

    // Get the number of outputs this function produces
    virtual size_t num_outputs() const {
        return 1;
    }

    // Get next edge at index
    const Edge& next_edge(size_t index) const {
        return next_edges_.at(index);
    }

    Edge& next_edge(size_t index) {
        return next_edges_.at(index);
    }

    // Get all next edges
    const edge_list& next_edges() const noexcept {
        return next_edges_;
    }

    edge_list& next_edges() noexcept {
        return next_edges_;
    }

    // Add a next edge
    void add_next_edge(Edge edge) {
        next_edges_.push_back(std::move(edge));
    }

    // Release all resources (call after backward with retain_graph=false)
    // This breaks circular references and allows memory to be freed
    // Subclasses should override release_saved_tensors() to clear their saved tensors
    virtual void release() {
        g_nodes_released++;
        next_edges_.clear();
        release_saved_tensors();  // Also clear saved tensors
    }

    // Override this in subclasses to release saved tensors
    virtual void release_saved_tensors() {
        // Default implementation does nothing
    }

    // Set next edge at index
    void set_next_edge(size_t index, Edge edge) {
        if (index >= next_edges_.size()) {
            next_edges_.resize(index + 1);
        }
        next_edges_[index] = std::move(edge);
    }

    // Add input metadata (creates edge from input tensor's grad_fn)
    void add_input_metadata(const Tensor& tensor);

    // ========================================================================
    // Sequence Number
    // ========================================================================

    uint64_t sequence_nr() const noexcept {
        return sequence_nr_;
    }

    // Set sequence number (used by engine for ordering)
    void set_sequence_nr(uint64_t sequence_nr) {
        sequence_nr_ = sequence_nr;
    }

    // ========================================================================
    // Operator() - Convenience wrapper for apply()
    // ========================================================================

    variable_list operator()(variable_list&& inputs) {
        return apply(std::move(inputs));
    }

    // ========================================================================
    // Utility
    // ========================================================================

    // Check if this node should be executed (has valid next edges)
    bool should_compute_output(size_t output_idx) const {
        if (output_idx >= next_edges_.size()) {
            return false;
        }
        return next_edges_[output_idx].is_valid();
    }

    // Create an edge pointing to this node's output
    Edge make_edge(uint32_t output_nr = 0) {
        return Edge(shared_from_this(), output_nr);
    }
};

// Initialize static counter
inline std::atomic<uint64_t> Node::sequence_nr_counter_{0};

// ============================================================================
// AccumulateGrad - Accumulates gradients for leaf tensors
// ============================================================================
// This is a special Node that accumulates gradients into a leaf tensor's
// grad field. It's the terminal node in the backward graph for leaf
// variables that require gradients.

struct AccumulateGrad : public Node {
private:
    // Weak reference to the tensor we're accumulating gradients for
    // We use weak_ptr to avoid preventing the tensor from being freed
    std::weak_ptr<c10::TensorImpl> weak_impl_;

public:
    explicit AccumulateGrad(const Tensor& tensor)
        : weak_impl_(tensor.getIntrusivePtr()) {
        g_accum_grad_created++;
    }

    ~AccumulateGrad() {
        g_accum_grad_destroyed++;
    }

    variable_list apply(variable_list&& grads) override;

    std::string name() const override {
        return "AccumulateGrad";
    }

    size_t num_outputs() const override {
        return 0;  // AccumulateGrad doesn't produce outputs
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

// Collect the next edges from a list of tensors
inline edge_list collect_next_edges(const variable_list& tensors) {
    edge_list edges;
    edges.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        if (tensor.defined() && tensor.requires_grad()) {
            // Use dynamic_cast to safely check for AutogradMetaImpl
            auto* meta = get_autograd_meta(tensor);
            if (meta && meta->grad_fn) {
                edges.emplace_back(meta->grad_fn, meta->output_nr_);
            } else {
                // Leaf tensor or base AutogradMeta - needs AccumulateGrad
                edges.emplace_back(nullptr, 0);
            }
        } else {
            edges.emplace_back(nullptr, 0);
        }
    }

    return edges;
}

// Check if any tensor in the list requires grad
inline bool any_requires_grad(const variable_list& tensors) {
    for (const auto& t : tensors) {
        if (t.defined() && t.requires_grad()) {
            return true;
        }
    }
    return false;
}

// Compute requires_grad for output based on inputs
inline bool compute_requires_grad(const variable_list& inputs) {
    return any_requires_grad(inputs);
}

inline bool compute_requires_grad(const Tensor& a) {
    return a.defined() && a.requires_grad();
}

inline bool compute_requires_grad(const Tensor& a, const Tensor& b) {
    return compute_requires_grad(a) || compute_requires_grad(b);
}

} // namespace autograd
} // namespace torch
