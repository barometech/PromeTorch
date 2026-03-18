#pragma once

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "aten/src/ATen/core/Tensor.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include <vector>
#include <memory>
#include <string>
#include <atomic>
#include <functional>
#include <mutex>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace torch {
namespace autograd {

using at::Tensor;
using variable_list = std::vector<Tensor>;
using edge_list = std::vector<Edge>;

// ============================================================================
// SmallEdgeList - Inline storage for edges (avoids heap allocation)
// ============================================================================
// Most autograd nodes have 1-3 input edges (unary ops: 1, binary ops: 2,
// ternary: 3). This stores up to N edges inline without malloc.
// Falls back to heap vector only for rare cases with > N edges.
// On Elbrus E2K, each avoided malloc saves a syscall + VLIW pipeline stall.

template<size_t InlineCapacity = 4>
class SmallEdgeList {
    Edge inline_storage_[InlineCapacity];
    uint8_t size_ = 0;
    uint8_t using_heap_ = 0;  // 0 = inline, 1 = heap
    std::vector<Edge>* heap_ = nullptr;

public:
    SmallEdgeList() = default;

    ~SmallEdgeList() {
        if (using_heap_ && heap_) {
            delete heap_;
        }
    }

    // Move constructor
    SmallEdgeList(SmallEdgeList&& other) noexcept
        : size_(other.size_), using_heap_(other.using_heap_) {
        if (using_heap_) {
            heap_ = other.heap_;
            other.heap_ = nullptr;
            other.using_heap_ = 0;
            other.size_ = 0;
        } else {
            for (uint8_t i = 0; i < size_; ++i) {
                inline_storage_[i] = std::move(other.inline_storage_[i]);
            }
            other.size_ = 0;
        }
    }

    // Move assignment
    SmallEdgeList& operator=(SmallEdgeList&& other) noexcept {
        if (this != &other) {
            clear();
            size_ = other.size_;
            using_heap_ = other.using_heap_;
            if (using_heap_) {
                heap_ = other.heap_;
                other.heap_ = nullptr;
                other.using_heap_ = 0;
                other.size_ = 0;
            } else {
                for (uint8_t i = 0; i < size_; ++i) {
                    inline_storage_[i] = std::move(other.inline_storage_[i]);
                }
                other.size_ = 0;
            }
        }
        return *this;
    }

    // No copy (edges contain shared_ptr, copying is expensive)
    SmallEdgeList(const SmallEdgeList&) = delete;
    SmallEdgeList& operator=(const SmallEdgeList&) = delete;

    size_t size() const { return using_heap_ ? heap_->size() : size_; }
    bool empty() const { return size() == 0; }

    void push_back(Edge edge) {
        if (using_heap_) {
            heap_->push_back(std::move(edge));
        } else if (size_ < InlineCapacity) {
            inline_storage_[size_++] = std::move(edge);
        } else {
            // Spill to heap
            heap_ = new std::vector<Edge>();
            heap_->reserve(size_ + 4);
            for (uint8_t i = 0; i < size_; ++i) {
                heap_->push_back(std::move(inline_storage_[i]));
            }
            heap_->push_back(std::move(edge));
            using_heap_ = 1;
        }
    }

    template<typename... Args>
    void emplace_back(Args&&... args) {
        push_back(Edge(std::forward<Args>(args)...));
    }

    const Edge& operator[](size_t i) const {
        return using_heap_ ? (*heap_)[i] : inline_storage_[i];
    }

    Edge& operator[](size_t i) {
        return using_heap_ ? (*heap_)[i] : inline_storage_[i];
    }

    const Edge& at(size_t i) const {
        if (i >= size()) throw std::out_of_range("SmallEdgeList::at");
        return (*this)[i];
    }

    Edge& at(size_t i) {
        if (i >= size()) throw std::out_of_range("SmallEdgeList::at");
        return (*this)[i];
    }

    void resize(size_t n) {
        if (using_heap_) {
            heap_->resize(n);
        } else if (n <= InlineCapacity) {
            // Clear edges beyond new size
            for (size_t i = n; i < size_; ++i) {
                inline_storage_[i] = Edge();
            }
            size_ = static_cast<uint8_t>(n);
        } else {
            // Spill to heap
            heap_ = new std::vector<Edge>(n);
            for (uint8_t i = 0; i < size_; ++i) {
                (*heap_)[i] = std::move(inline_storage_[i]);
            }
            using_heap_ = 1;
        }
    }

    void clear() {
        if (using_heap_) {
            delete heap_;
            heap_ = nullptr;
            using_heap_ = 0;
        } else {
            for (uint8_t i = 0; i < size_; ++i) {
                inline_storage_[i] = Edge();
            }
        }
        size_ = 0;
    }

    void reserve(size_t n) {
        if (n > InlineCapacity && !using_heap_) {
            // Pre-spill to heap if we know we'll need it
            heap_ = new std::vector<Edge>();
            heap_->reserve(n);
            for (uint8_t i = 0; i < size_; ++i) {
                heap_->push_back(std::move(inline_storage_[i]));
            }
            using_heap_ = 1;
        } else if (using_heap_) {
            heap_->reserve(n);
        }
    }

    // Iterator support (for range-based for loops)
    const Edge* begin() const { return using_heap_ ? heap_->data() : inline_storage_; }
    const Edge* end() const { return begin() + size(); }
    Edge* begin() { return using_heap_ ? heap_->data() : inline_storage_; }
    Edge* end() { return begin() + size(); }

    // Convert to edge_list (for compatibility with existing code)
    edge_list to_vector() const {
        edge_list result;
        result.reserve(size());
        for (size_t i = 0; i < size(); ++i) {
            result.push_back((*this)[i]);
        }
        return result;
    }
};

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
    // Uses SmallEdgeList: stores up to 4 edges inline (no malloc).
    // Most ops have 1-3 edges (unary=1, binary=2), so this covers 99%+ cases.
    SmallEdgeList<4> next_edges_;

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

    // Get all next edges (returns SmallEdgeList reference)
    const SmallEdgeList<4>& next_edges() const noexcept {
        return next_edges_;
    }

    SmallEdgeList<4>& next_edges() noexcept {
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

    // Clear edges for NodePool recycling (called by PooledDeleter)
    void next_edges_clear_pooled() {
        next_edges_.clear();
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
// IMPORTANT: Returns false if GradMode is disabled!
inline bool compute_requires_grad(const variable_list& inputs) {
    if (!GradMode::is_enabled()) return false;
    return any_requires_grad(inputs);
}

inline bool compute_requires_grad(const Tensor& a) {
    if (!GradMode::is_enabled()) return false;
    return a.defined() && a.requires_grad();
}

inline bool compute_requires_grad(const Tensor& a, const Tensor& b) {
    if (!GradMode::is_enabled()) return false;
    return (a.defined() && a.requires_grad()) || (b.defined() && b.requires_grad());
}

} // namespace autograd
} // namespace torch
