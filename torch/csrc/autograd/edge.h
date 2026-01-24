#pragma once

#include <memory>
#include <cstdint>

namespace torch {
namespace autograd {

// Forward declaration
struct Node;

// ============================================================================
// Edge - Connection in the Computational Graph
// ============================================================================
// An Edge represents a connection between a Node's output and another Node's
// input. It stores a shared_ptr to the function that produces the gradient
// and the index of which output of that function this edge corresponds to.

struct Edge {
    // The function this edge points to (the producer of the gradient)
    std::shared_ptr<Node> function;

    // Which output of the function this edge corresponds to
    // (functions can have multiple outputs)
    uint32_t input_nr;

    Edge() noexcept : function(nullptr), input_nr(0) {}

    Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
        : function(std::move(function_)), input_nr(input_nr_) {}

    // Check if edge is valid (has a function)
    bool is_valid() const noexcept {
        return function != nullptr;
    }

    // Comparison operators for use in containers
    bool operator==(const Edge& other) const noexcept {
        return function == other.function && input_nr == other.input_nr;
    }

    bool operator!=(const Edge& other) const noexcept {
        return !(*this == other);
    }
};

} // namespace autograd
} // namespace torch
