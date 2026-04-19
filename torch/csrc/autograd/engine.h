#pragma once

#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include <queue>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <utility>

namespace torch {
namespace autograd {

using at::Tensor;

// ============================================================================
// GraphTask - Represents a single backward pass
// ============================================================================

struct GraphTask {
    std::vector<Edge> output_edges;
    variable_list output_grads;
    bool retain_graph = false;
    bool create_graph = false;

    // All nodes visited during this backward pass (for cleanup)
    std::vector<Node*> all_nodes;

    void reset() {
        output_edges.clear();
        output_grads.clear();
        retain_graph = false;
        create_graph = false;
        // Reset graph state on all visited nodes
        for (Node* n : all_nodes) {
            n->reset_graph_state();
        }
        all_nodes.clear();
    }
};

// ============================================================================
// NodeTask - A node to be executed in the backward pass
// ============================================================================

struct NodeTask {
    std::shared_ptr<Node> fn;
    variable_list inputs;

    NodeTask(std::shared_ptr<Node> fn_, variable_list inputs_)
        : fn(std::move(fn_)), inputs(std::move(inputs_)) {}

    // For priority queue ordering (by sequence number)
    bool operator<(const NodeTask& other) const {
        // Higher sequence numbers should be processed first (reverse topo order)
        return fn->sequence_nr() < other.fn->sequence_nr();
    }
};

// ============================================================================
// Engine - The Backward Execution Engine
// ============================================================================
// The Engine is responsible for executing the backward pass. It traverses
// the computational graph in reverse topological order, calling each node's
// apply() method to compute gradients.
//
// The algorithm:
// 1. Count dependencies for all nodes reachable from outputs
// 2. Start with output nodes (grads provided by user)
// 3. Use a priority queue to process nodes in reverse topological order
// 4. For each node, call apply() and propagate gradients to next nodes
// 5. Accumulate gradients when multiple paths lead to the same node

class Engine {
public:
    static Engine& get_default_engine() {
        static Engine engine;
        return engine;
    }

    // Execute backward pass
    // outputs: tensors to differentiate
    // grad_outputs: gradients with respect to outputs
    // retain_graph: whether to retain computation graph for future backward
    // create_graph: whether to create graph of backward pass
    variable_list execute(
        const edge_list& roots,
        const variable_list& grad_outputs,
        bool retain_graph = false,
        bool create_graph = false,
        const edge_list& inputs = {}
    );

    // Simplified backward for a single tensor
    void backward(
        const Tensor& output,
        const Tensor& grad_output,
        bool retain_graph = false,
        bool create_graph = false
    );

    // Compute gradients for specific inputs
    variable_list grad(
        const variable_list& outputs,
        const variable_list& inputs,
        const variable_list& grad_outputs,
        bool retain_graph = false,
        bool create_graph = false,
        bool allow_unused = false
    );

private:
    Engine() = default;

    // FIX 1.2: removed cached_task_ member — data race on singleton.
    // Now thread_local in execute().

    // Count dependencies for all nodes
    void compute_dependencies(
        GraphTask& task,
        const edge_list& roots
    );

    // Validate inputs before execution
    void validate_inputs(
        const edge_list& roots,
        const variable_list& grad_outputs
    );

    // Execute a single node
    void execute_node(
        GraphTask& task,
        const NodeTask& node_task,
        std::priority_queue<NodeTask>& ready_queue
    );

    // Accumulate gradient for a node
    void accumulate_grad(
        GraphTask& task,
        Node* node,
        size_t input_nr,
        Tensor grad
    );

    // Check if a node is ready to execute
    bool is_ready(GraphTask& task, Node* node);
};

// ============================================================================
// Implementation
// ============================================================================

inline void Engine::compute_dependencies(
    GraphTask& task,
    const edge_list& roots
) {
    // BFS using Node-internal state (no hash maps!)
    std::queue<Node*> queue;

    for (const auto& root : roots) {
        if (root.function) {
            Node* n = root.function.get();
            if (!n->visited_) {
                n->visited_ = true;
                n->dependency_count_ = 0;
                n->accumulated_grad_.clear();
                task.all_nodes.push_back(n);
                queue.push(n);
            }
        }
    }

    while (!queue.empty()) {
        Node* fn = queue.front();
        queue.pop();

        for (const auto& edge : fn->next_edges()) {
            if (edge.function) {
                Node* next = edge.function.get();
                next->dependency_count_++;

                if (!next->visited_) {
                    next->visited_ = true;
                    next->accumulated_grad_.clear();
                    task.all_nodes.push_back(next);
                    queue.push(next);
                }
            }
        }
    }
}

inline void Engine::validate_inputs(
    const edge_list& roots,
    const variable_list& grad_outputs
) {
    if (roots.size() != grad_outputs.size()) {
        throw std::runtime_error(
            "backward: Number of gradients doesn't match number of outputs"
        );
    }

    for (size_t i = 0; i < roots.size(); ++i) {
        if (!roots[i].function && grad_outputs[i].defined()) {
            // This is okay - it means this output doesn't require grad
        }
    }
}

inline void Engine::accumulate_grad(
    GraphTask& task,
    Node* node,
    size_t input_nr,
    Tensor grad
) {
    if (!grad.defined()) {
        return;
    }

    auto& grads = node->accumulated_grad_;
    if (grads.empty()) {
        grads.resize(node->num_inputs());
    }

    if (input_nr >= grads.size()) {
        grads.resize(input_nr + 1);
    }

    if (!grads[input_nr].defined()) {
        // First gradient — just move, no copy, no allocation
        grads[input_nr] = std::move(grad);
    } else {
        // In-place accumulate using hot::add_inplace — bypasses Tensor dispatch.
        // Avoids: add_ -> operator overload -> type check -> allocate result.
        // On Elbrus E2K, each tensor allocation = malloc syscall.
        auto& existing = grads[input_nr];
        const int64_t n = existing.numel();
        if (n == grad.numel() && existing.is_contiguous() && grad.is_contiguous()
            && existing.dtype() == c10::ScalarType::Float
            && grad.dtype() == c10::ScalarType::Float) {
            // FIX Bug3: only float32 fast-path
            at::native::hot::add_inplace(
                existing.mutable_data_ptr<float>(),
                grad.data_ptr<float>(), n);
        } else {
            existing.add_(grad);
        }
    }
}

inline bool Engine::is_ready(GraphTask& /*task*/, Node* node) {
    return node->dependency_count_ <= 0;
}

inline void Engine::execute_node(
    GraphTask& task,
    const NodeTask& node_task,
    std::priority_queue<NodeTask>& ready_queue
) {
    auto& fn = node_task.fn;

    // create_graph=true: leave autograd enabled inside apply() so that gradient
    // tensors produced by backward also carry grad_fn and can be differentiated
    // again (double backward).  create_graph=false (default): run under
    // NoGradGuard so interior ops (mul, add) don't spawn spurious autograd nodes.
    variable_list grads;
    if (task.create_graph) {
        grads = fn->apply(variable_list(node_task.inputs));
    } else {
        NoGradGuard no_grad;
        grads = fn->apply(variable_list(node_task.inputs));
    }

    // Gradient tensors produced by backward are float32 contiguous CPU — mark trusted
    for (auto& g : grads) {
        if (g.defined() && g.is_cpu() && g.dtype() == c10::ScalarType::Float && g.is_contiguous()) {
            g.set_trusted(true);
        }
    }

    // Propagate gradients to next nodes
    const auto& edges = fn->next_edges();

    for (size_t i = 0; i < grads.size() && i < edges.size(); ++i) {
        const auto& edge = edges[i];
        if (!edge.function) {
            continue;
        }

        if (grads[i].defined()) {
            accumulate_grad(task, edge.function.get(), edge.input_nr, grads[i]);
        }

        // Decrement dependency count (Node-internal, no hash lookup!)
        Node* next_node = edge.function.get();
        next_node->dependency_count_--;

        // If all dependencies satisfied, add to ready queue
        if (next_node->dependency_count_ == 0) {
            ready_queue.emplace(edge.function, std::move(next_node->accumulated_grad_));
            next_node->accumulated_grad_.clear();
        }
    }

    // CRITICAL: Release node resources if not retaining graph
    // This breaks circular references and allows memory to be freed
    if (!task.retain_graph) {
        fn->release();
    }
}

inline variable_list Engine::execute(
    const edge_list& roots,
    const variable_list& grad_outputs,
    bool retain_graph,
    bool create_graph,
    const edge_list& inputs
) {
    validate_inputs(roots, grad_outputs);

    // FIX: stack-local GraphTask (thread_local broke re-entrancy for double backward)
    GraphTask task;
    task.reset();
    task.retain_graph = retain_graph;
    task.create_graph = create_graph;

    // Compute dependencies
    compute_dependencies(task, roots);

    // Initialize ready queue with root nodes
    std::priority_queue<NodeTask> ready_queue;

    for (size_t i = 0; i < roots.size(); ++i) {
        if (roots[i].function) {
            auto fn = roots[i].function;
            accumulate_grad(task, fn.get(), roots[i].input_nr, grad_outputs[i]);

            if (is_ready(task, fn.get())) {
                ready_queue.emplace(fn, std::move(fn->accumulated_grad_));
                fn->accumulated_grad_.clear();
            }
        }
    }

    // Process nodes in reverse topological order
    while (!ready_queue.empty()) {
        NodeTask node_task = std::move(const_cast<NodeTask&>(ready_queue.top()));
        ready_queue.pop();
        execute_node(task, node_task, ready_queue);
    }

    // Collect results for requested inputs (from Node-internal state)
    variable_list result;
    if (!inputs.empty()) {
        result.reserve(inputs.size());
        for (const auto& input : inputs) {
            Node* fn = input.function.get();
            if (fn && input.input_nr < fn->accumulated_grad_.size()) {
                result.push_back(fn->accumulated_grad_[input.input_nr]);
            } else {
                result.push_back(Tensor());
            }
        }
    }

    // Clean up Node-internal state for all visited nodes
    task.reset();

    return result;
}

inline void Engine::backward(
    const Tensor& output,
    const Tensor& grad_output,
    bool retain_graph,
    bool create_graph
) {
    edge_list roots;
    variable_list grads;

    auto edge = gradient_edge(output);
    if (edge.function) {
        roots.push_back(edge);
        if (grad_output.defined()) {
            grads.push_back(grad_output);
        } else {
            // Create ones tensor on same device as output
            Tensor ones = at::ones(output.sizes());
#ifdef PT_USE_NMCARD
            if (output.is_nmcard()) {
                ones = at::to_nmcard(ones);
            }
#endif
#ifdef PT_USE_CUDA
            if (output.is_cuda()) {
                ones = at::to_cuda(ones);
            }
#endif
            grads.push_back(ones);
        }
    }

    execute(roots, grads, retain_graph, create_graph);
}

inline variable_list Engine::grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs,
    bool retain_graph,
    bool create_graph,
    bool allow_unused
) {
    edge_list roots;
    variable_list grads;

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto edge = gradient_edge(outputs[i]);
        roots.push_back(edge);
        if (i < grad_outputs.size() && grad_outputs[i].defined()) {
            grads.push_back(grad_outputs[i]);
        } else {
            // Create ones tensor on same device as output
            Tensor ones = at::ones(outputs[i].sizes());
#ifdef PT_USE_NMCARD
            if (outputs[i].is_nmcard()) {
                ones = at::to_nmcard(ones);
            }
#endif
#ifdef PT_USE_CUDA
            if (outputs[i].is_cuda()) {
                ones = at::to_cuda(ones);
            }
#endif
            grads.push_back(ones);
        }
    }

    edge_list input_edges;
    for (const auto& input : inputs) {
        input_edges.push_back(gradient_edge(input));
    }

    return execute(roots, grads, retain_graph, create_graph, input_edges);
}

// ============================================================================
// Free functions for backward
// ============================================================================

inline void backward(
    const variable_list& tensors,
    const variable_list& grad_tensors = {},
    bool retain_graph = false,
    bool create_graph = false
) {
    edge_list roots;
    variable_list grads;

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto edge = gradient_edge(tensors[i]);
        if (edge.function) {
            roots.push_back(edge);
            if (i < grad_tensors.size() && grad_tensors[i].defined()) {
                grads.push_back(grad_tensors[i]);
            } else {
                // Create ones tensor on same device as output
                Tensor ones = at::ones(tensors[i].sizes());
#ifdef PT_USE_NMCARD
                if (tensors[i].is_nmcard()) {
                    ones = at::to_nmcard(ones);
                }
#endif
#ifdef PT_USE_CUDA
                if (tensors[i].is_cuda()) {
                    ones = at::to_cuda(ones);
                }
#endif
                grads.push_back(ones);
            }
        }
    }

    Engine::get_default_engine().execute(roots, grads, retain_graph, create_graph);
}

inline variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs = {},
    bool retain_graph = false,
    bool create_graph = false,
    bool allow_unused = false
) {
    return Engine::get_default_engine().grad(
        outputs, inputs, grad_outputs, retain_graph, create_graph, allow_unused
    );
}

} // namespace autograd
} // namespace torch
