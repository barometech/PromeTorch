#pragma once

#include "torch/csrc/autograd/node.h"
#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/native/cpu/PromeBLAS.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"

namespace torch {
namespace autograd {

using at::Tensor;

// ============================================================================
// FusedLinearBackward — fused mm + bias_add backward
// ============================================================================
// Forward:  output = input @ weight^T + bias
// Backward: grad_input  = grad_output @ weight
//           grad_weight = input^T @ grad_output
//           grad_bias   = grad_output.sum(dim=0)
// All computed in a single backward node (instead of MmBackward + AddBackward
// + TBackward = 3 nodes + 3 intermediate tensors).
//
// Saves: input [M, K], weight [N, K] (original, NOT transposed)
// Receives: grad_output [M, N]
// Outputs: grad_input [M, K], grad_weight [N, K], grad_bias [N]

struct FusedLinearBackward : public Node {
    Tensor input_;   // [M, K]
    Tensor weight_;  // [N, K]
    bool has_bias_;

    FusedLinearBackward(const Tensor& input, const Tensor& weight, bool has_bias)
        : input_(input), weight_(weight), has_bias_(has_bias) {}

    void release_saved_tensors() override {
        input_ = Tensor();
        weight_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // grad_output [M, N]
        if (!grad.defined()) {
            return {Tensor(), Tensor(), Tensor()};
        }

        // Saved tensors from Linear forward are ALWAYS contiguous:
        //   input_ comes from user (must be contiguous for mm)
        //   weight_ comes from at::empty (always contiguous)
        //   grad comes from upstream backward (contiguous by construction)
        // Skip .contiguous() checks — each is a branch + potential malloc+memcpy.
        const int64_t M = input_.size(0);
        const int64_t K = input_.size(1);
        const int64_t N = weight_.size(0);

        const float* grad_data = grad.data_ptr<float>();
        const float* x_data = input_.data_ptr<float>();
        const float* w_data = weight_.data_ptr<float>();

        // --- grad_input = grad_output @ weight  [M,N] @ [N,K] = [M,K] ---
        Tensor grad_input = at::empty({M, K});
        at::native::hot::sgemm(M, N, K, 1.0f,
                               grad_data, N, w_data, K,
                               0.0f, grad_input.mutable_data_ptr<float>(), K);

        // --- grad_weight = grad_output^T @ input = [N,M] @ [M,K] = [N,K] ---
        Tensor grad_weight = at::empty({N, K});
        at::native::hot::sgemm_tn(M, N, K, 1.0f,
                                   grad_data, N, x_data, K,
                                   0.0f, grad_weight.mutable_data_ptr<float>(), K);

        // --- grad_bias = grad_output.sum(dim=0) [N] ---
        Tensor grad_bias;
        if (has_bias_) {
            grad_bias = at::empty({N});
            at::native::hot::col_sum(grad_data, grad_bias.mutable_data_ptr<float>(), M, N);
        }

        // Release saved tensors
        input_ = Tensor();
        weight_ = Tensor();

        return {grad_input, grad_weight, grad_bias};
    }

    std::string name() const override { return "FusedLinearBackward"; }
};

// ============================================================================
// FusedLinearReluBackward — fused mm + bias_add + relu backward
// ============================================================================
// Forward:  output = relu(input @ weight^T + bias)
// Backward: grad_relu   = grad_output * (output > 0)   [relu mask from output]
//           grad_input  = grad_relu @ weight
//           grad_weight = input^T @ grad_relu
//           grad_bias   = grad_relu.sum(dim=0)
//
// ALL operations use hot:: compiled functions. ZERO intermediate Tensor
// allocations from mm()/t() — only the 4 output tensors are allocated.
//
// Saves: input [M, K], weight [N, K], output [M, N] (post-relu, for mask)
// Receives: grad_output [M, N]
// Outputs: grad_input [M, K], grad_weight [N, K], grad_bias [N]

struct FusedLinearReluBackward : public Node {
    Tensor input_;   // [M, K]
    Tensor weight_;  // [N, K]
    Tensor output_;  // [M, N] — post-relu output, used as mask
    bool has_bias_;

    FusedLinearReluBackward(const Tensor& input, const Tensor& weight,
                            const Tensor& output, bool has_bias)
        : input_(input), weight_(weight), output_(output), has_bias_(has_bias) {}

    void release_saved_tensors() override {
        input_ = Tensor();
        weight_ = Tensor();
        output_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // grad_output [M, N]
        if (!grad.defined()) {
            return {Tensor(), Tensor(), Tensor()};
        }

        // All saved tensors are contiguous by construction (from Linear forward).
        // Skip .contiguous() — eliminates 4 branch + potential malloc+memcpy.
        const int64_t M = input_.size(0);
        const int64_t K = input_.size(1);
        const int64_t N = weight_.size(0);

        const float* grad_data = grad.data_ptr<float>();
        const float* output_data = output_.data_ptr<float>();
        const float* w_data = weight_.data_ptr<float>();
        const float* x_data = input_.data_ptr<float>();

        // --- Step 1: Apply relu backward mask ---
        // grad_relu = grad_output * (output > 0)
        // This is the ONLY intermediate allocation (unavoidable: needed by 3 consumers)
        Tensor grad_relu = at::empty({M, N});
        float* gr_data = grad_relu.mutable_data_ptr<float>();
        at::native::hot::relu_mask_mul(grad_data, output_data, gr_data, M * N);

        // --- Step 2: grad_input = grad_relu @ weight [M,N] @ [N,K] = [M,K] ---
        Tensor grad_input = at::empty({M, K});
        at::native::hot::sgemm(M, N, K, 1.0f,
                               gr_data, N, w_data, K,
                               0.0f, grad_input.mutable_data_ptr<float>(), K);

        // --- Step 3: grad_weight = grad_relu^T @ input [N,M] @ [M,K] = [N,K] ---
        Tensor grad_weight = at::empty({N, K});
        at::native::hot::sgemm_tn(M, N, K, 1.0f,
                                   gr_data, N, x_data, K,
                                   0.0f, grad_weight.mutable_data_ptr<float>(), K);

        // --- Step 4: grad_bias = grad_relu.sum(dim=0) [N] ---
        Tensor grad_bias;
        if (has_bias_) {
            grad_bias = at::empty({N});
            at::native::hot::col_sum(gr_data, grad_bias.mutable_data_ptr<float>(), M, N);
        }

        // Release saved tensors
        input_ = Tensor();
        weight_ = Tensor();
        output_ = Tensor();

        return {grad_input, grad_weight, grad_bias};
    }

    std::string name() const override { return "FusedLinearReluBackward"; }
};

// ============================================================================
// FusedMLPBackward — backward for entire MLP layer chain
// ============================================================================
// For a multi-layer MLP: y = relu(relu(x @ W1^T + b1) @ W2^T + b2) @ W3^T + b3
// Instead of 3 * (ReluBackward + AddBackward + MmBackward) = 9 nodes,
// this produces 1 node with ZERO intermediate gradient tensors between layers.
//
// Each layer's backward is fused:
//   grad_masked = grad * relu_mask(output)
//   grad_input  = grad_masked @ weight        (sgemm)
//   grad_weight = grad_masked^T @ input       (sgemm_tn)
//   grad_bias   = grad_masked.sum(dim=0)      (col_sum)
//
// The grad_input of layer L becomes the grad_output of layer L-1.
// This eliminates all intermediate tensor allocations between layers.

struct FusedMLPBackward : public Node {
    struct LayerData {
        Tensor input;    // [M, in_features] — input to this layer
        Tensor weight;   // [out_features, in_features]
        Tensor output;   // [M, out_features] — post-relu output (for relu mask)
        bool has_bias;
        bool has_relu;   // Last layer typically has no relu
    };
    std::vector<LayerData> layers_;

    FusedMLPBackward(std::vector<LayerData> layers)
        : layers_(std::move(layers)) {}

    void release_saved_tensors() override {
        for (auto& layer : layers_) {
            layer.input = Tensor();
            layer.weight = Tensor();
            layer.output = Tensor();
        }
        layers_.clear();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_out = grads[0];
        if (!grad_out.defined()) {
            // Return empty grads for all inputs (input + per-layer weight + bias)
            size_t total = 1;  // input grad
            for (auto& l : layers_) total += (l.has_bias ? 2 : 1);
            return variable_list(total, Tensor());
        }

        // Process layers in reverse order (backward through the MLP)
        // grad_out flows backward: last layer -> ... -> first layer
        Tensor current_grad = grad_out.is_contiguous() ? grad_out : grad_out.contiguous();
        variable_list results;

        // Pre-allocate: we'll fill in reverse order
        // Order: grad_input, grad_w1, grad_b1, grad_w2, grad_b2, ...
        // We build per-layer results then assemble at the end
        struct LayerGrads {
            Tensor grad_weight;
            Tensor grad_bias;
        };
        std::vector<LayerGrads> layer_grads(layers_.size());

        for (int64_t l = static_cast<int64_t>(layers_.size()) - 1; l >= 0; --l) {
            auto& layer = layers_[l];

            // All saved tensors are contiguous by construction (from MLP forward).
            // Skip .contiguous() — eliminates 2-3 branches + potential malloc+memcpy per layer.
            const int64_t M = layer.input.size(0);
            const int64_t K = layer.input.size(1);
            const int64_t N = layer.weight.size(0);

            const float* w_data = layer.weight.data_ptr<float>();
            const float* x_data = layer.input.data_ptr<float>();

            // Apply relu mask if this layer has relu
            const float* gm_data;
            Tensor grad_masked;
            if (layer.has_relu && layer.output.defined()) {
                grad_masked = at::empty({M, N});
                gm_data = grad_masked.mutable_data_ptr<float>();
                at::native::hot::relu_mask_mul(
                    current_grad.data_ptr<float>(),
                    layer.output.data_ptr<float>(),
                    grad_masked.mutable_data_ptr<float>(),
                    M * N);
            } else {
                gm_data = current_grad.data_ptr<float>();
            }

            // grad_weight = grad_masked^T @ input  [N,K]
            layer_grads[l].grad_weight = at::empty({N, K});
            at::native::hot::sgemm_tn(M, N, K, 1.0f,
                                       gm_data, N, x_data, K,
                                       0.0f, layer_grads[l].grad_weight.mutable_data_ptr<float>(), K);

            // grad_bias = grad_masked.sum(dim=0)  [N]
            if (layer.has_bias) {
                layer_grads[l].grad_bias = at::empty({N});
                at::native::hot::col_sum(gm_data,
                                          layer_grads[l].grad_bias.mutable_data_ptr<float>(),
                                          M, N);
            }

            // grad_input = grad_masked @ weight  [M,K]
            // This becomes current_grad for the next (earlier) layer
            if (l > 0) {
                Tensor grad_input = at::empty({M, K});
                at::native::hot::sgemm(M, N, K, 1.0f,
                                       gm_data, N, w_data, K,
                                       0.0f, grad_input.mutable_data_ptr<float>(), K);
                current_grad = grad_input;
            } else {
                // First layer: grad_input is an output
                Tensor grad_input = at::empty({M, K});
                at::native::hot::sgemm(M, N, K, 1.0f,
                                       gm_data, N, w_data, K,
                                       0.0f, grad_input.mutable_data_ptr<float>(), K);
                current_grad = grad_input;
            }
        }

        // Assemble results: grad_input, then per-layer (grad_weight, grad_bias)
        results.push_back(current_grad);  // grad_input for first layer's input
        for (size_t l = 0; l < layers_.size(); ++l) {
            results.push_back(layer_grads[l].grad_weight);
            if (layers_[l].has_bias) {
                results.push_back(layer_grads[l].grad_bias);
            }
        }

        // Release saved tensors
        for (auto& layer : layers_) {
            layer.input = Tensor();
            layer.weight = Tensor();
            layer.output = Tensor();
        }

        return results;
    }

    std::string name() const override { return "FusedMLPBackward"; }
};

// ============================================================================
// LowRankLinearBackward — backward for low-rank linear: out = x @ B^T @ A^T + bias
// ============================================================================
// Forward: temp = x @ B^T  [M, rank]
//          out  = temp @ A^T [M, N]   (+ bias)
//
// Backward:
//   grad_x    = grad @ A @ B                        [M, K]
//   grad_A    = grad^T @ temp                       [N, rank]
//   grad_B    = (grad @ A)^T @ x                    [rank, K]
//   grad_bias = grad.sum(dim=0)                     [N]
//
// Saves: input [M, K], A [N, rank], B [rank, K], temp [M, rank]

struct LowRankLinearBackward : public Node {
    Tensor input_;  // [M, K]
    Tensor A_;      // [N, rank]
    Tensor B_;      // [rank, K]
    Tensor temp_;   // [M, rank] = input @ B^T
    bool has_bias_;

    LowRankLinearBackward(const Tensor& input, const Tensor& A, const Tensor& B,
                          const Tensor& temp, bool has_bias)
        : input_(input), A_(A), B_(B), temp_(temp), has_bias_(has_bias) {}

    void release_saved_tensors() override {
        input_ = Tensor();
        A_ = Tensor();
        B_ = Tensor();
        temp_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];  // grad_output [M, N]
        if (!grad.defined()) {
            return {Tensor(), Tensor(), Tensor(), Tensor()};
        }

        // All saved tensors are contiguous by construction.
        const int64_t M = input_.size(0);
        const int64_t K = input_.size(1);
        const int64_t N = A_.size(0);
        const int64_t rank = A_.size(1);

        const float* grad_data = grad.data_ptr<float>();
        const float* x_data = input_.data_ptr<float>();
        const float* a_data = A_.data_ptr<float>();
        const float* b_data = B_.data_ptr<float>();
        const float* temp_data = temp_.data_ptr<float>();

        // grad_input = grad @ A @ B  [M,N]@[N,rank]@[rank,K] = [M,K]
        Tensor grad_A_product = at::empty({M, rank});
        at::native::hot::sgemm(M, N, rank, 1.0f,
                               grad_data, N, a_data, rank,
                               0.0f, grad_A_product.mutable_data_ptr<float>(), rank);
        Tensor grad_input = at::empty({M, K});
        at::native::hot::sgemm(M, rank, K, 1.0f,
                               grad_A_product.data_ptr<float>(), rank, b_data, K,
                               0.0f, grad_input.mutable_data_ptr<float>(), K);

        // grad_A = grad^T @ temp  [N,M]@[M,rank] = [N, rank]
        Tensor grad_A = at::empty({N, rank});
        at::native::hot::sgemm_tn(M, N, rank, 1.0f,
                                   grad_data, N, temp_data, rank,
                                   0.0f, grad_A.mutable_data_ptr<float>(), rank);

        // grad_B = (grad @ A)^T @ input = [rank,M]@[M,K] = [rank, K]
        Tensor grad_B = at::empty({rank, K});
        at::native::hot::sgemm_tn(M, rank, K, 1.0f,
                                   grad_A_product.data_ptr<float>(), rank, x_data, K,
                                   0.0f, grad_B.mutable_data_ptr<float>(), K);

        // grad_bias = grad.sum(dim=0) [N]
        Tensor grad_bias;
        if (has_bias_) {
            grad_bias = at::empty({N});
            at::native::hot::col_sum(grad_data, grad_bias.mutable_data_ptr<float>(), M, N);
        }

        input_ = Tensor();
        A_ = Tensor();
        B_ = Tensor();
        temp_ = Tensor();

        return {grad_input, grad_A, grad_B, grad_bias};
    }

    std::string name() const override { return "LowRankLinearBackward"; }
};

} // namespace autograd
} // namespace torch
