#pragma once

#include "torch/csrc/autograd/node.h"
#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/native/cpu/PromeBLAS.h"
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

        Tensor grad_contig = grad.is_contiguous() ? grad : grad.contiguous();
        Tensor input_contig = input_.is_contiguous() ? input_ : input_.contiguous();
        Tensor weight_contig = weight_.is_contiguous() ? weight_ : weight_.contiguous();

        const int64_t M = input_contig.size(0);
        const int64_t K = input_contig.size(1);
        const int64_t N = weight_contig.size(0);

        const float* grad_data = grad_contig.data_ptr<float>();
        (void)K; // Used only for dimension documentation

        // --- grad_input = grad_output @ weight  [M, N] @ [N, K] = [M, K] ---
        Tensor grad_input = grad_contig.mm(weight_contig);

        // --- grad_weight = grad_output^T @ input = [N, M] @ [M, K] = [N, K] ---
        Tensor grad_weight = grad_contig.t().mm(input_contig);

        // --- grad_bias = grad_output.sum(dim=0) [N] ---
        Tensor grad_bias;
        if (has_bias_) {
            grad_bias = at::empty({N});
            float* bias_grad_data = grad_bias.mutable_data_ptr<float>();

            // AVX2-optimized column sum
            constexpr int VW = at::native::tuda::VecF::width;
            // Zero-init
            int64_t j = 0;
            for (; j + VW <= N; j += VW) {
                at::native::tuda::VecF::zero().store(bias_grad_data + j);
            }
            for (; j < N; ++j) bias_grad_data[j] = 0.0f;

            // Accumulate rows
            for (int64_t i = 0; i < M; ++i) {
                const float* row = grad_data + i * N;
                j = 0;
                for (; j + VW <= N; j += VW) {
                    auto acc = at::native::tuda::VecF::load(bias_grad_data + j);
                    auto val = at::native::tuda::VecF::load(row + j);
                    (acc + val).store(bias_grad_data + j);
                }
                for (; j < N; ++j) bias_grad_data[j] += row[j];
            }
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

        Tensor grad_contig = grad.is_contiguous() ? grad : grad.contiguous();
        Tensor input_contig = input_.is_contiguous() ? input_ : input_.contiguous();
        Tensor weight_contig = weight_.is_contiguous() ? weight_ : weight_.contiguous();
        Tensor output_contig = output_.is_contiguous() ? output_ : output_.contiguous();

        const int64_t M = input_contig.size(0);
        const int64_t N = weight_contig.size(0);

        const float* grad_data = grad_contig.data_ptr<float>();
        const float* output_data = output_contig.data_ptr<float>();

        // --- Step 1: Apply relu backward mask into a temporary ---
        // grad_relu = grad_output * (output > 0)
        Tensor grad_relu = at::empty({M, N});
        float* gr_data = grad_relu.mutable_data_ptr<float>();

        const int64_t total = M * N;
        for (int64_t idx = 0; idx < total; ++idx) {
            gr_data[idx] = (output_data[idx] > 0.0f) ? grad_data[idx] : 0.0f;
        }

        // --- Step 2: grad_input = grad_relu @ weight [M, K] ---
        Tensor grad_input = grad_relu.mm(weight_contig);

        // --- Step 3: grad_weight = grad_relu^T @ input [N, K] ---
        Tensor grad_weight = grad_relu.t().mm(input_contig);

        // --- Step 4: grad_bias = grad_relu.sum(dim=0) [N] ---
        Tensor grad_bias;
        if (has_bias_) {
            grad_bias = at::empty({N});
            float* bias_grad = grad_bias.mutable_data_ptr<float>();

            constexpr int VW = at::native::tuda::VecF::width;
            int64_t j = 0;
            for (; j + VW <= N; j += VW) {
                at::native::tuda::VecF::zero().store(bias_grad + j);
            }
            for (; j < N; ++j) bias_grad[j] = 0.0f;

            for (int64_t i = 0; i < M; ++i) {
                const float* row = gr_data + i * N;
                j = 0;
                for (; j + VW <= N; j += VW) {
                    auto acc = at::native::tuda::VecF::load(bias_grad + j);
                    auto val = at::native::tuda::VecF::load(row + j);
                    (acc + val).store(bias_grad + j);
                }
                for (; j < N; ++j) bias_grad[j] += row[j];
            }
        }

        // Release saved tensors
        input_ = Tensor();
        weight_ = Tensor();
        output_ = Tensor();

        return {grad_input, grad_weight, grad_bias};
    }

    std::string name() const override { return "FusedLinearReluBackward"; }
};

} // namespace autograd
} // namespace torch
