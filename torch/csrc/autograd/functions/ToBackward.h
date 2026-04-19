#pragma once

#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "aten/src/ATen/ATen.h"
#include "c10/core/ScalarType.h"

namespace torch {
namespace autograd {

using at::Tensor;

// ============================================================================
// ToBackward - Backward for dtype cast (e.g. FP32 -> FP16 in autocast)
// ============================================================================
// Forward: result = input.to(target_dtype)
// Backward: grad_input = grad_output.to(source_dtype)
//
// The cast itself has identity Jacobian within the overlap of the dtypes; the
// only thing backward needs to do is bring the upstream gradient back to the
// dtype of the source tensor so AccumulateGrad can store it consistently with
// other contributions to the same leaf. We save only source_dtype_ — the
// target dtype is implied by the upstream gradient.
struct ToBackward : public Node {
    c10::ScalarType source_dtype_;

    explicit ToBackward(c10::ScalarType source_dtype)
        : source_dtype_(source_dtype) {}

    variable_list apply(variable_list&& grads) override {
        auto& grad = grads[0];
        if (!grad.defined()) return {Tensor()};
        // Fast path: dtype already matches (no-op cast on backward).
        if (grad.dtype() == source_dtype_) {
            return {grad};
        }
        return {grad.to(source_dtype_)};
    }

    std::string name() const override { return "ToBackward"; }
};

} // namespace autograd
} // namespace torch
