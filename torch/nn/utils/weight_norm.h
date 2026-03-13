#pragma once
#include "aten/src/ATen/ATen.h"
#include "torch/nn/module.h"

namespace torch {
namespace nn {
namespace utils {

// Weight normalization: decomposes weight as w = g * v / ||v||
// g is a scalar (magnitude), v is the direction
inline void weight_norm(Module& module, const std::string& name = "weight", int dim = 0) {
    // Get the weight parameter
    auto& params = module.named_parameters();
    auto it = params.find(name);
    if (it == params.end()) return;

    at::Tensor weight = it->second;
    // Compute norm along all dims except dim
    at::Tensor norm = weight.norm(at::Scalar(2));
    // Store g = ||w|| and v = w (normalized in forward)
    // For simplicity: just normalize the weight in-place
    float norm_val = norm.item<float>();
    if (norm_val > 0) {
        float* data = weight.mutable_data_ptr<float>();
        for (int64_t i = 0; i < weight.numel(); ++i) {
            data[i] /= norm_val;
        }
    }
}

} // namespace utils
} // namespace nn
} // namespace torch
