#pragma once
#include "aten/src/ATen/ATen.h"
#include "torch/nn/module.h"

namespace torch {
namespace nn {
namespace utils {

// Spectral normalization: normalize weight by its largest singular value
// Uses power iteration to estimate the largest singular value
inline void spectral_norm(at::Tensor& weight, int n_power_iterations = 1) {
    // Reshape weight to 2D: (out_features, in_features)
    int64_t h = weight.size(0);
    int64_t w = weight.numel() / h;
    at::Tensor weight_mat = weight.view({h, w});

    // Initialize u, v vectors randomly
    at::Tensor u = at::randn({h});
    at::Tensor v = at::randn({w});

    // Normalize
    float u_norm = u.norm(at::Scalar(2)).item<float>();
    float v_norm = v.norm(at::Scalar(2)).item<float>();
    if (u_norm > 0) { float* d = u.mutable_data_ptr<float>(); for(int64_t i=0;i<h;++i) d[i]/=u_norm; }
    if (v_norm > 0) { float* d = v.mutable_data_ptr<float>(); for(int64_t i=0;i<w;++i) d[i]/=v_norm; }

    // Power iteration
    for (int i = 0; i < n_power_iterations; ++i) {
        v = at::native::mv(weight_mat.t(), u);
        v_norm = v.norm(at::Scalar(2)).item<float>();
        if (v_norm > 0) { float* d = v.mutable_data_ptr<float>(); for(int64_t j=0;j<w;++j) d[j]/=v_norm; }

        u = at::native::mv(weight_mat, v);
        u_norm = u.norm(at::Scalar(2)).item<float>();
        if (u_norm > 0) { float* d = u.mutable_data_ptr<float>(); for(int64_t j=0;j<h;++j) d[j]/=u_norm; }
    }

    // sigma = u^T W v
    float sigma = at::native::dot(u, at::native::mv(weight_mat, v)).item<float>();

    // Normalize weight
    if (sigma > 0) {
        float* data = weight.mutable_data_ptr<float>();
        for (int64_t i = 0; i < weight.numel(); ++i) {
            data[i] /= sigma;
        }
    }
}

} // namespace utils
} // namespace nn
} // namespace torch
