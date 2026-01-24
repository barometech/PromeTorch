#pragma once

#include "aten/src/ATen/ATen.h"
#include <cmath>
#include <random>

namespace torch {
namespace nn {
namespace init {

using at::Tensor;

// ============================================================================
// Fan Calculation
// ============================================================================

enum class FanMode {
    FanIn,
    FanOut
};

inline std::pair<int64_t, int64_t> calculate_fan_in_and_fan_out(const Tensor& tensor) {
    int64_t dim = tensor.dim();
    if (dim < 2) {
        throw std::runtime_error(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        );
    }

    int64_t num_input_fmaps = tensor.size(1);
    int64_t num_output_fmaps = tensor.size(0);

    int64_t receptive_field_size = 1;
    if (dim > 2) {
        for (int64_t i = 2; i < dim; ++i) {
            receptive_field_size *= tensor.size(i);
        }
    }

    int64_t fan_in = num_input_fmaps * receptive_field_size;
    int64_t fan_out = num_output_fmaps * receptive_field_size;

    return {fan_in, fan_out};
}

// ============================================================================
// Constant Initialization
// ============================================================================

inline Tensor& constant_(Tensor& tensor, double val) {
    tensor.fill_(at::Scalar(val));
    return tensor;
}

inline Tensor& zeros_(Tensor& tensor) {
    return constant_(tensor, 0.0);
}

inline Tensor& ones_(Tensor& tensor) {
    return constant_(tensor, 1.0);
}

// ============================================================================
// Uniform Initialization
// ============================================================================

inline Tensor& uniform_(Tensor& tensor, double a = 0.0, double b = 1.0) {
    float* data = tensor.mutable_data_ptr<float>();
    int64_t n = tensor.numel();

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(static_cast<float>(a), static_cast<float>(b));

    for (int64_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }

    return tensor;
}

// ============================================================================
// Normal Initialization
// ============================================================================

inline Tensor& normal_(Tensor& tensor, double mean = 0.0, double std = 1.0) {
    float* data = tensor.mutable_data_ptr<float>();
    int64_t n = tensor.numel();

    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(std));

    for (int64_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }

    return tensor;
}

// ============================================================================
// Xavier (Glorot) Initialization
// ============================================================================

inline Tensor& xavier_uniform_(Tensor& tensor, double gain = 1.0) {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    double std = gain * std::sqrt(2.0 / (fan_in + fan_out));
    double a = std::sqrt(3.0) * std;
    return uniform_(tensor, -a, a);
}

inline Tensor& xavier_normal_(Tensor& tensor, double gain = 1.0) {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    double std = gain * std::sqrt(2.0 / (fan_in + fan_out));
    return normal_(tensor, 0.0, std);
}

// ============================================================================
// Kaiming (He) Initialization
// ============================================================================

inline double calculate_gain(const std::string& nonlinearity, double param = 0.01) {
    if (nonlinearity == "linear" || nonlinearity == "sigmoid") {
        return 1.0;
    } else if (nonlinearity == "tanh") {
        return 5.0 / 3.0;
    } else if (nonlinearity == "relu") {
        return std::sqrt(2.0);
    } else if (nonlinearity == "leaky_relu") {
        return std::sqrt(2.0 / (1 + param * param));
    } else if (nonlinearity == "selu") {
        return 3.0 / 4.0;
    }
    throw std::runtime_error("Unknown nonlinearity: " + nonlinearity);
}

inline Tensor& kaiming_uniform_(
    Tensor& tensor,
    double a = 0,
    FanMode mode = FanMode::FanIn,
    const std::string& nonlinearity = "leaky_relu"
) {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    int64_t fan = (mode == FanMode::FanIn) ? fan_in : fan_out;
    double gain = calculate_gain(nonlinearity, a);
    double std = gain / std::sqrt(static_cast<double>(fan));
    double bound = std::sqrt(3.0) * std;
    return uniform_(tensor, -bound, bound);
}

inline Tensor& kaiming_normal_(
    Tensor& tensor,
    double a = 0,
    FanMode mode = FanMode::FanIn,
    const std::string& nonlinearity = "leaky_relu"
) {
    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
    int64_t fan = (mode == FanMode::FanIn) ? fan_in : fan_out;
    double gain = calculate_gain(nonlinearity, a);
    double std = gain / std::sqrt(static_cast<double>(fan));
    return normal_(tensor, 0.0, std);
}

// ============================================================================
// Orthogonal Initialization
// ============================================================================

inline Tensor& orthogonal_(Tensor& tensor, double gain = 1.0) {
    if (tensor.dim() < 2) {
        throw std::runtime_error("Only tensors with 2 or more dimensions are supported");
    }

    int64_t rows = tensor.size(0);
    int64_t cols = tensor.numel() / rows;

    // Create random matrix
    Tensor flat = at::randn({rows, cols});

    // QR decomposition (simplified - in production use LAPACK)
    // For now, use a simple Gram-Schmidt orthogonalization
    float* data = flat.mutable_data_ptr<float>();

    for (int64_t i = 0; i < std::min(rows, cols); ++i) {
        // Normalize column i
        float norm = 0.0f;
        for (int64_t j = 0; j < rows; ++j) {
            norm += data[j * cols + i] * data[j * cols + i];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-10f) {
            for (int64_t j = 0; j < rows; ++j) {
                data[j * cols + i] /= norm;
            }
        }

        // Orthogonalize remaining columns against column i
        for (int64_t k = i + 1; k < cols; ++k) {
            float dot = 0.0f;
            for (int64_t j = 0; j < rows; ++j) {
                dot += data[j * cols + i] * data[j * cols + k];
            }
            for (int64_t j = 0; j < rows; ++j) {
                data[j * cols + k] -= dot * data[j * cols + i];
            }
        }
    }

    // Copy to tensor and apply gain
    tensor.copy_(flat.reshape(tensor.sizes()));
    if (gain != 1.0) {
        tensor.mul_(at::Scalar(gain));
    }

    return tensor;
}

// ============================================================================
// Sparse Initialization
// ============================================================================

inline Tensor& sparse_(Tensor& tensor, double sparsity, double std = 0.01) {
    if (tensor.dim() != 2) {
        throw std::runtime_error("Only tensors with 2 dimensions are supported");
    }

    int64_t rows = tensor.size(0);
    int64_t cols = tensor.size(1);
    int64_t num_zeros = static_cast<int64_t>(std::ceil(sparsity * rows));

    normal_(tensor, 0.0, std);

    float* data = tensor.mutable_data_ptr<float>();

    static std::mt19937 gen(std::random_device{}());

    for (int64_t j = 0; j < cols; ++j) {
        // Randomly select indices to zero out
        std::vector<int64_t> indices(rows);
        for (int64_t i = 0; i < rows; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), gen);

        for (int64_t i = 0; i < num_zeros; ++i) {
            data[indices[i] * cols + j] = 0.0f;
        }
    }

    return tensor;
}

// ============================================================================
// Eye Initialization (for square matrices)
// ============================================================================

inline Tensor& eye_(Tensor& tensor) {
    if (tensor.dim() != 2) {
        throw std::runtime_error("Only 2D tensors are supported");
    }

    zeros_(tensor);
    int64_t n = std::min(tensor.size(0), tensor.size(1));
    float* data = tensor.mutable_data_ptr<float>();
    int64_t cols = tensor.size(1);

    for (int64_t i = 0; i < n; ++i) {
        data[i * cols + i] = 1.0f;
    }

    return tensor;
}

// ============================================================================
// Dirac Initialization (for convolutions)
// ============================================================================

inline Tensor& dirac_(Tensor& tensor, int64_t groups = 1) {
    int64_t dim = tensor.dim();
    if (dim < 3 || dim > 5) {
        throw std::runtime_error("Only 3D, 4D, 5D tensors are supported");
    }

    zeros_(tensor);

    int64_t out_channels = tensor.size(0);
    int64_t in_channels = tensor.size(1);

    int64_t min_dim = std::min(out_channels, in_channels);
    float* data = tensor.mutable_data_ptr<float>();

    // Set center element to 1 for identity-like behavior
    std::vector<int64_t> sizes = tensor.sizes().vec();
    int64_t center_offset = 0;

    // Calculate stride
    int64_t stride = 1;
    for (int64_t i = dim - 1; i >= 2; --i) {
        int64_t center = sizes[i] / 2;
        center_offset += center * stride;
        stride *= sizes[i];
    }

    stride = tensor.numel() / (out_channels * in_channels);

    for (int64_t i = 0; i < min_dim; ++i) {
        int64_t idx = i * in_channels * stride + i * stride + center_offset;
        data[idx] = 1.0f;
    }

    return tensor;
}

} // namespace init
} // namespace nn
} // namespace torch
