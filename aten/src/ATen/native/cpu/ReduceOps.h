#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <cmath>
#include <limits>
#include <tuple>

namespace at {
namespace native {

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum all elements
inline Tensor sum(const Tensor& self) {
    Tensor result = zeros({}, TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "sum", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        scalar_t total = 0;
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            total += data[i];
        }

        result.mutable_data_ptr<scalar_t>()[0] = total;
    });

    return result;
}

// Sum along dimension
inline Tensor sum(const Tensor& self, int64_t dim, bool keepdim = false) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    // Compute result shape
    std::vector<int64_t> result_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) result_shape.push_back(1);
        } else {
            result_shape.push_back(self.size(i));
        }
    }

    Tensor result = zeros(result_shape, TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "sum_dim", [&] {
        const scalar_t* in = self.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();

        // Calculate strides for iteration
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= self.size(i);
        }

        int64_t reduce_size = self.size(dim);

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            inner_size *= self.size(i);
        }

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                scalar_t sum_val = 0;
                for (int64_t r = 0; r < reduce_size; ++r) {
                    int64_t in_idx = (outer * reduce_size + r) * inner_size + inner;
                    sum_val += in[in_idx];
                }
                int64_t out_idx = outer * inner_size + inner;
                out[out_idx] = sum_val;
            }
        }
    });

    return result;
}

// Mean all elements
inline Tensor mean(const Tensor& self) {
    Tensor s = sum(self);

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "mean", [&] {
        s.mutable_data_ptr<scalar_t>()[0] /= static_cast<scalar_t>(self.numel());
    });

    return s;
}

// Mean along dimension
inline Tensor mean(const Tensor& self, int64_t dim, bool keepdim = false) {
    Tensor s = sum(self, dim, keepdim);

    int64_t actual_dim = dim < 0 ? dim + self.dim() : dim;
    int64_t reduce_size = self.size(actual_dim);

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "mean_dim", [&] {
        scalar_t* data = s.mutable_data_ptr<scalar_t>();
        int64_t n = s.numel();

        for (int64_t i = 0; i < n; ++i) {
            data[i] /= static_cast<scalar_t>(reduce_size);
        }
    });

    return s;
}

// Product all elements
inline Tensor prod(const Tensor& self) {
    Tensor result = ones({}, TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "prod", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        scalar_t total = 1;
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            total *= data[i];
        }

        result.mutable_data_ptr<scalar_t>()[0] = total;
    });

    return result;
}

// Max all elements
inline Tensor max(const Tensor& self) {
    PT_CHECK(self.numel() > 0);

    Tensor result = empty({}, TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "max", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        scalar_t max_val = data[0];
        int64_t n = self.numel();

        for (int64_t i = 1; i < n; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
            }
        }

        result.mutable_data_ptr<scalar_t>()[0] = max_val;
    });

    return result;
}

// Max along dimension (returns values and indices)
inline std::tuple<Tensor, Tensor> max(const Tensor& self, int64_t dim, bool keepdim = false) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    std::vector<int64_t> result_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) result_shape.push_back(1);
        } else {
            result_shape.push_back(self.size(i));
        }
    }

    Tensor values = empty(result_shape, TensorOptions().dtype(self.dtype()).device(self.device()));
    Tensor indices = empty(result_shape, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "max_dim", [&] {
        const scalar_t* in = self.data_ptr<scalar_t>();
        scalar_t* out_vals = values.mutable_data_ptr<scalar_t>();
        int64_t* out_idx = indices.mutable_data_ptr<int64_t>();

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= self.size(i);
        }

        int64_t reduce_size = self.size(dim);

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            inner_size *= self.size(i);
        }

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                int64_t first_idx = outer * reduce_size * inner_size + inner;
                scalar_t max_val = in[first_idx];
                int64_t max_idx = 0;

                for (int64_t r = 1; r < reduce_size; ++r) {
                    int64_t in_idx = (outer * reduce_size + r) * inner_size + inner;
                    if (in[in_idx] > max_val) {
                        max_val = in[in_idx];
                        max_idx = r;
                    }
                }

                int64_t out_pos = outer * inner_size + inner;
                out_vals[out_pos] = max_val;
                out_idx[out_pos] = max_idx;
            }
        }
    });

    return std::make_tuple(values, indices);
}

// Min all elements
inline Tensor min(const Tensor& self) {
    PT_CHECK(self.numel() > 0);

    Tensor result = empty({}, TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "min", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        scalar_t min_val = data[0];
        int64_t n = self.numel();

        for (int64_t i = 1; i < n; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
            }
        }

        result.mutable_data_ptr<scalar_t>()[0] = min_val;
    });

    return result;
}

// Min along dimension
inline std::tuple<Tensor, Tensor> min(const Tensor& self, int64_t dim, bool keepdim = false) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    std::vector<int64_t> result_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) result_shape.push_back(1);
        } else {
            result_shape.push_back(self.size(i));
        }
    }

    Tensor values = empty(result_shape, TensorOptions().dtype(self.dtype()).device(self.device()));
    Tensor indices = empty(result_shape, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "min_dim", [&] {
        const scalar_t* in = self.data_ptr<scalar_t>();
        scalar_t* out_vals = values.mutable_data_ptr<scalar_t>();
        int64_t* out_idx = indices.mutable_data_ptr<int64_t>();

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= self.size(i);
        }

        int64_t reduce_size = self.size(dim);

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            inner_size *= self.size(i);
        }

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                int64_t first_idx = outer * reduce_size * inner_size + inner;
                scalar_t min_val = in[first_idx];
                int64_t min_idx = 0;

                for (int64_t r = 1; r < reduce_size; ++r) {
                    int64_t in_idx = (outer * reduce_size + r) * inner_size + inner;
                    if (in[in_idx] < min_val) {
                        min_val = in[in_idx];
                        min_idx = r;
                    }
                }

                int64_t out_pos = outer * inner_size + inner;
                out_vals[out_pos] = min_val;
                out_idx[out_pos] = min_idx;
            }
        }
    });

    return std::make_tuple(values, indices);
}

// Argmax all elements
inline Tensor argmax(const Tensor& self) {
    PT_CHECK(self.numel() > 0);

    Tensor result = empty({}, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "argmax", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        scalar_t max_val = data[0];
        int64_t max_idx = 0;
        int64_t n = self.numel();

        for (int64_t i = 1; i < n; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }

        result.mutable_data_ptr<int64_t>()[0] = max_idx;
    });

    return result;
}

// Argmin all elements
inline Tensor argmin(const Tensor& self) {
    PT_CHECK(self.numel() > 0);

    Tensor result = empty({}, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "argmin", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        scalar_t min_val = data[0];
        int64_t min_idx = 0;
        int64_t n = self.numel();

        for (int64_t i = 1; i < n; ++i) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }

        result.mutable_data_ptr<int64_t>()[0] = min_idx;
    });

    return result;
}

// Argmax along dimension
inline Tensor argmax(const Tensor& self, int64_t dim, bool keepdim = false) {
    PT_CHECK(self.numel() > 0);
    int64_t ndim = self.dim();
    PT_CHECK(dim >= -ndim && dim < ndim);
    if (dim < 0) dim += ndim;

    // Get output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(self.size(i));
        }
    }

    Tensor result = empty(out_shape, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));
    int64_t dim_size = self.size(dim);

    // Calculate strides for iteration
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= self.size(i);
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= self.size(i);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "argmax_dim", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        int64_t* out_data = result.mutable_data_ptr<int64_t>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                int64_t max_idx = 0;
                scalar_t max_val = data[outer * dim_size * inner_size + inner];

                for (int64_t d = 1; d < dim_size; ++d) {
                    scalar_t val = data[outer * dim_size * inner_size + d * inner_size + inner];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = d;
                    }
                }
                out_data[outer * inner_size + inner] = max_idx;
            }
        }
    });

    return result;
}

// Argmin along dimension
inline Tensor argmin(const Tensor& self, int64_t dim, bool keepdim = false) {
    PT_CHECK(self.numel() > 0);
    int64_t ndim = self.dim();
    PT_CHECK(dim >= -ndim && dim < ndim);
    if (dim < 0) dim += ndim;

    // Get output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(self.size(i));
        }
    }

    Tensor result = empty(out_shape, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));
    int64_t dim_size = self.size(dim);

    // Calculate strides for iteration
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= self.size(i);
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= self.size(i);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "argmin_dim", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        int64_t* out_data = result.mutable_data_ptr<int64_t>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                int64_t min_idx = 0;
                scalar_t min_val = data[outer * dim_size * inner_size + inner];

                for (int64_t d = 1; d < dim_size; ++d) {
                    scalar_t val = data[outer * dim_size * inner_size + d * inner_size + inner];
                    if (val < min_val) {
                        min_val = val;
                        min_idx = d;
                    }
                }
                out_data[outer * inner_size + inner] = min_idx;
            }
        }
    });

    return result;
}

// Variance
inline Tensor var(const Tensor& self, bool unbiased = true) {
    Tensor m = mean(self);

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "var", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        scalar_t mean_val = m.data_ptr<scalar_t>()[0];
        scalar_t sum_sq = 0;
        int64_t n = self.numel();

        for (int64_t i = 0; i < n; ++i) {
            scalar_t diff = data[i] - mean_val;
            sum_sq += diff * diff;
        }

        int64_t divisor = unbiased ? (n - 1) : n;
        m.mutable_data_ptr<scalar_t>()[0] = sum_sq / static_cast<scalar_t>(divisor);
    });

    return m;
}

// Standard deviation
inline Tensor std(const Tensor& self, bool unbiased = true) {
    Tensor v = var(self, unbiased);

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "std", [&] {
        v.mutable_data_ptr<scalar_t>()[0] = std::sqrt(v.data_ptr<scalar_t>()[0]);
    });

    return v;
}

// Norm
inline Tensor norm(const Tensor& self, Scalar p = 2) {
    double p_val = p.toDouble();
    Tensor result = zeros({}, TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "norm", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        double sum = 0;
        int64_t n = self.numel();

        if (p_val == 2.0) {
            // L2 norm (Euclidean)
            for (int64_t i = 0; i < n; ++i) {
                double val = static_cast<double>(data[i]);
                sum += val * val;
            }
            result.mutable_data_ptr<scalar_t>()[0] = static_cast<scalar_t>(std::sqrt(sum));
        } else if (p_val == 1.0) {
            // L1 norm
            for (int64_t i = 0; i < n; ++i) {
                sum += std::abs(static_cast<double>(data[i]));
            }
            result.mutable_data_ptr<scalar_t>()[0] = static_cast<scalar_t>(sum);
        } else if (std::isinf(p_val)) {
            // L-infinity norm
            double max_val = 0;
            for (int64_t i = 0; i < n; ++i) {
                max_val = std::max(max_val, std::abs(static_cast<double>(data[i])));
            }
            result.mutable_data_ptr<scalar_t>()[0] = static_cast<scalar_t>(max_val);
        } else {
            // General Lp norm
            for (int64_t i = 0; i < n; ++i) {
                sum += std::pow(std::abs(static_cast<double>(data[i])), p_val);
            }
            result.mutable_data_ptr<scalar_t>()[0] = static_cast<scalar_t>(std::pow(sum, 1.0 / p_val));
        }
    });

    return result;
}

// All (logical AND)
inline bool all(const Tensor& self) {
    bool result = true;

    PT_DISPATCH_ALL_TYPES(self.dtype(), "all", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n && result; ++i) {
            if (data[i] == static_cast<scalar_t>(0)) {
                result = false;
            }
        }
    });

    return result;
}

// Any (logical OR)
inline bool any(const Tensor& self) {
    bool result = false;

    PT_DISPATCH_ALL_TYPES(self.dtype(), "any", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        int64_t n = self.numel();

        for (int64_t i = 0; i < n && !result; ++i) {
            if (data[i] != static_cast<scalar_t>(0)) {
                result = true;
            }
        }
    });

    return result;
}

} // namespace native
} // namespace at
