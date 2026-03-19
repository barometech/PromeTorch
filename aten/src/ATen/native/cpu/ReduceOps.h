#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/VectorizedOps.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/tuda/TudaMath.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include <cmath>
#include <limits>
#include <tuple>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {
namespace native {

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum all elements
inline Tensor sum(const Tensor& self) {
    Tensor result = zeros({}, TensorOptions().dtype(self.dtype()).device(self.device()));

    // Fast path for contiguous float — delegate to hot_loops.cpp
    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous()) {
        result.mutable_data_ptr<float>()[0] =
            hot::sum_loop(self.data_ptr<float>(), self.numel());
        return result;
    }

    Tensor contiguous_self = self.contiguous();
    PT_DISPATCH_ALL_TYPES(contiguous_self.dtype(), "sum", [&] {
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        scalar_t total = 0;
        int64_t n = contiguous_self.numel();
        _Pragma("omp parallel for reduction(+:total) schedule(static) if(n > 4096)")
        for (int64_t i = 0; i < n; ++i) total += data[i];
        result.mutable_data_ptr<scalar_t>()[0] = total;
    });

    return result;
}

// Sum along dimension
inline Tensor sum(const Tensor& self, int64_t dim, bool keepdim = false) {
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

    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = zeros(result_shape, TensorOptions().dtype(self.dtype()).device(self.device()));

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= input.size(i);
    int64_t reduce_size = input.size(dim);
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= input.size(i);

    // Fast path for float — delegate to hot_loops.cpp
    if (self.dtype() == c10::ScalarType::Float) {
        hot::sum_dim_loop(input.data_ptr<float>(), result.mutable_data_ptr<float>(),
                          outer_size, reduce_size, inner_size);
        return result;
    }

    PT_DISPATCH_ALL_TYPES(self.dtype(), "sum_dim", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t total_work = outer_size * inner_size;
        _Pragma("omp parallel for schedule(static) if(total_work > 64)")
        for (int64_t idx = 0; idx < total_work; ++idx) {
            int64_t outer = idx / inner_size;
            int64_t inner = idx % inner_size;
            scalar_t sum_val = 0;
            for (int64_t r = 0; r < reduce_size; ++r) {
                sum_val += in[(outer * reduce_size + r) * inner_size + inner];
            }
            out[outer * inner_size + inner] = sum_val;
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
        _Pragma("omp parallel for schedule(static) if(n > 4096)")
        for (int64_t i = 0; i < n; ++i) {
            data[i] /= static_cast<scalar_t>(reduce_size);
        }
    });

    return s;
}

// Product all elements
inline Tensor prod(const Tensor& self) {
    Tensor result = ones({}, TensorOptions().dtype(self.dtype()).device(self.device()));

    Tensor contiguous_self = self.contiguous();
    PT_DISPATCH_ALL_TYPES(contiguous_self.dtype(), "prod", [&] {
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        scalar_t total = 1;
        int64_t n = contiguous_self.numel();
        _Pragma("omp parallel for reduction(*:total) schedule(static) if(n > 4096)")
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

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous()) {
        result.mutable_data_ptr<float>()[0] =
            tuda::vec_max(self.data_ptr<float>(), self.numel());
        return result;
    }

    Tensor contiguous_self = self.contiguous();
    PT_DISPATCH_ALL_TYPES(contiguous_self.dtype(), "max", [&] {
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        scalar_t max_val = data[0];
        int64_t n = contiguous_self.numel();
        // OpenMP reduction(max:) requires OpenMP 3.1+ (not available on MSVC)
        #if defined(_OPENMP) && _OPENMP >= 201107
        if (n > 4096) {
            #pragma omp parallel for reduction(max:max_val) schedule(static)
            for (int64_t i = 1; i < n; ++i)
                if (data[i] > max_val) max_val = data[i];
        } else
        #endif
        {
            for (int64_t i = 1; i < n; ++i)
                if (data[i] > max_val) max_val = data[i];
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

        int64_t total_work = outer_size * inner_size;
        _Pragma("omp parallel for schedule(static) if(total_work > 64)")
        for (int64_t idx = 0; idx < total_work; ++idx) {
            int64_t outer = idx / inner_size;
            int64_t inner = idx % inner_size;
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
    });

    return std::make_tuple(values, indices);
}

// Min all elements
inline Tensor min(const Tensor& self) {
    PT_CHECK(self.numel() > 0);

    Tensor result = empty({}, TensorOptions().dtype(self.dtype()).device(self.device()));

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous()) {
        result.mutable_data_ptr<float>()[0] =
            tuda::vec_min(self.data_ptr<float>(), self.numel());
        return result;
    }

    Tensor contiguous_self = self.contiguous();
    PT_DISPATCH_ALL_TYPES(contiguous_self.dtype(), "min", [&] {
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        scalar_t min_val = data[0];
        int64_t n = contiguous_self.numel();
        #if defined(_OPENMP) && _OPENMP >= 201107
        if (n > 4096) {
            #pragma omp parallel for reduction(min:min_val) schedule(static)
            for (int64_t i = 1; i < n; ++i)
                if (data[i] < min_val) min_val = data[i];
        } else
        #endif
        {
            for (int64_t i = 1; i < n; ++i)
                if (data[i] < min_val) min_val = data[i];
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

        int64_t total_work = outer_size * inner_size;
        _Pragma("omp parallel for schedule(static) if(total_work > 64)")
        for (int64_t idx = 0; idx < total_work; ++idx) {
            int64_t outer = idx / inner_size;
            int64_t inner = idx % inner_size;
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
    });

    return std::make_tuple(values, indices);
}

// Argmax all elements
inline Tensor argmax(const Tensor& self) {
    PT_CHECK(self.numel() > 0);

    Tensor result = empty({}, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));

    Tensor contiguous_self = self.contiguous();
    PT_DISPATCH_ALL_TYPES(contiguous_self.dtype(), "argmax", [&] {
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        scalar_t max_val = data[0];
        int64_t max_idx = 0;
        int64_t n = contiguous_self.numel();

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

    Tensor contiguous_self = self.contiguous();
    PT_DISPATCH_ALL_TYPES(contiguous_self.dtype(), "argmin", [&] {
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        scalar_t min_val = data[0];
        int64_t min_idx = 0;
        int64_t n = contiguous_self.numel();

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
        int64_t total_work = outer_size * inner_size;
        _Pragma("omp parallel for schedule(static) if(total_work > 64)")
        for (int64_t idx = 0; idx < total_work; ++idx) {
            int64_t outer = idx / inner_size;
            int64_t inner = idx % inner_size;
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
        int64_t total_work = outer_size * inner_size;
        _Pragma("omp parallel for schedule(static) if(total_work > 64)")
        for (int64_t idx = 0; idx < total_work; ++idx) {
            int64_t outer = idx / inner_size;
            int64_t inner = idx % inner_size;
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
    });

    return result;
}

// Variance
inline Tensor var(const Tensor& self, bool unbiased = true) {
    Tensor m = mean(self);

    if (self.dtype() == c10::ScalarType::Float && self.is_contiguous()) {
        const float* data = self.data_ptr<float>();
        float mean_val = m.data_ptr<float>()[0];
        int64_t n = self.numel();
        constexpr int W = tuda::VecF::width;
        tuda::VecF vmean = tuda::VecF::broadcast(mean_val);
        tuda::VecF acc0 = tuda::VecF::zero(), acc1 = tuda::VecF::zero();
        int64_t i = 0;
        for (; i + 2*W <= n; i += 2*W) {
            tuda::VecF d0 = tuda::VecF::load(data + i) - vmean;
            tuda::VecF d1 = tuda::VecF::load(data + i + W) - vmean;
            acc0 = tuda::VecF::fmadd(d0, d0, acc0);
            acc1 = tuda::VecF::fmadd(d1, d1, acc1);
        }
        float sum_sq = (acc0 + acc1).hsum();
        for (; i < n; ++i) { float d = data[i] - mean_val; sum_sq += d * d; }
        int64_t divisor = unbiased ? (n - 1) : n;
        m.mutable_data_ptr<float>()[0] = sum_sq / static_cast<float>(divisor);
        return m;
    }

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "var", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        scalar_t mean_val = m.data_ptr<scalar_t>()[0];
        scalar_t sum_sq = 0;
        int64_t n = self.numel();
        _Pragma("omp parallel for reduction(+:sum_sq) schedule(static) if(n > 4096)")
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

    // TUDA VecF fast path for L2 norm on float
    if (p_val == 2.0 && self.dtype() == c10::ScalarType::Float && self.is_contiguous()) {
        const float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        constexpr int W = tuda::VecF::width;
        tuda::VecF acc0 = tuda::VecF::zero(), acc1 = tuda::VecF::zero();
        int64_t i = 0;
        for (; i + 2*W <= n; i += 2*W) {
            tuda::VecF v0 = tuda::VecF::load(data + i);
            tuda::VecF v1 = tuda::VecF::load(data + i + W);
            acc0 = tuda::VecF::fmadd(v0, v0, acc0);
            acc1 = tuda::VecF::fmadd(v1, v1, acc1);
        }
        float sum_sq = (acc0 + acc1).hsum();
        for (; i < n; ++i) sum_sq += data[i] * data[i];
        result.mutable_data_ptr<float>()[0] = std::sqrt(sum_sq);
        return result;
    }

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "norm", [&] {
        const scalar_t* data = self.data_ptr<scalar_t>();
        double sum = 0;
        int64_t n = self.numel();

        if (p_val == 2.0) {
            _Pragma("omp parallel for reduction(+:sum) schedule(static) if(n > 4096)")
            for (int64_t i = 0; i < n; ++i) {
                double val = static_cast<double>(data[i]);
                sum += val * val;
            }
            result.mutable_data_ptr<scalar_t>()[0] = static_cast<scalar_t>(std::sqrt(sum));
        } else if (p_val == 1.0) {
            // L1 norm
            _Pragma("omp parallel for reduction(+:sum) schedule(static) if(n > 4096)")
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
            _Pragma("omp parallel for reduction(+:sum) schedule(static) if(n > 4096)")
            for (int64_t i = 0; i < n; ++i) {
                sum += std::pow(std::abs(static_cast<double>(data[i])), p_val);
            }
            result.mutable_data_ptr<scalar_t>()[0] = static_cast<scalar_t>(std::pow(sum, 1.0 / p_val));
        }
    });

    return result;
}

// Norm along dimension
inline Tensor norm(const Tensor& self, Scalar p, int64_t dim, bool keepdim = false) {
    double p_val = p.toDouble();
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;

    // Compute output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(self.size(i));
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor cont = self.contiguous();
    Tensor result = zeros(out_shape, TensorOptions().dtype(self.dtype()).device(self.device()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "norm_dim", [&] {
        const scalar_t* data = cont.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();

        int64_t dim_size = cont.size(dim);
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= cont.size(i);
        for (int64_t i = dim + 1; i < ndim; ++i) inner *= cont.size(i);

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t in = 0; in < inner; ++in) {
                double acc = 0;
                for (int64_t d = 0; d < dim_size; ++d) {
                    double val = std::abs(static_cast<double>(data[(o * dim_size + d) * inner + in]));
                    if (p_val == 2.0) acc += val * val;
                    else if (p_val == 1.0) acc += val;
                    else if (std::isinf(p_val)) acc = std::max(acc, val);
                    else acc += std::pow(val, p_val);
                }
                if (p_val == 2.0) acc = std::sqrt(acc);
                else if (p_val != 1.0 && !std::isinf(p_val)) acc = std::pow(acc, 1.0 / p_val);
                out[o * inner + in] = static_cast<scalar_t>(acc);
            }
        }
    });

    return result;
}

// Var along dimension
inline Tensor var(const Tensor& self, int64_t dim, bool unbiased = true, bool keepdim = false) {
    // Compute mean along dim, then (x-mean)^2 sum / (N-1 or N)
    Tensor m = mean(self, dim, /*keepdim=*/true);
    // Expand mean to match self shape for broadcasting
    Tensor centered = sub(self, m);  // broadcasting handles it if keepdim=true on mean
    Tensor sq = mul(centered, centered);
    Tensor s = sum(sq, dim, keepdim);
    int64_t actual_dim = dim < 0 ? dim + self.dim() : dim;
    double divisor = unbiased ? (double)(self.size(actual_dim) - 1) : (double)self.size(actual_dim);
    if (divisor <= 0) divisor = 1.0;
    // Divide by divisor
    Tensor result = zeros(s.sizes(), TensorOptions().dtype(s.dtype()).device(s.device()));
    PT_DISPATCH_FLOATING_TYPES(s.dtype(), "var_dim_div", [&] {
        const scalar_t* src = s.data_ptr<scalar_t>();
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();
        int64_t nn = s.numel();
        _Pragma("omp parallel for schedule(static) if(nn > 4096)")
        for (int64_t i = 0; i < nn; ++i) {
            dst[i] = src[i] / static_cast<scalar_t>(divisor);
        }
    });
    return result;
}

// Std along dimension
inline Tensor std(const Tensor& self, int64_t dim, bool unbiased = true, bool keepdim = false) {
    return sqrt(var(self, dim, unbiased, keepdim));
}

// Product along dimension
inline Tensor prod(const Tensor& self, int64_t dim, bool keepdim = false) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(self.size(i));
    }
    if (out_shape.empty()) out_shape.push_back(1);
    Tensor cont = self.contiguous();
    Tensor result = ones(out_shape, TensorOptions().dtype(self.dtype()).device(self.device()));
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "prod_dim", [&] {
        const scalar_t* data = cont.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        int64_t dim_size = cont.size(dim);
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= cont.size(i);
        for (int64_t i = dim + 1; i < ndim; ++i) inner *= cont.size(i);
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t in = 0; in < inner; ++in) {
                scalar_t acc = 1;
                for (int64_t d = 0; d < dim_size; ++d) {
                    acc *= data[(o * dim_size + d) * inner + in];
                }
                out[o * inner + in] = acc;
            }
        }
    });
    return result;
}

// All (logical AND)
inline bool all(const Tensor& self) {
    bool result = true;

    Tensor contiguous_self = self.contiguous();
    PT_DISPATCH_ALL_TYPES(contiguous_self.dtype(), "all", [&] {
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        int64_t n = contiguous_self.numel();

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

    Tensor contiguous_self = self.contiguous();
    PT_DISPATCH_ALL_TYPES(contiguous_self.dtype(), "any", [&] {
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        int64_t n = contiguous_self.numel();

        for (int64_t i = 0; i < n && !result; ++i) {
            if (data[i] != static_cast<scalar_t>(0)) {
                result = true;
            }
        }
    });

    return result;
}

// ============================================================================
// Sort, Argsort, Topk, Cumsum, Cumprod
// ============================================================================

// Sort along dimension
inline std::tuple<Tensor, Tensor> sort(const Tensor& self, int64_t dim = -1, bool descending = false) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor values = input.clone();
    Tensor indices = empty(self.sizes(), TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));

    PT_DISPATCH_ALL_TYPES(self.dtype(), "sort", [&] {
        scalar_t* vals = values.mutable_data_ptr<scalar_t>();
        int64_t* idx = indices.mutable_data_ptr<int64_t>();

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= self.size(i);
        int64_t sort_size = self.size(dim);
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= self.size(i);

        // Temp buffer for sorting
        std::vector<int64_t> perm(sort_size);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                // Initialize permutation
                std::iota(perm.begin(), perm.end(), 0);

                // Sort permutation by values
                int64_t base = outer * sort_size * inner_size + inner;
                if (descending) {
                    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                        return vals[base + a * inner_size] > vals[base + b * inner_size];
                    });
                } else {
                    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                        return vals[base + a * inner_size] < vals[base + b * inner_size];
                    });
                }

                // Apply permutation - need temp copy of slice
                std::vector<scalar_t> tmp(sort_size);
                for (int64_t r = 0; r < sort_size; ++r) {
                    tmp[r] = vals[base + r * inner_size];
                }
                for (int64_t r = 0; r < sort_size; ++r) {
                    vals[base + r * inner_size] = tmp[perm[r]];
                    idx[base + r * inner_size] = perm[r];
                }
            }
        }
    });

    return std::make_tuple(values, indices);
}

// Argsort
inline Tensor argsort(const Tensor& self, int64_t dim = -1, bool descending = false) {
    auto sort_result_ = sort(self, dim, descending);
    return std::get<1>(sort_result_);
}

// Topk
inline std::tuple<Tensor, Tensor> topk(const Tensor& self, int64_t k, int64_t dim = -1,
                                         bool largest = true, bool sorted = true) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);
    PT_CHECK(k >= 0 && k <= self.size(dim));

    // Result shape: same as input but dim has size k
    std::vector<int64_t> result_shape(self.sizes().vec());
    result_shape[dim] = k;

    Tensor values = empty(result_shape, TensorOptions().dtype(self.dtype()).device(self.device()));
    Tensor indices = empty(result_shape, TensorOptions().dtype(c10::ScalarType::Long).device(self.device()));

    Tensor input = self.is_contiguous() ? self : self.contiguous();

    PT_DISPATCH_ALL_TYPES(self.dtype(), "topk", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out_vals = values.mutable_data_ptr<scalar_t>();
        int64_t* out_idx = indices.mutable_data_ptr<int64_t>();

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= self.size(i);
        int64_t dim_size = self.size(dim);
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= self.size(i);

        std::vector<int64_t> perm(dim_size);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                std::iota(perm.begin(), perm.end(), 0);
                int64_t in_base = outer * dim_size * inner_size + inner;

                if (largest) {
                    std::partial_sort(perm.begin(), perm.begin() + k, perm.end(),
                        [&](int64_t a, int64_t b) {
                            return in[in_base + a * inner_size] > in[in_base + b * inner_size];
                        });
                } else {
                    std::partial_sort(perm.begin(), perm.begin() + k, perm.end(),
                        [&](int64_t a, int64_t b) {
                            return in[in_base + a * inner_size] < in[in_base + b * inner_size];
                        });
                }

                if (sorted && largest) {
                    std::sort(perm.begin(), perm.begin() + k,
                        [&](int64_t a, int64_t b) {
                            return in[in_base + a * inner_size] > in[in_base + b * inner_size];
                        });
                } else if (sorted && !largest) {
                    std::sort(perm.begin(), perm.begin() + k,
                        [&](int64_t a, int64_t b) {
                            return in[in_base + a * inner_size] < in[in_base + b * inner_size];
                        });
                }

                int64_t out_base = outer * k * inner_size + inner;
                for (int64_t r = 0; r < k; ++r) {
                    out_vals[out_base + r * inner_size] = in[in_base + perm[r] * inner_size];
                    out_idx[out_base + r * inner_size] = perm[r];
                }
            }
        }
    });

    return std::make_tuple(values, indices);
}

// Cumulative sum along dimension
inline Tensor cumsum(const Tensor& self, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "cumsum", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= self.size(i);
        int64_t dim_size = self.size(dim);
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= self.size(i);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                scalar_t running = 0;
                for (int64_t r = 0; r < dim_size; ++r) {
                    int64_t idx = (outer * dim_size + r) * inner_size + inner;
                    running += in[idx];
                    out[idx] = running;
                }
            }
        }
    });

    return result;
}

// Cumulative product along dimension
inline Tensor cumprod(const Tensor& self, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    PT_CHECK(dim >= 0 && dim < ndim);

    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor result = empty_like(input);

    PT_DISPATCH_ALL_TYPES(self.dtype(), "cumprod", [&] {
        const scalar_t* in = input.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= self.size(i);
        int64_t dim_size = self.size(dim);
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= self.size(i);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                scalar_t running = 1;
                for (int64_t r = 0; r < dim_size; ++r) {
                    int64_t idx = (outer * dim_size + r) * inner_size + inner;
                    running *= in[idx];
                    out[idx] = running;
                }
            }
        }
    });

    return result;
}

} // namespace native
} // namespace at
