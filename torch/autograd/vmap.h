#pragma once

// ============================================================================
// vmap — Vectorized function map over a batch dimension
// ============================================================================
//
// Loops f over slices of input(s) along in_dim and stacks the results along
// out_dim. Simplest correct implementation: slice -> apply -> stack.
//
// Example:
//   auto y = torch::autograd::vmap([](at::Tensor x){ return x.mul(2); }, x);
//
// Notes:
//   * CPU-only path (Elbrus-friendly), no CUDA dependency.
//   * Optional OpenMP parallelism over the batch axis when _OPENMP is defined
//     and f is thread-safe.  By default we run serially because the user's f
//     may capture state we cannot reason about.
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"
#include "c10/macros/Macros.h"

#include <functional>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace torch {
namespace autograd {

namespace detail {

inline int64_t vmap_normalize_dim(int64_t dim, int64_t ndim, const char* what) {
    int64_t d = dim < 0 ? dim + ndim : dim;
    PT_CHECK_MSG(d >= 0 && d < ndim,
        "vmap: ", what, " dim ", dim,
        " out of range for tensor with ", ndim, " dims");
    return d;
}

// Move `from` axis to position 0 by transposing.
inline at::Tensor move_dim_to_front(const at::Tensor& t, int64_t from) {
    if (from == 0) return t;
    return at::native::transpose(t, 0, from);
}

// Move axis 0 of `t` to position `to` (in the output's coordinate system).
inline at::Tensor move_front_to_dim(const at::Tensor& t, int64_t to) {
    if (to == 0) return t;
    int64_t ndim = t.dim();
    int64_t d = to < 0 ? to + ndim : to;
    PT_CHECK_MSG(d >= 0 && d < ndim,
        "vmap: out_dim ", to, " out of range for output with ", ndim, " dims");
    // Transpose 0 <-> d. For >2 dims this is not the same as a full roll, but
    // it correctly places the batch axis at position d when the user only
    // cares about which axis carries the batch.  For a true "roll" semantic
    // see torch.movedim — left as a future enhancement.
    return at::native::transpose(t, 0, d);
}

} // namespace detail

// ----------------------------------------------------------------------------
// Single-input vmap
// ----------------------------------------------------------------------------
inline at::Tensor vmap(std::function<at::Tensor(at::Tensor)> f,
                       const at::Tensor& input,
                       int64_t in_dim = 0,
                       int64_t out_dim = 0) {
    PT_CHECK_MSG(input.defined(), "vmap: input tensor is undefined");
    PT_CHECK_MSG(input.dim() >= 1,
        "vmap: input must have at least 1 dimension, got ", input.dim());

    int64_t ndim = input.dim();
    int64_t in_d = detail::vmap_normalize_dim(in_dim, ndim, "in_dim");
    int64_t batch = input.size(in_d);
    PT_CHECK_MSG(batch >= 0, "vmap: negative batch size");

    // Move batch dim to front for uniform slicing.
    at::Tensor x = detail::move_dim_to_front(input, in_d);

    std::vector<at::Tensor> outs(static_cast<size_t>(batch));

#if defined(_OPENMP)
    // Each iteration calls user f on an independent slice; parallelize when
    // the user opted in to OpenMP.  We don't enforce thread-safety of f —
    // if f is not safe, the user can simply not link OpenMP.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < batch; ++i) {
        at::Tensor slice = x.select(0, i);
        outs[static_cast<size_t>(i)] = f(slice);
    }
#else
    for (int64_t i = 0; i < batch; ++i) {
        at::Tensor slice = x.select(0, i);
        outs[static_cast<size_t>(i)] = f(slice);
    }
#endif

    PT_CHECK_MSG(!outs.empty(),
        "vmap: input had zero-length batch dim, cannot infer output shape");
    for (size_t i = 0; i < outs.size(); ++i) {
        PT_CHECK_MSG(outs[i].defined(),
            "vmap: f returned undefined tensor at batch index ", i);
        PT_CHECK_MSG(outs[i].dim() == outs[0].dim(),
            "vmap: f returned tensors with mismatched dim at index ", i,
            ": expected ", outs[0].dim(), ", got ", outs[i].dim());
    }

    // Stack along axis 0, then move that axis to out_dim.
    at::Tensor stacked = at::native::stack(outs, /*dim=*/0);
    return detail::move_front_to_dim(stacked, out_dim);
}

// ----------------------------------------------------------------------------
// Multi-input vmap
// ----------------------------------------------------------------------------
inline at::Tensor vmap(std::function<at::Tensor(std::vector<at::Tensor>)> f,
                       const std::vector<at::Tensor>& inputs,
                       const std::vector<int64_t>& in_dims,
                       int64_t out_dim = 0) {
    PT_CHECK_MSG(!inputs.empty(), "vmap: inputs vector is empty");
    PT_CHECK_MSG(inputs.size() == in_dims.size(),
        "vmap: inputs.size()=", inputs.size(),
        " != in_dims.size()=", in_dims.size());

    // Normalize each in_dim, move batch axis to front, validate batch sizes.
    std::vector<at::Tensor> fronted;
    fronted.reserve(inputs.size());
    int64_t batch = -1;

    for (size_t k = 0; k < inputs.size(); ++k) {
        const at::Tensor& t = inputs[k];
        PT_CHECK_MSG(t.defined(), "vmap: input ", k, " is undefined");
        PT_CHECK_MSG(t.dim() >= 1,
            "vmap: input ", k, " must have at least 1 dim, got ", t.dim());

        int64_t d = detail::vmap_normalize_dim(in_dims[k], t.dim(), "in_dims");
        int64_t b = t.size(d);

        if (batch < 0) {
            batch = b;
        } else {
            PT_CHECK_MSG(b == batch,
                "vmap: input ", k, " has batch size ", b,
                " along in_dim ", in_dims[k],
                " but expected ", batch);
        }
        fronted.push_back(detail::move_dim_to_front(t, d));
    }

    PT_CHECK_MSG(batch >= 0, "vmap: could not determine batch size");

    std::vector<at::Tensor> outs(static_cast<size_t>(batch));

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < batch; ++i) {
        std::vector<at::Tensor> slices;
        slices.reserve(fronted.size());
        for (const auto& tf : fronted) {
            slices.push_back(tf.select(0, i));
        }
        outs[static_cast<size_t>(i)] = f(std::move(slices));
    }
#else
    for (int64_t i = 0; i < batch; ++i) {
        std::vector<at::Tensor> slices;
        slices.reserve(fronted.size());
        for (const auto& tf : fronted) {
            slices.push_back(tf.select(0, i));
        }
        outs[static_cast<size_t>(i)] = f(std::move(slices));
    }
#endif

    for (size_t i = 0; i < outs.size(); ++i) {
        PT_CHECK_MSG(outs[i].defined(),
            "vmap: f returned undefined tensor at batch index ", i);
        PT_CHECK_MSG(outs[i].dim() == outs[0].dim(),
            "vmap: f returned tensors with mismatched dim at index ", i);
    }

    at::Tensor stacked = at::native::stack(outs, /*dim=*/0);
    return detail::move_front_to_dim(stacked, out_dim);
}

} // namespace autograd
} // namespace torch
