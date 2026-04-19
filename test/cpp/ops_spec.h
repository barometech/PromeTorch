// ops_spec.h - Declarative spec for auto-generated op tests
// Covers unary, binary, reduction, shape, and linear algebra ops across
// multiple tensor shapes, layouts (contiguous / strided views), and dtypes.
//
// Reference implementations are either:
//   - hand-rolled naive loops over contiguous memory, or
//   - computed via at::native::op on a .contiguous() copy of the input
//     (acceptable: the test then validates the strided path against the
//     contiguous fast path).
//
// This header intentionally ONLY pulls in ATen public surface so it can be
// compiled as part of a single translation unit test binary.

#pragma once

#include <gtest/gtest.h>
#include "aten/src/ATen/ATen.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ops_spec {

using at::Tensor;
using at::Scalar;

// -------------------------------------------------------------------------
// Layout variants
// -------------------------------------------------------------------------
enum class Layout {
    kContiguous,     // natural creation
    kTransposed,     // transpose the last two dims (only for dim >= 2)
    kSliced,         // narrow() half in dim 0
    kPermuted,       // permute all dims reverse (only for dim >= 2)
};

// -------------------------------------------------------------------------
// Op category -- selects the test harness
// -------------------------------------------------------------------------
enum class Category {
    kUnary,          // out = op(in)
    kUnaryScalar,    // out = op(in, scalar)    e.g. clamp_min, pow_scalar
    kBinary,         // out = op(a, b)          same-shape
    kReduceAll,      // out = op(in)            reduce to scalar
    kReduceDim,      // out = op(in, dim)       reduce over dim
    kShape,          // shape-only transform, validate by recomputing
    kMatmul,         // out = a @ b             (2D x 2D or ND)
};

struct OpSpec {
    std::string name;
    Category category;
    float tolerance = 1e-4f;
    bool supports_negative_input = true;   // if false, abs the input
    bool positive_only = false;            // >0 (for log, sqrt of negatives)
    bool avoid_zero = false;               // for reciprocal / div / log
    Scalar scalar_arg = Scalar(0.5);

    // Forward function being tested. Returns actual output.
    std::function<Tensor(const std::vector<Tensor>&, const OpSpec&)> impl;

    // Reference function using naive/safe path. Returns expected output.
    std::function<Tensor(const std::vector<Tensor>&, const OpSpec&)> reference;

    // For reductions over a dim
    int64_t dim = 0;
};

// -------------------------------------------------------------------------
// Helpers: create test input with a stable (deterministic) pattern
// -------------------------------------------------------------------------
inline Tensor make_input(const std::vector<int64_t>& shape,
                         uint32_t seed,
                         bool positive_only,
                         bool avoid_zero) {
    Tensor t = at::empty(shape);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    float* p = t.mutable_data_ptr<float>();
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        float v = dist(rng);
        if (positive_only) v = std::fabs(v) + 0.05f;
        if (avoid_zero && std::fabs(v) < 0.1f) v = (v < 0 ? -0.1f : 0.1f);
        p[i] = v;
    }
    return t;
}

// Apply a layout transformation. The returned tensor may be non-contiguous.
inline Tensor apply_layout(const Tensor& in, Layout layout) {
    int64_t d = in.dim();
    switch (layout) {
        case Layout::kContiguous: return in;
        case Layout::kTransposed:
            if (d >= 2) return in.transpose(d - 2, d - 1);
            return in;
        case Layout::kSliced: {
            int64_t s0 = in.size(0);
            if (s0 >= 2) return in.narrow(0, 0, s0 / 2);
            return in;
        }
        case Layout::kPermuted: {
            if (d < 2) return in;
            std::vector<int64_t> dims(d);
            for (int64_t i = 0; i < d; ++i) dims[i] = d - 1 - i;
            return in.permute(dims);
        }
    }
    return in;
}

// Shape-string for test names
inline std::string shape_str(const std::vector<int64_t>& s) {
    std::stringstream ss;
    ss << "d" << s.size() << "_";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) ss << "x";
        ss << s[i];
    }
    return ss.str();
}

inline std::string layout_str(Layout l) {
    switch (l) {
        case Layout::kContiguous: return "cont";
        case Layout::kTransposed: return "trans";
        case Layout::kSliced:     return "slice";
        case Layout::kPermuted:   return "perm";
    }
    return "?";
}

// -------------------------------------------------------------------------
// Compare two tensors elementwise (both must be same shape).
// Works for any strided layout.
// -------------------------------------------------------------------------
inline ::testing::AssertionResult tensors_near(const Tensor& actual,
                                               const Tensor& expected,
                                               float tol) {
    if (!actual.defined() || !expected.defined()) {
        return ::testing::AssertionFailure() << "undefined tensor";
    }
    if (actual.sizes() != expected.sizes()) {
        std::stringstream ss;
        ss << "shape mismatch: actual=[";
        for (auto s : actual.sizes().vec()) ss << s << ",";
        ss << "] expected=[";
        for (auto s : expected.sizes().vec()) ss << s << ",";
        ss << "]";
        return ::testing::AssertionFailure() << ss.str();
    }
    Tensor a_c = actual.contiguous();
    Tensor e_c = expected.contiguous();
    const float* ap = a_c.data_ptr<float>();
    const float* ep = e_c.data_ptr<float>();
    int64_t n = a_c.numel();
    for (int64_t i = 0; i < n; ++i) {
        float av = ap[i];
        float ev = ep[i];
        if (std::isnan(av) && std::isnan(ev)) continue;
        float diff = std::fabs(av - ev);
        float scale = std::max(1.0f, std::max(std::fabs(av), std::fabs(ev)));
        if (diff > tol * scale) {
            return ::testing::AssertionFailure()
                   << "mismatch at i=" << i
                   << " actual=" << av << " expected=" << ev
                   << " diff=" << diff << " tol=" << tol;
        }
    }
    return ::testing::AssertionSuccess();
}

// Spec registry ----------------------------------------------------------
std::vector<OpSpec> unary_specs();
std::vector<OpSpec> unary_scalar_specs();
std::vector<OpSpec> binary_specs();
std::vector<OpSpec> reduce_all_specs();
std::vector<OpSpec> reduce_dim_specs();
std::vector<OpSpec> shape_specs();
std::vector<OpSpec> matmul_specs();

// All shapes grouped by dim
inline std::vector<std::vector<int64_t>> shapes_1d() {
    return { {1}, {8}, {33}, {128}, {513} };
}
inline std::vector<std::vector<int64_t>> shapes_2d() {
    return { {2,3}, {8,16}, {17,33}, {64,64}, {5,129} };
}
inline std::vector<std::vector<int64_t>> shapes_3d() {
    return { {2,3,4}, {4,8,16}, {3,5,7}, {8,8,17}, {2,33,9} };
}
inline std::vector<std::vector<int64_t>> shapes_4d() {
    return { {2,3,4,5}, {2,4,8,16}, {3,3,5,7}, {1,8,8,17}, {2,2,11,13} };
}
inline std::vector<std::vector<int64_t>> shapes_2d_square() {
    return { {4,4}, {8,8}, {16,16}, {32,32}, {64,64}, {17,17}, {33,33} };
}

// Convenience: 20 mixed shapes for unary/binary ops
inline std::vector<std::vector<int64_t>> shapes_mixed() {
    std::vector<std::vector<int64_t>> v;
    auto append = [&](std::vector<std::vector<int64_t>> xs) {
        for (auto& x : xs) v.push_back(std::move(x));
    };
    append(shapes_1d());
    append(shapes_2d());
    append(shapes_3d());
    append(shapes_4d());
    return v;  // 20 shapes
}

// Layouts we test for each op
inline std::vector<Layout> layouts_all() {
    return { Layout::kContiguous,
             Layout::kTransposed,
             Layout::kSliced,
             Layout::kPermuted };
}

}  // namespace ops_spec
