// test_ops_generated.cpp - Auto-generated comprehensive op test suite.
//
// Uses Google Test value-parameterized tests (INSTANTIATE_TEST_SUITE_P) to
// expand the cartesian product of:
//     ops (~75 unary + binary + reduce + shape + matmul)
//   x shapes (1D / 2D / 3D / 4D)
//   x layouts (contiguous / transposed / sliced / permuted)
// into thousands of concrete gtest cases.
//
// Reference implementations are in ops_spec.cpp and generally compare the
// strided-view path against the same op applied to a .contiguous() copy.

#include "ops_spec.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using ops_spec::OpSpec;
using ops_spec::Layout;
using ops_spec::Category;
using ops_spec::make_input;
using ops_spec::apply_layout;
using ops_spec::tensors_near;
using ops_spec::shape_str;
using ops_spec::layout_str;
using at::Tensor;

// ===========================================================================
// Helpers to build the parameter list for each category
// ===========================================================================

struct Param {
    OpSpec spec;
    std::vector<int64_t> shape;
    Layout layout;
    uint32_t seed;

    std::string test_name() const {
        std::stringstream ss;
        ss << spec.name << "__" << shape_str(shape) << "__" << layout_str(layout);
        std::string s = ss.str();
        // sanitize to valid gtest test name (alnum + underscore)
        for (auto& c : s) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
        }
        return s;
    }
};

static uint32_t hash_str(const std::string& s) {
    uint32_t h = 2166136261u;
    for (char c : s) { h ^= (uint8_t)c; h *= 16777619u; }
    return h;
}

static bool is_layout_compatible(const std::vector<int64_t>& shape, Layout l) {
    int64_t d = (int64_t)shape.size();
    switch (l) {
        case Layout::kContiguous: return true;
        case Layout::kTransposed: return d >= 2;
        case Layout::kSliced:     return shape[0] >= 2;
        case Layout::kPermuted:   return d >= 2;
    }
    return true;
}

// Some layouts only make sense for certain op categories.
// For reductions with a fixed dim, transposed/permuted would change which dim
// is reduced. Since our reference uses .contiguous() and the SAME op with the
// SAME dim, the expected semantics match — the compute must still be correct.
// So we keep all layouts but skip dim values that exceed shape rank.
static bool is_spec_compatible(const OpSpec& spec,
                               const std::vector<int64_t>& shape,
                               Layout layout) {
    int64_t d = (int64_t)shape.size();
    if (spec.category == Category::kReduceDim) {
        if (spec.dim >= d) return false;
    }
    if (spec.category == Category::kMatmul) {
        if (d != 2) return false;
    }
    // Shape ops like unsqueeze/flatten should work on any shape
    (void)layout;
    return true;
}

// Generate the full list of params for a given category by zipping
// (spec, shape, layout).
static std::vector<Param> build_params(
        const std::vector<OpSpec>& specs,
        const std::vector<std::vector<int64_t>>& shapes,
        const std::vector<Layout>& layouts) {
    std::vector<Param> out;
    for (const auto& s : specs) {
        for (const auto& sh : shapes) {
            for (auto l : layouts) {
                if (!is_layout_compatible(sh, l)) continue;
                if (!is_spec_compatible(s, sh, l)) continue;
                Param p{s, sh, l, 0};
                p.seed = hash_str(p.test_name());
                out.push_back(std::move(p));
            }
        }
    }
    return out;
}

// ===========================================================================
// Unary op fixture
// ===========================================================================
class UnaryOpTest : public ::testing::TestWithParam<Param> {};

TEST_P(UnaryOpTest, Forward) {
    const Param& p = GetParam();
    Tensor base = make_input(p.shape, p.seed, p.spec.positive_only, p.spec.avoid_zero);
    Tensor input = apply_layout(base, p.layout);
    Tensor actual = p.spec.impl({input}, p.spec);
    Tensor expected = p.spec.reference({input}, p.spec);
    EXPECT_TRUE(tensors_near(actual, expected, p.spec.tolerance));
}

static std::vector<Param> unary_params() {
    return build_params(ops_spec::unary_specs(),
                        ops_spec::shapes_mixed(),
                        ops_spec::layouts_all());
}

INSTANTIATE_TEST_SUITE_P(
    GenUnary, UnaryOpTest,
    ::testing::ValuesIn(unary_params()),
    [](const ::testing::TestParamInfo<Param>& info) {
        return info.param.test_name();
    });

// ===========================================================================
// Unary-scalar op fixture (same harness, different spec set)
// ===========================================================================
class UnaryScalarOpTest : public ::testing::TestWithParam<Param> {};

TEST_P(UnaryScalarOpTest, Forward) {
    const Param& p = GetParam();
    Tensor base = make_input(p.shape, p.seed, p.spec.positive_only, p.spec.avoid_zero);
    Tensor input = apply_layout(base, p.layout);
    Tensor actual = p.spec.impl({input}, p.spec);
    Tensor expected = p.spec.reference({input}, p.spec);
    EXPECT_TRUE(tensors_near(actual, expected, p.spec.tolerance));
}

static std::vector<Param> unary_scalar_params() {
    return build_params(ops_spec::unary_scalar_specs(),
                        ops_spec::shapes_mixed(),
                        ops_spec::layouts_all());
}

INSTANTIATE_TEST_SUITE_P(
    GenUnaryScalar, UnaryScalarOpTest,
    ::testing::ValuesIn(unary_scalar_params()),
    [](const ::testing::TestParamInfo<Param>& info) {
        return info.param.test_name();
    });

// ===========================================================================
// Binary op fixture
// ===========================================================================
class BinaryOpTest : public ::testing::TestWithParam<Param> {};

TEST_P(BinaryOpTest, Forward) {
    const Param& p = GetParam();
    Tensor a_base = make_input(p.shape, p.seed,     p.spec.positive_only, p.spec.avoid_zero);
    Tensor b_base = make_input(p.shape, p.seed + 1, p.spec.positive_only, p.spec.avoid_zero);
    Tensor a = apply_layout(a_base, p.layout);
    Tensor b = apply_layout(b_base, p.layout);
    Tensor actual = p.spec.impl({a, b}, p.spec);
    Tensor expected = p.spec.reference({a, b}, p.spec);
    EXPECT_TRUE(tensors_near(actual, expected, p.spec.tolerance));
}

static std::vector<Param> binary_params() {
    return build_params(ops_spec::binary_specs(),
                        ops_spec::shapes_mixed(),
                        ops_spec::layouts_all());
}

INSTANTIATE_TEST_SUITE_P(
    GenBinary, BinaryOpTest,
    ::testing::ValuesIn(binary_params()),
    [](const ::testing::TestParamInfo<Param>& info) {
        return info.param.test_name();
    });

// ===========================================================================
// Reduce-all op fixture
// ===========================================================================
class ReduceAllOpTest : public ::testing::TestWithParam<Param> {};

TEST_P(ReduceAllOpTest, Forward) {
    const Param& p = GetParam();
    Tensor base = make_input(p.shape, p.seed, p.spec.positive_only, false);
    Tensor input = apply_layout(base, p.layout);
    Tensor actual = p.spec.impl({input}, p.spec);
    Tensor expected = p.spec.reference({input}, p.spec);
    EXPECT_TRUE(tensors_near(actual, expected, p.spec.tolerance));
}

static std::vector<Param> reduce_all_params() {
    return build_params(ops_spec::reduce_all_specs(),
                        ops_spec::shapes_mixed(),
                        ops_spec::layouts_all());
}

INSTANTIATE_TEST_SUITE_P(
    GenReduceAll, ReduceAllOpTest,
    ::testing::ValuesIn(reduce_all_params()),
    [](const ::testing::TestParamInfo<Param>& info) {
        return info.param.test_name();
    });

// ===========================================================================
// Reduce-dim op fixture
// ===========================================================================
class ReduceDimOpTest : public ::testing::TestWithParam<Param> {};

TEST_P(ReduceDimOpTest, Forward) {
    const Param& p = GetParam();
    Tensor base = make_input(p.shape, p.seed, p.spec.positive_only, false);
    Tensor input = apply_layout(base, p.layout);
    Tensor actual = p.spec.impl({input}, p.spec);
    Tensor expected = p.spec.reference({input}, p.spec);
    EXPECT_TRUE(tensors_near(actual, expected, p.spec.tolerance));
}

static std::vector<Param> reduce_dim_params() {
    // For reduce-dim we only use shapes with dim >= 2 so that dim=0..2 is valid
    std::vector<std::vector<int64_t>> shapes;
    for (auto& s : ops_spec::shapes_2d()) shapes.push_back(s);
    for (auto& s : ops_spec::shapes_3d()) shapes.push_back(s);
    for (auto& s : ops_spec::shapes_4d()) shapes.push_back(s);
    return build_params(ops_spec::reduce_dim_specs(), shapes, ops_spec::layouts_all());
}

INSTANTIATE_TEST_SUITE_P(
    GenReduceDim, ReduceDimOpTest,
    ::testing::ValuesIn(reduce_dim_params()),
    [](const ::testing::TestParamInfo<Param>& info) {
        return info.param.test_name();
    });

// ===========================================================================
// Shape op fixture
// ===========================================================================
class ShapeOpTest : public ::testing::TestWithParam<Param> {};

TEST_P(ShapeOpTest, Forward) {
    const Param& p = GetParam();
    Tensor base = make_input(p.shape, p.seed, false, false);
    Tensor input = apply_layout(base, p.layout);
    Tensor actual = p.spec.impl({input}, p.spec);
    Tensor expected = p.spec.reference({input}, p.spec);
    EXPECT_TRUE(tensors_near(actual, expected, p.spec.tolerance));
}

static std::vector<Param> shape_params() {
    return build_params(ops_spec::shape_specs(),
                        ops_spec::shapes_mixed(),
                        ops_spec::layouts_all());
}

INSTANTIATE_TEST_SUITE_P(
    GenShape, ShapeOpTest,
    ::testing::ValuesIn(shape_params()),
    [](const ::testing::TestParamInfo<Param>& info) {
        return info.param.test_name();
    });

// ===========================================================================
// Matmul op fixture — uses square 2D shapes only
// ===========================================================================
class MatmulOpTest : public ::testing::TestWithParam<Param> {};

TEST_P(MatmulOpTest, Forward) {
    const Param& p = GetParam();
    Tensor a_base = make_input(p.shape, p.seed,     false, false);
    Tensor b_base = make_input(p.shape, p.seed + 7, false, false);
    Tensor a = apply_layout(a_base, p.layout);
    Tensor b = apply_layout(b_base, p.layout);
    // For matmul, we need compatible inner dims after layout transform.
    // Using square shapes means a.shape[1] == b.shape[0] after transpose.
    Tensor actual = p.spec.impl({a, b}, p.spec);
    Tensor expected = p.spec.reference({a, b}, p.spec);
    EXPECT_TRUE(tensors_near(actual, expected, p.spec.tolerance));
}

static std::vector<Param> matmul_params() {
    return build_params(ops_spec::matmul_specs(),
                        ops_spec::shapes_2d_square(),
                        ops_spec::layouts_all());
}

INSTANTIATE_TEST_SUITE_P(
    GenMatmul, MatmulOpTest,
    ::testing::ValuesIn(matmul_params()),
    [](const ::testing::TestParamInfo<Param>& info) {
        return info.param.test_name();
    });

// ===========================================================================
// Sanity regression: a handful of explicit TEST cases for fixed values, to
// catch catastrophic breakage (parameterized tests share one fixture so a
// registry bug would wipe out thousands of cases at once).
// ===========================================================================

TEST(GenSanity, AddWorks) {
    auto a = at::tensor({1.0f, 2.0f, 3.0f});
    auto b = at::tensor({4.0f, 5.0f, 6.0f});
    auto r = a.add(b);
    EXPECT_NEAR(r.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 7.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 9.0f, 1e-5);
}

TEST(GenSanity, MulStridedMatchesContiguous) {
    auto a = at::randn({8, 16});
    auto b = at::randn({8, 16});
    auto at_ = a.transpose(0, 1);
    auto bt = b.transpose(0, 1);
    auto expected = a.contiguous().mul(b.contiguous()).transpose(0, 1).contiguous();
    auto actual = at_.mul(bt).contiguous();
    EXPECT_TRUE(tensors_near(actual, expected, 1e-5f));
}

TEST(GenSanity, SumDim0) {
    auto t = at::ones({4, 8});
    auto r = t.sum(0, false);
    EXPECT_EQ(r.numel(), 8);
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(r.data_ptr<float>()[i], 4.0f, 1e-5f);
    }
}
