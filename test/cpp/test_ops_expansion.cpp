#include <gtest/gtest.h>
#include "aten/src/ATen/ATen.h"

#include <cmath>
#include <vector>

using namespace at;

// ============================================================================
// Shape / view helpers
// ============================================================================

TEST(OpsExpansion, Unbind) {
    Tensor t = at::arange(0, 12);
    t = at::native::reshape(t, {3, 4});
    auto rows = at::native::unbind(t, 0);
    ASSERT_EQ(rows.size(), 3u);
    for (int64_t r = 0; r < 3; ++r) {
        EXPECT_EQ(rows[r].size(0), 4);
        const float* d = rows[r].data_ptr<float>();
        for (int64_t c = 0; c < 4; ++c) {
            EXPECT_FLOAT_EQ(d[c], static_cast<float>(r * 4 + c));
        }
    }
}

TEST(OpsExpansion, Movedim) {
    Tensor t = at::empty({2, 3, 4});
    Tensor m = at::native::movedim(t, 0, 2);
    EXPECT_EQ(m.size(0), 3);
    EXPECT_EQ(m.size(1), 4);
    EXPECT_EQ(m.size(2), 2);
}

TEST(OpsExpansion, ViewReshapeExpandAs) {
    Tensor a = at::arange(0, 6);
    Tensor shape_like = at::empty({2, 3});
    Tensor v = at::native::view_as(a, shape_like);
    EXPECT_EQ(v.size(0), 2);
    EXPECT_EQ(v.size(1), 3);

    Tensor r = at::native::reshape_as(a, shape_like);
    EXPECT_EQ(r.size(0), 2);
    EXPECT_EQ(r.size(1), 3);

    Tensor x = at::ones({1, 3});
    Tensor y = at::empty({4, 3});
    Tensor e = at::native::expand_as(x, y);
    EXPECT_EQ(e.size(0), 4);
    EXPECT_EQ(e.size(1), 3);
}

TEST(OpsExpansion, BroadcastTo) {
    Tensor x = at::ones({3});
    Tensor b = at::native::broadcast_to(x, {2, 3});
    EXPECT_EQ(b.size(0), 2);
    EXPECT_EQ(b.size(1), 3);
}

TEST(OpsExpansion, SplitSizes) {
    Tensor t = at::arange(0, 10);
    auto parts = at::native::split_sizes(t, {3, 3, 4}, 0);
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0].size(0), 3);
    EXPECT_EQ(parts[1].size(0), 3);
    EXPECT_EQ(parts[2].size(0), 4);
    EXPECT_FLOAT_EQ(parts[2].data_ptr<float>()[0], 6.0f);
}

// ============================================================================
// Pad
// ============================================================================

TEST(OpsExpansion, PadConstant1D) {
    Tensor t = at::arange(1, 4);  // [1,2,3]
    Tensor p = at::native::pad_constant(t, {2, 1}, Scalar(0));
    EXPECT_EQ(p.size(0), 6);
    const float* d = p.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 1.0f);
    EXPECT_FLOAT_EQ(d[3], 2.0f);
    EXPECT_FLOAT_EQ(d[4], 3.0f);
    EXPECT_FLOAT_EQ(d[5], 0.0f);
}

TEST(OpsExpansion, PadReplicate1D) {
    Tensor t = at::arange(1, 4);
    Tensor p = at::native::pad_replicate(t, {2, 2});
    const float* d = p.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0], 1.0f);
    EXPECT_FLOAT_EQ(d[1], 1.0f);
    EXPECT_FLOAT_EQ(d[2], 1.0f);
    EXPECT_FLOAT_EQ(d[3], 2.0f);
    EXPECT_FLOAT_EQ(d[4], 3.0f);
    EXPECT_FLOAT_EQ(d[5], 3.0f);
    EXPECT_FLOAT_EQ(d[6], 3.0f);
}

TEST(OpsExpansion, PadReflect1D) {
    Tensor t = at::arange(1, 5);  // [1,2,3,4]
    Tensor p = at::native::pad_reflect(t, {2, 2});
    const float* d = p.data_ptr<float>();
    // expected: [3,2,1,2,3,4,3,2]
    float exp[] = {3, 2, 1, 2, 3, 4, 3, 2};
    for (int i = 0; i < 8; ++i) EXPECT_FLOAT_EQ(d[i], exp[i]);
}

// ============================================================================
// unfold_window / fold_window
// ============================================================================

TEST(OpsExpansion, UnfoldWindow1D) {
    Tensor t = at::arange(0, 10);
    Tensor w = at::native::unfold_window(t, 0, 3, 2);
    // windows: [0,1,2],[2,3,4],[4,5,6],[6,7,8]
    EXPECT_EQ(w.dim(), 2);
    EXPECT_EQ(w.size(0), 4);
    EXPECT_EQ(w.size(1), 3);
    const float* d = w.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 2.0f);
    EXPECT_FLOAT_EQ(d[3], 2.0f);
    EXPECT_FLOAT_EQ(d[8], 6.0f);
}

TEST(OpsExpansion, FoldWindow1D) {
    Tensor t = at::arange(0, 10);
    Tensor w = at::native::unfold_window(t, 0, 3, 3);
    Tensor back = at::native::fold_window(w, 0, 9, 3);
    EXPECT_EQ(back.size(0), 9);
    for (int i = 0; i < 9; ++i)
        EXPECT_FLOAT_EQ(back.data_ptr<float>()[i], static_cast<float>(i));
}

// ============================================================================
// Tile
// ============================================================================

TEST(OpsExpansion, Tile) {
    Tensor t = at::arange(1, 4);
    Tensor rep = at::native::tile(t, {2});
    EXPECT_EQ(rep.size(0), 6);
    const float* d = rep.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0], 1);
    EXPECT_FLOAT_EQ(d[3], 1);
    EXPECT_FLOAT_EQ(d[5], 3);
}

// ============================================================================
// Cat / stack / cartesian_prod
// ============================================================================

TEST(OpsExpansion, StackAlongDim) {
    Tensor a = at::arange(0, 3);
    Tensor b = at::arange(3, 6);
    Tensor s = at::native::stack_along_dim({a, b}, 1);
    EXPECT_EQ(s.size(0), 3);
    EXPECT_EQ(s.size(1), 2);
}

TEST(OpsExpansion, CartesianProd) {
    Tensor a = at::arange(0, 2);  // [0,1]
    Tensor b = at::arange(0, 3);  // [0,1,2]
    Tensor cp = at::native::cartesian_prod({a, b});
    EXPECT_EQ(cp.size(0), 6);
    EXPECT_EQ(cp.size(1), 2);
}

// ============================================================================
// Index ops
// ============================================================================

TEST(OpsExpansion, IndexAdd) {
    Tensor x = at::zeros({4, 2});
    Tensor src = at::ones({2, 2});
    Tensor idx = at::empty({2}, c10::ScalarType::Long);
    idx.mutable_data_ptr<int64_t>()[0] = 0;
    idx.mutable_data_ptr<int64_t>()[1] = 2;
    Tensor out = at::native::index_add(x, 0, idx, src);
    // rows 0 and 2 should be 1, rows 1 and 3 should be 0
    const float* d = out.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0 * 2 + 0], 1.0f);
    EXPECT_FLOAT_EQ(d[1 * 2 + 0], 0.0f);
    EXPECT_FLOAT_EQ(d[2 * 2 + 0], 1.0f);
    EXPECT_FLOAT_EQ(d[3 * 2 + 0], 0.0f);
}

TEST(OpsExpansion, IndexCopy) {
    Tensor x = at::zeros({4, 2});
    Tensor src = at::full({2, 2}, Scalar(7));
    Tensor idx = at::empty({2}, c10::ScalarType::Long);
    idx.mutable_data_ptr<int64_t>()[0] = 1;
    idx.mutable_data_ptr<int64_t>()[1] = 3;
    Tensor out = at::native::index_copy(x, 0, idx, src);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[0 * 2 + 0], 0.0f);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[1 * 2 + 0], 7.0f);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[2 * 2 + 0], 0.0f);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[3 * 2 + 0], 7.0f);
}

TEST(OpsExpansion, MaskedScatter) {
    Tensor x = at::zeros({4});
    Tensor mask = at::empty({4}, c10::ScalarType::Bool);
    bool* m = mask.mutable_data_ptr<bool>();
    m[0] = true; m[1] = false; m[2] = true; m[3] = false;
    Tensor src = at::arange(10, 12);  // [10, 11]
    Tensor out = at::native::masked_scatter(x, mask, src);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[0], 10.0f);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[2], 11.0f);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[3], 0.0f);
}

TEST(OpsExpansion, ScatterAddDim) {
    Tensor x = at::zeros({3});
    Tensor src = at::ones({3});
    Tensor idx = at::empty({3}, c10::ScalarType::Long);
    idx.mutable_data_ptr<int64_t>()[0] = 0;
    idx.mutable_data_ptr<int64_t>()[1] = 0;
    idx.mutable_data_ptr<int64_t>()[2] = 2;
    Tensor out = at::native::scatter_add_dim(x, 0, idx, src);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(out.data_ptr<float>()[2], 1.0f);
}

// ============================================================================
// Reduce dim-variants
// ============================================================================

TEST(OpsExpansion, ArgmaxDim) {
    Tensor t = at::empty({2, 3});
    float vals[] = {1, 3, 2, 5, 0, 4};
    std::memcpy(t.mutable_data_ptr<float>(), vals, sizeof(vals));
    Tensor am = at::native::argmax_dim(t, 1, false);
    EXPECT_EQ(am.size(0), 2);
    EXPECT_EQ(am.data_ptr<int64_t>()[0], 1);
    EXPECT_EQ(am.data_ptr<int64_t>()[1], 0);
}

TEST(OpsExpansion, TopkAlongDim) {
    Tensor t = at::empty({5});
    float vals[] = {5, 3, 7, 1, 4};
    std::memcpy(t.mutable_data_ptr<float>(), vals, sizeof(vals));
    auto [v, i] = at::native::topk_along_dim(t, 2, 0, true, true);
    EXPECT_EQ(v.size(0), 2);
    EXPECT_FLOAT_EQ(v.data_ptr<float>()[0], 7.0f);
    EXPECT_FLOAT_EQ(v.data_ptr<float>()[1], 5.0f);
}

TEST(OpsExpansion, AllAnyReduce) {
    Tensor t = at::ones({5});
    Tensor allr = at::native::all_reduce(t);
    EXPECT_TRUE(allr.data_ptr<bool>()[0]);
    Tensor zeros = at::zeros({5});
    EXPECT_FALSE(at::native::all_reduce(zeros).data_ptr<bool>()[0]);
    EXPECT_FALSE(at::native::any_reduce(zeros).data_ptr<bool>()[0]);
}

// ============================================================================
// Linalg helpers
// ============================================================================

TEST(OpsExpansion, Diagonal) {
    Tensor t = at::empty({3, 3});
    float vals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::memcpy(t.mutable_data_ptr<float>(), vals, sizeof(vals));
    Tensor d = at::native::diagonal(t, 0);
    EXPECT_EQ(d.size(0), 3);
    EXPECT_FLOAT_EQ(d.data_ptr<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(d.data_ptr<float>()[1], 5.0f);
    EXPECT_FLOAT_EQ(d.data_ptr<float>()[2], 9.0f);
}

TEST(OpsExpansion, OuterProduct) {
    Tensor a = at::arange(1, 4);  // [1,2,3]
    Tensor b = at::arange(1, 3);  // [1,2]
    Tensor o = at::native::outer_product(a, b);
    EXPECT_EQ(o.size(0), 3);
    EXPECT_EQ(o.size(1), 2);
    EXPECT_FLOAT_EQ(o.data_ptr<float>()[0], 1.0f);   // 1*1
    EXPECT_FLOAT_EQ(o.data_ptr<float>()[1], 2.0f);   // 1*2
    EXPECT_FLOAT_EQ(o.data_ptr<float>()[2], 2.0f);   // 2*1
    EXPECT_FLOAT_EQ(o.data_ptr<float>()[5], 6.0f);   // 3*2
}

TEST(OpsExpansion, Kron) {
    // [[1,2],[3,4]] kron [[0,5],[6,7]] = 4x4 block matrix
    Tensor a = at::empty({2, 2});
    Tensor b = at::empty({2, 2});
    float av[] = {1, 2, 3, 4};
    float bv[] = {0, 5, 6, 7};
    std::memcpy(a.mutable_data_ptr<float>(), av, sizeof(av));
    std::memcpy(b.mutable_data_ptr<float>(), bv, sizeof(bv));
    Tensor k = at::native::kron(a, b);
    EXPECT_EQ(k.size(0), 4);
    EXPECT_EQ(k.size(1), 4);
    // block (0,0) = 1 * b  →  k[0][0]=0, k[0][1]=5
    EXPECT_FLOAT_EQ(k.data_ptr<float>()[0 * 4 + 0], 0.0f);
    EXPECT_FLOAT_EQ(k.data_ptr<float>()[0 * 4 + 1], 5.0f);
    // block (1,1) = 4 * b  →  k[2][2]=0, k[2][3]=20
    EXPECT_FLOAT_EQ(k.data_ptr<float>()[2 * 4 + 2], 0.0f);
    EXPECT_FLOAT_EQ(k.data_ptr<float>()[2 * 4 + 3], 20.0f);
    EXPECT_FLOAT_EQ(k.data_ptr<float>()[3 * 4 + 3], 28.0f);  // 4 * 7
}

// ============================================================================
// Logical
// ============================================================================

TEST(OpsExpansion, LogicalOps) {
    Tensor a = at::empty({3});
    float av[] = {1, 0, 1};
    std::memcpy(a.mutable_data_ptr<float>(), av, sizeof(av));
    Tensor b = at::empty({3});
    float bv[] = {1, 1, 0};
    std::memcpy(b.mutable_data_ptr<float>(), bv, sizeof(bv));
    Tensor AND = at::native::logical_and(a, b);
    Tensor OR  = at::native::logical_or(a, b);
    Tensor XOR = at::native::logical_xor(a, b);
    EXPECT_TRUE(AND.data_ptr<bool>()[0]);
    EXPECT_FALSE(AND.data_ptr<bool>()[1]);
    EXPECT_FALSE(AND.data_ptr<bool>()[2]);
    EXPECT_TRUE(OR.data_ptr<bool>()[0]);
    EXPECT_TRUE(OR.data_ptr<bool>()[1]);
    EXPECT_TRUE(OR.data_ptr<bool>()[2]);
    EXPECT_FALSE(XOR.data_ptr<bool>()[0]);
    EXPECT_TRUE(XOR.data_ptr<bool>()[1]);
    EXPECT_TRUE(XOR.data_ptr<bool>()[2]);

    Tensor NOT = at::native::logical_not(a);
    EXPECT_FALSE(NOT.data_ptr<bool>()[0]);
    EXPECT_TRUE(NOT.data_ptr<bool>()[1]);
    EXPECT_FALSE(NOT.data_ptr<bool>()[2]);
}

// ============================================================================
// Float classification
// ============================================================================

TEST(OpsExpansion, IsFiniteInfNan) {
    Tensor t = at::empty({4});
    float* p = t.mutable_data_ptr<float>();
    p[0] = 1.0f;
    p[1] = std::numeric_limits<float>::infinity();
    p[2] = -std::numeric_limits<float>::infinity();
    p[3] = std::nanf("");
    Tensor f = at::native::isfinite(t);
    Tensor i = at::native::isinf(t);
    Tensor n = at::native::isnan(t);
    EXPECT_TRUE(f.data_ptr<bool>()[0]);
    EXPECT_FALSE(f.data_ptr<bool>()[1]);
    EXPECT_FALSE(f.data_ptr<bool>()[3]);
    EXPECT_TRUE(i.data_ptr<bool>()[1]);
    EXPECT_TRUE(i.data_ptr<bool>()[2]);
    EXPECT_TRUE(n.data_ptr<bool>()[3]);
}

TEST(OpsExpansion, IsClose) {
    Tensor a = at::empty({3});
    Tensor b = at::empty({3});
    float av[] = {1.0f, 2.0f, 3.0f};
    float bv[] = {1.0f + 1e-7f, 2.5f, 3.0f};
    std::memcpy(a.mutable_data_ptr<float>(), av, sizeof(av));
    std::memcpy(b.mutable_data_ptr<float>(), bv, sizeof(bv));
    Tensor c = at::native::isclose(a, b, 1e-5, 1e-6);
    EXPECT_TRUE(c.data_ptr<bool>()[0]);
    EXPECT_FALSE(c.data_ptr<bool>()[1]);
    EXPECT_TRUE(c.data_ptr<bool>()[2]);
}

// ============================================================================
// Lerp / hypot / atan2
// ============================================================================

TEST(OpsExpansion, Lerp) {
    Tensor s = at::zeros({3});
    Tensor e = at::full({3}, Scalar(10));
    Tensor r = at::native::lerp(s, e, Scalar(0.5));
    for (int i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(r.data_ptr<float>()[i], 5.0f);
}

TEST(OpsExpansion, Hypot) {
    Tensor a = at::empty({2});
    Tensor b = at::empty({2});
    float av[] = {3.0f, 5.0f};
    float bv[] = {4.0f, 12.0f};
    std::memcpy(a.mutable_data_ptr<float>(), av, sizeof(av));
    std::memcpy(b.mutable_data_ptr<float>(), bv, sizeof(bv));
    Tensor h = at::native::hypot(a, b);
    EXPECT_NEAR(h.data_ptr<float>()[0], 5.0f, 1e-5f);
    EXPECT_NEAR(h.data_ptr<float>()[1], 13.0f, 1e-5f);
}

TEST(OpsExpansion, Atan2) {
    Tensor y = at::empty({2});
    Tensor x = at::empty({2});
    float yv[] = {1.0f, 0.0f};
    float xv[] = {1.0f, -1.0f};
    std::memcpy(y.mutable_data_ptr<float>(), yv, sizeof(yv));
    std::memcpy(x.mutable_data_ptr<float>(), xv, sizeof(xv));
    Tensor r = at::native::atan2(y, x);
    EXPECT_NEAR(r.data_ptr<float>()[0], static_cast<float>(M_PI / 4), 1e-5f);
    EXPECT_NEAR(r.data_ptr<float>()[1], static_cast<float>(M_PI), 1e-5f);
}
