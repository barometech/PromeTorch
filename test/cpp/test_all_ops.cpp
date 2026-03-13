#include <gtest/gtest.h>
#include "aten/src/ATen/ATen.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>

using namespace at;
using namespace torch;

// =============================================================================
// Math Unary Ops (20 tests)
// =============================================================================

TEST(UnaryOps, Neg) {
    auto t = tensor({1.0f, -2.0f, 3.0f});
    auto r = t.neg();
    EXPECT_EQ(r.numel(), 3);
    EXPECT_NEAR(r.data_ptr<float>()[0], -1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], -3.0f, 1e-5);
}

TEST(UnaryOps, Abs) {
    auto t = tensor({-1.0f, 2.0f, -3.0f});
    auto r = t.abs();
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 3.0f, 1e-5);
}

TEST(UnaryOps, Sqrt) {
    auto t = tensor({4.0f, 9.0f, 16.0f});
    auto r = t.sqrt();
    EXPECT_NEAR(r.data_ptr<float>()[0], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 3.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 4.0f, 1e-5);
}

TEST(UnaryOps, Rsqrt) {
    auto t = tensor({4.0f, 1.0f, 0.25f});
    auto r = t.rsqrt();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.5f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 2.0f, 1e-5);
}

TEST(UnaryOps, Square) {
    auto t = tensor({2.0f, 3.0f, 4.0f});
    auto r = t.square();
    EXPECT_NEAR(r.data_ptr<float>()[0], 4.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 9.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 16.0f, 1e-5);
}

TEST(UnaryOps, Exp) {
    auto t = tensor({0.0f, 1.0f});
    auto r = t.exp();
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], std::exp(1.0f), 1e-5);
}

TEST(UnaryOps, Log) {
    float e = std::exp(1.0f);
    auto t = tensor({1.0f, e});
    auto r = t.log();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 1.0f, 1e-4);
}

TEST(UnaryOps, Log2) {
    auto t = tensor({1.0f, 2.0f, 4.0f});
    auto r = t.log2();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 2.0f, 1e-5);
}

TEST(UnaryOps, Log10) {
    auto t = tensor({1.0f, 10.0f, 100.0f});
    auto r = t.log10();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 2.0f, 1e-5);
}

TEST(UnaryOps, Sin) {
    auto t = tensor({0.0f});
    auto r = t.sin();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
}

TEST(UnaryOps, Cos) {
    auto t = tensor({0.0f});
    auto r = t.cos();
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
}

TEST(UnaryOps, Tan) {
    auto t = tensor({0.0f});
    auto r = t.tan();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
}

TEST(UnaryOps, Tanh) {
    auto t = tensor({0.0f});
    auto r = t.tanh();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
}

TEST(UnaryOps, Sigmoid) {
    auto t = tensor({0.0f});
    auto r = t.sigmoid();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.5f, 1e-5);
}

TEST(UnaryOps, Relu) {
    auto t = tensor({-1.0f, 0.0f, 1.0f});
    auto r = t.relu();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 0.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 1.0f, 1e-5);
}

TEST(UnaryOps, Ceil) {
    auto t = tensor({1.2f, -1.7f});
    auto r = t.ceil();
    EXPECT_NEAR(r.data_ptr<float>()[0], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], -1.0f, 1e-5);
}

TEST(UnaryOps, Floor) {
    auto t = tensor({1.2f, -1.7f});
    auto r = t.floor();
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], -2.0f, 1e-5);
}

TEST(UnaryOps, Round) {
    auto t = tensor({1.4f, 1.5f, 1.6f});
    auto r = t.round();
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 2.0f, 1e-5);
}

TEST(UnaryOps, Sign) {
    auto t = tensor({-5.0f, 0.0f, 3.0f});
    auto r = t.sign();
    EXPECT_NEAR(r.data_ptr<float>()[0], -1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 0.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 1.0f, 1e-5);
}

TEST(UnaryOps, Reciprocal) {
    auto t = tensor({2.0f, 4.0f, 5.0f});
    auto r = t.reciprocal();
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.5f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 0.25f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 0.2f, 1e-5);
}

// =============================================================================
// In-place Ops (5 tests)
// =============================================================================

TEST(InplaceOps, NegInplace) {
    auto t = tensor({1.0f, -2.0f, 3.0f});
    t.neg_();
    EXPECT_NEAR(t.data_ptr<float>()[0], -1.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[2], -3.0f, 1e-5);
}

TEST(InplaceOps, ReluInplace) {
    auto t = tensor({-1.0f, 0.0f, 1.0f});
    t.relu_();
    EXPECT_NEAR(t.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[1], 0.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[2], 1.0f, 1e-5);
}

TEST(InplaceOps, SigmoidInplace) {
    auto t = tensor({0.0f});
    t.sigmoid_();
    EXPECT_NEAR(t.data_ptr<float>()[0], 0.5f, 1e-5);
}

TEST(InplaceOps, AddTensorInplace) {
    auto a = tensor({1.0f, 2.0f, 3.0f});
    auto b = tensor({10.0f, 20.0f, 30.0f});
    a.add_(b);
    EXPECT_NEAR(a.data_ptr<float>()[0], 11.0f, 1e-5);
    EXPECT_NEAR(a.data_ptr<float>()[1], 22.0f, 1e-5);
    EXPECT_NEAR(a.data_ptr<float>()[2], 33.0f, 1e-5);
}

TEST(InplaceOps, MulScalarInplace) {
    auto t = tensor({2.0f, 3.0f, 4.0f});
    t.mul_(Scalar(10.0));
    EXPECT_NEAR(t.data_ptr<float>()[0], 20.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[1], 30.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[2], 40.0f, 1e-5);
}

// =============================================================================
// Binary Ops (15 tests)
// =============================================================================

TEST(BinaryOps, AddTensor) {
    auto a = tensor({1.0f, 2.0f, 3.0f});
    auto b = tensor({4.0f, 5.0f, 6.0f});
    auto r = a.add(b);
    EXPECT_NEAR(r.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 7.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 9.0f, 1e-5);
}

TEST(BinaryOps, SubTensor) {
    auto a = tensor({10.0f, 20.0f, 30.0f});
    auto b = tensor({1.0f, 2.0f, 3.0f});
    auto r = a.sub(b);
    EXPECT_NEAR(r.data_ptr<float>()[0], 9.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 18.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 27.0f, 1e-5);
}

TEST(BinaryOps, MulTensor) {
    auto a = tensor({2.0f, 3.0f, 4.0f});
    auto b = tensor({5.0f, 6.0f, 7.0f});
    auto r = a.mul(b);
    EXPECT_NEAR(r.data_ptr<float>()[0], 10.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 18.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 28.0f, 1e-5);
}

TEST(BinaryOps, DivTensor) {
    auto a = tensor({10.0f, 20.0f, 30.0f});
    auto b = tensor({2.0f, 4.0f, 5.0f});
    auto r = a.div(b);
    EXPECT_NEAR(r.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 6.0f, 1e-5);
}

TEST(BinaryOps, AddScalar) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto r = t.add(Scalar(10.0));
    EXPECT_NEAR(r.data_ptr<float>()[0], 11.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 12.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 13.0f, 1e-5);
}

TEST(BinaryOps, MulScalar) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto r = t.mul(Scalar(3.0));
    EXPECT_NEAR(r.data_ptr<float>()[0], 3.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 6.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 9.0f, 1e-5);
}

TEST(BinaryOps, DivScalar) {
    auto t = tensor({10.0f, 20.0f, 30.0f});
    auto r = t.div(Scalar(10.0));
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 3.0f, 1e-5);
}

TEST(BinaryOps, PowTensor) {
    auto base = tensor({2.0f, 3.0f, 4.0f});
    auto exp = tensor({3.0f, 2.0f, 0.5f});
    auto r = base.pow(exp);
    EXPECT_NEAR(r.data_ptr<float>()[0], 8.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 9.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 2.0f, 1e-5);
}

TEST(BinaryOps, PowScalar) {
    auto t = tensor({2.0f, 3.0f, 4.0f});
    auto r = t.pow(Scalar(2.0));
    EXPECT_NEAR(r.data_ptr<float>()[0], 4.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 9.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 16.0f, 1e-5);
}

TEST(BinaryOps, Maximum) {
    auto a = tensor({1.0f, 5.0f, 3.0f});
    auto b = tensor({4.0f, 2.0f, 6.0f});
    auto r = at::maximum(a, b);
    EXPECT_NEAR(r.data_ptr<float>()[0], 4.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 6.0f, 1e-5);
}

TEST(BinaryOps, Minimum) {
    auto a = tensor({1.0f, 5.0f, 3.0f});
    auto b = tensor({4.0f, 2.0f, 6.0f});
    auto r = at::minimum(a, b);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 3.0f, 1e-5);
}

TEST(BinaryOps, AddcmulInplace) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto t1 = tensor({2.0f, 3.0f, 4.0f});
    auto t2 = tensor({3.0f, 4.0f, 5.0f});
    t.addcmul_(t1, t2, Scalar(0.1));
    // result = t + 0.1 * t1 * t2
    EXPECT_NEAR(t.data_ptr<float>()[0], 1.0f + 0.1f * 6.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[1], 2.0f + 0.1f * 12.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[2], 3.0f + 0.1f * 20.0f, 1e-5);
}

TEST(BinaryOps, AddcdivInplace) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto t1 = tensor({6.0f, 8.0f, 10.0f});
    auto t2 = tensor({2.0f, 4.0f, 5.0f});
    t.addcdiv_(t1, t2, Scalar(0.5));
    // result = t + 0.5 * t1 / t2
    EXPECT_NEAR(t.data_ptr<float>()[0], 1.0f + 0.5f * 3.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[1], 2.0f + 0.5f * 2.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[2], 3.0f + 0.5f * 2.0f, 1e-5);
}

// =============================================================================
// Clamp (4 tests)
// =============================================================================

TEST(Clamp, ClampMinMax) {
    auto t = tensor({-2.0f, 0.5f, 3.0f});
    auto r = t.clamp(Scalar(0.0), Scalar(1.0));
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 0.5f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 1.0f, 1e-5);
}

TEST(Clamp, ClampMin) {
    auto t = tensor({-2.0f, 0.5f, 3.0f});
    auto r = t.clamp_min(Scalar(0.0));
    EXPECT_NEAR(r.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 0.5f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 3.0f, 1e-5);
}

TEST(Clamp, ClampMax) {
    auto t = tensor({-2.0f, 0.5f, 3.0f});
    auto r = t.clamp_max(Scalar(1.0));
    EXPECT_NEAR(r.data_ptr<float>()[0], -2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 0.5f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 1.0f, 1e-5);
}

TEST(Clamp, ClampOptional) {
    auto t = tensor({-2.0f, 0.5f, 3.0f});
    auto r = t.clamp(std::optional<Scalar>(Scalar(-1.0)),
                     std::optional<Scalar>(Scalar(2.0)));
    EXPECT_NEAR(r.data_ptr<float>()[0], -1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 0.5f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 2.0f, 1e-5);
}

// =============================================================================
// Comparison (6 tests)
// =============================================================================

TEST(Comparison, Eq) {
    auto a = tensor({1.0f, 2.0f, 3.0f});
    auto b = tensor({1.0f, 5.0f, 3.0f});
    auto r = a.eq(b);
    EXPECT_EQ(r.data_ptr<float>()[0], 1.0f);
    EXPECT_EQ(r.data_ptr<float>()[1], 0.0f);
    EXPECT_EQ(r.data_ptr<float>()[2], 1.0f);
}

TEST(Comparison, Ne) {
    auto a = tensor({1.0f, 2.0f, 3.0f});
    auto b = tensor({1.0f, 5.0f, 3.0f});
    auto r = a.ne(b);
    EXPECT_EQ(r.data_ptr<float>()[0], 0.0f);
    EXPECT_EQ(r.data_ptr<float>()[1], 1.0f);
    EXPECT_EQ(r.data_ptr<float>()[2], 0.0f);
}

TEST(Comparison, Lt) {
    auto a = tensor({1.0f, 5.0f, 3.0f});
    auto b = tensor({2.0f, 3.0f, 3.0f});
    auto r = a.lt(b);
    EXPECT_EQ(r.data_ptr<float>()[0], 1.0f);
    EXPECT_EQ(r.data_ptr<float>()[1], 0.0f);
    EXPECT_EQ(r.data_ptr<float>()[2], 0.0f);
}

TEST(Comparison, Le) {
    auto a = tensor({1.0f, 5.0f, 3.0f});
    auto b = tensor({2.0f, 3.0f, 3.0f});
    auto r = a.le(b);
    EXPECT_EQ(r.data_ptr<float>()[0], 1.0f);
    EXPECT_EQ(r.data_ptr<float>()[1], 0.0f);
    EXPECT_EQ(r.data_ptr<float>()[2], 1.0f);
}

TEST(Comparison, Gt) {
    auto a = tensor({1.0f, 5.0f, 3.0f});
    auto b = tensor({2.0f, 3.0f, 3.0f});
    auto r = a.gt(b);
    EXPECT_EQ(r.data_ptr<float>()[0], 0.0f);
    EXPECT_EQ(r.data_ptr<float>()[1], 1.0f);
    EXPECT_EQ(r.data_ptr<float>()[2], 0.0f);
}

TEST(Comparison, GeWithScalar) {
    auto a = tensor({1.0f, 2.0f, 3.0f});
    auto r = a.ge(Scalar(2.0));
    EXPECT_EQ(r.data_ptr<float>()[0], 0.0f);
    EXPECT_EQ(r.data_ptr<float>()[1], 1.0f);
    EXPECT_EQ(r.data_ptr<float>()[2], 1.0f);
}

// =============================================================================
// Reduction Ops (25 tests)
// =============================================================================

TEST(ReduceOps, SumAll) {
    auto t = tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto r = t.sum();
    EXPECT_NEAR(r.data_ptr<float>()[0], 10.0f, 1e-5);
}

TEST(ReduceOps, SumDim) {
    // 2x3 matrix: [[1,2,3],[4,5,6]]
    auto t = tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto r = t.sum(0);  // sum over rows -> [5, 7, 9]
    EXPECT_EQ(r.numel(), 3);
    EXPECT_NEAR(r.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 7.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 9.0f, 1e-5);
}

TEST(ReduceOps, SumDim1) {
    auto t = tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto r = t.sum(1);  // sum over cols -> [6, 15]
    EXPECT_EQ(r.numel(), 2);
    EXPECT_NEAR(r.data_ptr<float>()[0], 6.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 15.0f, 1e-5);
}

TEST(ReduceOps, MeanAll) {
    auto t = tensor({2.0f, 4.0f, 6.0f});
    auto r = t.mean();
    EXPECT_NEAR(r.data_ptr<float>()[0], 4.0f, 1e-5);
}

TEST(ReduceOps, MeanDim) {
    auto t = tensor({{2.0f, 4.0f}, {6.0f, 8.0f}});
    auto r = t.mean(0);  // mean over rows -> [4, 6]
    EXPECT_NEAR(r.data_ptr<float>()[0], 4.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 6.0f, 1e-5);
}

TEST(ReduceOps, ProdAll) {
    auto t = tensor({2.0f, 3.0f, 4.0f});
    auto r = t.prod();
    EXPECT_NEAR(r.data_ptr<float>()[0], 24.0f, 1e-5);
}

TEST(ReduceOps, MaxAll) {
    auto t = tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f});
    auto r = t.max();
    EXPECT_NEAR(r.data_ptr<float>()[0], 5.0f, 1e-5);
}

TEST(ReduceOps, MaxDim) {
    auto t = tensor({{1.0f, 3.0f}, {4.0f, 2.0f}});
    auto [values, indices] = t.max(1);
    EXPECT_NEAR(values.data_ptr<float>()[0], 3.0f, 1e-5);
    EXPECT_NEAR(values.data_ptr<float>()[1], 4.0f, 1e-5);
    EXPECT_EQ(indices.data_ptr<int64_t>()[0], 1);
    EXPECT_EQ(indices.data_ptr<int64_t>()[1], 0);
}

TEST(ReduceOps, MinAll) {
    auto t = tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f});
    auto r = t.min();
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
}

TEST(ReduceOps, MinDim) {
    auto t = tensor({{1.0f, 3.0f}, {4.0f, 2.0f}});
    auto [values, indices] = t.min(1);
    EXPECT_NEAR(values.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(values.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_EQ(indices.data_ptr<int64_t>()[0], 0);
    EXPECT_EQ(indices.data_ptr<int64_t>()[1], 1);
}

TEST(ReduceOps, ArgmaxAll) {
    auto t = tensor({3.0f, 1.0f, 5.0f, 2.0f});
    auto r = t.argmax();
    EXPECT_EQ(r.data_ptr<int64_t>()[0], 2);
}

TEST(ReduceOps, ArgmaxDim) {
    auto t = tensor({{1.0f, 3.0f}, {4.0f, 2.0f}});
    auto r = t.argmax(1);
    EXPECT_EQ(r.data_ptr<int64_t>()[0], 1);
    EXPECT_EQ(r.data_ptr<int64_t>()[1], 0);
}

TEST(ReduceOps, ArgminAll) {
    auto t = tensor({3.0f, 1.0f, 5.0f, 2.0f});
    auto r = t.argmin();
    EXPECT_EQ(r.data_ptr<int64_t>()[0], 1);
}

TEST(ReduceOps, ArgminDim) {
    auto t = tensor({{1.0f, 3.0f}, {4.0f, 2.0f}});
    auto r = t.argmin(1);
    EXPECT_EQ(r.data_ptr<int64_t>()[0], 0);
    EXPECT_EQ(r.data_ptr<int64_t>()[1], 1);
}

TEST(ReduceOps, VarAll) {
    auto t = tensor({2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f});
    auto r = t.var(true);  // unbiased
    // mean=5, var = sum((x-5)^2)/(8-1) = 32/7 = 4.571428...
    EXPECT_NEAR(r.data_ptr<float>()[0], 32.0f / 7.0f, 1e-4);
}

TEST(ReduceOps, StdAll) {
    auto t = tensor({2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f});
    auto r = t.std(true);
    EXPECT_NEAR(r.data_ptr<float>()[0], std::sqrt(32.0f / 7.0f), 1e-4);
}

TEST(ReduceOps, NormL2) {
    auto t = tensor({3.0f, 4.0f});
    auto r = t.norm(Scalar(2.0));
    EXPECT_NEAR(r.data_ptr<float>()[0], 5.0f, 1e-5);
}

TEST(ReduceOps, AllTrue) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    EXPECT_TRUE(t.all());
}

TEST(ReduceOps, AnyTrue) {
    auto t = tensor({0.0f, 0.0f, 1.0f});
    EXPECT_TRUE(t.any());
}

TEST(ReduceOps, Cumsum) {
    auto t = tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto r = t.cumsum(0);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 3.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 6.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[3], 10.0f, 1e-5);
}

TEST(ReduceOps, Cumprod) {
    auto t = tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto r = t.cumprod(0);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 6.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[3], 24.0f, 1e-5);
}

TEST(ReduceOps, Sort) {
    auto t = tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f});
    auto [values, indices] = t.sort(0);
    EXPECT_NEAR(values.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(values.data_ptr<float>()[1], 1.0f, 1e-5);
    EXPECT_NEAR(values.data_ptr<float>()[2], 3.0f, 1e-5);
    EXPECT_NEAR(values.data_ptr<float>()[3], 4.0f, 1e-5);
    EXPECT_NEAR(values.data_ptr<float>()[4], 5.0f, 1e-5);
}

TEST(ReduceOps, Argsort) {
    auto t = tensor({3.0f, 1.0f, 2.0f});
    auto r = t.argsort(0);
    EXPECT_EQ(r.data_ptr<int64_t>()[0], 1);
    EXPECT_EQ(r.data_ptr<int64_t>()[1], 2);
    EXPECT_EQ(r.data_ptr<int64_t>()[2], 0);
}

TEST(ReduceOps, Topk) {
    auto t = tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f});
    auto [values, indices] = t.topk(3, 0, true, true);
    EXPECT_EQ(values.numel(), 3);
    EXPECT_NEAR(values.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(values.data_ptr<float>()[1], 4.0f, 1e-5);
    EXPECT_NEAR(values.data_ptr<float>()[2], 3.0f, 1e-5);
}

// =============================================================================
// Shape Ops (20 tests)
// =============================================================================

TEST(ShapeOps, View) {
    auto t = arange(Scalar(0), Scalar(6));
    auto r = t.view({2, 3});
    EXPECT_EQ(r.dim(), 2);
    EXPECT_EQ(r.size(0), 2);
    EXPECT_EQ(r.size(1), 3);
}

TEST(ShapeOps, Reshape) {
    auto t = arange(Scalar(0), Scalar(12));
    auto r = t.reshape({3, 4});
    EXPECT_EQ(r.dim(), 2);
    EXPECT_EQ(r.size(0), 3);
    EXPECT_EQ(r.size(1), 4);
}

TEST(ShapeOps, Flatten) {
    auto t = ones({2, 3, 4});
    auto r = t.flatten();
    EXPECT_EQ(r.dim(), 1);
    EXPECT_EQ(r.numel(), 24);
}

TEST(ShapeOps, Squeeze) {
    auto t = ones({1, 3, 1, 4});
    auto r = t.squeeze();
    EXPECT_EQ(r.dim(), 2);
    EXPECT_EQ(r.size(0), 3);
    EXPECT_EQ(r.size(1), 4);
}

TEST(ShapeOps, SqueezeDim) {
    auto t = ones({1, 3, 4});
    auto r = t.squeeze(0);
    EXPECT_EQ(r.dim(), 2);
    EXPECT_EQ(r.size(0), 3);
    EXPECT_EQ(r.size(1), 4);
}

TEST(ShapeOps, Unsqueeze) {
    auto t = ones({3, 4});
    auto r = t.unsqueeze(0);
    EXPECT_EQ(r.dim(), 3);
    EXPECT_EQ(r.size(0), 1);
    EXPECT_EQ(r.size(1), 3);
    EXPECT_EQ(r.size(2), 4);
}

TEST(ShapeOps, Transpose) {
    auto t = tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto r = t.transpose(0, 1);
    EXPECT_EQ(r.size(0), 3);
    EXPECT_EQ(r.size(1), 2);
}

TEST(ShapeOps, Permute) {
    auto t = ones({2, 3, 4});
    auto r = t.permute({2, 0, 1});
    EXPECT_EQ(r.size(0), 4);
    EXPECT_EQ(r.size(1), 2);
    EXPECT_EQ(r.size(2), 3);
}

TEST(ShapeOps, T) {
    auto t = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto r = t.t();
    // Transpose is a view so need contiguous to read in order
    auto rc = r.contiguous();
    EXPECT_EQ(rc.size(0), 2);
    EXPECT_EQ(rc.size(1), 2);
    EXPECT_NEAR(rc.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(rc.data_ptr<float>()[1], 3.0f, 1e-5);
    EXPECT_NEAR(rc.data_ptr<float>()[2], 2.0f, 1e-5);
    EXPECT_NEAR(rc.data_ptr<float>()[3], 4.0f, 1e-5);
}

TEST(ShapeOps, Cat) {
    auto a = tensor({1.0f, 2.0f});
    auto b = tensor({3.0f, 4.0f});
    auto r = torch::cat({a, b}, 0);
    EXPECT_EQ(r.numel(), 4);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 3.0f, 1e-5);
}

TEST(ShapeOps, Stack) {
    auto a = tensor({1.0f, 2.0f});
    auto b = tensor({3.0f, 4.0f});
    auto r = torch::stack({a, b}, 0);
    EXPECT_EQ(r.dim(), 2);
    EXPECT_EQ(r.size(0), 2);
    EXPECT_EQ(r.size(1), 2);
}

TEST(ShapeOps, Split) {
    auto t = arange(Scalar(0), Scalar(6));
    auto parts = t.split(2, 0);
    EXPECT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0].numel(), 2);
    EXPECT_EQ(parts[1].numel(), 2);
    EXPECT_EQ(parts[2].numel(), 2);
}

TEST(ShapeOps, Chunk) {
    auto t = arange(Scalar(0), Scalar(6));
    auto parts = t.chunk(3, 0);
    EXPECT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0].numel(), 2);
}

TEST(ShapeOps, Expand) {
    auto t = tensor({1.0f, 2.0f, 3.0f}).unsqueeze(0);  // [1, 3]
    auto r = t.expand({4, 3});
    EXPECT_EQ(r.size(0), 4);
    EXPECT_EQ(r.size(1), 3);
    auto rc = r.contiguous();
    EXPECT_NEAR(rc.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(rc.data_ptr<float>()[3], 1.0f, 1e-5);
}

TEST(ShapeOps, Repeat) {
    auto t = tensor({1.0f, 2.0f});
    auto r = t.repeat({3});
    EXPECT_EQ(r.numel(), 6);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 1.0f, 1e-5);
}

TEST(ShapeOps, Select) {
    auto t = tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto r = t.select(0, 1);  // select row 1
    auto rc = r.contiguous();
    EXPECT_EQ(rc.numel(), 3);
    EXPECT_NEAR(rc.data_ptr<float>()[0], 4.0f, 1e-5);
}

TEST(ShapeOps, Narrow) {
    auto t = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto r = t.narrow(0, 1, 3);  // elements at index 1,2,3
    auto rc = r.contiguous();
    EXPECT_EQ(rc.numel(), 3);
    EXPECT_NEAR(rc.data_ptr<float>()[0], 2.0f, 1e-5);
    EXPECT_NEAR(rc.data_ptr<float>()[2], 4.0f, 1e-5);
}

TEST(ShapeOps, Slice) {
    auto t = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto r = t.slice(0, 0, 5, 2);  // elements at index 0,2,4
    auto rc = r.contiguous();
    EXPECT_EQ(rc.numel(), 3);
    EXPECT_NEAR(rc.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(rc.data_ptr<float>()[1], 3.0f, 1e-5);
    EXPECT_NEAR(rc.data_ptr<float>()[2], 5.0f, 1e-5);
}

TEST(ShapeOps, Roll) {
    auto t = tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto r = torch::roll(t, {1}, {0});
    EXPECT_NEAR(r.data_ptr<float>()[0], 4.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 1.0f, 1e-5);
}

TEST(ShapeOps, Flip) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto r = torch::flip(t, {0});
    EXPECT_NEAR(r.data_ptr<float>()[0], 3.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 1.0f, 1e-5);
}

TEST(ShapeOps, CloneContiguous) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto c = t.clone();
    EXPECT_EQ(c.numel(), 3);
    EXPECT_NEAR(c.data_ptr<float>()[0], 1.0f, 1e-5);
    // Modifying clone should not affect original
    c.data_ptr<float>()[0] = 99.0f;
    EXPECT_NEAR(t.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_TRUE(c.is_contiguous());
}

// =============================================================================
// Index Ops (10 tests)
// =============================================================================

TEST(IndexOps, IndexSelect) {
    auto t = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    auto idx = at::tensor(std::vector<int64_t>{0, 2},
                          TensorOptions().dtype(c10::ScalarType::Long));
    auto r = at::native::index_select(t, 0, idx);
    EXPECT_EQ(r.size(0), 2);
    EXPECT_EQ(r.size(1), 2);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 5.0f, 1e-5);
}

TEST(IndexOps, IndexWithTensor) {
    auto t = tensor({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
    auto idx = at::tensor(std::vector<int64_t>{1, 3, 4},
                          TensorOptions().dtype(c10::ScalarType::Long));
    auto r = at::native::index_with_tensor(t, 0, idx);
    EXPECT_EQ(r.numel(), 3);
    EXPECT_NEAR(r.data_ptr<float>()[0], 20.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 40.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 50.0f, 1e-5);
}

TEST(IndexOps, IndexPut) {
    auto t = zeros({5});
    auto idx = at::tensor(std::vector<int64_t>{1, 3},
                          TensorOptions().dtype(c10::ScalarType::Long));
    auto vals = tensor({10.0f, 30.0f});
    at::native::index_put_(t, 0, idx, vals);
    EXPECT_NEAR(t.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[1], 10.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[3], 30.0f, 1e-5);
}

TEST(IndexOps, ScatterAdd) {
    auto t = zeros({5});
    auto idx = at::tensor(std::vector<int64_t>{0, 1, 0},
                          TensorOptions().dtype(c10::ScalarType::Long));
    auto src = tensor({1.0f, 2.0f, 3.0f});
    at::native::scatter_add_(t, 0, idx, src);
    EXPECT_NEAR(t.data_ptr<float>()[0], 4.0f, 1e-5);  // 1+3
    EXPECT_NEAR(t.data_ptr<float>()[1], 2.0f, 1e-5);
}

TEST(IndexOps, ScatterReduceSum) {
    auto t = zeros({4});
    auto idx = at::tensor(std::vector<int64_t>{0, 1, 0, 2},
                          TensorOptions().dtype(c10::ScalarType::Long));
    auto src = tensor({1.0f, 2.0f, 3.0f, 4.0f});
    torch::scatter_reduce_(t, 0, idx, src, "sum");
    EXPECT_NEAR(t.data_ptr<float>()[0], 4.0f, 1e-5);  // 1+3
    EXPECT_NEAR(t.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[2], 4.0f, 1e-5);
}

TEST(IndexOps, BooleanIndex) {
    auto t = tensor({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
    auto mask = at::empty({5}, TensorOptions().dtype(c10::ScalarType::Bool));
    bool* m = mask.mutable_data_ptr<bool>();
    m[0] = true; m[1] = false; m[2] = true; m[3] = false; m[4] = true;
    auto r = at::native::boolean_index(t, mask);
    EXPECT_EQ(r.numel(), 3);
    EXPECT_NEAR(r.data_ptr<float>()[0], 10.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 30.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 50.0f, 1e-5);
}

TEST(IndexOps, Gather) {
    auto t = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto idx = at::tensor({{0LL, 0LL}, {1LL, 0LL}},
                          TensorOptions().dtype(c10::ScalarType::Long));
    auto r = at::native::gather(t, 1, idx);
    EXPECT_EQ(r.size(0), 2);
    EXPECT_EQ(r.size(1), 2);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 4.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[3], 3.0f, 1e-5);
}

TEST(IndexOps, MaskedFill) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto mask = at::empty({3}, TensorOptions().dtype(c10::ScalarType::Bool));
    bool* m = mask.mutable_data_ptr<bool>();
    m[0] = false; m[1] = true; m[2] = false;
    at::native::masked_fill_(t, mask, Scalar(-1.0));
    EXPECT_NEAR(t.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[1], -1.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[2], 3.0f, 1e-5);
}

TEST(IndexOps, Searchsorted) {
    auto sorted = tensor({1.0f, 3.0f, 5.0f, 7.0f, 9.0f});
    auto values = tensor({2.0f, 4.0f, 8.0f});
    auto r = torch::searchsorted(sorted, values);
    EXPECT_EQ(r.data_ptr<int64_t>()[0], 1);  // 2 goes before 3
    EXPECT_EQ(r.data_ptr<int64_t>()[1], 2);  // 4 goes before 5
    EXPECT_EQ(r.data_ptr<int64_t>()[2], 4);  // 8 goes before 9
}

// =============================================================================
// Linear Algebra (15+ tests)
// =============================================================================

TEST(LinAlg, Mm) {
    // A=[2,3], B=[3,2]
    auto A = tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto B = tensor({{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}});
    auto C = A.mm(B);
    EXPECT_EQ(C.size(0), 2);
    EXPECT_EQ(C.size(1), 2);
    // C[0,0] = 1*7+2*9+3*11 = 58
    EXPECT_NEAR(C.data_ptr<float>()[0], 58.0f, 1e-4);
    // C[0,1] = 1*8+2*10+3*12 = 64
    EXPECT_NEAR(C.data_ptr<float>()[1], 64.0f, 1e-4);
    // C[1,0] = 4*7+5*9+6*11 = 139
    EXPECT_NEAR(C.data_ptr<float>()[2], 139.0f, 1e-4);
    // C[1,1] = 4*8+5*10+6*12 = 154
    EXPECT_NEAR(C.data_ptr<float>()[3], 154.0f, 1e-4);
}

TEST(LinAlg, Mv) {
    auto A = tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    auto v = tensor({1.0f, 2.0f, 3.0f});
    auto r = A.mv(v);
    EXPECT_EQ(r.numel(), 2);
    // r[0] = 1+4+9 = 14
    EXPECT_NEAR(r.data_ptr<float>()[0], 14.0f, 1e-4);
    // r[1] = 4+10+18 = 32
    EXPECT_NEAR(r.data_ptr<float>()[1], 32.0f, 1e-4);
}

TEST(LinAlg, Bmm) {
    auto A = ones({2, 2, 3});
    auto B = ones({2, 3, 2});
    // Fill with known values
    float* a_data = A.mutable_data_ptr<float>();
    for (int i = 0; i < 12; ++i) a_data[i] = static_cast<float>(i + 1);
    float* b_data = B.mutable_data_ptr<float>();
    for (int i = 0; i < 12; ++i) b_data[i] = static_cast<float>(i + 1);
    auto C = A.bmm(B);
    EXPECT_EQ(C.size(0), 2);
    EXPECT_EQ(C.size(1), 2);
    EXPECT_EQ(C.size(2), 2);
    // Batch 0: A=[1,2,3;4,5,6], B=[1,2;3,4;5,6]
    // C[0,0,0] = 1*1+2*3+3*5 = 22
    EXPECT_NEAR(C.data_ptr<float>()[0], 22.0f, 1e-4);
}

TEST(LinAlg, MatmulMM) {
    auto A = tensor({{1.0f, 0.0f}, {0.0f, 1.0f}});
    auto B = tensor({{5.0f, 6.0f}, {7.0f, 8.0f}});
    auto r = A.matmul(B);
    EXPECT_NEAR(r.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[3], 8.0f, 1e-5);
}

TEST(LinAlg, Dot) {
    auto a = tensor({1.0f, 2.0f, 3.0f});
    auto b = tensor({4.0f, 5.0f, 6.0f});
    auto r = a.dot(b);
    // 1*4+2*5+3*6 = 32
    EXPECT_NEAR(r.data_ptr<float>()[0], 32.0f, 1e-5);
}

TEST(LinAlg, Outer) {
    auto a = tensor({1.0f, 2.0f});
    auto b = tensor({3.0f, 4.0f, 5.0f});
    auto r = at::native::outer(a, b);
    EXPECT_EQ(r.size(0), 2);
    EXPECT_EQ(r.size(1), 3);
    EXPECT_NEAR(r.data_ptr<float>()[0], 3.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 4.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[3], 6.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[4], 8.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[5], 10.0f, 1e-5);
}

TEST(LinAlg, Addmm) {
    auto M = zeros({2, 2});
    auto A = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto B = tensor({{5.0f, 6.0f}, {7.0f, 8.0f}});
    // C = 0*M + 1*A@B
    auto r = at::native::addmm(M, A, B, Scalar(0.0), Scalar(1.0));
    // A@B = [19,22;43,50]
    EXPECT_NEAR(r.data_ptr<float>()[0], 19.0f, 1e-4);
    EXPECT_NEAR(r.data_ptr<float>()[1], 22.0f, 1e-4);
    EXPECT_NEAR(r.data_ptr<float>()[2], 43.0f, 1e-4);
    EXPECT_NEAR(r.data_ptr<float>()[3], 50.0f, 1e-4);
}

TEST(LinAlg, Trace) {
    auto t = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto r = torch::trace(t);
    EXPECT_NEAR(r.data_ptr<float>()[0], 5.0f, 1e-5);
}

TEST(LinAlg, EinsumMM) {
    auto A = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto B = tensor({{5.0f, 6.0f}, {7.0f, 8.0f}});
    auto r = torch::einsum("ij,jk->ik", {A, B});
    // Same as mm: [19,22;43,50]
    EXPECT_NEAR(r.data_ptr<float>()[0], 19.0f, 1e-4);
    EXPECT_NEAR(r.data_ptr<float>()[3], 50.0f, 1e-4);
}

TEST(LinAlg, LU) {
    auto A = tensor({{2.0f, 1.0f}, {5.0f, 3.0f}});
    auto [L, U, P] = torch::linalg::lu(A);
    // Verify P@L@U = A (approximately)
    auto PLU = P.mm(L).mm(U);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(PLU.data_ptr<float>()[i], A.data_ptr<float>()[i], 1e-4);
    }
}

TEST(LinAlg, QR) {
    auto A = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto [Q, R] = torch::linalg::qr(A);
    // Verify Q@R = A
    auto QR = Q.mm(R);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(QR.data_ptr<float>()[i], A.data_ptr<float>()[i], 1e-3);
    }
}

TEST(LinAlg, Solve) {
    // A@x = b -> x = solve(A, b)
    auto A = tensor({{2.0f, 1.0f}, {5.0f, 3.0f}});
    auto b = tensor({4.0f, 7.0f});
    auto x = torch::solve(A, b);
    // Verify A@x = b
    auto Ax = A.mv(x);
    EXPECT_NEAR(Ax.data_ptr<float>()[0], b.data_ptr<float>()[0], 1e-3);
    EXPECT_NEAR(Ax.data_ptr<float>()[1], b.data_ptr<float>()[1], 1e-3);
}

TEST(LinAlg, Inverse) {
    auto A = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto Ainv = torch::inverse(A);
    auto I = A.mm(Ainv);
    EXPECT_NEAR(I.data_ptr<float>()[0], 1.0f, 1e-3);
    EXPECT_NEAR(I.data_ptr<float>()[1], 0.0f, 1e-3);
    EXPECT_NEAR(I.data_ptr<float>()[2], 0.0f, 1e-3);
    EXPECT_NEAR(I.data_ptr<float>()[3], 1.0f, 1e-3);
}

TEST(LinAlg, Det) {
    auto A = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto d = torch::det(A);
    // det = 1*4-2*3 = -2
    EXPECT_NEAR(d.data_ptr<float>()[0], -2.0f, 1e-3);
}

TEST(LinAlg, Lstsq) {
    // Overdetermined: 3 equations, 2 unknowns
    auto A = tensor({{1.0f, 1.0f}, {1.0f, 2.0f}, {1.0f, 3.0f}});
    auto b = tensor({1.0f, 2.0f, 2.0f});
    auto x = torch::lstsq(A, b);
    EXPECT_EQ(x.numel(), 2);
    // x should minimize ||Ax-b||^2
    auto residual = A.mv(x).sub(b);
    float norm2 = 0;
    for (int i = 0; i < 3; ++i) {
        float r = residual.data_ptr<float>()[i];
        norm2 += r * r;
    }
    // Residual should be small (this is a least-squares solution)
    EXPECT_LT(norm2, 1.0f);
}

TEST(LinAlg, SVD) {
    auto A = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    auto [U, S, Vh] = torch::svd(A);
    // Reconstruct: A ~ U @ diag(S) @ Vh
    // Build diagonal S matrix
    auto Smat = zeros({(int64_t)U.size(1), (int64_t)Vh.size(0)});
    for (int64_t i = 0; i < S.numel(); ++i) {
        Smat.mutable_data_ptr<float>()[i * Vh.size(0) + i] = S.data_ptr<float>()[i];
    }
    auto reconstructed = U.mm(Smat).mm(Vh);
    for (int64_t i = 0; i < A.numel(); ++i) {
        EXPECT_NEAR(reconstructed.data_ptr<float>()[i], A.data_ptr<float>()[i], 1e-3);
    }
}

TEST(LinAlg, Pinverse) {
    auto A = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});
    auto Apinv = torch::pinverse(A);
    // A @ pinv(A) @ A should be approximately A
    auto AApA = A.mm(Apinv).mm(A);
    for (int64_t i = 0; i < A.numel(); ++i) {
        EXPECT_NEAR(AApA.data_ptr<float>()[i], A.data_ptr<float>()[i], 1e-2);
    }
}

TEST(LinAlg, Eig) {
    // Symmetric matrix: eigenvalues are real
    auto A = tensor({{2.0f, 1.0f}, {1.0f, 2.0f}});
    auto [eigenvalues, eigenvectors] = torch::eig(A);
    // Eigenvalues of [[2,1],[1,2]] are 1 and 3
    float e0 = eigenvalues.data_ptr<float>()[0];
    float e1 = eigenvalues.data_ptr<float>()[1];
    float emin = std::min(e0, e1);
    float emax = std::max(e0, e1);
    EXPECT_NEAR(emin, 1.0f, 1e-2);
    EXPECT_NEAR(emax, 3.0f, 1e-2);
}

// =============================================================================
// Factory Functions (10+ tests)
// =============================================================================

TEST(Factory, Zeros) {
    auto t = zeros({2, 3});
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(t.data_ptr<float>()[i], 0.0f, 1e-5);
}

TEST(Factory, Ones) {
    auto t = ones({3, 2});
    EXPECT_EQ(t.numel(), 6);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(t.data_ptr<float>()[i], 1.0f, 1e-5);
}

TEST(Factory, Full) {
    auto t = full({2, 3}, Scalar(7.0));
    EXPECT_EQ(t.numel(), 6);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(t.data_ptr<float>()[i], 7.0f, 1e-5);
}

TEST(Factory, Eye) {
    auto t = eye(3, 3);
    EXPECT_EQ(t.size(0), 3);
    EXPECT_EQ(t.size(1), 3);
    float* d = t.data_ptr<float>();
    EXPECT_NEAR(d[0], 1.0f, 1e-5);
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    EXPECT_NEAR(d[4], 1.0f, 1e-5);
    EXPECT_NEAR(d[8], 1.0f, 1e-5);
}

TEST(Factory, Arange) {
    auto t = arange(Scalar(0), Scalar(5));
    EXPECT_EQ(t.numel(), 5);
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(t.data_ptr<float>()[i], static_cast<float>(i), 1e-5);
}

TEST(Factory, Linspace) {
    auto t = linspace(Scalar(0.0), Scalar(1.0), 5);
    EXPECT_EQ(t.numel(), 5);
    EXPECT_NEAR(t.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[2], 0.5f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[4], 1.0f, 1e-5);
}

TEST(Factory, Logspace) {
    auto t = logspace(Scalar(0.0), Scalar(2.0), 3, 10.0);
    EXPECT_EQ(t.numel(), 3);
    EXPECT_NEAR(t.data_ptr<float>()[0], 1.0f, 1e-4);
    EXPECT_NEAR(t.data_ptr<float>()[1], 10.0f, 1e-4);
    EXPECT_NEAR(t.data_ptr<float>()[2], 100.0f, 1e-3);
}

TEST(Factory, Rand) {
    auto t = rand({100});
    EXPECT_EQ(t.numel(), 100);
    float* d = t.data_ptr<float>();
    for (int i = 0; i < 100; ++i) {
        EXPECT_GE(d[i], 0.0f);
        EXPECT_LT(d[i], 1.0f);
    }
}

TEST(Factory, Randn) {
    auto t = randn({1000});
    EXPECT_EQ(t.numel(), 1000);
    // Check mean is roughly 0 and std is roughly 1
    float sum = 0;
    float* d = t.data_ptr<float>();
    for (int i = 0; i < 1000; ++i) sum += d[i];
    float mean = sum / 1000.0f;
    EXPECT_NEAR(mean, 0.0f, 0.15f);
}

TEST(Factory, Randint) {
    auto t = randint(0, 10, {50});
    EXPECT_EQ(t.numel(), 50);
    int64_t* d = t.data_ptr<int64_t>();
    for (int i = 0; i < 50; ++i) {
        EXPECT_GE(d[i], 0);
        EXPECT_LT(d[i], 10);
    }
}

TEST(Factory, Randperm) {
    auto t = randperm(10);
    EXPECT_EQ(t.numel(), 10);
    int64_t* d = t.data_ptr<int64_t>();
    std::set<int64_t> vals(d, d + 10);
    EXPECT_EQ(vals.size(), 10);
    for (int64_t i = 0; i < 10; ++i)
        EXPECT_TRUE(vals.count(i) == 1);
}

TEST(Factory, Multinomial) {
    auto probs = tensor({0.1f, 0.2f, 0.3f, 0.4f});
    auto r = at::multinomial(probs, 10, true);
    EXPECT_EQ(r.numel(), 10);
    int64_t* d = r.data_ptr<int64_t>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_GE(d[i], 0);
        EXPECT_LT(d[i], 4);
    }
}

// =============================================================================
// Type/Device (5 tests)
// =============================================================================

TEST(TypeDevice, ToDouble) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto r = t.to(c10::ScalarType::Double);
    EXPECT_EQ(r.dtype(), c10::ScalarType::Double);
    EXPECT_NEAR(r.data_ptr<double>()[0], 1.0, 1e-10);
    EXPECT_NEAR(r.data_ptr<double>()[1], 2.0, 1e-10);
}

TEST(TypeDevice, CloneWithDtype) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto r = t.to(c10::ScalarType::Double);
    EXPECT_EQ(r.dtype(), c10::ScalarType::Double);
    // Original should remain float
    EXPECT_EQ(t.dtype(), c10::ScalarType::Float);
}

TEST(TypeDevice, OperatorOverloadsArith) {
    auto a = tensor({1.0f, 2.0f, 3.0f});
    auto b = tensor({4.0f, 5.0f, 6.0f});
    auto sum = a + b;
    auto diff = a - b;
    auto prod = a * b;
    auto quot = b / a;
    EXPECT_NEAR(sum.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(diff.data_ptr<float>()[1], -3.0f, 1e-5);
    EXPECT_NEAR(prod.data_ptr<float>()[2], 18.0f, 1e-5);
    EXPECT_NEAR(quot.data_ptr<float>()[0], 4.0f, 1e-5);
    // Unary neg
    auto neg = -a;
    EXPECT_NEAR(neg.data_ptr<float>()[0], -1.0f, 1e-5);
}

TEST(TypeDevice, ComparisonOperators) {
    auto a = tensor({1.0f, 2.0f, 3.0f});
    auto b = tensor({2.0f, 2.0f, 1.0f});
    auto eq = (a == b);
    auto ne = (a != b);
    auto lt = (a < b);
    auto le = (a <= b);
    auto gt = (a > b);
    auto ge = (a >= b);
    EXPECT_EQ(eq.data_ptr<float>()[1], 1.0f);
    EXPECT_EQ(ne.data_ptr<float>()[0], 1.0f);
    EXPECT_EQ(lt.data_ptr<float>()[0], 1.0f);
    EXPECT_EQ(le.data_ptr<float>()[1], 1.0f);
    EXPECT_EQ(gt.data_ptr<float>()[2], 1.0f);
    EXPECT_EQ(ge.data_ptr<float>()[1], 1.0f);
}

TEST(TypeDevice, ScalarTensorArith) {
    auto t = tensor({2.0f, 4.0f, 6.0f});
    auto r = t + Scalar(1.0);
    EXPECT_NEAR(r.data_ptr<float>()[0], 3.0f, 1e-5);
    auto r2 = t * Scalar(0.5);
    EXPECT_NEAR(r2.data_ptr<float>()[1], 2.0f, 1e-5);
}

// =============================================================================
// Broadcasting (5 tests)
// =============================================================================

TEST(Broadcasting, DifferentShapes) {
    // [3,1] + [1,4] = [3,4]
    auto a = tensor({1.0f, 2.0f, 3.0f}).reshape({3, 1});
    auto b = tensor({10.0f, 20.0f, 30.0f, 40.0f}).reshape({1, 4});
    auto r = a.add(b);
    EXPECT_EQ(r.size(0), 3);
    EXPECT_EQ(r.size(1), 4);
    auto rc = r.contiguous();
    EXPECT_NEAR(rc.data_ptr<float>()[0], 11.0f, 1e-5);
    EXPECT_NEAR(rc.data_ptr<float>()[3], 41.0f, 1e-5);
}

TEST(Broadcasting, MatrixVector) {
    // [2,3] * [3] = [2,3]
    auto a = ones({2, 3});
    auto b = tensor({1.0f, 2.0f, 3.0f});
    auto r = a.mul(b);
    EXPECT_EQ(r.size(0), 2);
    EXPECT_EQ(r.size(1), 3);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 3.0f, 1e-5);
}

TEST(Broadcasting, ScalarPlusTensor) {
    auto t = tensor({1.0f, 2.0f, 3.0f});
    auto r = Scalar(10.0) + t;
    EXPECT_NEAR(r.data_ptr<float>()[0], 11.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 12.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 13.0f, 1e-5);
}

TEST(Broadcasting, ScalarTimesTensor) {
    auto t = tensor({2.0f, 3.0f, 4.0f});
    auto r = Scalar(3.0) * t;
    EXPECT_NEAR(r.data_ptr<float>()[0], 6.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 9.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 12.0f, 1e-5);
}

TEST(Broadcasting, ScalarDivTensor) {
    auto t = tensor({2.0f, 4.0f, 5.0f});
    auto r = Scalar(20.0) / t;
    EXPECT_NEAR(r.data_ptr<float>()[0], 10.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 5.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 4.0f, 1e-5);
}

// =============================================================================
// Special (5 tests)
// =============================================================================

TEST(Special, Where) {
    auto cond = at::empty({3}, TensorOptions().dtype(c10::ScalarType::Bool));
    bool* m = cond.mutable_data_ptr<bool>();
    m[0] = true; m[1] = false; m[2] = true;
    auto x = tensor({1.0f, 2.0f, 3.0f});
    auto y = tensor({10.0f, 20.0f, 30.0f});
    auto r = torch::where(cond, x, y);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 20.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 3.0f, 1e-5);
}

TEST(Special, Meshgrid) {
    auto x = tensor({1.0f, 2.0f, 3.0f});
    auto y = tensor({4.0f, 5.0f});
    auto grids = torch::meshgrid({x, y}, "ij");
    EXPECT_EQ(grids.size(), 2);
    EXPECT_EQ(grids[0].size(0), 3);
    EXPECT_EQ(grids[0].size(1), 2);
    EXPECT_EQ(grids[1].size(0), 3);
    EXPECT_EQ(grids[1].size(1), 2);
}

TEST(Special, Unique) {
    auto t = tensor({3.0f, 1.0f, 2.0f, 1.0f, 3.0f});
    auto [unique_vals, inverse, counts] = torch::unique(t, true, false, false);
    EXPECT_EQ(unique_vals.numel(), 3);
    auto uv = unique_vals.data_ptr<float>();
    EXPECT_NEAR(uv[0], 1.0f, 1e-5);
    EXPECT_NEAR(uv[1], 2.0f, 1e-5);
    EXPECT_NEAR(uv[2], 3.0f, 1e-5);
}

TEST(Special, TriuTril) {
    auto t = tensor({{1.0f, 2.0f, 3.0f},
                     {4.0f, 5.0f, 6.0f},
                     {7.0f, 8.0f, 9.0f}});
    auto upper = t.triu();
    auto lower = t.tril();
    // Upper should have zeros below diagonal
    EXPECT_NEAR(upper.data_ptr<float>()[3], 0.0f, 1e-5);  // [1,0]
    EXPECT_NEAR(upper.data_ptr<float>()[6], 0.0f, 1e-5);  // [2,0]
    EXPECT_NEAR(upper.data_ptr<float>()[7], 0.0f, 1e-5);  // [2,1]
    EXPECT_NEAR(upper.data_ptr<float>()[0], 1.0f, 1e-5);  // [0,0]
    // Lower should have zeros above diagonal
    EXPECT_NEAR(lower.data_ptr<float>()[1], 0.0f, 1e-5);  // [0,1]
    EXPECT_NEAR(lower.data_ptr<float>()[2], 0.0f, 1e-5);  // [0,2]
    EXPECT_NEAR(lower.data_ptr<float>()[5], 0.0f, 1e-5);  // [1,2]
    EXPECT_NEAR(lower.data_ptr<float>()[0], 1.0f, 1e-5);  // [0,0]
}

TEST(Special, Diag) {
    // 2D->1D: extract diagonal
    auto t = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto d = t.diag();
    EXPECT_EQ(d.numel(), 2);
    EXPECT_NEAR(d.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(d.data_ptr<float>()[1], 4.0f, 1e-5);

    // 1D->2D: create diagonal matrix
    auto v = tensor({5.0f, 6.0f, 7.0f});
    auto m = v.diag();
    EXPECT_EQ(m.dim(), 2);
    EXPECT_EQ(m.size(0), 3);
    EXPECT_EQ(m.size(1), 3);
    EXPECT_NEAR(m.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(m.data_ptr<float>()[4], 6.0f, 1e-5);
    EXPECT_NEAR(m.data_ptr<float>()[8], 7.0f, 1e-5);
    EXPECT_NEAR(m.data_ptr<float>()[1], 0.0f, 1e-5);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
