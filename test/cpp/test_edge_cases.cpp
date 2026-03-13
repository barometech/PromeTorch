#include <gtest/gtest.h>
#include "aten/src/ATen/ATen.h"
#include <cmath>

using namespace at;
using namespace torch;

// ============================================================================
// Scalar (0-dim) Tensor Tests
// ============================================================================

TEST(EdgeCases, ScalarTensorCreation) {
    Tensor s = scalar_tensor(Scalar(3.14));
    EXPECT_EQ(s.dim(), 0);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_NEAR(s.item<float>(), 3.14f, 1e-5);
}

TEST(EdgeCases, ScalarArithmetic) {
    Tensor a = scalar_tensor(Scalar(2.0));
    Tensor b = scalar_tensor(Scalar(3.0));
    Tensor c = a + b;
    EXPECT_EQ(c.dim(), 0);
    EXPECT_NEAR(c.item<float>(), 5.0f, 1e-5);
}

TEST(EdgeCases, ScalarReduction) {
    Tensor s = scalar_tensor(Scalar(7.0));
    Tensor r = s.sum();
    EXPECT_EQ(r.dim(), 0);
    EXPECT_NEAR(r.item<float>(), 7.0f, 1e-5);
}

TEST(EdgeCases, ScalarUnary) {
    Tensor s = scalar_tensor(Scalar(4.0));
    Tensor r = at::native::sqrt(s);
    EXPECT_EQ(r.dim(), 0);
    EXPECT_NEAR(r.item<float>(), 2.0f, 1e-5);
}

// ============================================================================
// Single Element Tensor Tests
// ============================================================================

TEST(EdgeCases, SingleElementView) {
    Tensor t = tensor({42.0f});
    Tensor v = t.view({1, 1});
    EXPECT_EQ(v.dim(), 2);
    EXPECT_EQ(v.size(0), 1);
    EXPECT_EQ(v.size(1), 1);
    EXPECT_NEAR(v.data_ptr<float>()[0], 42.0f, 1e-5);
}

TEST(EdgeCases, SingleElementMm) {
    Tensor a = tensor({{2.0f}});
    Tensor b = tensor({{3.0f}});
    Tensor c = at::native::mm(a, b);
    EXPECT_NEAR(c.data_ptr<float>()[0], 6.0f, 1e-5);
}

// ============================================================================
// Non-contiguous Tensor Tests
// ============================================================================

TEST(EdgeCases, TransposeIsNonContiguous) {
    Tensor t = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor tt = t.t();
    // Transpose creates a view, not a copy
    EXPECT_FALSE(tt.is_contiguous());
    // Making contiguous should fix it
    Tensor tc = tt.contiguous();
    EXPECT_TRUE(tc.is_contiguous());
    EXPECT_NEAR(tc.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(tc.data_ptr<float>()[1], 3.0f, 1e-5);
    EXPECT_NEAR(tc.data_ptr<float>()[2], 2.0f, 1e-5);
    EXPECT_NEAR(tc.data_ptr<float>()[3], 4.0f, 1e-5);
}

TEST(EdgeCases, NonContiguousMm) {
    // mm on transposed tensors should work correctly
    Tensor a = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor b = tensor({{5.0f, 6.0f}, {7.0f, 8.0f}});
    // a.t() is non-contiguous
    Tensor c = at::native::mm(a.t(), b);
    EXPECT_EQ(c.size(0), 2);
    EXPECT_EQ(c.size(1), 2);
    // a.t() = [[1,3],[2,4]], b = [[5,6],[7,8]]
    // c[0,0] = 1*5 + 3*7 = 26
    EXPECT_NEAR(c.data_ptr<float>()[0], 26.0f, 1e-5);
}

TEST(EdgeCases, NonContiguousUnary) {
    Tensor t = tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor s = t.select(1, 0); // column 0: {1, 3} -- non-contiguous
    Tensor r = at::native::exp(s);
    EXPECT_EQ(r.numel(), 2);
    EXPECT_NEAR(r.data_ptr<float>()[0], std::exp(1.0f), 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], std::exp(3.0f), 1e-5);
}

TEST(EdgeCases, NarrowIsView) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor n = t.narrow(0, 1, 3); // {2, 3, 4}
    EXPECT_EQ(n.numel(), 3);
    EXPECT_NEAR(n.data_ptr<float>()[0], 2.0f, 1e-5);
    // Shares storage
    EXPECT_EQ(n.storage_offset(), 1);
}

// ============================================================================
// Broadcasting Edge Cases
// ============================================================================

TEST(EdgeCases, BroadcastScalarToTensor) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f});
    Tensor s = scalar_tensor(Scalar(10.0));
    Tensor r = t + s;
    EXPECT_EQ(r.numel(), 3);
    EXPECT_NEAR(r.data_ptr<float>()[0], 11.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 13.0f, 1e-5);
}

TEST(EdgeCases, BroadcastDifferentDims) {
    Tensor a = tensor({{1.0f}, {2.0f}, {3.0f}}); // [3,1]
    Tensor b = tensor({10.0f, 20.0f});            // [2]
    Tensor c = a + b;                              // [3,2]
    EXPECT_EQ(c.size(0), 3);
    EXPECT_EQ(c.size(1), 2);
    EXPECT_NEAR(c.data_ptr<float>()[0], 11.0f, 1e-5); // 1+10
    EXPECT_NEAR(c.data_ptr<float>()[1], 21.0f, 1e-5); // 1+20
}

// ============================================================================
// Dtype Promotion Tests
// ============================================================================

TEST(EdgeCases, FloatPlusDouble) {
    Tensor a = tensor({1.0f, 2.0f}); // float
    Tensor b = at::tensor({3.0, 4.0}, TensorOptions().dtype(c10::ScalarType::Double));
    // Result should be Double (higher precision)
    Tensor c = a.to(c10::ScalarType::Double) + b;
    EXPECT_EQ(c.dtype(), c10::ScalarType::Double);
    EXPECT_NEAR(c.data_ptr<double>()[0], 4.0, 1e-10);
}

TEST(EdgeCases, IntToFloat) {
    Tensor a = at::tensor({1, 2, 3}, TensorOptions().dtype(c10::ScalarType::Long));
    Tensor b = a.to(c10::ScalarType::Float);
    EXPECT_EQ(b.dtype(), c10::ScalarType::Float);
    EXPECT_NEAR(b.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(b.data_ptr<float>()[2], 3.0f, 1e-5);
}

// ============================================================================
// Large Tensor Stress Tests
// ============================================================================

TEST(EdgeCases, LargeTensorSum) {
    Tensor t = ones({1000, 1000});
    Tensor s = t.sum();
    EXPECT_NEAR(s.item<float>(), 1000000.0f, 1.0f);
}

TEST(EdgeCases, LargeTensorMean) {
    Tensor t = full({500, 500}, Scalar(3.0));
    Tensor m = t.mean();
    EXPECT_NEAR(m.item<float>(), 3.0f, 1e-4);
}

// ============================================================================
// Copy and Clone Edge Cases
// ============================================================================

TEST(EdgeCases, CloneIndependence) {
    Tensor a = tensor({1.0f, 2.0f, 3.0f});
    Tensor b = a.clone();
    b.mutable_data_ptr<float>()[0] = 99.0f;
    // a should be unchanged
    EXPECT_NEAR(a.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(b.data_ptr<float>()[0], 99.0f, 1e-5);
}

TEST(EdgeCases, DetachStopsGrad) {
    Tensor a = tensor({1.0f, 2.0f}).set_requires_grad(true);
    Tensor b = a.detach();
    EXPECT_FALSE(b.requires_grad());
}

TEST(EdgeCases, FillAndZero) {
    Tensor t = empty({5});
    t.fill_(Scalar(7.0));
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(t.data_ptr<float>()[i], 7.0f, 1e-5);
    }
    t.zero_();
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(t.data_ptr<float>()[i], 0.0f, 1e-5);
    }
}

// ============================================================================
// Repeat and Expand Edge Cases
// ============================================================================

TEST(EdgeCases, ExpandDoesNotCopy) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f}).unsqueeze(0); // [1, 3]
    Tensor e = t.expand({4, 3}); // [4, 3]
    EXPECT_EQ(e.size(0), 4);
    EXPECT_EQ(e.size(1), 3);
    // All rows should be the same
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(e.data_ptr<float>()[0], 1.0f, 1e-5); // stride[0]=0 for expanded dim
    }
}

TEST(EdgeCases, RepeatCopies) {
    Tensor t = tensor({1.0f, 2.0f}); // [2]
    Tensor r = t.repeat({3}); // [6]
    EXPECT_EQ(r.numel(), 6);
    EXPECT_NEAR(r.data_ptr<float>()[0], 1.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(r.data_ptr<float>()[2], 1.0f, 1e-5);
}

// ============================================================================
// Slice Step Edge Cases
// ============================================================================

TEST(EdgeCases, SliceWithStep) {
    Tensor t = at::arange(Scalar(0), Scalar(10), Scalar(1)); // {0,1,...,9}
    Tensor s = t.slice(0, 0, 10, 2); // {0, 2, 4, 6, 8} — non-contiguous view!
    EXPECT_EQ(s.numel(), 5);
    EXPECT_EQ(s.stride(0), 2); // step=2 means stride=2
    Tensor sc = s.contiguous(); // make contiguous to read sequentially
    EXPECT_NEAR(sc.data_ptr<float>()[0], 0.0f, 1e-5);
    EXPECT_NEAR(sc.data_ptr<float>()[1], 2.0f, 1e-5);
    EXPECT_NEAR(sc.data_ptr<float>()[4], 8.0f, 1e-5);
}

// ============================================================================
// Channels-Last Memory Format
// ============================================================================

TEST(EdgeCases, ChannelsLastContiguous) {
    Tensor t = randn({1, 3, 4, 4}); // NCHW
    Tensor cl = t.contiguous(c10::MemoryFormat::ChannelsLast);
    EXPECT_TRUE(cl.is_contiguous(c10::MemoryFormat::ChannelsLast));
    EXPECT_EQ(cl.sizes(), t.sizes());
}

// ============================================================================
// Eye and Arange Edge Cases
// ============================================================================

TEST(EdgeCases, EyeRectangular) {
    Tensor e = eye(3, 5);
    EXPECT_EQ(e.size(0), 3);
    EXPECT_EQ(e.size(1), 5);
    EXPECT_NEAR(e.data_ptr<float>()[0], 1.0f, 1e-5); // [0,0]
    EXPECT_NEAR(e.data_ptr<float>()[1], 0.0f, 1e-5); // [0,1]
}

TEST(EdgeCases, ArangeNegativeStep) {
    Tensor t = at::arange(Scalar(5.0), Scalar(0.0), Scalar(-1.0));
    EXPECT_EQ(t.numel(), 5);
    EXPECT_NEAR(t.data_ptr<float>()[0], 5.0f, 1e-5);
    EXPECT_NEAR(t.data_ptr<float>()[4], 1.0f, 1e-5);
}
