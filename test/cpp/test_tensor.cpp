#include <gtest/gtest.h>
#include "aten/src/ATen/ATen.h"
#include <cmath>

using namespace at;
using namespace torch;

// ============================================================================
// Tensor Creation Tests
// ============================================================================

TEST(TensorCreationTest, Empty) {
    Tensor t = empty({2, 3});
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorCreationTest, Zeros) {
    Tensor t = zeros({3, 4});
    EXPECT_EQ(t.numel(), 12);

    float* data = t.data_ptr<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST(TensorCreationTest, Ones) {
    Tensor t = ones({2, 2});
    EXPECT_EQ(t.numel(), 4);

    float* data = t.data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(data[i], 1.0f);
    }
}

TEST(TensorCreationTest, Full) {
    Tensor t = full({2, 3}, Scalar(3.14));
    EXPECT_EQ(t.numel(), 6);

    float* data = t.data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], 3.14f, 1e-5f);
    }
}

TEST(TensorCreationTest, Arange) {
    Tensor t = arange(0, 10, 2);
    EXPECT_EQ(t.dim(), 1);
    EXPECT_EQ(t.size(0), 5);

    float* data = t.data_ptr<float>();
    EXPECT_EQ(data[0], 0.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], 4.0f);
    EXPECT_EQ(data[3], 6.0f);
    EXPECT_EQ(data[4], 8.0f);
}

TEST(TensorCreationTest, Linspace) {
    Tensor t = linspace(0, 1, 5);
    EXPECT_EQ(t.size(0), 5);

    float* data = t.data_ptr<float>();
    EXPECT_NEAR(data[0], 0.0f, 1e-5f);
    EXPECT_NEAR(data[1], 0.25f, 1e-5f);
    EXPECT_NEAR(data[2], 0.5f, 1e-5f);
    EXPECT_NEAR(data[3], 0.75f, 1e-5f);
    EXPECT_NEAR(data[4], 1.0f, 1e-5f);
}

TEST(TensorCreationTest, Eye) {
    Tensor t = eye(3);
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.size(0), 3);
    EXPECT_EQ(t.size(1), 3);

    float* data = t.data_ptr<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_EQ(data[i * 3 + j], expected);
        }
    }
}

TEST(TensorCreationTest, RandShape) {
    Tensor t = rand({10, 20});
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.size(0), 10);
    EXPECT_EQ(t.size(1), 20);

    // Check values are in [0, 1)
    float* data = t.data_ptr<float>();
    for (int i = 0; i < 200; ++i) {
        EXPECT_GE(data[i], 0.0f);
        EXPECT_LT(data[i], 1.0f);
    }
}

TEST(TensorCreationTest, TensorFromList) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_EQ(t.dim(), 1);
    EXPECT_EQ(t.size(0), 4);

    float* data = t.data_ptr<float>();
    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], 3.0f);
    EXPECT_EQ(data[3], 4.0f);
}

// ============================================================================
// Tensor Options Tests
// ============================================================================

TEST(TensorOptionsTest, Dtype) {
    Tensor t = zeros({2, 2}, dtype(c10::ScalarType::Double));
    EXPECT_EQ(t.dtype(), c10::ScalarType::Double);
    EXPECT_EQ(t.itemsize(), sizeof(double));
}

TEST(TensorOptionsTest, RequiresGrad) {
    Tensor t = zeros({2, 2}, requires_grad(true));
    EXPECT_TRUE(t.requires_grad());
}

// ============================================================================
// Unary Operations Tests
// ============================================================================

TEST(UnaryOpsTest, Neg) {
    Tensor t = tensor({1.0f, -2.0f, 3.0f});
    Tensor result = t.neg();

    float* data = result.data_ptr<float>();
    EXPECT_EQ(data[0], -1.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], -3.0f);
}

TEST(UnaryOpsTest, Abs) {
    Tensor t = tensor({-1.0f, 2.0f, -3.0f});
    Tensor result = t.abs();

    float* data = result.data_ptr<float>();
    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], 3.0f);
}

TEST(UnaryOpsTest, Sqrt) {
    Tensor t = tensor({1.0f, 4.0f, 9.0f});
    Tensor result = t.sqrt();

    float* data = result.data_ptr<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5f);
    EXPECT_NEAR(data[1], 2.0f, 1e-5f);
    EXPECT_NEAR(data[2], 3.0f, 1e-5f);
}

TEST(UnaryOpsTest, Exp) {
    Tensor t = tensor({0.0f, 1.0f, 2.0f});
    Tensor result = t.exp();

    float* data = result.data_ptr<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5f);
    EXPECT_NEAR(data[1], std::exp(1.0f), 1e-5f);
    EXPECT_NEAR(data[2], std::exp(2.0f), 1e-5f);
}

TEST(UnaryOpsTest, Log) {
    Tensor t = tensor({1.0f, std::exp(1.0f), std::exp(2.0f)});
    Tensor result = t.log();

    float* data = result.data_ptr<float>();
    EXPECT_NEAR(data[0], 0.0f, 1e-5f);
    EXPECT_NEAR(data[1], 1.0f, 1e-5f);
    EXPECT_NEAR(data[2], 2.0f, 1e-5f);
}

TEST(UnaryOpsTest, Relu) {
    Tensor t = tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    Tensor result = t.relu();

    float* data = result.data_ptr<float>();
    EXPECT_EQ(data[0], 0.0f);
    EXPECT_EQ(data[1], 0.0f);
    EXPECT_EQ(data[2], 0.0f);
    EXPECT_EQ(data[3], 1.0f);
    EXPECT_EQ(data[4], 2.0f);
}

TEST(UnaryOpsTest, Sigmoid) {
    Tensor t = tensor({0.0f});
    Tensor result = t.sigmoid();

    EXPECT_NEAR(result.data_ptr<float>()[0], 0.5f, 1e-5f);
}

TEST(UnaryOpsTest, Tanh) {
    Tensor t = tensor({0.0f});
    Tensor result = t.tanh();

    EXPECT_NEAR(result.data_ptr<float>()[0], 0.0f, 1e-5f);
}

// ============================================================================
// Binary Operations Tests
// ============================================================================

TEST(BinaryOpsTest, Add) {
    Tensor a = tensor({1.0f, 2.0f, 3.0f});
    Tensor b = tensor({4.0f, 5.0f, 6.0f});
    Tensor result = a.add(b);

    float* data = result.data_ptr<float>();
    EXPECT_EQ(data[0], 5.0f);
    EXPECT_EQ(data[1], 7.0f);
    EXPECT_EQ(data[2], 9.0f);
}

TEST(BinaryOpsTest, AddScalar) {
    Tensor a = tensor({1.0f, 2.0f, 3.0f});
    Tensor result = a.add(Scalar(10.0));

    float* data = result.data_ptr<float>();
    EXPECT_EQ(data[0], 11.0f);
    EXPECT_EQ(data[1], 12.0f);
    EXPECT_EQ(data[2], 13.0f);
}

TEST(BinaryOpsTest, Sub) {
    Tensor a = tensor({10.0f, 20.0f, 30.0f});
    Tensor b = tensor({1.0f, 2.0f, 3.0f});
    Tensor result = a.sub(b);

    float* data = result.data_ptr<float>();
    EXPECT_EQ(data[0], 9.0f);
    EXPECT_EQ(data[1], 18.0f);
    EXPECT_EQ(data[2], 27.0f);
}

TEST(BinaryOpsTest, Mul) {
    Tensor a = tensor({2.0f, 3.0f, 4.0f});
    Tensor b = tensor({5.0f, 6.0f, 7.0f});
    Tensor result = a.mul(b);

    float* data = result.data_ptr<float>();
    EXPECT_EQ(data[0], 10.0f);
    EXPECT_EQ(data[1], 18.0f);
    EXPECT_EQ(data[2], 28.0f);
}

TEST(BinaryOpsTest, Div) {
    Tensor a = tensor({10.0f, 20.0f, 30.0f});
    Tensor b = tensor({2.0f, 4.0f, 5.0f});
    Tensor result = a.div(b);

    float* data = result.data_ptr<float>();
    EXPECT_EQ(data[0], 5.0f);
    EXPECT_EQ(data[1], 5.0f);
    EXPECT_EQ(data[2], 6.0f);
}

TEST(BinaryOpsTest, Operators) {
    Tensor a = tensor({1.0f, 2.0f, 3.0f});
    Tensor b = tensor({4.0f, 5.0f, 6.0f});

    Tensor sum = a + b;
    Tensor diff = b - a;
    Tensor prod = a * b;
    Tensor quot = b / a;

    EXPECT_EQ(sum.data_ptr<float>()[0], 5.0f);
    EXPECT_EQ(diff.data_ptr<float>()[0], 3.0f);
    EXPECT_EQ(prod.data_ptr<float>()[0], 4.0f);
    EXPECT_NEAR(quot.data_ptr<float>()[0], 4.0f, 1e-5f);
}

TEST(BinaryOpsTest, InPlace) {
    Tensor a = tensor({1.0f, 2.0f, 3.0f});
    Tensor b = tensor({4.0f, 5.0f, 6.0f});

    a += b;
    EXPECT_EQ(a.data_ptr<float>()[0], 5.0f);
    EXPECT_EQ(a.data_ptr<float>()[1], 7.0f);
    EXPECT_EQ(a.data_ptr<float>()[2], 9.0f);
}

// ============================================================================
// Reduction Operations Tests
// ============================================================================

TEST(ReductionOpsTest, Sum) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f, 4.0f});
    Tensor result = t.sum();

    EXPECT_EQ(result.numel(), 1);
    EXPECT_EQ(result.data_ptr<float>()[0], 10.0f);
}

TEST(ReductionOpsTest, Mean) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f, 4.0f});
    Tensor result = t.mean();

    EXPECT_NEAR(result.data_ptr<float>()[0], 2.5f, 1e-5f);
}

TEST(ReductionOpsTest, Max) {
    Tensor t = tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f});
    Tensor result = t.max();

    EXPECT_EQ(result.data_ptr<float>()[0], 5.0f);
}

TEST(ReductionOpsTest, Min) {
    Tensor t = tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f});
    Tensor result = t.min();

    EXPECT_EQ(result.data_ptr<float>()[0], 1.0f);
}

TEST(ReductionOpsTest, Argmax) {
    Tensor t = tensor({3.0f, 1.0f, 4.0f, 1.0f, 5.0f});
    Tensor result = t.argmax();

    EXPECT_EQ(result.data_ptr<int64_t>()[0], 4);  // Index of 5.0
}

TEST(ReductionOpsTest, Var) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor result = t.var();

    // Unbiased variance of [1,2,3,4,5] = 2.5
    EXPECT_NEAR(result.data_ptr<float>()[0], 2.5f, 1e-5f);
}

TEST(ReductionOpsTest, Std) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor result = t.std();

    EXPECT_NEAR(result.data_ptr<float>()[0], std::sqrt(2.5f), 1e-5f);
}

TEST(ReductionOpsTest, Norm) {
    Tensor t = tensor({3.0f, 4.0f});
    Tensor result = t.norm(Scalar(2));

    EXPECT_NEAR(result.data_ptr<float>()[0], 5.0f, 1e-5f);  // sqrt(9+16) = 5
}

// ============================================================================
// Shape Operations Tests
// ============================================================================

TEST(ShapeOpsTest, View) {
    Tensor t = arange(0, 12);
    Tensor viewed = t.view({3, 4});

    EXPECT_EQ(viewed.dim(), 2);
    EXPECT_EQ(viewed.size(0), 3);
    EXPECT_EQ(viewed.size(1), 4);
    EXPECT_EQ(viewed.numel(), 12);

    // Data should be same (view shares storage)
    EXPECT_EQ(viewed.data_ptr<float>()[0], 0.0f);
    EXPECT_EQ(viewed.data_ptr<float>()[4], 4.0f);
}

TEST(ShapeOpsTest, ViewInfer) {
    Tensor t = arange(0, 12);
    Tensor viewed = t.view({3, -1});

    EXPECT_EQ(viewed.size(0), 3);
    EXPECT_EQ(viewed.size(1), 4);
}

TEST(ShapeOpsTest, Reshape) {
    Tensor t = arange(0, 6);
    Tensor reshaped = t.reshape({2, 3});

    EXPECT_EQ(reshaped.dim(), 2);
    EXPECT_EQ(reshaped.size(0), 2);
    EXPECT_EQ(reshaped.size(1), 3);
}

TEST(ShapeOpsTest, Squeeze) {
    Tensor t = ones({1, 3, 1, 4, 1});
    Tensor squeezed = t.squeeze();

    EXPECT_EQ(squeezed.dim(), 2);
    EXPECT_EQ(squeezed.size(0), 3);
    EXPECT_EQ(squeezed.size(1), 4);
}

TEST(ShapeOpsTest, SqueezeDim) {
    Tensor t = ones({1, 3, 1, 4});
    Tensor squeezed = t.squeeze(0);

    EXPECT_EQ(squeezed.dim(), 3);
    EXPECT_EQ(squeezed.size(0), 3);
}

TEST(ShapeOpsTest, Unsqueeze) {
    Tensor t = ones({3, 4});
    Tensor unsqueezed = t.unsqueeze(0);

    EXPECT_EQ(unsqueezed.dim(), 3);
    EXPECT_EQ(unsqueezed.size(0), 1);
    EXPECT_EQ(unsqueezed.size(1), 3);
    EXPECT_EQ(unsqueezed.size(2), 4);
}

TEST(ShapeOpsTest, Transpose) {
    Tensor t = arange(0, 6).view({2, 3});
    Tensor transposed = t.transpose(0, 1);

    EXPECT_EQ(transposed.size(0), 3);
    EXPECT_EQ(transposed.size(1), 2);
}

TEST(ShapeOpsTest, T) {
    Tensor t = arange(0, 6).view({2, 3});
    Tensor transposed = t.t();

    EXPECT_EQ(transposed.size(0), 3);
    EXPECT_EQ(transposed.size(1), 2);
}

TEST(ShapeOpsTest, Flatten) {
    Tensor t = ones({2, 3, 4});
    Tensor flattened = t.flatten();

    EXPECT_EQ(flattened.dim(), 1);
    EXPECT_EQ(flattened.size(0), 24);
}

// ============================================================================
// Linear Algebra Tests
// ============================================================================

TEST(LinearAlgebraTest, Dot) {
    Tensor a = tensor({1.0f, 2.0f, 3.0f});
    Tensor b = tensor({4.0f, 5.0f, 6.0f});
    Tensor result = a.dot(b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_EQ(result.data_ptr<float>()[0], 32.0f);
}

TEST(LinearAlgebraTest, Mm) {
    // [2, 3] @ [3, 4] = [2, 4]
    Tensor a = ones({2, 3});
    Tensor b = ones({3, 4});
    Tensor result = a.mm(b);

    EXPECT_EQ(result.dim(), 2);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 4);

    // Each element should be 3 (sum of 3 ones)
    float* data = result.data_ptr<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(data[i], 3.0f);
    }
}

TEST(LinearAlgebraTest, Mv) {
    // [2, 3] @ [3] = [2]
    Tensor a = ones({2, 3});
    Tensor b = tensor({1.0f, 2.0f, 3.0f});
    Tensor result = a.mv(b);

    EXPECT_EQ(result.dim(), 1);
    EXPECT_EQ(result.size(0), 2);

    // Each element should be 1+2+3 = 6
    EXPECT_EQ(result.data_ptr<float>()[0], 6.0f);
    EXPECT_EQ(result.data_ptr<float>()[1], 6.0f);
}

TEST(LinearAlgebraTest, Matmul) {
    Tensor a = ones({2, 3});
    Tensor b = ones({3, 4});
    Tensor result = a.matmul(b);

    EXPECT_EQ(result.dim(), 2);
    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 4);
}

TEST(LinearAlgebraTest, Bmm) {
    // Batch matrix multiplication
    Tensor a = ones({4, 2, 3});  // 4 batches of [2, 3]
    Tensor b = ones({4, 3, 5});  // 4 batches of [3, 5]
    Tensor result = a.bmm(b);

    EXPECT_EQ(result.dim(), 3);
    EXPECT_EQ(result.size(0), 4);
    EXPECT_EQ(result.size(1), 2);
    EXPECT_EQ(result.size(2), 5);
}

// ============================================================================
// Indexing Tests
// ============================================================================

TEST(IndexingTest, Select) {
    Tensor t = arange(0, 12).view({3, 4});
    Tensor selected = t.select(0, 1);  // Second row

    EXPECT_EQ(selected.dim(), 1);
    EXPECT_EQ(selected.size(0), 4);
    EXPECT_EQ(selected.data_ptr<float>()[0], 4.0f);
}

TEST(IndexingTest, Narrow) {
    Tensor t = arange(0, 10);
    Tensor narrowed = t.narrow(0, 2, 5);  // Elements 2-6

    EXPECT_EQ(narrowed.size(0), 5);
    EXPECT_EQ(narrowed.data_ptr<float>()[0], 2.0f);
    EXPECT_EQ(narrowed.data_ptr<float>()[4], 6.0f);
}

TEST(IndexingTest, IndexOperator) {
    Tensor t = arange(0, 12).view({3, 4});
    Tensor row = t[1];  // Second row

    EXPECT_EQ(row.dim(), 1);
    EXPECT_EQ(row.size(0), 4);
}

// ============================================================================
// Clone and Copy Tests
// ============================================================================

TEST(CopyTest, Clone) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f});
    Tensor cloned = t.clone();

    // Should have same data
    EXPECT_EQ(cloned.data_ptr<float>()[0], 1.0f);

    // But different storage
    EXPECT_NE(cloned.data_ptr<float>(), t.data_ptr<float>());
}

TEST(CopyTest, Detach) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f});
    t.set_requires_grad(true);

    Tensor detached = t.detach();

    EXPECT_FALSE(detached.requires_grad());
}

// ============================================================================
// Type Conversion Tests
// ============================================================================

TEST(TypeConversionTest, ToDouble) {
    Tensor t = tensor({1.0f, 2.0f, 3.0f});
    Tensor converted = t.to(c10::ScalarType::Double);

    EXPECT_EQ(converted.dtype(), c10::ScalarType::Double);
    EXPECT_NEAR(converted.data_ptr<double>()[0], 1.0, 1e-10);
}

TEST(TypeConversionTest, ToInt) {
    Tensor t = tensor({1.5f, 2.7f, 3.9f});
    Tensor converted = t.to(c10::ScalarType::Int);

    EXPECT_EQ(converted.dtype(), c10::ScalarType::Int);
    EXPECT_EQ(converted.data_ptr<int32_t>()[0], 1);
    EXPECT_EQ(converted.data_ptr<int32_t>()[1], 2);
    EXPECT_EQ(converted.data_ptr<int32_t>()[2], 3);
}
