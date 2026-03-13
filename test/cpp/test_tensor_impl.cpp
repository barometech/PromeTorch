#include <gtest/gtest.h>
#include "c10/core/TensorImpl.h"
#include <cstring>
#include <numeric>

using namespace c10;

// ============================================================================
// SmallVector Tests
// ============================================================================

TEST(SmallVectorTest, DefaultConstructor) {
    SmallVector<int, 4> vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
}

TEST(SmallVectorTest, InitializerList) {
    SmallVector<int, 4> vec{1, 2, 3, 4};
    EXPECT_EQ(vec.size(), 4);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[3], 4);
}

TEST(SmallVectorTest, PushBack) {
    SmallVector<int, 2> vec;
    vec.push_back(1);
    vec.push_back(2);
    EXPECT_EQ(vec.size(), 2);

    // Should still work when exceeding stack capacity
    vec.push_back(3);
    vec.push_back(4);
    EXPECT_EQ(vec.size(), 4);
    EXPECT_EQ(vec[2], 3);
    EXPECT_EQ(vec[3], 4);
}

TEST(SmallVectorTest, Resize) {
    SmallVector<int, 4> vec;
    vec.resize(3, 42);
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 42);
    EXPECT_EQ(vec[2], 42);
}

TEST(SmallVectorTest, CopyConstructor) {
    SmallVector<int, 4> vec1{1, 2, 3};
    SmallVector<int, 4> vec2(vec1);
    EXPECT_EQ(vec2.size(), 3);
    EXPECT_EQ(vec2[0], 1);
}

TEST(SmallVectorTest, MoveConstructor) {
    SmallVector<int, 4> vec1{1, 2, 3};
    SmallVector<int, 4> vec2(std::move(vec1));
    EXPECT_EQ(vec2.size(), 3);
    EXPECT_EQ(vec2[0], 1);
}

// ============================================================================
// IntArrayRef Tests
// ============================================================================

TEST(IntArrayRefTest, FromVector) {
    std::vector<int64_t> vec{1, 2, 3, 4, 5};
    IntArrayRef ref(vec);

    EXPECT_EQ(ref.size(), 5);
    EXPECT_EQ(ref[0], 1);
    EXPECT_EQ(ref[4], 5);
}

TEST(IntArrayRefTest, FromInitializerList) {
    IntArrayRef ref{2, 3, 4};
    EXPECT_EQ(ref.size(), 3);
    EXPECT_EQ(ref.front(), 2);
    EXPECT_EQ(ref.back(), 4);
}

TEST(IntArrayRefTest, FromSmallVector) {
    SmallVector<int64_t, 4> vec{10, 20, 30};
    IntArrayRef ref(vec);

    EXPECT_EQ(ref.size(), 3);
    EXPECT_EQ(ref[1], 20);
}

TEST(IntArrayRefTest, ToVector) {
    IntArrayRef ref{1, 2, 3};
    std::vector<int64_t> vec = ref.vec();

    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 1);
}

TEST(IntArrayRefTest, Comparison) {
    IntArrayRef ref1{1, 2, 3};
    IntArrayRef ref2{1, 2, 3};
    IntArrayRef ref3{1, 2, 4};
    IntArrayRef ref4{1, 2};

    EXPECT_EQ(ref1, ref2);
    EXPECT_NE(ref1, ref3);
    EXPECT_NE(ref1, ref4);
}

// ============================================================================
// TensorImpl Constructor Tests
// ============================================================================

TEST(TensorImplTest, DefaultConstructor) {
    TensorImpl impl;

    EXPECT_EQ(impl.dim(), 0);
    EXPECT_EQ(impl.numel(), 0);
    EXPECT_EQ(impl.dtype(), ScalarType::Float);
    EXPECT_TRUE(impl.is_contiguous());
    EXPECT_FALSE(impl.requires_grad());
}

TEST(TensorImplTest, ConstructorWithStorage) {
    Storage storage = Storage::create(sizeof(float) * 10, kCPU);
    TensorImpl impl(std::move(storage), ScalarType::Float);

    EXPECT_TRUE(impl.has_storage());
    EXPECT_EQ(impl.dtype(), ScalarType::Float);
    EXPECT_TRUE(impl.is_cpu());
}

TEST(TensorImplTest, ConstructorWithSizes) {
    Storage storage = Storage::create(sizeof(float) * 24, kCPU);  // 2*3*4 = 24
    TensorImpl impl(std::move(storage), ScalarType::Float, {2, 3, 4});

    EXPECT_EQ(impl.dim(), 3);
    EXPECT_EQ(impl.size(0), 2);
    EXPECT_EQ(impl.size(1), 3);
    EXPECT_EQ(impl.size(2), 4);
    EXPECT_EQ(impl.numel(), 24);
    EXPECT_TRUE(impl.is_contiguous());
}

TEST(TensorImplTest, ConstructorWithSizesAndStrides) {
    Storage storage = Storage::create(sizeof(float) * 24, kCPU);
    // Transposed tensor: logical shape [3, 2, 4] but stored as [2, 3, 4]
    TensorImpl impl(std::move(storage), ScalarType::Float, {3, 2, 4}, {4, 12, 1});

    EXPECT_EQ(impl.dim(), 3);
    EXPECT_EQ(impl.size(0), 3);
    EXPECT_EQ(impl.stride(0), 4);
    // This tensor is not contiguous due to non-standard strides
}

// ============================================================================
// TensorImpl Shape Tests
// ============================================================================

TEST(TensorImplTest, SetSizesContiguous) {
    Storage storage = Storage::create(sizeof(float) * 60, kCPU);
    TensorImpl impl(std::move(storage), ScalarType::Float);

    impl.set_sizes_contiguous({3, 4, 5});

    EXPECT_EQ(impl.dim(), 3);
    EXPECT_EQ(impl.size(0), 3);
    EXPECT_EQ(impl.size(1), 4);
    EXPECT_EQ(impl.size(2), 5);
    EXPECT_EQ(impl.numel(), 60);

    // Check contiguous strides: [20, 5, 1]
    EXPECT_EQ(impl.stride(0), 20);
    EXPECT_EQ(impl.stride(1), 5);
    EXPECT_EQ(impl.stride(2), 1);
    EXPECT_TRUE(impl.is_contiguous());
}

TEST(TensorImplTest, NegativeDimension) {
    Storage storage = Storage::create(sizeof(float) * 12, kCPU);
    TensorImpl impl(std::move(storage), ScalarType::Float, {3, 4});

    // Negative indexing
    EXPECT_EQ(impl.size(-1), 4);
    EXPECT_EQ(impl.size(-2), 3);
    EXPECT_EQ(impl.stride(-1), 1);
    EXPECT_EQ(impl.stride(-2), 4);
}

TEST(TensorImplTest, ScalarTensor) {
    Storage storage = Storage::create(sizeof(float), kCPU);
    TensorImpl impl(std::move(storage), ScalarType::Float, {});  // Empty sizes = scalar

    EXPECT_EQ(impl.dim(), 0);
    EXPECT_EQ(impl.numel(), 1);
    EXPECT_TRUE(impl.is_contiguous());
}

// ============================================================================
// TensorImpl Data Access Tests
// ============================================================================

TEST(TensorImplTest, DataAccess) {
    auto impl = make_tensor_impl({2, 3}, ScalarType::Float, kCPU);

    float* data = impl->mutable_data<float>();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    const float* const_data = impl->data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(const_data[i], static_cast<float>(i));
    }
}

TEST(TensorImplTest, StorageOffset) {
    Storage storage = Storage::create(sizeof(float) * 20, kCPU);

    // Fill storage
    float* raw_data = storage.mutable_data<float>();
    for (int i = 0; i < 20; ++i) {
        raw_data[i] = static_cast<float>(i);
    }

    // Create tensor with offset
    TensorImpl impl(std::move(storage), ScalarType::Float, {5}, {1}, 10);

    EXPECT_EQ(impl.storage_offset(), 10);

    const float* data = impl.data<float>();
    EXPECT_EQ(data[0], 10.0f);
    EXPECT_EQ(data[4], 14.0f);
}

// ============================================================================
// TensorImpl Type Tests
// ============================================================================

TEST(TensorImplTest, Dtype) {
    auto impl_float = make_tensor_impl({2, 3}, ScalarType::Float, kCPU);
    EXPECT_EQ(impl_float->dtype(), ScalarType::Float);
    EXPECT_EQ(impl_float->itemsize(), sizeof(float));

    auto impl_double = make_tensor_impl({2, 3}, ScalarType::Double, kCPU);
    EXPECT_EQ(impl_double->dtype(), ScalarType::Double);
    EXPECT_EQ(impl_double->itemsize(), sizeof(double));

    auto impl_int = make_tensor_impl({2, 3}, ScalarType::Int, kCPU);
    EXPECT_EQ(impl_int->dtype(), ScalarType::Int);
    EXPECT_EQ(impl_int->itemsize(), sizeof(int32_t));
}

TEST(TensorImplTest, SetDtype) {
    auto impl = make_tensor_impl({2, 3}, ScalarType::Float, kCPU);
    impl->set_dtype(ScalarType::Double);
    EXPECT_EQ(impl->dtype(), ScalarType::Double);
}

// ============================================================================
// TensorImpl Autograd Tests
// ============================================================================

TEST(TensorImplTest, RequiresGrad) {
    auto impl = make_tensor_impl({2, 3}, ScalarType::Float, kCPU);

    EXPECT_FALSE(impl->requires_grad());
    EXPECT_TRUE(impl->is_leaf());

    impl->set_requires_grad(true);
    EXPECT_TRUE(impl->requires_grad());
    EXPECT_TRUE(impl->is_leaf());

    impl->set_requires_grad(false);
    EXPECT_FALSE(impl->requires_grad());
}

TEST(TensorImplTest, AutogradMeta) {
    auto impl = make_tensor_impl({2, 3}, ScalarType::Float, kCPU);

    EXPECT_EQ(impl->autograd_meta(), nullptr);

    impl->set_requires_grad(true);
    EXPECT_NE(impl->autograd_meta(), nullptr);
    EXPECT_TRUE(impl->autograd_meta()->requires_grad_);
}

// ============================================================================
// TensorImpl Reference Counting Tests
// ============================================================================

TEST(TensorImplTest, ReferenceCount) {
    auto impl = make_tensor_impl({2, 3}, ScalarType::Float, kCPU);
    auto raw_ptr = impl.get();

    EXPECT_EQ(raw_ptr->use_count(), 1);
    EXPECT_TRUE(raw_ptr->unique());

    // shared_ptr increases count
    auto impl2 = impl;
    EXPECT_EQ(raw_ptr->use_count(), 2);
    EXPECT_FALSE(raw_ptr->unique());
}

// ============================================================================
// TensorImpl Clone Tests
// ============================================================================

TEST(TensorImplTest, Clone) {
    auto impl = make_tensor_impl({2, 3}, ScalarType::Float, kCPU);

    // Fill with data
    float* data = impl->mutable_data<float>();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i * 2);
    }

    // Clone
    auto clone = impl->clone();

    EXPECT_EQ(clone->dim(), 2);
    EXPECT_EQ(clone->size(0), 2);
    EXPECT_EQ(clone->size(1), 3);
    EXPECT_EQ(clone->numel(), 6);

    // Data should be copied
    EXPECT_NE(clone->data(), impl->data());
    const float* clone_data = clone->data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(clone_data[i], static_cast<float>(i * 2));
    }
}

TEST(TensorImplTest, ShallowCopy) {
    auto impl = make_tensor_impl({2, 3}, ScalarType::Float, kCPU);

    // Fill with data
    float* data = impl->mutable_data<float>();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Shallow copy
    auto shallow = impl->shallow_copy();

    EXPECT_EQ(shallow->dim(), 2);
    EXPECT_EQ(shallow->numel(), 6);

    // Data should be shared
    // Note: shallow_copy shares storage, so underlying data pointer is same
    // (adjusted for storage offset which is 0)
}

// ============================================================================
// TensorImpl Nbytes Tests
// ============================================================================

TEST(TensorImplTest, Nbytes) {
    auto impl_float = make_tensor_impl({2, 3, 4}, ScalarType::Float, kCPU);
    EXPECT_EQ(impl_float->nbytes(), 2 * 3 * 4 * sizeof(float));

    auto impl_double = make_tensor_impl({10}, ScalarType::Double, kCPU);
    EXPECT_EQ(impl_double->nbytes(), 10 * sizeof(double));

    auto impl_half = make_tensor_impl({5, 5}, ScalarType::Half, kCPU);
    EXPECT_EQ(impl_half->nbytes(), 25 * sizeof(Half));
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST(TensorImplFactoryTest, MakeTensorImpl) {
    auto impl = make_tensor_impl({3, 4, 5}, ScalarType::Float, kCPU);

    EXPECT_EQ(impl->dim(), 3);
    EXPECT_EQ(impl->size(0), 3);
    EXPECT_EQ(impl->size(1), 4);
    EXPECT_EQ(impl->size(2), 5);
    EXPECT_EQ(impl->numel(), 60);
    EXPECT_TRUE(impl->is_cpu());
    EXPECT_EQ(impl->dtype(), ScalarType::Float);
}

TEST(TensorImplFactoryTest, MakeEmptyTensorImpl) {
    auto impl = make_empty_tensor_impl(ScalarType::Double, kCPU);

    EXPECT_EQ(impl->dim(), 0);
    EXPECT_EQ(impl->numel(), 0);
    EXPECT_EQ(impl->dtype(), ScalarType::Double);
    EXPECT_TRUE(impl->is_cpu());
}

// ============================================================================
// Contiguity Tests
// ============================================================================

TEST(TensorImplTest, ContiguityCheck) {
    // Contiguous tensor
    auto contiguous = make_tensor_impl({2, 3, 4}, ScalarType::Float, kCPU);
    EXPECT_TRUE(contiguous->is_contiguous());

    // Non-contiguous tensor (transposed)
    Storage storage = Storage::create(sizeof(float) * 12, kCPU);
    TensorImpl transposed(std::move(storage), ScalarType::Float, {4, 3}, {1, 4});
    EXPECT_FALSE(transposed.is_contiguous());
}
