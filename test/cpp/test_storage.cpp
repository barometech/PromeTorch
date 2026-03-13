#include <gtest/gtest.h>
#include "c10/core/Storage.h"
#include <cstring>

using namespace c10;

// ============================================================================
// StorageImpl Tests
// ============================================================================

TEST(StorageImplTest, Constructor) {
    auto alloc = &CPUAllocator::get();
    StorageImpl* impl = new StorageImpl(1024, alloc, false);

    EXPECT_EQ(impl->nbytes(), 1024);
    EXPECT_NE(impl->data(), nullptr);
    EXPECT_EQ(impl->allocator(), alloc);
    EXPECT_FALSE(impl->resizable());
    EXPECT_TRUE(impl->device().is_cpu());

    impl->release();
}

TEST(StorageImplTest, ReferenceCount) {
    auto alloc = &CPUAllocator::get();
    StorageImpl* impl = new StorageImpl(1024, alloc);

    EXPECT_EQ(impl->use_count(), 1);
    EXPECT_TRUE(impl->unique());

    impl->retain();
    EXPECT_EQ(impl->use_count(), 2);
    EXPECT_FALSE(impl->unique());

    impl->release();
    EXPECT_EQ(impl->use_count(), 1);
    EXPECT_TRUE(impl->unique());

    impl->release();  // Should delete
}

TEST(StorageImplTest, DataAccess) {
    auto alloc = &CPUAllocator::get();
    StorageImpl* impl = new StorageImpl(sizeof(float) * 10, alloc);

    float* data = impl->data<float>();
    EXPECT_NE(data, nullptr);

    // Write some data
    for (int i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Read back
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(impl->data<float>()[i], static_cast<float>(i));
    }

    impl->release();
}

TEST(StorageImplTest, Resize) {
    auto alloc = &CPUAllocator::get();
    StorageImpl* impl = new StorageImpl(100, alloc, true);  // resizable

    // Write some data
    std::memset(impl->mutable_data(), 0xAB, 100);

    // Resize larger
    impl->resize(200);
    EXPECT_EQ(impl->nbytes(), 200);

    // First 100 bytes should be preserved
    unsigned char* data = static_cast<unsigned char*>(impl->data());
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(data[i], 0xAB);
    }

    impl->release();
}

TEST(StorageImplTest, ResizeNonResizable) {
    auto alloc = &CPUAllocator::get();
    StorageImpl* impl = new StorageImpl(100, alloc, false);  // not resizable

    EXPECT_THROW(impl->resize(200), std::runtime_error);

    impl->release();
}

// ============================================================================
// Storage Tests
// ============================================================================

TEST(StorageTest, DefaultConstructor) {
    Storage storage;
    EXPECT_FALSE(storage.defined());
    EXPECT_FALSE(static_cast<bool>(storage));
}

TEST(StorageTest, ConstructorWithSize) {
    Storage storage(1024, &CPUAllocator::get());
    EXPECT_TRUE(storage.defined());
    EXPECT_EQ(storage.nbytes(), 1024);
    EXPECT_NE(storage.data(), nullptr);
}

TEST(StorageTest, CopyConstructor) {
    Storage storage1(1024, &CPUAllocator::get());
    EXPECT_EQ(storage1.use_count(), 1);

    Storage storage2(storage1);
    EXPECT_EQ(storage1.use_count(), 2);
    EXPECT_EQ(storage2.use_count(), 2);
    EXPECT_EQ(storage1.data(), storage2.data());  // Same data
}

TEST(StorageTest, MoveConstructor) {
    Storage storage1(1024, &CPUAllocator::get());
    void* data_ptr = storage1.data();

    Storage storage2(std::move(storage1));
    EXPECT_FALSE(storage1.defined());
    EXPECT_TRUE(storage2.defined());
    EXPECT_EQ(storage2.data(), data_ptr);
    EXPECT_EQ(storage2.use_count(), 1);
}

TEST(StorageTest, CopyAssignment) {
    Storage storage1(1024, &CPUAllocator::get());
    Storage storage2(512, &CPUAllocator::get());

    storage2 = storage1;
    EXPECT_EQ(storage1.use_count(), 2);
    EXPECT_EQ(storage2.use_count(), 2);
    EXPECT_EQ(storage1.data(), storage2.data());
}

TEST(StorageTest, MoveAssignment) {
    Storage storage1(1024, &CPUAllocator::get());
    Storage storage2(512, &CPUAllocator::get());
    void* data_ptr = storage1.data();

    storage2 = std::move(storage1);
    EXPECT_FALSE(storage1.defined());
    EXPECT_TRUE(storage2.defined());
    EXPECT_EQ(storage2.data(), data_ptr);
}

TEST(StorageTest, Device) {
    Storage storage(1024, &CPUAllocator::get());
    EXPECT_TRUE(storage.device().is_cpu());
    EXPECT_EQ(storage.device_type(), DeviceType::CPU);
}

TEST(StorageTest, DataAccess) {
    Storage storage(sizeof(double) * 5, &CPUAllocator::get());

    double* data = storage.mutable_data<double>();
    for (int i = 0; i < 5; ++i) {
        data[i] = i * 1.5;
    }

    const double* const_data = storage.data<double>();
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(const_data[i], i * 1.5);
    }
}

TEST(StorageTest, SetNbytes) {
    Storage storage(1024, &CPUAllocator::get());
    EXPECT_EQ(storage.nbytes(), 1024);

    storage.set_nbytes(2048);
    EXPECT_EQ(storage.nbytes(), 2048);  // Just changes metadata, not actual allocation
}

// ============================================================================
// Storage Factory Tests
// ============================================================================

TEST(StorageFactoryTest, Create) {
    Storage storage = Storage::create(1024, kCPU);
    EXPECT_TRUE(storage.defined());
    EXPECT_EQ(storage.nbytes(), 1024);
    EXPECT_TRUE(storage.device().is_cpu());
}

TEST(StorageFactoryTest, CreateWithData) {
    float* data = static_cast<float*>(malloc(sizeof(float) * 10));
    for (int i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto deleter = [](void* d, void*) { free(d); };

    Storage storage = Storage::create_with_data(
        data,
        sizeof(float) * 10,
        deleter,
        kCPU
    );

    EXPECT_TRUE(storage.defined());
    EXPECT_EQ(storage.data(), data);
    EXPECT_EQ(storage.data<float>()[5], 5.0f);
}

// ============================================================================
// Storage Comparison Tests
// ============================================================================

TEST(StorageComparisonTest, Equality) {
    Storage storage1(1024, &CPUAllocator::get());
    Storage storage2 = storage1;  // Same impl
    Storage storage3(1024, &CPUAllocator::get());  // Different impl

    EXPECT_EQ(storage1, storage2);
    EXPECT_NE(storage1, storage3);
}

TEST(StorageComparisonTest, NullStorage) {
    Storage null1;
    Storage null2;
    Storage valid(1024, &CPUAllocator::get());

    EXPECT_EQ(null1, null2);
    EXPECT_NE(null1, valid);
}
