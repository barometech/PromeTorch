#include <gtest/gtest.h>
#include "c10/core/Allocator.h"
#include <cstring>

using namespace c10;

// ============================================================================
// DataPtr Tests
// ============================================================================

TEST(DataPtrTest, DefaultConstructor) {
    DataPtr ptr;
    EXPECT_EQ(ptr.get(), nullptr);
    EXPECT_FALSE(ptr);
}

TEST(DataPtrTest, MoveConstructor) {
    auto deleter_called = false;
    auto deleter = [](void* data, void* ctx) {
        *static_cast<bool*>(ctx) = true;
        free(data);
    };

    void* data = malloc(100);
    {
        DataPtr ptr1(data, &deleter_called, deleter, kCPU);
        EXPECT_TRUE(ptr1);
        EXPECT_EQ(ptr1.get(), data);

        DataPtr ptr2(std::move(ptr1));
        EXPECT_FALSE(ptr1);  // ptr1 is now empty
        EXPECT_TRUE(ptr2);
        EXPECT_EQ(ptr2.get(), data);
    }
    EXPECT_TRUE(deleter_called);  // Deleter should be called when ptr2 goes out of scope
}

TEST(DataPtrTest, MoveAssignment) {
    bool deleter1_called = false;
    bool deleter2_called = false;

    auto deleter = [](void* data, void* ctx) {
        *static_cast<bool*>(ctx) = true;
        free(data);
    };

    DataPtr ptr1(malloc(100), &deleter1_called, deleter, kCPU);
    DataPtr ptr2(malloc(100), &deleter2_called, deleter, kCPU);

    ptr1 = std::move(ptr2);

    EXPECT_TRUE(deleter1_called);   // Old ptr1 data should be deleted
    EXPECT_FALSE(deleter2_called);  // ptr2 data now owned by ptr1
}

TEST(DataPtrTest, Clear) {
    bool deleter_called = false;
    auto deleter = [](void* data, void* ctx) {
        *static_cast<bool*>(ctx) = true;
        free(data);
    };

    DataPtr ptr(malloc(100), &deleter_called, deleter, kCPU);
    EXPECT_TRUE(ptr);

    ptr.clear();
    EXPECT_FALSE(ptr);
    EXPECT_TRUE(deleter_called);
}

TEST(DataPtrTest, Cast) {
    float* data = static_cast<float*>(malloc(sizeof(float) * 10));
    data[0] = 3.14f;

    auto deleter = [](void* d, void*) { free(d); };
    DataPtr ptr(data, nullptr, deleter, kCPU);

    float* typed_ptr = ptr.cast<float>();
    EXPECT_EQ(typed_ptr[0], 3.14f);
}

TEST(DataPtrTest, Device) {
    auto deleter = [](void* d, void*) { free(d); };
    DataPtr cpu_ptr(malloc(100), nullptr, deleter, kCPU);
    EXPECT_TRUE(cpu_ptr.device().is_cpu());

    DataPtr cuda_ptr(malloc(100), nullptr, deleter, kCUDA(0));
    EXPECT_TRUE(cuda_ptr.device().is_cuda());
}

// ============================================================================
// CPUAllocator Tests
// ============================================================================

TEST(CPUAllocatorTest, Singleton) {
    CPUAllocator& alloc1 = CPUAllocator::get();
    CPUAllocator& alloc2 = CPUAllocator::get();
    EXPECT_EQ(&alloc1, &alloc2);
}

TEST(CPUAllocatorTest, Allocate) {
    CPUAllocator& alloc = CPUAllocator::get();

    DataPtr ptr = alloc.allocate(1024);
    EXPECT_NE(ptr.get(), nullptr);
    EXPECT_TRUE(ptr.device().is_cpu());

    // Test that memory is writable
    std::memset(ptr.mutable_get(), 0xAB, 1024);
}

TEST(CPUAllocatorTest, AllocateZero) {
    CPUAllocator& alloc = CPUAllocator::get();

    DataPtr ptr = alloc.allocate(0);
    EXPECT_EQ(ptr.get(), nullptr);
}

TEST(CPUAllocatorTest, Alignment) {
    CPUAllocator& alloc = CPUAllocator::get();

    // Allocate several blocks and check alignment
    for (size_t size : {1, 7, 64, 128, 1000, 4096}) {
        DataPtr ptr = alloc.allocate(size);
        if (ptr.get() != nullptr) {
            // Should be 64-byte aligned for AVX-512
            EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr.get()) % 64, 0)
                << "Allocation of " << size << " bytes is not 64-byte aligned";
        }
    }
}

TEST(CPUAllocatorTest, RawAllocateAndDeallocate) {
    CPUAllocator& alloc = CPUAllocator::get();

    void* ptr = alloc.raw_allocate(1024);
    EXPECT_NE(ptr, nullptr);

    // Write to memory
    std::memset(ptr, 0x42, 1024);

    // Deallocate
    alloc.raw_deallocate(ptr);
}

TEST(CPUAllocatorTest, RawDeallocateNull) {
    CPUAllocator& alloc = CPUAllocator::get();
    alloc.raw_deallocate(nullptr);  // Should not crash
}

// ============================================================================
// AllocatorRegistry Tests
// ============================================================================

TEST(AllocatorRegistryTest, Singleton) {
    AllocatorRegistry& reg1 = AllocatorRegistry::get();
    AllocatorRegistry& reg2 = AllocatorRegistry::get();
    EXPECT_EQ(&reg1, &reg2);
}

TEST(AllocatorRegistryTest, CPUAllocatorRegistered) {
    AllocatorRegistry& reg = AllocatorRegistry::get();
    EXPECT_TRUE(reg.hasAllocator(DeviceType::CPU));

    Allocator* alloc = reg.getAllocator(DeviceType::CPU);
    EXPECT_NE(alloc, nullptr);
    EXPECT_EQ(alloc, &CPUAllocator::get());
}

TEST(AllocatorRegistryTest, UnregisteredDevice) {
    AllocatorRegistry& reg = AllocatorRegistry::get();

    // CUDA allocator is not registered in CPU-only build
    // This might throw or return false
    bool has_cuda = reg.hasAllocator(DeviceType::CUDA);
    if (!has_cuda) {
        EXPECT_THROW(reg.getAllocator(DeviceType::CUDA), std::runtime_error);
    }
}

// ============================================================================
// GetAllocator Helper Tests
// ============================================================================

TEST(GetAllocatorTest, CPU) {
    Allocator* alloc = GetAllocator(DeviceType::CPU);
    EXPECT_NE(alloc, nullptr);
    EXPECT_EQ(alloc, &CPUAllocator::get());
}

TEST(GetAllocatorTest, Device) {
    Allocator* alloc = GetAllocator(kCPU);
    EXPECT_NE(alloc, nullptr);
}

// ============================================================================
// AllocateMemory Helper Tests
// ============================================================================

TEST(AllocateMemoryTest, CPU) {
    DataPtr ptr = AllocateMemory(512, kCPU);
    EXPECT_NE(ptr.get(), nullptr);
    EXPECT_TRUE(ptr.device().is_cpu());
}

// ============================================================================
// InefficientStdFunctionContext Tests
// ============================================================================

TEST(InefficientStdFunctionContextTest, CustomDeleter) {
    bool deleted = false;

    {
        void* data = malloc(100);
        DataPtr ptr = InefficientStdFunctionContext::makeDataPtr(
            data,
            [&deleted](void* d) {
                deleted = true;
                free(d);
            },
            kCPU
        );
        EXPECT_NE(ptr.get(), nullptr);
    }

    EXPECT_TRUE(deleted);
}

TEST(InefficientStdFunctionContextTest, LambdaCapture) {
    int capture_value = 42;
    bool verified = false;

    {
        void* data = malloc(100);
        DataPtr ptr = InefficientStdFunctionContext::makeDataPtr(
            data,
            [&verified, capture_value](void* d) {
                verified = (capture_value == 42);
                free(d);
            },
            kCPU
        );
    }

    EXPECT_TRUE(verified);
}
