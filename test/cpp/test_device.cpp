#include <gtest/gtest.h>
#include "c10/core/Device.h"

using namespace c10;

// ============================================================================
// DeviceType Tests
// ============================================================================

TEST(DeviceTypeTest, DeviceTypeName) {
    EXPECT_STREQ(DeviceTypeName(DeviceType::CPU), "CPU");
    EXPECT_STREQ(DeviceTypeName(DeviceType::CUDA), "CUDA");
    EXPECT_STREQ(DeviceTypeName(DeviceType::HIP), "HIP");
    EXPECT_STREQ(DeviceTypeName(DeviceType::XPU), "XPU");
    EXPECT_STREQ(DeviceTypeName(DeviceType::MPS), "MPS");
    EXPECT_STREQ(DeviceTypeName(DeviceType::Meta), "Meta");

    // Lowercase
    EXPECT_STREQ(DeviceTypeName(DeviceType::CPU, true), "cpu");
    EXPECT_STREQ(DeviceTypeName(DeviceType::CUDA, true), "cuda");
}

TEST(DeviceTypeTest, IsValidDeviceType) {
    EXPECT_TRUE(isValidDeviceType(DeviceType::CPU));
    EXPECT_TRUE(isValidDeviceType(DeviceType::CUDA));
    EXPECT_TRUE(isValidDeviceType(DeviceType::Meta));
}

// ============================================================================
// Device Tests
// ============================================================================

TEST(DeviceTest, DefaultConstructor) {
    Device device;
    EXPECT_EQ(device.type(), DeviceType::CPU);
    EXPECT_EQ(device.index(), kNoDeviceIndex);
    EXPECT_TRUE(device.is_cpu());
}

TEST(DeviceTest, ConstructorWithType) {
    Device cpu(DeviceType::CPU);
    EXPECT_TRUE(cpu.is_cpu());
    EXPECT_FALSE(cpu.has_index());

    Device cuda(DeviceType::CUDA, 0);
    EXPECT_TRUE(cuda.is_cuda());
    EXPECT_EQ(cuda.index(), 0);
    EXPECT_TRUE(cuda.has_index());

    Device cuda1(DeviceType::CUDA, 1);
    EXPECT_EQ(cuda1.index(), 1);
}

TEST(DeviceTest, StringParsing) {
    Device cpu("cpu");
    EXPECT_TRUE(cpu.is_cpu());

    Device cuda("cuda");
    EXPECT_TRUE(cuda.is_cuda());
    EXPECT_FALSE(cuda.has_index());

    Device cuda0("cuda:0");
    EXPECT_TRUE(cuda0.is_cuda());
    EXPECT_EQ(cuda0.index(), 0);

    Device cuda1("cuda:1");
    EXPECT_EQ(cuda1.index(), 1);

    Device meta("meta");
    EXPECT_TRUE(meta.is_meta());

    // Case insensitive
    Device CUDA("CUDA:0");
    EXPECT_TRUE(CUDA.is_cuda());
    EXPECT_EQ(CUDA.index(), 0);
}

TEST(DeviceTest, Str) {
    Device cpu(DeviceType::CPU);
    EXPECT_EQ(cpu.str(), "cpu");

    Device cuda0(DeviceType::CUDA, 0);
    EXPECT_EQ(cuda0.str(), "cuda:0");

    Device cuda1(DeviceType::CUDA, 1);
    EXPECT_EQ(cuda1.str(), "cuda:1");

    Device meta(DeviceType::Meta);
    EXPECT_EQ(meta.str(), "meta");
}

TEST(DeviceTest, Comparison) {
    Device cpu1(DeviceType::CPU);
    Device cpu2(DeviceType::CPU);
    Device cuda0(DeviceType::CUDA, 0);
    Device cuda1(DeviceType::CUDA, 1);

    EXPECT_EQ(cpu1, cpu2);
    EXPECT_NE(cpu1, cuda0);
    EXPECT_NE(cuda0, cuda1);
    EXPECT_LT(cpu1, cuda0);  // CPU < CUDA
    EXPECT_LT(cuda0, cuda1); // cuda:0 < cuda:1
}

TEST(DeviceTest, SetIndex) {
    Device cuda(DeviceType::CUDA);
    EXPECT_FALSE(cuda.has_index());

    cuda.set_index(2);
    EXPECT_TRUE(cuda.has_index());
    EXPECT_EQ(cuda.index(), 2);
}

TEST(DeviceTest, Checkers) {
    Device cpu(DeviceType::CPU);
    EXPECT_TRUE(cpu.is_cpu());
    EXPECT_FALSE(cpu.is_cuda());
    EXPECT_FALSE(cpu.is_meta());

    Device cuda(DeviceType::CUDA, 0);
    EXPECT_FALSE(cuda.is_cpu());
    EXPECT_TRUE(cuda.is_cuda());

    Device meta(DeviceType::Meta);
    EXPECT_TRUE(meta.is_meta());
}

TEST(DeviceTest, Hash) {
    Device cpu(DeviceType::CPU);
    Device cuda0(DeviceType::CUDA, 0);
    Device cuda1(DeviceType::CUDA, 1);

    DeviceHash hasher;
    EXPECT_NE(hasher(cpu), hasher(cuda0));
    EXPECT_NE(hasher(cuda0), hasher(cuda1));

    // Same device should have same hash
    Device cpu2(DeviceType::CPU);
    EXPECT_EQ(hasher(cpu), hasher(cpu2));
}

TEST(DeviceTest, StdHash) {
    Device cpu(DeviceType::CPU);
    Device cuda0(DeviceType::CUDA, 0);

    std::hash<Device> hasher;
    EXPECT_NE(hasher(cpu), hasher(cuda0));
}

// ============================================================================
// Constants Tests
// ============================================================================

TEST(DeviceConstantsTest, kCPU) {
    EXPECT_TRUE(kCPU.is_cpu());
    EXPECT_FALSE(kCPU.has_index());
}

TEST(DeviceConstantsTest, kCUDA) {
    Device cuda0 = kCUDA(0);
    EXPECT_TRUE(cuda0.is_cuda());
    EXPECT_EQ(cuda0.index(), 0);

    Device cuda1 = kCUDA(1);
    EXPECT_EQ(cuda1.index(), 1);
}

// ============================================================================
// Invalid Device Tests
// ============================================================================

TEST(DeviceTest, InvalidStringThrows) {
    EXPECT_THROW(Device("invalid"), std::runtime_error);
    EXPECT_THROW(Device(""), std::runtime_error);
    EXPECT_THROW(Device("cuda:abc"), std::runtime_error);
}
