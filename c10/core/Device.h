#pragma once

#include <cstdint>
#include <string>
#include <functional>
#include <ostream>
#include "c10/macros/Macros.h"
#include "c10/util/Exception.h"

namespace c10 {

// ============================================================================
// Device Type Enumeration
// ============================================================================

enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1,
    MKLDNN = 2,     // Intel MKL-DNN
    OPENCL = 3,
    OPENGL = 4,
    IDEEP = 5,      // Intel IDEEP
    HIP = 6,        // AMD ROCm HIP
    FPGA = 7,
    MSNPU = 8,
    XLA = 9,        // Google TPU
    Vulkan = 10,
    Metal = 11,     // Apple Metal
    XPU = 12,       // Intel XPU
    MPS = 13,       // Apple Metal Performance Shaders
    Meta = 14,      // Meta tensors (no storage)
    HPU = 15,       // Habana Gaudi
    VE = 16,        // NEC Vector Engine
    Lazy = 17,      // Lazy tensors
    IPU = 18,       // Graphcore IPU
    MTIA = 19,      // Meta Training and Inference Accelerator
    PrivateUse1 = 20,  // Reserved for custom backends (NMCard)
    PrivateUse2 = 21,  // Reserved for custom backends (LinQ)

    // Compile time constant for number of device types
    COMPILE_TIME_MAX_DEVICE_TYPES = 22
};

constexpr int kNumDeviceTypes = static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

// ============================================================================
// Device Type Utilities
// ============================================================================

inline const char* DeviceTypeName(DeviceType type, bool lower_case = false) {
    switch (type) {
        case DeviceType::CPU:
            return lower_case ? "cpu" : "CPU";
        case DeviceType::CUDA:
            return lower_case ? "cuda" : "CUDA";
        case DeviceType::MKLDNN:
            return lower_case ? "mkldnn" : "MKLDNN";
        case DeviceType::OPENCL:
            return lower_case ? "opencl" : "OpenCL";
        case DeviceType::OPENGL:
            return lower_case ? "opengl" : "OpenGL";
        case DeviceType::IDEEP:
            return lower_case ? "ideep" : "IDEEP";
        case DeviceType::HIP:
            return lower_case ? "hip" : "HIP";
        case DeviceType::FPGA:
            return lower_case ? "fpga" : "FPGA";
        case DeviceType::XLA:
            return lower_case ? "xla" : "XLA";
        case DeviceType::Vulkan:
            return lower_case ? "vulkan" : "Vulkan";
        case DeviceType::Metal:
            return lower_case ? "metal" : "Metal";
        case DeviceType::XPU:
            return lower_case ? "xpu" : "XPU";
        case DeviceType::MPS:
            return lower_case ? "mps" : "MPS";
        case DeviceType::Meta:
            return lower_case ? "meta" : "Meta";
        case DeviceType::HPU:
            return lower_case ? "hpu" : "HPU";
        case DeviceType::VE:
            return lower_case ? "ve" : "VE";
        case DeviceType::Lazy:
            return lower_case ? "lazy" : "Lazy";
        case DeviceType::IPU:
            return lower_case ? "ipu" : "IPU";
        case DeviceType::MTIA:
            return lower_case ? "mtia" : "MTIA";
        case DeviceType::PrivateUse1:
            return lower_case ? "nmcard" : "NMCard";
        case DeviceType::PrivateUse2:
            return lower_case ? "linq" : "LinQ";
        default:
            return lower_case ? "unknown" : "Unknown";
    }
}

inline bool isValidDeviceType(DeviceType type) {
    return static_cast<int8_t>(type) >= 0 &&
           static_cast<int8_t>(type) < static_cast<int8_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
}

inline std::ostream& operator<<(std::ostream& os, DeviceType type) {
    return os << DeviceTypeName(type);
}

// ============================================================================
// Device Index
// ============================================================================

using DeviceIndex = int8_t;
constexpr DeviceIndex kNoDeviceIndex = -1;

// ============================================================================
// Device Class
// ============================================================================

class PT_API Device {
public:
    // Constructors
    constexpr Device() : type_(DeviceType::CPU), index_(kNoDeviceIndex) {}

    constexpr explicit Device(DeviceType type, DeviceIndex index = kNoDeviceIndex)
        : type_(type), index_(index) {
        // Note: validation moved to non-constexpr contexts
    }

    // Parse from string: "cpu", "cuda", "cuda:0", "cuda:1", etc.
    explicit Device(const std::string& device_string) {
        parse(device_string);
    }

    explicit Device(const char* device_string) {
        parse(std::string(device_string));
    }

    // Accessors
    DeviceType type() const noexcept { return type_; }
    DeviceIndex index() const noexcept { return index_; }

    // Checkers
    bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }
    bool is_cuda() const noexcept { return type_ == DeviceType::CUDA; }
    bool is_hip() const noexcept { return type_ == DeviceType::HIP; }
    bool is_xpu() const noexcept { return type_ == DeviceType::XPU; }
    bool is_mps() const noexcept { return type_ == DeviceType::MPS; }
    bool is_meta() const noexcept { return type_ == DeviceType::Meta; }
    bool is_vulkan() const noexcept { return type_ == DeviceType::Vulkan; }
    bool is_metal() const noexcept { return type_ == DeviceType::Metal; }
    bool is_nmcard() const noexcept { return type_ == DeviceType::PrivateUse1; }
    bool is_linq() const noexcept { return type_ == DeviceType::PrivateUse2; }

    bool has_index() const noexcept { return index_ != kNoDeviceIndex; }

    // Set index
    void set_index(DeviceIndex index) {
        index_ = index;
    }

    // Comparison operators
    bool operator==(const Device& other) const noexcept {
        return type_ == other.type_ && index_ == other.index_;
    }

    bool operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }

    bool operator<(const Device& other) const noexcept {
        if (type_ != other.type_) {
            return static_cast<int8_t>(type_) < static_cast<int8_t>(other.type_);
        }
        return index_ < other.index_;
    }

    // String representation
    std::string str() const {
        std::string result = DeviceTypeName(type_, true);
        if (has_index()) {
            result += ":" + std::to_string(index_);
        }
        return result;
    }

private:
    DeviceType type_;
    DeviceIndex index_;

    void validate() const {
        PT_CHECK_MSG(
            isValidDeviceType(type_),
            "Invalid device type: " + std::to_string(static_cast<int>(type_))
        );

        // CPU doesn't need an index
        if (type_ == DeviceType::CPU) {
            PT_CHECK_MSG(
                index_ == kNoDeviceIndex || index_ == 0,
                "CPU device index must be -1 or 0, got " + std::to_string(index_)
            );
        }

        // GPU devices need non-negative index (or kNoDeviceIndex)
        if (type_ == DeviceType::CUDA || type_ == DeviceType::HIP) {
            PT_CHECK_MSG(
                index_ >= kNoDeviceIndex,
                "Device index must be non-negative or -1, got " + std::to_string(index_)
            );
        }
    }

    void parse(const std::string& device_string) {
        if (device_string.empty()) {
            PT_ERROR("Device string cannot be empty");
        }

        // Find colon separator
        size_t colon_pos = device_string.find(':');

        std::string type_str;
        std::string index_str;

        if (colon_pos == std::string::npos) {
            type_str = device_string;
            index_ = kNoDeviceIndex;
        } else {
            type_str = device_string.substr(0, colon_pos);
            index_str = device_string.substr(colon_pos + 1);
        }

        // Convert to lowercase for comparison
        std::string type_lower = type_str;
        for (char& c : type_lower) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        // Parse device type
        if (type_lower == "cpu") {
            type_ = DeviceType::CPU;
        } else if (type_lower == "cuda") {
            type_ = DeviceType::CUDA;
        } else if (type_lower == "hip") {
            type_ = DeviceType::HIP;
        } else if (type_lower == "xpu") {
            type_ = DeviceType::XPU;
        } else if (type_lower == "mps") {
            type_ = DeviceType::MPS;
        } else if (type_lower == "meta") {
            type_ = DeviceType::Meta;
        } else if (type_lower == "vulkan") {
            type_ = DeviceType::Vulkan;
        } else if (type_lower == "metal") {
            type_ = DeviceType::Metal;
        } else if (type_lower == "nmcard") {
            type_ = DeviceType::PrivateUse1;
        } else if (type_lower == "linq") {
            type_ = DeviceType::PrivateUse2;
        } else {
            PT_ERROR("Unknown device type: ", type_str);
        }

        // Parse index
        if (!index_str.empty()) {
            try {
                index_ = static_cast<DeviceIndex>(std::stoi(index_str));
            } catch (...) {
                PT_ERROR("Invalid device index: ", index_str);
            }
        }

        validate();
    }
};

// ============================================================================
// Stream operators
// ============================================================================

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    return os << device.str();
}

// ============================================================================
// Hash support for std::unordered_map
// ============================================================================

struct DeviceHash {
    size_t operator()(const Device& device) const noexcept {
        return std::hash<int64_t>()(
            (static_cast<int64_t>(device.type()) << 8) | (device.index() & 0xFF)
        );
    }
};

// ============================================================================
// Device Guard (RAII for device switching)
// ============================================================================

class PT_API DeviceGuard {
public:
    explicit DeviceGuard(Device device);
    ~DeviceGuard();

    // No copy
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    // Move allowed
    DeviceGuard(DeviceGuard&& other) noexcept;
    DeviceGuard& operator=(DeviceGuard&& other) noexcept;

    // Get current device
    Device current_device() const noexcept { return current_device_; }
    Device original_device() const noexcept { return original_device_; }

    // Reset to specific device
    void reset_device(Device device);

    // Reset to original device
    void reset();

private:
    Device original_device_;
    Device current_device_;
    bool active_ = false;
};

// ============================================================================
// Common device constants
// ============================================================================

constexpr Device kCPU{DeviceType::CPU};

inline Device kCUDA(DeviceIndex index = 0) {
    return Device{DeviceType::CUDA, index};
}

inline Device kNMCard(DeviceIndex index = 0) {
    return Device{DeviceType::PrivateUse1, index};
}

inline Device kLinQ(DeviceIndex index = 0) {
    return Device{DeviceType::PrivateUse2, index};
}

} // namespace c10

// ============================================================================
// std::hash specialization
// ============================================================================

namespace std {
template<>
struct hash<c10::Device> {
    size_t operator()(const c10::Device& device) const noexcept {
        return c10::DeviceHash{}(device);
    }
};
} // namespace std
