#pragma once

#include "torch/io/gguf_dequant.h"
#include "aten/src/ATen/ATen.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cstring>
#include <variant>
#include <algorithm>

namespace torch {
namespace io {
namespace gguf {

// ============================================================================
// GGUF Constants
// ============================================================================

static constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" as little-endian uint32
static constexpr uint32_t GGUF_VERSION_3 = 3;
static constexpr size_t GGUF_DEFAULT_ALIGNMENT = 32;

// ============================================================================
// GGUF Value Types
// ============================================================================

enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// ============================================================================
// GGUF Metadata Value (variant)
// ============================================================================

struct GGUFValue {
    GGUFValueType type;

    // Store all possible types
    uint64_t u64 = 0;
    int64_t i64 = 0;
    double f64 = 0.0;
    bool b = false;
    std::string str;
    std::vector<GGUFValue> arr;

    // Convenience getters
    uint32_t as_uint32() const { return static_cast<uint32_t>(u64); }
    int32_t as_int32() const { return static_cast<int32_t>(i64); }
    uint64_t as_uint64() const { return u64; }
    int64_t as_int64() const { return i64; }
    float as_float() const { return static_cast<float>(f64); }
    double as_double() const { return f64; }
    bool as_bool() const { return b; }
    const std::string& as_string() const { return str; }
    const std::vector<GGUFValue>& as_array() const { return arr; }
};

// ============================================================================
// GGUF Tensor Info
// ============================================================================

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;  // ne[0], ne[1], ... (ggml order)
    GGMLType type;
    uint64_t offset;  // Offset from start of data section

    int64_t n_elements() const {
        int64_t n = 1;
        for (auto d : dims) n *= static_cast<int64_t>(d);
        return n;
    }

    // Convert to PyTorch-convention shape (reversed dims)
    std::vector<int64_t> shape() const {
        std::vector<int64_t> s(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            s[i] = static_cast<int64_t>(dims[dims.size() - 1 - i]);
        }
        return s;
    }

    int64_t data_bytes() const {
        return ggml_tensor_bytes(type, n_elements());
    }
};

// ============================================================================
// GGUF File Reader
// ============================================================================

class GGUFReader {
public:
    // Header
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;

    // Parsed data
    std::unordered_map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensorInfo> tensors;
    std::unordered_map<std::string, size_t> tensor_index;  // name → index in tensors

    // File info
    std::string file_path;
    uint64_t data_offset = 0;  // Absolute offset to data section in file

    // ========================================================================
    // Open and parse a GGUF file
    // ========================================================================

    void open(const std::string& path) {
        file_path = path;
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("GGUF: Cannot open file: " + path);
        }

        parse_header(f);
        parse_metadata(f);
        parse_tensor_infos(f);

        // Data section starts at alignment boundary after tensor infos
        uint64_t pos = static_cast<uint64_t>(f.tellg());
        data_offset = align_offset(pos, GGUF_DEFAULT_ALIGNMENT);

        std::cout << "[GGUF] Loaded: " << path << std::endl;
        std::cout << "[GGUF] Version: " << version
                  << ", Tensors: " << tensor_count
                  << ", Metadata KVs: " << metadata_kv_count << std::endl;
        std::cout << "[GGUF] Data offset: " << data_offset << " bytes" << std::endl;
    }

    // ========================================================================
    // Get metadata by key
    // ========================================================================

    bool has_metadata(const std::string& key) const {
        return metadata.find(key) != metadata.end();
    }

    const GGUFValue& get_metadata(const std::string& key) const {
        auto it = metadata.find(key);
        if (it == metadata.end()) {
            throw std::runtime_error("GGUF: Metadata key not found: " + key);
        }
        return it->second;
    }

    std::string get_string(const std::string& key, const std::string& default_val = "") const {
        if (!has_metadata(key)) return default_val;
        return get_metadata(key).as_string();
    }

    int64_t get_int(const std::string& key, int64_t default_val = 0) const {
        if (!has_metadata(key)) return default_val;
        auto& v = get_metadata(key);
        switch (v.type) {
            case GGUFValueType::UINT32: return static_cast<int64_t>(v.u64);
            case GGUFValueType::INT32:  return v.i64;
            case GGUFValueType::UINT64: return static_cast<int64_t>(v.u64);
            case GGUFValueType::INT64:  return v.i64;
            case GGUFValueType::UINT8:  return static_cast<int64_t>(v.u64);
            case GGUFValueType::UINT16: return static_cast<int64_t>(v.u64);
            case GGUFValueType::INT8:   return v.i64;
            case GGUFValueType::INT16:  return v.i64;
            default: return default_val;
        }
    }

    float get_float(const std::string& key, float default_val = 0.0f) const {
        if (!has_metadata(key)) return default_val;
        return get_metadata(key).as_float();
    }

    // ========================================================================
    // Get architecture-prefixed metadata
    // ========================================================================

    std::string architecture() const {
        return get_string("general.architecture", "unknown");
    }

    int64_t get_arch_int(const std::string& key, int64_t default_val = 0) const {
        std::string arch = architecture();
        return get_int(arch + "." + key, default_val);
    }

    float get_arch_float(const std::string& key, float default_val = 0.0f) const {
        std::string arch = architecture();
        return get_float(arch + "." + key, default_val);
    }

    // ========================================================================
    // Check if tensor exists
    // ========================================================================

    bool has_tensor(const std::string& name) const {
        return tensor_index.find(name) != tensor_index.end();
    }

    const GGUFTensorInfo& get_tensor_info(const std::string& name) const {
        auto it = tensor_index.find(name);
        if (it == tensor_index.end()) {
            throw std::runtime_error("GGUF: Tensor not found: " + name);
        }
        return tensors[it->second];
    }

    // ========================================================================
    // Load and dequantize a single tensor → at::Tensor (float32)
    // ========================================================================

    at::Tensor load_tensor(const std::string& name) const {
        const auto& info = get_tensor_info(name);
        return load_tensor(info);
    }

    at::Tensor load_tensor(const GGUFTensorInfo& info) const {
        int64_t n_elements = info.n_elements();
        int64_t raw_bytes = info.data_bytes();

        // Read raw quantized data from file
        std::ifstream f(file_path, std::ios::binary);
        if (!f) throw std::runtime_error("GGUF: Cannot reopen file: " + file_path);

        uint64_t abs_offset = data_offset + info.offset;
        f.seekg(static_cast<std::streamoff>(abs_offset));
        if (!f) throw std::runtime_error("GGUF: Seek failed for tensor: " + info.name);

        std::vector<uint8_t> raw(raw_bytes);
        f.read(reinterpret_cast<char*>(raw.data()), raw_bytes);
        if (!f) {
            throw std::runtime_error("GGUF: Read failed for tensor: " + info.name +
                " (needed " + std::to_string(raw_bytes) + " bytes at offset " +
                std::to_string(abs_offset) + ")");
        }

        // Create output tensor with PyTorch-convention shape
        auto shape = info.shape();
        at::Tensor tensor = at::empty(shape);
        float* dst = tensor.mutable_data_ptr<float>();

        // Dequantize
        dequantize(info.type, raw.data(), dst, n_elements);

        return tensor;
    }

    // ========================================================================
    // Load all tensors (with progress)
    // ========================================================================

    std::unordered_map<std::string, at::Tensor> load_all_tensors(bool verbose = true) const {
        std::unordered_map<std::string, at::Tensor> result;
        size_t loaded = 0;

        for (const auto& info : tensors) {
            if (!is_type_supported(info.type)) {
                if (verbose) {
                    std::cout << "[GGUF] Skipping unsupported type: " << info.name
                              << " (" << ggml_type_name(info.type) << ")" << std::endl;
                }
                continue;
            }

            result[info.name] = load_tensor(info);
            loaded++;

            if (verbose && loaded % 20 == 0) {
                std::cout << "[GGUF] Loaded " << loaded << "/" << tensors.size()
                          << " tensors..." << std::endl;
            }
        }

        if (verbose) {
            std::cout << "[GGUF] Done. Loaded " << loaded << " tensors." << std::endl;
        }
        return result;
    }

    // ========================================================================
    // Print summary
    // ========================================================================

    void print_metadata() const {
        std::cout << "\n=== GGUF Metadata ===" << std::endl;
        for (const auto& [key, val] : metadata) {
            std::cout << "  " << key << " = ";
            print_value(val);
            std::cout << std::endl;
        }
    }

    void print_tensors() const {
        std::cout << "\n=== GGUF Tensors (" << tensors.size() << ") ===" << std::endl;
        for (const auto& t : tensors) {
            std::cout << "  " << t.name << " [";
            auto shape = t.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << shape[i];
            }
            std::cout << "] " << ggml_type_name(t.type)
                      << " (" << t.data_bytes() << " bytes)" << std::endl;
        }
    }

private:
    // ========================================================================
    // Binary reading helpers
    // ========================================================================

    template<typename T>
    static T read_val(std::ifstream& f) {
        T val;
        f.read(reinterpret_cast<char*>(&val), sizeof(T));
        if (!f) throw std::runtime_error("GGUF: Unexpected end of file");
        return val;
    }

    static std::string read_string(std::ifstream& f) {
        uint64_t len = read_val<uint64_t>(f);
        if (len > 1024 * 1024) {  // Sanity check: 1MB max string
            throw std::runtime_error("GGUF: String too long: " + std::to_string(len));
        }
        std::string s(len, '\0');
        f.read(s.data(), len);
        if (!f) throw std::runtime_error("GGUF: Failed to read string");
        return s;
    }

    static uint64_t align_offset(uint64_t offset, uint64_t alignment) {
        return ((offset + alignment - 1) / alignment) * alignment;
    }

    // ========================================================================
    // Parse header
    // ========================================================================

    void parse_header(std::ifstream& f) {
        uint32_t magic = read_val<uint32_t>(f);
        if (magic != GGUF_MAGIC) {
            throw std::runtime_error("GGUF: Invalid magic number (not a GGUF file)");
        }

        version = read_val<uint32_t>(f);
        if (version < 2 || version > 3) {
            throw std::runtime_error("GGUF: Unsupported version: " + std::to_string(version));
        }

        tensor_count = read_val<uint64_t>(f);
        metadata_kv_count = read_val<uint64_t>(f);
    }

    // ========================================================================
    // Parse metadata key-value pairs
    // ========================================================================

    GGUFValue read_value(std::ifstream& f, GGUFValueType type) {
        GGUFValue val;
        val.type = type;

        switch (type) {
            case GGUFValueType::UINT8:
                val.u64 = read_val<uint8_t>(f);
                break;
            case GGUFValueType::INT8:
                val.i64 = read_val<int8_t>(f);
                break;
            case GGUFValueType::UINT16:
                val.u64 = read_val<uint16_t>(f);
                break;
            case GGUFValueType::INT16:
                val.i64 = read_val<int16_t>(f);
                break;
            case GGUFValueType::UINT32:
                val.u64 = read_val<uint32_t>(f);
                break;
            case GGUFValueType::INT32:
                val.i64 = read_val<int32_t>(f);
                break;
            case GGUFValueType::FLOAT32:
                val.f64 = read_val<float>(f);
                break;
            case GGUFValueType::BOOL:
                val.b = read_val<uint8_t>(f) != 0;
                break;
            case GGUFValueType::STRING:
                val.str = read_string(f);
                break;
            case GGUFValueType::UINT64:
                val.u64 = read_val<uint64_t>(f);
                break;
            case GGUFValueType::INT64:
                val.i64 = read_val<int64_t>(f);
                break;
            case GGUFValueType::FLOAT64:
                val.f64 = read_val<double>(f);
                break;
            case GGUFValueType::ARRAY: {
                GGUFValueType elem_type = static_cast<GGUFValueType>(read_val<uint32_t>(f));
                uint64_t count = read_val<uint64_t>(f);
                val.arr.reserve(static_cast<size_t>((std::min)(count, uint64_t(1000000))));
                for (uint64_t i = 0; i < count; ++i) {
                    val.arr.push_back(read_value(f, elem_type));
                }
                break;
            }
            default:
                throw std::runtime_error("GGUF: Unknown value type: " +
                    std::to_string(static_cast<uint32_t>(type)));
        }
        return val;
    }

    void parse_metadata(std::ifstream& f) {
        for (uint64_t i = 0; i < metadata_kv_count; ++i) {
            std::string key = read_string(f);
            GGUFValueType type = static_cast<GGUFValueType>(read_val<uint32_t>(f));
            GGUFValue val = read_value(f, type);
            metadata[key] = std::move(val);
        }
    }

    // ========================================================================
    // Parse tensor info entries
    // ========================================================================

    void parse_tensor_infos(std::ifstream& f) {
        tensors.resize(tensor_count);
        for (uint64_t i = 0; i < tensor_count; ++i) {
            auto& t = tensors[i];
            t.name = read_string(f);

            uint32_t n_dims = read_val<uint32_t>(f);
            t.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                t.dims[d] = read_val<uint64_t>(f);
            }

            t.type = static_cast<GGMLType>(read_val<uint32_t>(f));
            t.offset = read_val<uint64_t>(f);

            tensor_index[t.name] = static_cast<size_t>(i);
        }
    }

    // ========================================================================
    // Print helpers
    // ========================================================================

    static void print_value(const GGUFValue& val) {
        switch (val.type) {
            case GGUFValueType::UINT8:
            case GGUFValueType::UINT16:
            case GGUFValueType::UINT32:
            case GGUFValueType::UINT64:
                std::cout << val.u64;
                break;
            case GGUFValueType::INT8:
            case GGUFValueType::INT16:
            case GGUFValueType::INT32:
            case GGUFValueType::INT64:
                std::cout << val.i64;
                break;
            case GGUFValueType::FLOAT32:
            case GGUFValueType::FLOAT64:
                std::cout << val.f64;
                break;
            case GGUFValueType::BOOL:
                std::cout << (val.b ? "true" : "false");
                break;
            case GGUFValueType::STRING:
                if (val.str.size() <= 80) {
                    std::cout << "\"" << val.str << "\"";
                } else {
                    std::cout << "\"" << val.str.substr(0, 77) << "...\" (" << val.str.size() << " chars)";
                }
                break;
            case GGUFValueType::ARRAY:
                std::cout << "[array, " << val.arr.size() << " elements]";
                break;
            default:
                std::cout << "<unknown>";
        }
    }
};

} // namespace gguf
} // namespace io
} // namespace torch
