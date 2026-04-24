#pragma once

// ============================================================================
// GGUF file loader — header parser, tensor-table reader, mmap backend.
// The GGUF container format (magic "GGUF", metadata KV, tensor descriptors,
// alignment rules) is specified by GGML/llama.cpp (MIT). Layout here follows
// that spec; see THIRD_PARTY_NOTICES.md.
// ============================================================================

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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

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
// Memory-Mapped File Handle
// Maps GGUF file into virtual address space for zero-copy weight access.
// OS handles paging — only accessed pages consume physical RAM.
// Multiple processes sharing the same file share physical pages (CoW).
// ============================================================================

class MmapHandle {
    void* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = NULL;
#else
    int fd_ = -1;
#endif

public:
    MmapHandle() = default;
    MmapHandle(const MmapHandle&) = delete;
    MmapHandle& operator=(const MmapHandle&) = delete;

    MmapHandle(MmapHandle&& other) noexcept
        : data_(other.data_), size_(other.size_)
#ifdef _WIN32
        , file_handle_(other.file_handle_), mapping_(other.mapping_)
#else
        , fd_(other.fd_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#ifdef _WIN32
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.mapping_ = NULL;
#else
        other.fd_ = -1;
#endif
    }

    MmapHandle& operator=(MmapHandle&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
#ifdef _WIN32
            file_handle_ = other.file_handle_;
            mapping_ = other.mapping_;
            other.file_handle_ = INVALID_HANDLE_VALUE;
            other.mapping_ = NULL;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    bool open(const std::string& path) {
        close();  // Close any previous mapping

#ifdef _WIN32
        // Windows: CreateFile → CreateFileMapping → MapViewOfFile
        file_handle_ = CreateFileA(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_SEQUENTIAL_SCAN,  // Hint: sequential access pattern
            nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            std::cerr << "[mmap] CreateFile failed: " << GetLastError() << std::endl;
            return false;
        }

        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_handle_, &file_size)) {
            std::cerr << "[mmap] GetFileSizeEx failed: " << GetLastError() << std::endl;
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
        size_ = static_cast<size_t>(file_size.QuadPart);

        mapping_ = CreateFileMappingA(
            file_handle_,
            nullptr,
            PAGE_READONLY,
            static_cast<DWORD>(size_ >> 32),
            static_cast<DWORD>(size_ & 0xFFFFFFFF),
            nullptr);
        if (!mapping_) {
            std::cerr << "[mmap] CreateFileMapping failed: " << GetLastError() << std::endl;
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }

        data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (!data_) {
            std::cerr << "[mmap] MapViewOfFile failed: " << GetLastError() << std::endl;
            CloseHandle(mapping_);
            mapping_ = NULL;
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }

        return true;
#else
        // POSIX: open → fstat → mmap
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::cerr << "[mmap] open() failed: " << strerror(errno) << std::endl;
            return false;
        }

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            std::cerr << "[mmap] fstat() failed: " << strerror(errno) << std::endl;
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);

        data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            std::cerr << "[mmap] mmap() failed: " << strerror(errno) << std::endl;
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        // Hint: random access. LLM decode visits every weight matrix once per
        // token, so the hot access pattern is per-decode-step random across
        // 2.5 GB, NOT a single linear scan. MADV_SEQUENTIAL tells the kernel
        // to AGGRESSIVELY DROP pages behind the cursor — exactly wrong for us
        // because the same pages are revisited on the next token's forward.
        // (round2 agent_7 finding, gguf_loader.h:261).
        madvise(data_, size_, MADV_RANDOM);

        return true;
#endif
    }

    void close() {
        if (!data_) return;

#ifdef _WIN32
        UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        mapping_ = NULL;
#else
        ::munmap(data_, size_);
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
        data_ = nullptr;
        size_ = 0;
    }

    // Lock a region in physical RAM (prevents paging out)
    // Use for critical weights: first/last layer norms, output projection
    bool lock_region(const void* ptr, size_t len) const {
        if (!data_ || !ptr) return false;
        // Verify ptr is within our mapping
        const char* base = static_cast<const char*>(data_);
        const char* p = static_cast<const char*>(ptr);
        if (p < base || p + len > base + size_) return false;

#ifdef _WIN32
        // VirtualLock requires pages to be committed (mmap'd pages are)
        // May fail if working set quota exceeded — not fatal
        if (!VirtualLock(const_cast<void*>(ptr), len)) {
            // Non-fatal: just means OS may page it out under pressure
            return false;
        }
        return true;
#else
        if (mlock(ptr, len) != 0) {
            return false;
        }
        return true;
#endif
    }

    const void* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

    // Get pointer at offset within the mapped file
    const void* at_offset(uint64_t offset) const {
        if (!data_ || offset >= size_) return nullptr;
        return static_cast<const char*>(data_) + offset;
    }

    ~MmapHandle() { close(); }
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

    // Memory-mapped file handle (optional, for zero-copy weight access)
    MmapHandle mmap_handle_;

    // ========================================================================
    // Memory-map the GGUF file for zero-copy tensor access
    // Must be called AFTER open() (needs data_offset to be computed)
    // Returns true on success. On failure, falls back to read-based loading.
    // ========================================================================

    bool mmap_file() {
        if (file_path.empty()) {
            std::cerr << "[mmap] No file path — call open() first" << std::endl;
            return false;
        }
        if (mmap_handle_.is_open()) return true;  // Already mapped

        if (!mmap_handle_.open(file_path)) {
            std::cerr << "[mmap] Failed to mmap: " << file_path << std::endl;
            return false;
        }

        std::cout << "[mmap] Mapped " << (mmap_handle_.size() / (1024*1024))
                  << " MB: " << file_path << std::endl;
        return true;
    }

    // ========================================================================
    // Get direct pointer to tensor data within mmap'd region (zero-copy)
    // Returns nullptr if mmap not active or tensor not found
    // ========================================================================

    const void* get_tensor_data_ptr(const std::string& name) const {
        if (!mmap_handle_.is_open()) return nullptr;
        if (!has_tensor(name)) return nullptr;
        const auto& info = get_tensor_info(name);
        uint64_t abs_offset = data_offset + info.offset;
        return mmap_handle_.at_offset(abs_offset);
    }

    // Get tensor data size in bytes
    int64_t get_tensor_data_bytes(const std::string& name) const {
        if (!has_tensor(name)) return 0;
        return get_tensor_info(name).data_bytes();
    }

    // Lock tensor data in physical RAM (mlock/VirtualLock)
    bool lock_tensor(const std::string& name) const {
        const void* ptr = get_tensor_data_ptr(name);
        if (!ptr) return false;
        int64_t bytes = get_tensor_data_bytes(name);
        return mmap_handle_.lock_region(ptr, static_cast<size_t>(bytes));
    }

    bool is_mmap_active() const { return mmap_handle_.is_open(); }

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
    // Load raw tensor bytes (no dequantization) for GPU quantized inference
    // Returns raw quantized bytes + quant type + shape
    // ========================================================================

    struct RawTensorData {
        std::vector<uint8_t> data;
        GGMLType type;
        std::vector<int64_t> shape;  // PyTorch convention [rows, cols]
        int64_t n_elements;
    };

    RawTensorData load_raw_tensor(const std::string& name) const {
        const auto& info = get_tensor_info(name);
        int64_t raw_bytes = info.data_bytes();

        std::ifstream f(file_path, std::ios::binary);
        if (!f) throw std::runtime_error("GGUF: Cannot reopen file: " + file_path);

        uint64_t abs_offset = data_offset + info.offset;
        f.seekg(static_cast<std::streamoff>(abs_offset));

        RawTensorData result;
        result.data.resize(raw_bytes);
        f.read(reinterpret_cast<char*>(result.data.data()), raw_bytes);
        if (!f) throw std::runtime_error("GGUF: Read failed for raw tensor: " + info.name);

        result.type = info.type;
        result.shape = info.shape();
        result.n_elements = info.n_elements();
        return result;
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
