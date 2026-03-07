#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <string>
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <cstring>

namespace torch {

using at::Tensor;
using StateDict = std::unordered_map<std::string, Tensor>;

// ============================================================================
// Binary Format:
//   Magic: "PTOR" (4 bytes)
//   Version: uint32_t
//   Tensor count: uint64_t
//   Per tensor:
//     name_len: uint32_t
//     name: char[name_len]
//     dtype: uint8_t
//     ndim: uint32_t
//     sizes: int64_t[ndim]
//     data_nbytes: uint64_t
//     data: raw bytes (contiguous)
// ============================================================================

namespace detail {

constexpr const char MAGIC[4] = {'P', 'T', 'O', 'R'};
constexpr uint32_t FORMAT_VERSION = 1;

inline void write_bytes(std::ofstream& f, const void* data, size_t n) {
    f.write(static_cast<const char*>(data), n);
    if (!f) throw std::runtime_error("serialization: write failed");
}

inline void read_bytes(std::ifstream& f, void* data, size_t n) {
    f.read(static_cast<char*>(data), n);
    if (!f) throw std::runtime_error("serialization: read failed (unexpected EOF)");
}

inline void write_tensor(std::ofstream& f, const std::string& name, const Tensor& tensor) {
    // Name
    uint32_t name_len = static_cast<uint32_t>(name.size());
    write_bytes(f, &name_len, sizeof(name_len));
    write_bytes(f, name.data(), name_len);

    // Dtype
    uint8_t dtype = static_cast<uint8_t>(tensor.dtype());
    write_bytes(f, &dtype, sizeof(dtype));

    // Dimensions
    uint32_t ndim = static_cast<uint32_t>(tensor.dim());
    write_bytes(f, &ndim, sizeof(ndim));

    // Sizes
    for (uint32_t i = 0; i < ndim; ++i) {
        int64_t s = tensor.size(i);
        write_bytes(f, &s, sizeof(s));
    }

    // Data (always contiguous)
    Tensor contig = tensor.contiguous();
    uint64_t nbytes = static_cast<uint64_t>(contig.nbytes());
    write_bytes(f, &nbytes, sizeof(nbytes));
    write_bytes(f, contig.data_ptr(), nbytes);
}

inline std::pair<std::string, Tensor> read_tensor(std::ifstream& f) {
    // Name
    uint32_t name_len;
    read_bytes(f, &name_len, sizeof(name_len));
    std::string name(name_len, '\0');
    read_bytes(f, name.data(), name_len);

    // Dtype
    uint8_t dtype_raw;
    read_bytes(f, &dtype_raw, sizeof(dtype_raw));
    auto dtype = static_cast<c10::ScalarType>(dtype_raw);

    // Dimensions
    uint32_t ndim;
    read_bytes(f, &ndim, sizeof(ndim));

    // Sizes
    std::vector<int64_t> sizes(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        read_bytes(f, &sizes[i], sizeof(int64_t));
    }

    // Data
    uint64_t nbytes;
    read_bytes(f, &nbytes, sizeof(nbytes));

    Tensor tensor = at::empty(sizes, at::TensorOptions().dtype(dtype));
    read_bytes(f, tensor.data_ptr(), nbytes);

    return {name, tensor};
}

} // namespace detail

// ============================================================================
// Public API
// ============================================================================

// Save a single tensor
inline void save(const Tensor& tensor, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save: cannot open file " + path);

    detail::write_bytes(f, detail::MAGIC, 4);
    detail::write_bytes(f, &detail::FORMAT_VERSION, sizeof(uint32_t));

    uint64_t count = 1;
    detail::write_bytes(f, &count, sizeof(count));

    detail::write_tensor(f, "tensor", tensor);
}

// Load a single tensor
inline Tensor load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load: cannot open file " + path);

    char magic[4];
    detail::read_bytes(f, magic, 4);
    if (std::memcmp(magic, detail::MAGIC, 4) != 0) {
        throw std::runtime_error("load: invalid file format (bad magic)");
    }

    uint32_t version;
    detail::read_bytes(f, &version, sizeof(version));
    if (version != detail::FORMAT_VERSION) {
        throw std::runtime_error("load: unsupported format version");
    }

    uint64_t count;
    detail::read_bytes(f, &count, sizeof(count));

    auto [name, tensor] = detail::read_tensor(f);
    return tensor;
}

// Save a state dict (name -> tensor map)
inline void save_state_dict(const StateDict& state_dict, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_state_dict: cannot open file " + path);

    detail::write_bytes(f, detail::MAGIC, 4);
    detail::write_bytes(f, &detail::FORMAT_VERSION, sizeof(uint32_t));

    uint64_t count = static_cast<uint64_t>(state_dict.size());
    detail::write_bytes(f, &count, sizeof(count));

    for (const auto& [name, tensor] : state_dict) {
        detail::write_tensor(f, name, tensor);
    }
}

// Load a state dict
inline StateDict load_state_dict(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_state_dict: cannot open file " + path);

    char magic[4];
    detail::read_bytes(f, magic, 4);
    if (std::memcmp(magic, detail::MAGIC, 4) != 0) {
        throw std::runtime_error("load_state_dict: invalid file format (bad magic)");
    }

    uint32_t version;
    detail::read_bytes(f, &version, sizeof(version));
    if (version != detail::FORMAT_VERSION) {
        throw std::runtime_error("load_state_dict: unsupported format version");
    }

    uint64_t count;
    detail::read_bytes(f, &count, sizeof(count));

    StateDict state;
    for (uint64_t i = 0; i < count; ++i) {
        auto [name, tensor] = detail::read_tensor(f);
        state[name] = tensor;
    }

    return state;
}

} // namespace torch
