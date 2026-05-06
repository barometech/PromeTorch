// ============================================================================
// pt8_reader.h — header-only mmap-based reader for PromeTorch .pt8 files.
//
// Agent C (Round 4 Mission, 2026-04-30). Pairs with:
//   - Agent A spec:    vliw_mission/round4_lossless_30/format_spec_v1.md
//   - Agent B writer:  tools/gguf2pt8/converter.{h,cpp}
//
// On-disk layout (matches what tools/gguf2pt8/converter.cpp actually writes —
// which deviates from Agent A's draft v1 §3 fixed-96-byte spec in favour of a
// variable-length tail table; the reader implementation here is the
// authoritative interpretation that BOTH writer and runtime must agree on):
//
//   +-------------------------------------------+ offset 0
//   | FILE HEADER (256 bytes)                   |
//   |   off  size  field                        |
//   |     0   u32   magic = 'PT8\0' (0x00385450)|
//   |     4   u32   version = 1                 |
//   |     8   u64   flags (currently 0)         |
//   |    16   u64   tensor_table_offset         |
//   |    24   u64   tensor_count                |
//   |   32..255 reserved (zero)                 |
//   +-------------------------------------------+
//   |  TENSOR DATA (each tensor 64-byte aligned,|
//   |  written in submit-order by writer thread)|
//   +-------------------------------------------+ tensor_table_offset
//   |  TENSOR TABLE (variable length, packed)   |
//   |   for each of `tensor_count` entries:     |
//   |     u32   name_length                     |
//   |     u8[]  name (UTF-8, no NUL)            |
//   |     u32   pt8_type                        |
//   |     u32   n_dims                          |
//   |     u64[] dims      (n_dims × 8 B)        |
//   |     u64   data_offset (absolute)          |
//   |     u64   data_size                       |
//   |     u64   row_stride (bytes per row, 0 N/A)|
//   |     u32   meta_length                     |
//   |     u8[]  meta_blob (per-encoder side data)|
//   +-------------------------------------------+ EOF
//
// Lifetime / ownership:
//   open() mmaps the whole file, parses header + tail table, and builds a
//   name -> entry hash map. tensor_data() returns a zero-copy const pointer
//   into the mmap region; the pointer is valid for the lifetime of the
//   PT8Reader.
//
// NUMA awareness:
//   The mmap itself is a single per-process MAP_PRIVATE handle. Per-NUMA
//   replication of hot weights is performed by the caller (gguf_model.h
//   replicate_weights_for_numa) via memcpy into node-bound buffers, identical
//   to the GGUF path. We DO NOT do mbind here — caller decides policy via
//   PT_NUMA_REPLICATE env (default OFF preserves zero-copy semantics).
// ============================================================================

#pragma once

#include "torch/io/gguf_loader.h"   // gguf::MmapHandle (cross-platform mmap)

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace torch {
namespace io {

// ----------------------------------------------------------------------------
// Format constants — must agree with prometorch::convert::PT8_* in
// tools/gguf2pt8/converter.h. Duplicated here so this header has no
// dependency on the converter library (loader can be built without the
// converter binary).
// ----------------------------------------------------------------------------

constexpr uint32_t PT8_MAGIC          = 0x00385450u;  // "PT8\0" little-endian
constexpr uint32_t PT8_VERSION        = 1u;
constexpr size_t   PT8_HEADER_BYTES   = 256;
constexpr size_t   PT8_DATA_ALIGNMENT = 64;

// pt8 dtype tags — keep in sync with prometorch::convert::Pt8Type.
enum Pt8Type : uint32_t {
    PT8_TYPE_F32           = 0,
    PT8_TYPE_F16           = 1,
    PT8_TYPE_BF16          = 2,

    // Reserved Agent-A primary lossless layouts (registered by encoders/).
    // The numeric codes are stable contracts; if a writer emits one of these,
    // every reader in the same minor version must accept it.
    PT8_TYPE_Q4K_SOA4      = 100,  // 4-row interleaved Q4 (PT8_Q4_SOA4 in spec §10)
    PT8_TYPE_Q6K_NATIVE    = 101,
    PT8_TYPE_Q5K_NATIVE    = 102,
    PT8_TYPE_Q8_0_SOA4     = 103,  // 4-row interleaved Q8 (Q8_SoA4, ready path)
    PT8_TYPE_Q4_0_SOA4     = 104,

    PT8_TYPE_UNKNOWN       = 0xFFFFFFFFu,
};

// ----------------------------------------------------------------------------
// In-memory header / record types
// ----------------------------------------------------------------------------

struct PT8Header {
    uint32_t magic   = 0;
    uint32_t version = 0;
    uint64_t flags   = 0;
    uint64_t tensor_table_offset = 0;
    uint64_t tensor_count = 0;
};

struct PT8TensorRecord {
    std::string          name;
    uint32_t             pt8_type   = PT8_TYPE_UNKNOWN;
    std::vector<int64_t> dims;
    uint64_t             data_offset = 0;
    uint64_t             data_size   = 0;
    uint64_t             row_stride  = 0;
    std::vector<uint8_t> meta_blob;
};

// ----------------------------------------------------------------------------
// PT8Reader — owns an MmapHandle, returns zero-copy const pointers.
// ----------------------------------------------------------------------------

class PT8Reader {
public:
    PT8Reader() = default;
    PT8Reader(const PT8Reader&) = delete;
    PT8Reader& operator=(const PT8Reader&) = delete;
    PT8Reader(PT8Reader&&) noexcept = default;
    PT8Reader& operator=(PT8Reader&&) noexcept = default;

    // Open a .pt8 file. Returns false on bad magic / version / out-of-range
    // table. On success header() and tensors() are populated.
    bool open(const std::string& path) {
        if (!mmap_.open(path)) {
            std::cerr << "[pt8] mmap open failed: " << path << std::endl;
            return false;
        }
        if (mmap_.size() < PT8_HEADER_BYTES) {
            std::cerr << "[pt8] file smaller than header (" << mmap_.size()
                      << " < " << PT8_HEADER_BYTES << ")" << std::endl;
            return false;
        }
        const uint8_t* base = static_cast<const uint8_t*>(mmap_.data());

        std::memcpy(&header_.magic,                base + 0,  4);
        std::memcpy(&header_.version,              base + 4,  4);
        std::memcpy(&header_.flags,                base + 8,  8);
        std::memcpy(&header_.tensor_table_offset,  base + 16, 8);
        std::memcpy(&header_.tensor_count,         base + 24, 8);

        if (header_.magic != PT8_MAGIC) {
            std::cerr << "[pt8] bad magic 0x" << std::hex << header_.magic
                      << std::dec << " (expected 0x" << std::hex << PT8_MAGIC
                      << std::dec << ")" << std::endl;
            return false;
        }
        if (header_.version != PT8_VERSION) {
            std::cerr << "[pt8] version mismatch: file=" << header_.version
                      << " reader=" << PT8_VERSION << std::endl;
            return false;
        }
        if (header_.tensor_table_offset == 0 ||
            header_.tensor_table_offset >= mmap_.size()) {
            std::cerr << "[pt8] tensor_table_offset out of range: "
                      << header_.tensor_table_offset
                      << " (file_size=" << mmap_.size() << ")" << std::endl;
            return false;
        }

        // Walk the variable-length tail table.
        const uint8_t* p   = base + header_.tensor_table_offset;
        const uint8_t* end = base + mmap_.size();
        records_.reserve(static_cast<size_t>(header_.tensor_count));

        for (uint64_t i = 0; i < header_.tensor_count; ++i) {
            if (!parse_record_(p, end)) {
                std::cerr << "[pt8] tail-table parse failed at entry " << i
                          << std::endl;
                return false;
            }
        }

        // Build name → record map for O(1) lookup.
        for (size_t i = 0; i < records_.size(); ++i) {
            by_name_[records_[i].name] = &records_[i];
        }
        return true;
    }

    bool is_open() const { return mmap_.is_open(); }
    const PT8Header& header() const { return header_; }
    const std::vector<PT8TensorRecord>& tensors() const { return records_; }
    size_t mmap_size() const { return mmap_.size(); }
    const void* mmap_data() const { return mmap_.data(); }

    // Returns a pointer to the named tensor's data, or nullptr if not found.
    // Pointer is valid for the lifetime of this PT8Reader.
    const void* tensor_data(const std::string& name) const {
        auto it = by_name_.find(name);
        if (it == by_name_.end()) return nullptr;
        const PT8TensorRecord* r = it->second;
        if (!mmap_.is_open()) return nullptr;
        if (r->data_offset + r->data_size > mmap_.size()) return nullptr;
        return static_cast<const uint8_t*>(mmap_.data()) + r->data_offset;
    }

    // Returns the tensor's data size in bytes, or 0 if not found.
    size_t tensor_size(const std::string& name) const {
        auto it = by_name_.find(name);
        if (it == by_name_.end()) return 0;
        return static_cast<size_t>(it->second->data_size);
    }

    // Returns nullptr if not found. Useful for scanning / dispatch.
    const PT8TensorRecord* find(const std::string& name) const {
        auto it = by_name_.find(name);
        return (it == by_name_.end()) ? nullptr : it->second;
    }

    bool has(const std::string& name) const {
        return by_name_.find(name) != by_name_.end();
    }

    // Best-effort detection: open `path`, peek 8 bytes, return true if
    // magic + version match. Used by gguf_model.h to auto-detect format
    // when PT_FORMAT_AUTO=1 (default).
    static bool is_pt8_file(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        uint8_t hdr[8] = {0};
        size_t n = std::fread(hdr, 1, 8, f);
        std::fclose(f);
        if (n < 8) return false;
        uint32_t magic = 0, version = 0;
        std::memcpy(&magic,   hdr + 0, 4);
        std::memcpy(&version, hdr + 4, 4);
        return magic == PT8_MAGIC && version == PT8_VERSION;
    }

    // Lock the named tensor's mmap range in physical RAM (mlock/VirtualLock).
    // Returns the number of bytes locked on success, 0 on failure.
    size_t lock_tensor(const std::string& name) const {
        auto it = by_name_.find(name);
        if (it == by_name_.end()) return 0;
        const void* p = tensor_data(name);
        if (!p) return 0;
        return mmap_.lock_region(p, static_cast<size_t>(it->second->data_size))
                   ? static_cast<size_t>(it->second->data_size)
                   : 0;
    }

private:
    // Variable-length record parser. Advances `p` past the record on success.
    bool parse_record_(const uint8_t*& p, const uint8_t* end) {
        PT8TensorRecord r;

        if (p + 4 > end) return false;
        uint32_t name_len = 0; std::memcpy(&name_len, p, 4); p += 4;
        if (p + name_len > end) return false;
        r.name.assign(reinterpret_cast<const char*>(p), name_len);
        p += name_len;

        if (p + 4 > end) return false;
        std::memcpy(&r.pt8_type, p, 4); p += 4;

        if (p + 4 > end) return false;
        uint32_t n_dims = 0; std::memcpy(&n_dims, p, 4); p += 4;
        if (n_dims > 8) return false;  // sanity: no tensor has > 8 dims
        if (p + 8 * n_dims > end) return false;
        r.dims.resize(n_dims);
        for (uint32_t i = 0; i < n_dims; ++i) {
            uint64_t d = 0; std::memcpy(&d, p, 8); p += 8;
            r.dims[i] = static_cast<int64_t>(d);
        }

        if (p + 24 > end) return false;
        std::memcpy(&r.data_offset, p +  0, 8);
        std::memcpy(&r.data_size,   p +  8, 8);
        std::memcpy(&r.row_stride,  p + 16, 8);
        p += 24;

        if (p + 4 > end) return false;
        uint32_t meta_len = 0; std::memcpy(&meta_len, p, 4); p += 4;
        if (p + meta_len > end) return false;
        if (meta_len > 0) {
            r.meta_blob.assign(p, p + meta_len);
            p += meta_len;
        }

        records_.push_back(std::move(r));
        return true;
    }

    gguf::MmapHandle              mmap_;
    PT8Header                     header_{};
    std::vector<PT8TensorRecord>  records_;
    // Pointers into records_ — stable because we reserve() up-front and
    // don't grow after open(). Re-built per open().
    std::unordered_map<std::string, const PT8TensorRecord*> by_name_;
};

// ----------------------------------------------------------------------------
// Helpers for the GGUFModel integration (kept in this header so the loader
// is self-contained).
// ----------------------------------------------------------------------------

// PT8 → ggml type id. Used to drive the existing GEMV dispatch when a PT8
// tensor stores raw GGML quant blocks (passthrough variants, e.g. Q4_K).
// Returns 0 (GGML_TYPE_F32) for unsupported / opaque PT8-native layouts;
// caller should branch on pt8_type instead in those cases.
inline uint32_t pt8_type_to_ggml(uint32_t pt8_type) {
    switch (pt8_type) {
        case PT8_TYPE_F32:  return 0;   // GGML_TYPE_F32
        case PT8_TYPE_F16:  return 1;   // GGML_TYPE_F16
        case PT8_TYPE_BF16: return 30;  // GGML_TYPE_BF16
        // SoA4 layouts have no GGML counterpart — caller must use them
        // through the Q8SoA4 / Q4SoA4 specific path.
        default:            return 0xFFFFFFFFu;
    }
}

// Pretty-printable name. Used in [pt8] log lines.
inline const char* pt8_type_name(uint32_t pt8_type) {
    switch (pt8_type) {
        case PT8_TYPE_F32:        return "F32";
        case PT8_TYPE_F16:        return "F16";
        case PT8_TYPE_BF16:       return "BF16";
        case PT8_TYPE_Q4K_SOA4:   return "Q4K_SOA4";
        case PT8_TYPE_Q6K_NATIVE: return "Q6K_NATIVE";
        case PT8_TYPE_Q5K_NATIVE: return "Q5K_NATIVE";
        case PT8_TYPE_Q8_0_SOA4:  return "Q8_0_SOA4";
        case PT8_TYPE_Q4_0_SOA4:  return "Q4_0_SOA4";
        default:                  return "?";
    }
}

// Returns true if the pt8_type stores its data in the Q8_SoA4 byte layout
// described in q8_soa_repack.h (176 B / super-row block × 32 K-elems × 4
// rows). When true, the loader can mmap the bytes directly into a
// cpu_quant::Q8SoA4 view without re-packing — saving the ~7 s repack step.
inline bool pt8_is_q8_soa4_layout(uint32_t pt8_type) {
    return pt8_type == PT8_TYPE_Q8_0_SOA4;
}

// Returns true if the pt8_type stores its data in the Q4_SoA4 byte layout
// described in format_spec_v1.md §10 (88 B / 4-row × 32-K block, 0.6875
// B/param). Reserved for the Round 4 ⭐ hot-path; caller must check this
// before assuming Q8 SoA4 byte-equivalence.
inline bool pt8_is_q4_soa4_layout(uint32_t pt8_type) {
    return pt8_type == PT8_TYPE_Q4K_SOA4 || pt8_type == PT8_TYPE_Q4_0_SOA4;
}

}  // namespace io
}  // namespace torch
