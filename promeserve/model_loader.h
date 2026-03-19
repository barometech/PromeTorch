#pragma once

// ============================================================================
// PromeServe — Universal Model Loader
//
// Supports loading model weights from multiple formats:
//   1. GGUF       — .gguf files, Ollama cache (via existing GGUFReader)
//   2. SafeTensors — HuggingFace .safetensors (JSON header + mmap-friendly data)
//   3. PyTorch     — .bin/.pt files (zip-based, pickle-free raw tensor read)
//   4. ONNX        — .onnx files (protobuf-lite header, weight extraction)
//
// Auto-detects format by magic bytes. No external dependencies.
// ============================================================================

#include "torch/io/gguf_loader.h"
#include "aten/src/ATen/ATen.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

namespace promeserve {

// ============================================================================
// Common data types across formats
// ============================================================================

enum class TensorDType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    FLOAT64,
    BOOL,
    UNKNOWN
};

static size_t dtype_size(TensorDType dt) {
    switch (dt) {
        case TensorDType::FLOAT32:  return 4;
        case TensorDType::FLOAT16:  return 2;
        case TensorDType::BFLOAT16: return 2;
        case TensorDType::INT8:     return 1;
        case TensorDType::INT16:    return 2;
        case TensorDType::UINT8:    return 1;
        case TensorDType::BOOL:     return 1;
        case TensorDType::INT32:    return 4;
        case TensorDType::INT64:    return 8;
        case TensorDType::FLOAT64:  return 8;
        default: return 0;
    }
}

static const char* dtype_name(TensorDType dt) {
    switch (dt) {
        case TensorDType::FLOAT32:  return "F32";
        case TensorDType::FLOAT16:  return "F16";
        case TensorDType::BFLOAT16: return "BF16";
        case TensorDType::INT8:     return "I8";
        case TensorDType::INT16:    return "I16";
        case TensorDType::UINT8:    return "U8";
        case TensorDType::INT32:    return "I32";
        case TensorDType::INT64:    return "I64";
        case TensorDType::FLOAT64:  return "F64";
        case TensorDType::BOOL:     return "BOOL";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// TensorInfo — metadata for a single tensor in any format
// ============================================================================

struct TensorInfo {
    std::string name;
    TensorDType dtype = TensorDType::UNKNOWN;
    std::vector<int64_t> shape;
    uint64_t data_offset = 0;    // byte offset into data section
    uint64_t data_size = 0;      // byte count

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

// ============================================================================
// ModelConfig — architecture parameters extracted from model metadata
// ============================================================================

struct ModelConfig {
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t intermediate_size = 0;
    int64_t vocab_size = 0;
    int64_t head_dim = 0;
    int64_t context_length = 0;
    float rms_norm_eps = 1e-6f;
    float rope_freq_base = 10000.0f;
    bool tie_word_embeddings = false;

    std::map<std::string, std::string> extra;  // format-specific metadata
};

// ============================================================================
// ModelWeights — the universal result of loading any model format
// ============================================================================

struct ModelWeights {
    std::string architecture;  // llama, qwen2, gemma, mistral, etc.
    std::string format;        // "gguf", "safetensors", "pytorch", "onnx"
    std::string source_path;

    std::map<std::string, TensorInfo> tensors;  // name -> metadata
    ModelConfig config;

    // The file handle + data base offset for lazy tensor loading
    std::string file_path;
    uint64_t data_base_offset = 0;

    // For multi-file models (sharded safetensors / pytorch bins)
    std::vector<std::string> shard_files;

    bool has_tensor(const std::string& name) const {
        return tensors.find(name) != tensors.end();
    }

    void print_summary() const {
        std::cout << "[ModelLoader] Format: " << format
                  << ", Architecture: " << architecture
                  << ", Tensors: " << tensors.size() << std::endl;
        if (config.hidden_size > 0)
            std::cout << "  hidden_size=" << config.hidden_size
                      << " layers=" << config.num_layers
                      << " heads=" << config.num_heads
                      << " vocab=" << config.vocab_size << std::endl;
    }
};

// Forward declarations (defined below, used by format loaders)
static std::string infer_architecture(const std::map<std::string, TensorInfo>& tensors);
static void infer_config_from_tensors(ModelWeights& mw);

// ============================================================================
// Format detection
// ============================================================================

enum class ModelFormat {
    GGUF,
    SAFETENSORS,
    PYTORCH,
    ONNX,
    UNKNOWN
};

static ModelFormat detect_format(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return ModelFormat::UNKNOWN;

    uint8_t header[8] = {};
    f.read(reinterpret_cast<char*>(header), 8);
    size_t n = static_cast<size_t>(f.gcount());

    if (n < 4) return ModelFormat::UNKNOWN;

    // GGUF: magic "GGUF" = 0x47 0x47 0x55 0x46
    if (header[0] == 0x47 && header[1] == 0x47 &&
        header[2] == 0x55 && header[3] == 0x46) {
        return ModelFormat::GGUF;
    }

    // PyTorch zip: magic "PK" (0x50 0x4B)
    if (header[0] == 0x50 && header[1] == 0x4B) {
        return ModelFormat::PYTORCH;
    }

    // PyTorch pickle: 0x80 (PROTO opcode)
    if (header[0] == 0x80) {
        return ModelFormat::PYTORCH;
    }

    // ONNX: protobuf field 1 (wire type 2 = length-delimited) = 0x0A or 0x08
    // Check file extension too for disambiguation
    {
        std::string ext;
        auto dot = path.rfind('.');
        if (dot != std::string::npos) {
            ext = path.substr(dot);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        }
        if (ext == ".onnx") return ModelFormat::ONNX;
        if (ext == ".safetensors") return ModelFormat::SAFETENSORS;
        if (ext == ".bin" || ext == ".pt" || ext == ".pth") return ModelFormat::PYTORCH;
    }

    // SafeTensors: first 8 bytes = uint64_t LE header_size, then '{'
    // The header_size is typically < 100MB, and byte 8 should be '{'
    if (n >= 8) {
        uint64_t header_size;
        memcpy(&header_size, header, 8);
        // Sanity: header between 2 bytes and 100MB
        if (header_size >= 2 && header_size < 100 * 1024 * 1024) {
            // Peek at byte 8 to see if it's '{'
            char c;
            f.seekg(8);
            if (f.get(c) && c == '{') {
                return ModelFormat::SAFETENSORS;
            }
        }
    }

    return ModelFormat::UNKNOWN;
}

static const char* format_name(ModelFormat fmt) {
    switch (fmt) {
        case ModelFormat::GGUF:        return "gguf";
        case ModelFormat::SAFETENSORS: return "safetensors";
        case ModelFormat::PYTORCH:     return "pytorch";
        case ModelFormat::ONNX:        return "onnx";
        default: return "unknown";
    }
}

// ============================================================================
// Minimal JSON parser — just enough for SafeTensors headers
// Parses: objects, strings, arrays of ints, simple values
// ============================================================================

namespace json_mini {

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
    enum Type { NONE, STRING, NUMBER, OBJECT, ARRAY, BOOL_VAL } type = NONE;
    std::string str_val;
    int64_t num_val = 0;
    JsonObject obj_val;
    JsonArray arr_val;
    bool bool_val = false;
};

// Skip whitespace, return current char or 0 on end
static char skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    return (p < end) ? *p : 0;
}

static std::string parse_string(const char*& p, const char* end) {
    if (p >= end || *p != '"') throw std::runtime_error("JSON: expected '\"'");
    ++p;
    std::string result;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (p >= end) break;
            switch (*p) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case 'u':  // skip unicode escapes — just emit placeholder
                    result += '?';
                    for (int i = 0; i < 4 && p + 1 < end; ++i) ++p;
                    break;
                default: result += *p;
            }
        } else {
            result += *p;
        }
        ++p;
    }
    if (p < end) ++p;  // skip closing '"'
    return result;
}

static JsonValue parse_value(const char*& p, const char* end);

static JsonObject parse_object(const char*& p, const char* end) {
    JsonObject obj;
    if (p >= end || *p != '{') throw std::runtime_error("JSON: expected '{'");
    ++p;
    skip_ws(p, end);
    if (p < end && *p == '}') { ++p; return obj; }

    while (p < end) {
        skip_ws(p, end);
        std::string key = parse_string(p, end);
        skip_ws(p, end);
        if (p >= end || *p != ':') throw std::runtime_error("JSON: expected ':'");
        ++p;
        skip_ws(p, end);
        obj[key] = parse_value(p, end);
        skip_ws(p, end);
        if (p < end && *p == ',') { ++p; continue; }
        if (p < end && *p == '}') { ++p; break; }
    }
    return obj;
}

static JsonArray parse_array(const char*& p, const char* end) {
    JsonArray arr;
    if (p >= end || *p != '[') throw std::runtime_error("JSON: expected '['");
    ++p;
    skip_ws(p, end);
    if (p < end && *p == ']') { ++p; return arr; }

    while (p < end) {
        skip_ws(p, end);
        arr.push_back(parse_value(p, end));
        skip_ws(p, end);
        if (p < end && *p == ',') { ++p; continue; }
        if (p < end && *p == ']') { ++p; break; }
    }
    return arr;
}

static JsonValue parse_value(const char*& p, const char* end) {
    skip_ws(p, end);
    JsonValue val;
    if (p >= end) return val;

    if (*p == '"') {
        val.type = JsonValue::STRING;
        val.str_val = parse_string(p, end);
    } else if (*p == '{') {
        val.type = JsonValue::OBJECT;
        val.obj_val = parse_object(p, end);
    } else if (*p == '[') {
        val.type = JsonValue::ARRAY;
        val.arr_val = parse_array(p, end);
    } else if (*p == 't' || *p == 'f') {
        val.type = JsonValue::BOOL_VAL;
        if (end - p >= 4 && strncmp(p, "true", 4) == 0) {
            val.bool_val = true; p += 4;
        } else if (end - p >= 5 && strncmp(p, "false", 5) == 0) {
            val.bool_val = false; p += 5;
        }
    } else if (*p == 'n') {
        // null
        if (end - p >= 4) p += 4;
    } else if (*p == '-' || (*p >= '0' && *p <= '9')) {
        val.type = JsonValue::NUMBER;
        bool neg = false;
        if (*p == '-') { neg = true; ++p; }
        int64_t n = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            n = n * 10 + (*p - '0');
            ++p;
        }
        // skip fractional/exponent parts (we only need ints for shapes/offsets)
        if (p < end && *p == '.') {
            ++p;
            while (p < end && *p >= '0' && *p <= '9') ++p;
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            ++p;
            if (p < end && (*p == '+' || *p == '-')) ++p;
            while (p < end && *p >= '0' && *p <= '9') ++p;
        }
        val.num_val = neg ? -n : n;
    }
    return val;
}

static JsonObject parse(const std::string& text) {
    const char* p = text.data();
    const char* end = p + text.size();
    skip_ws(p, end);
    return parse_object(p, end);
}

}  // namespace json_mini

// ============================================================================
// SafeTensors loader
// ============================================================================

static TensorDType safetensors_dtype(const std::string& s) {
    if (s == "F32")  return TensorDType::FLOAT32;
    if (s == "F16")  return TensorDType::FLOAT16;
    if (s == "BF16") return TensorDType::BFLOAT16;
    if (s == "I8")   return TensorDType::INT8;
    if (s == "U8")   return TensorDType::UINT8;
    if (s == "I32")  return TensorDType::INT32;
    if (s == "I64")  return TensorDType::INT64;
    if (s == "F64")  return TensorDType::FLOAT64;
    if (s == "BOOL") return TensorDType::BOOL;
    return TensorDType::UNKNOWN;
}

static ModelWeights load_safetensors(const std::string& path) {
    ModelWeights result;
    result.format = "safetensors";
    result.source_path = path;
    result.file_path = path;

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("SafeTensors: cannot open: " + path);

    // Read header size (8 bytes, uint64 LE)
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    if (!f) throw std::runtime_error("SafeTensors: cannot read header size");
    if (header_size > 100 * 1024 * 1024) {
        throw std::runtime_error("SafeTensors: header too large: " +
                                 std::to_string(header_size));
    }

    // Read header JSON
    std::string header_json(header_size, '\0');
    f.read(header_json.data(), static_cast<std::streamsize>(header_size));
    if (!f) throw std::runtime_error("SafeTensors: cannot read header JSON");

    result.data_base_offset = 8 + header_size;

    // Parse JSON header
    auto root = json_mini::parse(header_json);

    for (auto& [key, val] : root) {
        if (key == "__metadata__") {
            // Extract architecture hints from metadata
            if (val.type == json_mini::JsonValue::OBJECT) {
                for (auto& [mk, mv] : val.obj_val) {
                    if (mv.type == json_mini::JsonValue::STRING) {
                        result.config.extra[mk] = mv.str_val;
                    }
                }
            }
            continue;
        }

        // Each tensor entry: { "dtype": "F16", "shape": [4096, 4096], "data_offsets": [0, 33554432] }
        if (val.type != json_mini::JsonValue::OBJECT) continue;

        TensorInfo ti;
        ti.name = key;

        auto& obj = val.obj_val;

        // dtype
        auto dt_it = obj.find("dtype");
        if (dt_it != obj.end() && dt_it->second.type == json_mini::JsonValue::STRING) {
            ti.dtype = safetensors_dtype(dt_it->second.str_val);
        }

        // shape
        auto sh_it = obj.find("shape");
        if (sh_it != obj.end() && sh_it->second.type == json_mini::JsonValue::ARRAY) {
            for (auto& dim : sh_it->second.arr_val) {
                ti.shape.push_back(dim.num_val);
            }
        }

        // data_offsets [begin, end]
        auto off_it = obj.find("data_offsets");
        if (off_it != obj.end() && off_it->second.type == json_mini::JsonValue::ARRAY &&
            off_it->second.arr_val.size() >= 2) {
            uint64_t begin = static_cast<uint64_t>(off_it->second.arr_val[0].num_val);
            uint64_t end   = static_cast<uint64_t>(off_it->second.arr_val[1].num_val);
            ti.data_offset = begin;
            ti.data_size = end - begin;
        }

        result.tensors[key] = std::move(ti);
    }

    // Infer architecture from tensor names
    result.architecture = infer_architecture(result.tensors);

    // Infer config from tensor shapes
    infer_config_from_tensors(result);

    std::cout << "[SafeTensors] Loaded header: " << result.tensors.size()
              << " tensors from " << path << std::endl;
    return result;
}

// ============================================================================
// PyTorch .bin/.pt loader (zip-based, pickle-free tensor extraction)
//
// PyTorch saves are zip files containing:
//   - archive/data.pkl (pickle — we skip this)
//   - archive/data/0, archive/data/1, ... (raw tensor data)
//   - archive/data.json or tensor metadata in pkl
//
// For .bin (HuggingFace format): it's a zip with:
//   - archive/data.pkl (maps tensor names to storage keys)
//   - archive/data/N (raw float data per tensor)
//
// We scan zip entries, identify raw data files, and read tensor shapes from
// the minimal pickle header (STACK_GLOBAL "torch._utils._rebuild_tensor_v2").
// ============================================================================

// Minimal zip local file header reader
struct ZipLocalHeader {
    std::string filename;
    uint64_t compressed_size = 0;
    uint64_t uncompressed_size = 0;
    uint64_t data_offset = 0;  // absolute file offset of data
    uint16_t compression = 0;  // 0 = stored (no compression)
};

static std::vector<ZipLocalHeader> scan_zip_entries(const std::string& path) {
    std::vector<ZipLocalHeader> entries;
    std::ifstream f(path, std::ios::binary);
    if (!f) return entries;

    // Find End of Central Directory (scan from end)
    f.seekg(0, std::ios::end);
    int64_t file_size = static_cast<int64_t>(f.tellg());
    if (file_size < 22) return entries;

    // Search for EOCD signature 0x06054b50 in last 65KB
    int64_t search_start = (std::max)(int64_t(0), file_size - 65536);
    int64_t eocd_pos = -1;
    f.seekg(search_start);
    std::vector<uint8_t> buf(static_cast<size_t>(file_size - search_start));
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));

    for (int64_t i = static_cast<int64_t>(buf.size()) - 22; i >= 0; --i) {
        if (buf[i] == 0x50 && buf[i+1] == 0x4B &&
            buf[i+2] == 0x05 && buf[i+3] == 0x06) {
            eocd_pos = search_start + i;
            break;
        }
    }
    if (eocd_pos < 0) return entries;

    // Read central directory offset from EOCD
    f.seekg(eocd_pos + 16);
    uint32_t cd_offset = 0;
    f.read(reinterpret_cast<char*>(&cd_offset), 4);

    // Read central directory entries (signature 0x02014b50)
    f.seekg(cd_offset);
    while (f) {
        uint32_t sig = 0;
        f.read(reinterpret_cast<char*>(&sig), 4);
        if (sig != 0x02014b50) break;

        uint8_t cd_header[42];
        f.read(reinterpret_cast<char*>(cd_header), 42);

        uint16_t compression   = *reinterpret_cast<uint16_t*>(&cd_header[6]);
        uint32_t comp_size     = *reinterpret_cast<uint32_t*>(&cd_header[16]);
        uint32_t uncomp_size   = *reinterpret_cast<uint32_t*>(&cd_header[20]);
        uint16_t fname_len     = *reinterpret_cast<uint16_t*>(&cd_header[24]);
        uint16_t extra_len     = *reinterpret_cast<uint16_t*>(&cd_header[26]);
        uint16_t comment_len   = *reinterpret_cast<uint16_t*>(&cd_header[28]);
        uint32_t local_offset  = *reinterpret_cast<uint32_t*>(&cd_header[38]);

        std::string filename(fname_len, '\0');
        f.read(filename.data(), fname_len);
        f.seekg(extra_len + comment_len, std::ios::cur);

        // Now read local file header to get actual data offset
        auto saved = f.tellg();
        f.seekg(local_offset);
        uint32_t local_sig = 0;
        f.read(reinterpret_cast<char*>(&local_sig), 4);
        if (local_sig == 0x04034b50) {
            uint8_t lh[26];
            f.read(reinterpret_cast<char*>(lh), 26);
            uint16_t lh_fname_len = *reinterpret_cast<uint16_t*>(&lh[22]);
            uint16_t lh_extra_len = *reinterpret_cast<uint16_t*>(&lh[24]);

            ZipLocalHeader zh;
            zh.filename = filename;
            zh.compression = compression;
            zh.compressed_size = comp_size;
            zh.uncompressed_size = uncomp_size;
            zh.data_offset = static_cast<uint64_t>(local_offset) + 30 + lh_fname_len + lh_extra_len;
            entries.push_back(std::move(zh));
        }
        f.seekg(saved);
    }

    return entries;
}

static ModelWeights load_pytorch(const std::string& path) {
    ModelWeights result;
    result.format = "pytorch";
    result.source_path = path;
    result.file_path = path;

    auto entries = scan_zip_entries(path);
    if (entries.empty()) {
        throw std::runtime_error("PyTorch: not a valid zip file or empty: " + path);
    }

    // Find data entries: files matching "*/data/*" (raw tensor storage)
    // and the pickle file for name mapping
    std::map<std::string, ZipLocalHeader> data_entries;  // "0", "1", ... -> header
    for (auto& e : entries) {
        // Extract the part after "data/"
        auto pos = e.filename.find("data/");
        if (pos != std::string::npos) {
            std::string key = e.filename.substr(pos + 5);
            if (!key.empty() && key.back() != '/') {
                data_entries[key] = e;
            }
        }
    }

    // For HuggingFace .bin files, tensor names are embedded in the pickle.
    // We do a best-effort scan of the pickle bytes to extract tensor names
    // paired with storage keys.
    //
    // Strategy: find "data.pkl" entry, scan for SHORT_BINUNICODE strings
    // near STACK_GLOBAL "torch._utils._rebuild_tensor_v2" calls.

    ZipLocalHeader* pkl_entry = nullptr;
    for (auto& e : entries) {
        if (e.filename.find("data.pkl") != std::string::npos) {
            pkl_entry = &e;
            break;
        }
    }

    // Tensor name extraction from pickle
    struct PklTensor {
        std::string name;
        std::string storage_key;  // "0", "1", etc.
        TensorDType dtype;
        std::vector<int64_t> shape;
    };
    std::vector<PklTensor> pkl_tensors;

    if (pkl_entry && pkl_entry->compression == 0) {
        std::ifstream f(path, std::ios::binary);
        f.seekg(static_cast<std::streamoff>(pkl_entry->data_offset));
        std::vector<uint8_t> pkl(static_cast<size_t>(pkl_entry->uncompressed_size));
        f.read(reinterpret_cast<char*>(pkl.data()), static_cast<std::streamsize>(pkl.size()));

        // Scan for SHORT_BINUNICODE (opcode 0x8C) strings
        // Pattern in HuggingFace pickles:
        //   storage key string, dtype string, shape tuple, tensor name string
        std::vector<std::string> strings;
        for (size_t i = 0; i < pkl.size(); ++i) {
            if (pkl[i] == 0x8C && i + 1 < pkl.size()) {
                uint8_t len = pkl[i + 1];
                if (i + 2 + len <= pkl.size()) {
                    strings.push_back(std::string(
                        reinterpret_cast<char*>(&pkl[i + 2]), len));
                    i += 1 + len;
                }
            }
        }

        // Match storage keys to tensor names:
        // In typical HF pickle, pattern is: "storage_key", then later the tensor name
        // with "data/" prefix references.
        // Simple heuristic: strings containing "." are tensor names,
        // pure digits are storage keys.
        std::vector<std::string> tensor_names;
        std::vector<std::string> storage_keys;
        for (auto& s : strings) {
            bool all_digits = !s.empty();
            for (char c : s) { if (c < '0' || c > '9') { all_digits = false; break; } }
            if (all_digits && data_entries.count(s)) {
                storage_keys.push_back(s);
            } else if (s.find('.') != std::string::npos && s.find("torch") == std::string::npos &&
                       s.find("_utils") == std::string::npos && s.find("collections") == std::string::npos) {
                tensor_names.push_back(s);
            }
        }

        // Pair them up (they appear in matching order in the pickle)
        size_t n = (std::min)(tensor_names.size(), storage_keys.size());
        for (size_t i = 0; i < n; ++i) {
            PklTensor pt;
            pt.name = tensor_names[i];
            pt.storage_key = storage_keys[i];
            pt.dtype = TensorDType::FLOAT32;  // default assumption for .bin
            pkl_tensors.push_back(std::move(pt));
        }
    }

    // Build tensor map
    if (!pkl_tensors.empty()) {
        for (auto& pt : pkl_tensors) {
            auto it = data_entries.find(pt.storage_key);
            if (it == data_entries.end()) continue;

            TensorInfo ti;
            ti.name = pt.name;
            ti.dtype = pt.dtype;
            ti.data_offset = it->second.data_offset;
            ti.data_size = it->second.uncompressed_size;
            // Shape unknown from pickle scan — infer as 1D
            ti.shape = { static_cast<int64_t>(ti.data_size / dtype_size(ti.dtype)) };
            result.tensors[pt.name] = std::move(ti);
        }
    } else {
        // Fallback: expose raw data entries by storage key
        for (auto& [key, zh] : data_entries) {
            TensorInfo ti;
            ti.name = key;
            ti.dtype = TensorDType::FLOAT32;
            ti.data_offset = zh.data_offset;
            ti.data_size = zh.uncompressed_size;
            ti.shape = { static_cast<int64_t>(ti.data_size / 4) };
            result.tensors[key] = std::move(ti);
        }
    }

    result.architecture = infer_architecture(result.tensors);
    infer_config_from_tensors(result);

    std::cout << "[PyTorch] Loaded: " << result.tensors.size()
              << " tensors from " << path << std::endl;
    return result;
}

// ============================================================================
// ONNX loader (protobuf-lite, no external deps)
//
// ONNX uses protobuf. We hand-parse only what we need:
//   - Field 1 (ir_version): varint
//   - Field 7 (graph): embedded message containing:
//     - Field 5 (initializer): repeated TensorProto containing weights
// ============================================================================

namespace protobuf_mini {

// Read varint from buffer, advance pos
static uint64_t read_varint(const uint8_t* data, size_t size, size_t& pos) {
    uint64_t val = 0;
    int shift = 0;
    while (pos < size) {
        uint8_t b = data[pos++];
        val |= static_cast<uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
        if (shift > 63) throw std::runtime_error("Protobuf: varint too long");
    }
    return val;
}

// Read a length-delimited field, return pointer + length
struct Blob {
    const uint8_t* data;
    size_t size;
};

// Parse one protobuf field: returns field_number, wire_type, and advances pos
struct Field {
    uint32_t number;
    uint32_t wire_type;
    uint64_t varint_val;   // for wire_type 0
    Blob blob;             // for wire_type 2
    uint64_t fixed64;      // for wire_type 1
    uint32_t fixed32;      // for wire_type 5
};

static bool read_field(const uint8_t* data, size_t size, size_t& pos, Field& f) {
    if (pos >= size) return false;
    uint64_t tag = read_varint(data, size, pos);
    f.number = static_cast<uint32_t>(tag >> 3);
    f.wire_type = static_cast<uint32_t>(tag & 7);
    f.varint_val = 0;
    f.blob = {nullptr, 0};
    f.fixed64 = 0;
    f.fixed32 = 0;

    switch (f.wire_type) {
        case 0: // varint
            f.varint_val = read_varint(data, size, pos);
            break;
        case 1: // 64-bit
            if (pos + 8 > size) return false;
            memcpy(&f.fixed64, data + pos, 8);
            pos += 8;
            break;
        case 2: { // length-delimited
            uint64_t len = read_varint(data, size, pos);
            if (pos + len > size) return false;
            f.blob.data = data + pos;
            f.blob.size = static_cast<size_t>(len);
            pos += static_cast<size_t>(len);
            break;
        }
        case 5: // 32-bit
            if (pos + 4 > size) return false;
            memcpy(&f.fixed32, data + pos, 4);
            pos += 4;
            break;
        default:
            return false;  // unsupported wire type
    }
    return true;
}

}  // namespace protobuf_mini

static TensorDType onnx_elem_type_to_dtype(int32_t elem_type) {
    switch (elem_type) {
        case 1:  return TensorDType::FLOAT32;
        case 2:  return TensorDType::UINT8;
        case 3:  return TensorDType::INT8;
        case 5:  return TensorDType::INT16;
        case 6:  return TensorDType::INT32;
        case 7:  return TensorDType::INT64;
        case 10: return TensorDType::FLOAT16;
        case 11: return TensorDType::FLOAT64;
        case 16: return TensorDType::BFLOAT16;
        default: return TensorDType::UNKNOWN;
    }
}

// Parse an ONNX TensorProto from raw protobuf bytes
static TensorInfo parse_onnx_tensor_proto(const uint8_t* data, size_t size,
                                           uint64_t file_base_offset) {
    TensorInfo ti;
    size_t pos = 0;
    int32_t elem_type = 0;
    uint64_t raw_data_offset = 0;
    uint64_t raw_data_size = 0;

    while (pos < size) {
        protobuf_mini::Field f;
        if (!protobuf_mini::read_field(data, size, pos, f)) break;

        switch (f.number) {
            case 1:  // dims (repeated int64, packed or individual)
                if (f.wire_type == 0) {
                    ti.shape.push_back(static_cast<int64_t>(f.varint_val));
                } else if (f.wire_type == 2) {
                    // packed repeated int64
                    size_t p2 = 0;
                    while (p2 < f.blob.size) {
                        uint64_t dim = protobuf_mini::read_varint(f.blob.data, f.blob.size, p2);
                        ti.shape.push_back(static_cast<int64_t>(dim));
                    }
                }
                break;
            case 2:  // data_type (int32)
                elem_type = static_cast<int32_t>(f.varint_val);
                break;
            case 8:  // name (string)
                if (f.wire_type == 2) {
                    ti.name = std::string(reinterpret_cast<const char*>(f.blob.data), f.blob.size);
                }
                break;
            case 13: // raw_data (bytes) — inline tensor data
                if (f.wire_type == 2) {
                    // The raw data is embedded in the protobuf message.
                    // We record its absolute file offset.
                    raw_data_offset = file_base_offset + static_cast<uint64_t>(f.blob.data - data);
                    raw_data_size = f.blob.size;
                }
                break;
            // Fields 4,5,6,7,9,10 hold typed data (float_data, int32_data, etc.)
            // We skip these — raw_data (field 13) is the standard path for large models
        }
    }

    ti.dtype = onnx_elem_type_to_dtype(elem_type);
    if (raw_data_size > 0) {
        ti.data_offset = raw_data_offset;
        ti.data_size = raw_data_size;
    } else {
        // No raw_data — compute from shape + dtype
        ti.data_size = static_cast<uint64_t>(ti.numel()) * dtype_size(ti.dtype);
    }

    return ti;
}

static ModelWeights load_onnx(const std::string& path) {
    ModelWeights result;
    result.format = "onnx";
    result.source_path = path;
    result.file_path = path;

    // Read entire file (ONNX models are typically < 2GB for the protobuf part)
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("ONNX: cannot open: " + path);

    size_t file_size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(file_size);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(file_size));

    // Parse top-level ModelProto fields
    size_t pos = 0;
    while (pos < file_size) {
        protobuf_mini::Field field;
        if (!protobuf_mini::read_field(buf.data(), file_size, pos, field)) break;

        if (field.number == 7 && field.wire_type == 2) {
            // GraphProto — parse its sub-fields for initializers
            size_t gpos = 0;
            while (gpos < field.blob.size) {
                protobuf_mini::Field gf;
                if (!protobuf_mini::read_field(field.blob.data, field.blob.size, gpos, gf)) break;

                if (gf.number == 5 && gf.wire_type == 2) {
                    // initializer: TensorProto
                    uint64_t tensor_base = static_cast<uint64_t>(gf.blob.data - buf.data());
                    TensorInfo ti = parse_onnx_tensor_proto(gf.blob.data, gf.blob.size, 0);
                    // Fix: data_offset is relative to blob — make absolute
                    if (ti.data_size > 0 && ti.data_offset == 0) {
                        // Data was inline in protobuf, offset already absolute from parse
                    }
                    // Reparse with correct base
                    ti = parse_onnx_tensor_proto(gf.blob.data, gf.blob.size, tensor_base);
                    if (!ti.name.empty()) {
                        result.tensors[ti.name] = std::move(ti);
                    }
                }
            }
        }
    }

    result.architecture = infer_architecture(result.tensors);
    infer_config_from_tensors(result);

    std::cout << "[ONNX] Loaded: " << result.tensors.size()
              << " tensors from " << path << std::endl;
    return result;
}

// ============================================================================
// GGUF loader — wraps existing GGUFReader into ModelWeights
// ============================================================================

static ModelWeights load_gguf(const std::string& path) {
    ModelWeights result;
    result.format = "gguf";
    result.source_path = path;
    result.file_path = path;

    torch::io::gguf::GGUFReader reader;
    reader.open(path);

    result.architecture = reader.architecture();
    result.data_base_offset = reader.data_offset;

    // Copy tensor info
    for (auto& t : reader.tensors) {
        TensorInfo ti;
        ti.name = t.name;
        ti.shape = t.shape();
        ti.data_offset = t.offset;
        ti.data_size = static_cast<uint64_t>(t.data_bytes());
        // GGUF tensors are quantized — mark as unknown dtype (needs dequant)
        ti.dtype = TensorDType::UNKNOWN;
        result.tensors[t.name] = std::move(ti);
    }

    // Extract config from GGUF metadata
    result.config.hidden_size = reader.get_arch_int("embedding_length", 0);
    result.config.num_layers = reader.get_arch_int("block_count", 0);
    result.config.num_heads = reader.get_arch_int("attention.head_count", 0);
    result.config.num_kv_heads = reader.get_arch_int("attention.head_count_kv",
                                                      result.config.num_heads);
    result.config.intermediate_size = reader.get_arch_int("feed_forward_length", 0);
    result.config.context_length = reader.get_arch_int("context_length", 8192);
    result.config.rms_norm_eps = reader.get_arch_float("attention.layer_norm_rms_epsilon", 1e-6f);
    result.config.rope_freq_base = reader.get_arch_float("rope.freq_base", 10000.0f);
    result.config.vocab_size = static_cast<int64_t>(reader.tensors.empty() ? 0 :
        reader.tensors[0].dims.empty() ? 0 : reader.tensors[0].dims.back());

    // Try to get vocab_size from metadata
    int64_t meta_vocab = reader.get_int("tokenizer.ggml.vocab_size", 0);
    if (meta_vocab > 0) result.config.vocab_size = meta_vocab;

    std::cout << "[GGUF] Loaded: " << result.tensors.size()
              << " tensors from " << path << std::endl;
    return result;
}

// ============================================================================
// Architecture inference from tensor names
// ============================================================================

static std::string infer_architecture(const std::map<std::string, TensorInfo>& tensors) {
    // Check for architecture-specific tensor name patterns
    for (auto& [name, _] : tensors) {
        // Llama / Mistral patterns
        if (name.find("model.layers.") != std::string::npos &&
            name.find("mlp.gate_proj") != std::string::npos) {
            // Could be llama or mistral — check for sliding window attention
            return "llama";
        }
        // Qwen2 pattern
        if (name.find("model.layers.") != std::string::npos &&
            name.find("mlp.gate_proj") != std::string::npos &&
            name.find("model.embed_tokens") != std::string::npos) {
            // Check if there's a qwen-specific pattern
            for (auto& [n2, _2] : tensors) {
                if (n2.find("model.layers.0.mlp.gate_proj") != std::string::npos) {
                    // Both llama and qwen2 share this pattern
                    // Distinguish by other keys if possible
                    break;
                }
            }
        }
        // GPT-2 / GPT-NeoX
        if (name.find("transformer.h.") != std::string::npos ||
            name.find("gpt_neox.layers.") != std::string::npos) {
            return "gpt_neox";
        }
        // BERT
        if (name.find("bert.encoder.layer.") != std::string::npos) {
            return "bert";
        }
        // Gemma
        if (name.find("model.layers.") != std::string::npos &&
            name.find("mlp.gate_proj") != std::string::npos) {
            // Check for gemma-specific naming
            for (auto& [n2, _2] : tensors) {
                if (n2.find("model.embed_tokens") != std::string::npos) {
                    return "llama";  // generic llama-family
                }
            }
        }
        // Phi
        if (name.find("model.layers.") != std::string::npos &&
            name.find("mlp.fc1") != std::string::npos) {
            return "phi";
        }
        // Falcon
        if (name.find("transformer.word_embeddings") != std::string::npos) {
            return "falcon";
        }
        // GGUF-style names (blk.0.attn_q.weight)
        if (name.find("blk.") != std::string::npos) {
            return "gguf_generic";
        }
    }
    return "unknown";
}

// ============================================================================
// Config inference from tensor shapes
// ============================================================================

static void infer_config_from_tensors(ModelWeights& mw) {
    // Try to infer hidden_size from embedding tensor
    for (auto& [name, ti] : mw.tensors) {
        if ((name.find("embed_tokens") != std::string::npos ||
             name.find("wte") != std::string::npos ||
             name.find("word_embeddings") != std::string::npos) &&
            ti.shape.size() == 2) {
            mw.config.vocab_size = ti.shape[0];
            mw.config.hidden_size = ti.shape[1];
            break;
        }
    }

    // Count layers
    int64_t max_layer = -1;
    for (auto& [name, _] : mw.tensors) {
        // Match "layers.N." or "h.N." or "blk.N."
        auto try_extract = [&](const std::string& prefix) {
            auto pos = name.find(prefix);
            if (pos != std::string::npos) {
                pos += prefix.size();
                int64_t n = 0;
                while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
                    n = n * 10 + (name[pos] - '0');
                    ++pos;
                }
                if (n > max_layer) max_layer = n;
            }
        };
        try_extract("layers.");
        try_extract("h.");
        try_extract("blk.");
    }
    if (max_layer >= 0 && mw.config.num_layers == 0) {
        mw.config.num_layers = max_layer + 1;
    }

    // Infer num_heads from q_proj shape
    for (auto& [name, ti] : mw.tensors) {
        if ((name.find("q_proj") != std::string::npos ||
             name.find("attn_q") != std::string::npos) &&
            ti.shape.size() == 2 && mw.config.hidden_size > 0) {
            // q_proj is [num_heads * head_dim, hidden_size]
            int64_t q_out = ti.shape[0];
            // Common head dims: 64, 80, 96, 128
            for (int64_t hd : {128, 96, 80, 64}) {
                if (q_out % hd == 0) {
                    mw.config.num_heads = q_out / hd;
                    mw.config.head_dim = hd;
                    break;
                }
            }
            break;
        }
    }

    // Infer intermediate_size from gate_proj or fc1
    for (auto& [name, ti] : mw.tensors) {
        if ((name.find("gate_proj") != std::string::npos ||
             name.find("fc1") != std::string::npos) &&
            ti.shape.size() == 2) {
            mw.config.intermediate_size = ti.shape[0];
            break;
        }
    }

    // Infer num_kv_heads from k_proj
    if (mw.config.num_kv_heads == 0 && mw.config.head_dim > 0) {
        for (auto& [name, ti] : mw.tensors) {
            if ((name.find("k_proj") != std::string::npos ||
                 name.find("attn_k") != std::string::npos) &&
                ti.shape.size() == 2) {
                mw.config.num_kv_heads = ti.shape[0] / mw.config.head_dim;
                break;
            }
        }
    }
}

// ============================================================================
// Multi-file / sharded model support
// ============================================================================

static std::vector<std::string> find_shard_files(const std::string& path) {
    std::vector<std::string> shards;
    namespace fs = std::filesystem;

    fs::path p(path);
    if (!fs::exists(p)) return shards;

    // If path is a directory, scan for model files
    if (fs::is_directory(p)) {
        // Look for safetensors shards: model-00001-of-00005.safetensors
        for (auto& entry : fs::directory_iterator(p)) {
            std::string fname = entry.path().filename().string();
            if (fname.find(".safetensors") != std::string::npos) {
                shards.push_back(entry.path().string());
            }
        }
        if (!shards.empty()) {
            std::sort(shards.begin(), shards.end());
            return shards;
        }

        // Look for pytorch shards: pytorch_model-00001-of-00005.bin
        for (auto& entry : fs::directory_iterator(p)) {
            std::string fname = entry.path().filename().string();
            if (fname.find(".bin") != std::string::npos &&
                fname.find("pytorch_model") != std::string::npos) {
                shards.push_back(entry.path().string());
            }
        }
        if (!shards.empty()) {
            std::sort(shards.begin(), shards.end());
            return shards;
        }

        // Single model file
        for (auto& entry : fs::directory_iterator(p)) {
            std::string fname = entry.path().filename().string();
            if (fname == "model.safetensors" || fname == "pytorch_model.bin" ||
                fname == "model.onnx") {
                shards.push_back(entry.path().string());
                return shards;
            }
        }
    }

    // Single file
    shards.push_back(path);
    return shards;
}

// ============================================================================
// Main entry point: auto-detect and load any model format
// ============================================================================

static ModelWeights load_model(const std::string& path) {
    namespace fs = std::filesystem;

    // Handle directory paths — find shards
    if (fs::is_directory(path)) {
        auto shards = find_shard_files(path);
        if (shards.empty()) {
            throw std::runtime_error("ModelLoader: no model files found in: " + path);
        }

        // Load first shard to get format and config
        ModelWeights result = load_model(shards[0]);
        result.source_path = path;
        result.shard_files = shards;

        // Merge additional shards
        for (size_t i = 1; i < shards.size(); ++i) {
            ModelWeights shard = load_model(shards[i]);
            for (auto& [name, ti] : shard.tensors) {
                // Prefix shard info for data loading
                ti.data_offset = ti.data_offset;  // keep absolute
                result.tensors[name] = std::move(ti);
            }
        }

        std::cout << "[ModelLoader] Loaded " << shards.size() << " shards, "
                  << result.tensors.size() << " total tensors" << std::endl;
        return result;
    }

    // Single file — detect format
    ModelFormat fmt = detect_format(path);
    std::cout << "[ModelLoader] Detected format: " << format_name(fmt)
              << " for " << path << std::endl;

    switch (fmt) {
        case ModelFormat::GGUF:
            return load_gguf(path);
        case ModelFormat::SAFETENSORS:
            return load_safetensors(path);
        case ModelFormat::PYTORCH:
            return load_pytorch(path);
        case ModelFormat::ONNX:
            return load_onnx(path);
        default:
            throw std::runtime_error("ModelLoader: unknown format for: " + path);
    }
}

// ============================================================================
// Tensor data reader — load raw bytes for a specific tensor
// Returns float32 data (with conversion from F16/BF16 if needed)
// ============================================================================

static at::Tensor read_tensor_data(const ModelWeights& mw, const std::string& name) {
    auto it = mw.tensors.find(name);
    if (it == mw.tensors.end()) {
        throw std::runtime_error("ModelLoader: tensor not found: " + name);
    }
    const TensorInfo& ti = it->second;

    // Determine which file to read from
    std::string file_path = mw.file_path;

    std::ifstream f(file_path, std::ios::binary);
    if (!f) throw std::runtime_error("ModelLoader: cannot open: " + file_path);

    uint64_t abs_offset = mw.data_base_offset + ti.data_offset;
    f.seekg(static_cast<std::streamoff>(abs_offset));
    if (!f) throw std::runtime_error("ModelLoader: seek failed for: " + name);

    // Create output tensor
    at::Tensor result = at::empty(ti.shape);
    float* dst = result.mutable_data_ptr<float>();
    int64_t n = ti.numel();

    if (ti.dtype == TensorDType::FLOAT32) {
        // Direct read
        f.read(reinterpret_cast<char*>(dst), n * 4);
    } else if (ti.dtype == TensorDType::FLOAT16) {
        // Read F16, convert to F32
        std::vector<uint16_t> buf(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(buf.data()), n * 2);
        for (int64_t i = 0; i < n; ++i) {
            // IEEE 754 half-precision to single-precision
            uint16_t h = buf[static_cast<size_t>(i)];
            uint32_t sign = (h & 0x8000u) << 16;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x03FF;
            uint32_t f32;
            if (exp == 0) {
                if (mant == 0) { f32 = sign; }
                else {
                    // subnormal
                    exp = 1;
                    while (!(mant & 0x0400)) { mant <<= 1; exp--; }
                    mant &= 0x03FF;
                    f32 = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                f32 = sign | 0x7F800000u | (mant << 13);
            } else {
                f32 = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            memcpy(&dst[i], &f32, 4);
        }
    } else if (ti.dtype == TensorDType::BFLOAT16) {
        // Read BF16, convert to F32 (just shift left by 16 bits)
        std::vector<uint16_t> buf(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(buf.data()), n * 2);
        for (int64_t i = 0; i < n; ++i) {
            uint32_t f32 = static_cast<uint32_t>(buf[static_cast<size_t>(i)]) << 16;
            memcpy(&dst[i], &f32, 4);
        }
    } else if (ti.dtype == TensorDType::FLOAT64) {
        std::vector<double> buf(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(buf.data()), n * 8);
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = static_cast<float>(buf[static_cast<size_t>(i)]);
        }
    } else if (ti.dtype == TensorDType::INT32) {
        std::vector<int32_t> buf(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(buf.data()), n * 4);
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = static_cast<float>(buf[static_cast<size_t>(i)]);
        }
    } else if (ti.dtype == TensorDType::INT8 || ti.dtype == TensorDType::UINT8) {
        std::vector<uint8_t> buf(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(buf.data()), n);
        for (int64_t i = 0; i < n; ++i) {
            if (ti.dtype == TensorDType::INT8)
                dst[i] = static_cast<float>(static_cast<int8_t>(buf[static_cast<size_t>(i)]));
            else
                dst[i] = static_cast<float>(buf[static_cast<size_t>(i)]);
        }
    } else {
        throw std::runtime_error("ModelLoader: unsupported dtype " +
                                 std::string(dtype_name(ti.dtype)) + " for tensor: " + name);
    }

    return result;
}

}  // namespace promeserve
