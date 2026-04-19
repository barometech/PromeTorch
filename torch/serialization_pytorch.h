#pragma once
// ============================================================================
// serialization_pytorch.h  —  PyTorch .pt / .pth compatible save & load.
//
// Writes files that stock PyTorch can open with torch.load() and can read
// .pt files produced by torch.save().
//
// PyTorch's .pt (since 1.6) = ZIP archive with this layout:
//   archive_name/data.pkl                Pickle protocol 2 of state_dict.
//   archive_name/data/<key>              Raw little-endian tensor bytes.
//   archive_name/version                 "3\n"
//   archive_name/byteorder   (opt.)      "little\n"
//
// The pickle stream stores each tensor as:
//   torch._utils._rebuild_tensor_v2(
//       <persistent storage reference>,     ← ('storage', FloatStorage, '<key>',
//       storage_offset (int),                  'cpu', numel)
//       size (tuple), stride (tuple),
//       requires_grad (bool),
//       backward_hooks (OrderedDict)
//   )
//
// The writer and reader here speak just enough of Pickle protocol 2 to
// round-trip real-world state_dicts. Unknown opcodes raise; unknown classes
// raise. No exec, no import — safe against malicious .pt files.
//
// Keep this file self-contained (header-only, no deps beyond existing torch
// headers + <filesystem>) so it can live next to torch/serialization.h and
// be compiled on Elbrus (LCC 1.29) just like the rest of the header-only
// `torch_nn` side.
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {

using at::Tensor;

namespace pt_pickle {

// --------------------------------------------------------------------------
// CRC32  (IEEE 802.3 polynomial 0xEDB88320)  — needed for ZIP entries.
// --------------------------------------------------------------------------
inline uint32_t crc32(const uint8_t* data, size_t n, uint32_t seed = 0) {
    static uint32_t table[256];
    static bool built = false;
    if (!built) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        built = true;
    }
    uint32_t c = seed ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// --------------------------------------------------------------------------
// Minimal ZIP writer — STORE (method 0) only; sufficient because PyTorch
// itself does not require DEFLATE and the raw tensor bytes compress poorly.
// --------------------------------------------------------------------------
struct ZipEntry {
    std::string name;
    uint64_t local_header_offset;
    uint64_t size;
    uint32_t crc;
};

struct ZipWriter {
    std::ofstream out;
    std::vector<ZipEntry> entries;
    uint64_t pos = 0;

    explicit ZipWriter(const std::string& path) : out(path, std::ios::binary) {
        if (!out) throw std::runtime_error("ZipWriter: cannot open " + path);
    }

    void write_raw(const void* p, size_t n) {
        out.write(static_cast<const char*>(p), n);
        if (!out) throw std::runtime_error("ZipWriter: write failed");
        pos += n;
    }

    static void put16(std::vector<uint8_t>& b, uint16_t v) {
        b.push_back(v & 0xFF); b.push_back((v >> 8) & 0xFF);
    }
    static void put32(std::vector<uint8_t>& b, uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((v >> (i * 8)) & 0xFF);
    }

    void add_entry(const std::string& name, const void* data, size_t size) {
        ZipEntry e;
        e.name = name;
        e.local_header_offset = pos;
        e.size = size;
        e.crc = crc32(static_cast<const uint8_t*>(data), size);

        std::vector<uint8_t> hdr;
        put32(hdr, 0x04034b50);          // local file signature
        put16(hdr, 20);                  // version
        put16(hdr, 0);                   // flags
        put16(hdr, 0);                   // compression STORE
        put16(hdr, 0); put16(hdr, 0);    // mtime / mdate
        put32(hdr, e.crc);
        put32(hdr, static_cast<uint32_t>(size));  // compressed size
        put32(hdr, static_cast<uint32_t>(size));  // uncompressed size
        put16(hdr, static_cast<uint16_t>(name.size()));
        put16(hdr, 0);                   // extra len
        write_raw(hdr.data(), hdr.size());
        write_raw(name.data(), name.size());
        write_raw(data, size);
        entries.push_back(e);
    }

    void finish() {
        uint64_t cd_off = pos;
        for (const auto& e : entries) {
            std::vector<uint8_t> hdr;
            put32(hdr, 0x02014b50);      // central dir signature
            put16(hdr, 20); put16(hdr, 20);
            put16(hdr, 0); put16(hdr, 0);
            put16(hdr, 0); put16(hdr, 0);
            put32(hdr, e.crc);
            put32(hdr, static_cast<uint32_t>(e.size));
            put32(hdr, static_cast<uint32_t>(e.size));
            put16(hdr, static_cast<uint16_t>(e.name.size()));
            put16(hdr, 0); put16(hdr, 0);
            put16(hdr, 0); put16(hdr, 0);
            put32(hdr, 0);
            put32(hdr, static_cast<uint32_t>(e.local_header_offset));
            write_raw(hdr.data(), hdr.size());
            write_raw(e.name.data(), e.name.size());
        }
        uint64_t cd_size = pos - cd_off;

        std::vector<uint8_t> eocd;
        put32(eocd, 0x06054b50);
        put16(eocd, 0); put16(eocd, 0);
        put16(eocd, static_cast<uint16_t>(entries.size()));
        put16(eocd, static_cast<uint16_t>(entries.size()));
        put32(eocd, static_cast<uint32_t>(cd_size));
        put32(eocd, static_cast<uint32_t>(cd_off));
        put16(eocd, 0);
        write_raw(eocd.data(), eocd.size());
        out.close();
    }
};

// --------------------------------------------------------------------------
// ZIP reader — STORE only, no ZIP64, matches what PyTorch emits.
// --------------------------------------------------------------------------
struct ZipReader {
    std::vector<uint8_t> data;
    std::unordered_map<std::string, std::pair<size_t, size_t>> files; // name → (off,size)

    explicit ZipReader(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("ZipReader: cannot open " + path);
        std::streamsize n = f.tellg();
        f.seekg(0);
        data.resize(n);
        f.read(reinterpret_cast<char*>(data.data()), n);

        // Locate EOCD (scan backward, max 64 KB).
        int64_t eocd = -1;
        int64_t upper = static_cast<int64_t>(data.size());
        int64_t low = upper > 65557 ? upper - 65557 : 0;
        for (int64_t i = upper - 22; i >= low; --i) {
            if (load32(static_cast<size_t>(i)) == 0x06054b50) { eocd = i; break; }
        }
        if (eocd < 0) throw std::runtime_error("ZipReader: EOCD not found");
        uint32_t cd_off = load32(static_cast<size_t>(eocd) + 16);
        uint16_t count = load16(static_cast<size_t>(eocd) + 10);
        size_t p = cd_off;
        for (int i = 0; i < count; ++i) {
            if (load32(p) != 0x02014b50) throw std::runtime_error("ZipReader: bad CDH");
            uint32_t csize = load32(p + 20);
            uint16_t nlen  = load16(p + 28);
            uint16_t elen  = load16(p + 30);
            uint16_t clen  = load16(p + 32);
            uint32_t lh    = load32(p + 42);
            std::string name(reinterpret_cast<const char*>(&data[p + 46]), nlen);
            // Local header: 30 + fname_len + extra_len → data start.
            uint16_t lh_nlen = load16(lh + 26);
            uint16_t lh_elen = load16(lh + 28);
            size_t dstart = lh + 30 + lh_nlen + lh_elen;
            files[name] = { dstart, csize };
            p += 46 + nlen + elen + clen;
        }
    }

    uint16_t load16(size_t o) const { return data[o] | (data[o + 1] << 8); }
    uint32_t load32(size_t o) const {
        return data[o] | (data[o + 1] << 8) | (data[o + 2] << 16) | (data[o + 3] << 24);
    }

    bool has(const std::string& name) const { return files.count(name) > 0; }

    // Return pointer + size for zero-copy reads.
    std::pair<const uint8_t*, size_t> get(const std::string& name) const {
        auto it = files.find(name);
        if (it == files.end()) throw std::runtime_error("ZipReader: missing " + name);
        return { &data[it->second.first], it->second.second };
    }

    // Find any entry whose name ends with `/suffix`.
    std::string find_suffix(const std::string& suffix) const {
        for (const auto& kv : files)
            if (kv.first.size() >= suffix.size() &&
                kv.first.compare(kv.first.size() - suffix.size(), suffix.size(), suffix) == 0)
                return kv.first;
        return "";
    }
};

// --------------------------------------------------------------------------
// Pickle opcodes used by the writer & reader.
// --------------------------------------------------------------------------
constexpr uint8_t MARK           = '(';
constexpr uint8_t STOP           = '.';
constexpr uint8_t BININT         = 'J';
constexpr uint8_t BININT1        = 'K';
constexpr uint8_t BININT2        = 'M';
constexpr uint8_t LONG1          = 0x8a;
constexpr uint8_t BINSTRING      = 'T';
constexpr uint8_t SHORT_BINSTRING= 'U';
constexpr uint8_t SHORT_BINUNICODE = 0x8c;
constexpr uint8_t BINUNICODE     = 'X';
constexpr uint8_t EMPTY_TUPLE    = ')';
constexpr uint8_t TUPLE          = 't';
constexpr uint8_t TUPLE1         = 0x85;
constexpr uint8_t TUPLE2         = 0x86;
constexpr uint8_t TUPLE3         = 0x87;
constexpr uint8_t NEWTRUE        = 0x88;
constexpr uint8_t NEWFALSE       = 0x89;
constexpr uint8_t NONE           = 'N';
constexpr uint8_t REDUCE         = 'R';
constexpr uint8_t BUILD          = 'b';
constexpr uint8_t GLOBAL         = 'c';
constexpr uint8_t EMPTY_DICT     = '}';
constexpr uint8_t SETITEMS       = 'u';
constexpr uint8_t SETITEM        = 's';
constexpr uint8_t EMPTY_LIST     = ']';
constexpr uint8_t LIST           = 'l';
constexpr uint8_t APPENDS        = 'e';
constexpr uint8_t BINPUT         = 'q';
constexpr uint8_t LONG_BINPUT    = 'r';
constexpr uint8_t BINGET         = 'h';
constexpr uint8_t LONG_BINGET    = 'j';
constexpr uint8_t PROTO          = 0x80;
constexpr uint8_t BINPERSID      = 'Q';

// --------------------------------------------------------------------------
// Pickle writer  — enough of protocol 2 for state_dicts.
// --------------------------------------------------------------------------
struct PickleWriter {
    std::vector<uint8_t> buf;
    int memo_id = 0;

    void put_byte(uint8_t b) { buf.push_back(b); }
    void put_bytes(const void* p, size_t n) {
        const uint8_t* c = static_cast<const uint8_t*>(p);
        buf.insert(buf.end(), c, c + n);
    }
    void put32(uint32_t v) {
        for (int i = 0; i < 4; ++i) buf.push_back((v >> (i * 8)) & 0xFF);
    }
    void put64(uint64_t v) {
        for (int i = 0; i < 8; ++i) buf.push_back((v >> (i * 8)) & 0xFF);
    }
    void memoize() {
        if (memo_id < 256) {
            put_byte(BINPUT); put_byte(static_cast<uint8_t>(memo_id));
        } else {
            put_byte(LONG_BINPUT); put32(static_cast<uint32_t>(memo_id));
        }
        memo_id++;
    }
    void proto_header() { put_byte(PROTO); put_byte(2); }
    void stop() { put_byte(STOP); }

    void write_unicode(const std::string& s) {
        if (s.size() < 256) {
            put_byte(SHORT_BINUNICODE);
            put_byte(static_cast<uint8_t>(s.size()));
        } else {
            put_byte(BINUNICODE);
            put32(static_cast<uint32_t>(s.size()));
        }
        put_bytes(s.data(), s.size());
    }
    void write_int(int64_t v) {
        if (v >= 0 && v < 256) { put_byte(BININT1); put_byte(static_cast<uint8_t>(v)); }
        else if (v >= 0 && v < 65536) {
            put_byte(BININT2);
            put_byte(v & 0xFF); put_byte((v >> 8) & 0xFF);
        } else {
            put_byte(BININT);
            for (int i = 0; i < 4; ++i) put_byte((v >> (i * 8)) & 0xFF);
        }
    }
    void write_bool(bool b) { put_byte(b ? NEWTRUE : NEWFALSE); }
    void write_none() { put_byte(NONE); }
    void write_global(const std::string& mod, const std::string& cls) {
        put_byte(GLOBAL);
        put_bytes(mod.data(), mod.size()); put_byte('\n');
        put_bytes(cls.data(), cls.size()); put_byte('\n');
        memoize();
    }

    void flush_to(std::vector<uint8_t>& out) { out = std::move(buf); }
};

// --------------------------------------------------------------------------
// ScalarType ↔ torch.Storage class name  (the pickled GLOBAL).
// --------------------------------------------------------------------------
inline std::string storage_class_name(c10::ScalarType t) {
    switch (t) {
        case c10::ScalarType::Float:    return "FloatStorage";
        case c10::ScalarType::Double:   return "DoubleStorage";
        case c10::ScalarType::Half:     return "HalfStorage";
        case c10::ScalarType::BFloat16: return "BFloat16Storage";
        case c10::ScalarType::Long:     return "LongStorage";
        case c10::ScalarType::Int:      return "IntStorage";
        case c10::ScalarType::Short:    return "ShortStorage";
        case c10::ScalarType::Char:     return "CharStorage";
        case c10::ScalarType::Byte:     return "ByteStorage";
        case c10::ScalarType::Bool:     return "BoolStorage";
        default:
            throw std::runtime_error("save_pytorch: unsupported dtype");
    }
}

inline c10::ScalarType storage_name_to_dtype(const std::string& n) {
    if (n == "FloatStorage")    return c10::ScalarType::Float;
    if (n == "DoubleStorage")   return c10::ScalarType::Double;
    if (n == "HalfStorage")     return c10::ScalarType::Half;
    if (n == "BFloat16Storage") return c10::ScalarType::BFloat16;
    if (n == "LongStorage")     return c10::ScalarType::Long;
    if (n == "IntStorage")      return c10::ScalarType::Int;
    if (n == "ShortStorage")    return c10::ScalarType::Short;
    if (n == "CharStorage")     return c10::ScalarType::Char;
    if (n == "ByteStorage")     return c10::ScalarType::Byte;
    if (n == "BoolStorage")     return c10::ScalarType::Bool;
    throw std::runtime_error("load_pytorch: unsupported storage class " + n);
}

// ==========================================================================
// WRITER:  serialize the state_dict pickle stream.
//
// State_dict is an OrderedDict{ name: Tensor }. Each tensor is rebuilt by a
// REDUCE call to torch._utils._rebuild_tensor_v2 with args:
//   (storage_persistent_ref, storage_offset, size_tuple, stride_tuple,
//    requires_grad, backward_hooks)
//
// storage_persistent_ref is emitted via BINPERSID as
//   ('storage', <storage_class>, '<string key>', 'cpu', <numel>)
// ==========================================================================
inline std::vector<uint8_t> build_data_pkl(
    const std::unordered_map<std::string, Tensor>& sd,
    std::vector<std::string>& storage_keys) {

    PickleWriter w;
    w.proto_header();

    // Emit GLOBAL for torch._utils._rebuild_tensor_v2 once, memoize at id 0.
    w.write_global("torch._utils", "_rebuild_tensor_v2");
    int rebuild_memo = w.memo_id - 1;

    // Emit EMPTY_DICT (the state_dict itself).  We'll use SETITEMS.
    // PyTorch saves plain dicts, not OrderedDict, which torch.load accepts.
    w.put_byte(EMPTY_DICT);
    w.memoize();
    int dict_memo = w.memo_id - 1;
    (void)dict_memo;
    w.put_byte(MARK);

    int key_counter = 0;
    for (const auto& kv : sd) {
        const std::string& name = kv.first;
        const Tensor& t = kv.second;
        Tensor c = t.contiguous();
        std::string key = std::to_string(key_counter++);
        storage_keys.push_back(key);

        // --- key (dict key: tensor name) ------------------------------------
        w.write_unicode(name);
        w.memoize();

        // --- value: REDUCE(_rebuild_tensor_v2, (pers_storage, 0, size, stride, rg, OD())) ---
        //   GET rebuild callable from memo.
        if (rebuild_memo < 256) { w.put_byte(BINGET); w.put_byte(static_cast<uint8_t>(rebuild_memo)); }
        else { w.put_byte(LONG_BINGET); w.put32(static_cast<uint32_t>(rebuild_memo)); }

        // Args tuple:  ( pers_storage , offset , sizes , stride , rg , {} )
        w.put_byte(MARK);

        // --- persistent storage reference ----------------------------------
        // BINPERSID pops a tuple-marker list from stack; here we push a
        // 5-tuple: ('storage', StorageClass, key, 'cpu', numel)  then BINPERSID.
        w.put_byte(MARK);
        w.write_unicode("storage");
        w.write_global("torch", storage_class_name(c.dtype()));
        w.write_unicode(key);
        w.write_unicode("cpu");
        int64_t numel = 1;
        for (int i = 0; i < c.dim(); ++i) numel *= c.size(i);
        w.write_int(numel);
        w.put_byte(TUPLE);
        w.memoize();
        w.put_byte(BINPERSID);

        // storage_offset
        w.write_int(0);

        // size tuple
        w.put_byte(MARK);
        for (int i = 0; i < c.dim(); ++i) w.write_int(c.size(i));
        w.put_byte(TUPLE);
        w.memoize();

        // stride tuple
        w.put_byte(MARK);
        for (int i = 0; i < c.dim(); ++i) w.write_int(c.stride(i));
        w.put_byte(TUPLE);
        w.memoize();

        // requires_grad
        w.write_bool(false);

        // backward hooks: empty OrderedDict (pickled as empty dict is fine).
        w.put_byte(EMPTY_DICT);
        w.memoize();

        w.put_byte(TUPLE);        // close arg tuple
        w.memoize();
        w.put_byte(REDUCE);
        w.memoize();
    }

    w.put_byte(SETITEMS);
    w.stop();
    std::vector<uint8_t> out;
    w.flush_to(out);
    return out;
}

// ==========================================================================
// WRITER top-level:  save_pytorch().
// ==========================================================================
inline bool save_pytorch(const std::unordered_map<std::string, Tensor>& state_dict,
                         const std::string& path) {
    std::string archive = "archive";
    ZipWriter zw(path);

    std::vector<std::string> keys;
    auto pkl = build_data_pkl(state_dict, keys);

    zw.add_entry(archive + "/data.pkl", pkl.data(), pkl.size());

    // Tensor data blobs — same iteration order as build_data_pkl.
    size_t i = 0;
    for (const auto& kv : state_dict) {
        Tensor c = kv.second.contiguous();
        zw.add_entry(archive + "/data/" + keys[i++], c.data_ptr(), c.nbytes());
    }

    std::string version = "3\n";
    zw.add_entry(archive + "/version", version.data(), version.size());
    std::string byteorder = "little\n";
    zw.add_entry(archive + "/byteorder", byteorder.data(), byteorder.size());

    zw.finish();
    return true;
}

// ==========================================================================
// READER.
// ==========================================================================

// Universal pickle "value" node — tagged union sufficient for state_dicts.
struct PNode {
    enum Kind { NONE_, BOOL, INT, FLOAT, STR, BYTES, TUPLE, LIST, DICT, PERSID, GLOBAL, REDUCE, MARK_ };
    Kind kind = NONE_;
    bool b = false;
    int64_t i = 0;
    double f = 0.0;
    std::string s;
    std::vector<PNode> items;                      // TUPLE / LIST / PERSID
    std::vector<std::pair<PNode, PNode>> kv;       // DICT
    std::string gmod, gcls;                        // GLOBAL
    // REDUCE:  items[0] = callable (usually GLOBAL),  items[1] = args tuple.
    // PERSID: items[0] = raw persistent_id tuple (resolved by loader).

    static PNode mk_int(int64_t v)    { PNode p; p.kind = INT;   p.i = v; return p; }
    static PNode mk_str(std::string v){ PNode p; p.kind = STR;   p.s = std::move(v); return p; }
    static PNode mk_bool(bool v)      { PNode p; p.kind = BOOL;  p.b = v; return p; }
    static PNode mk_none()            { PNode p; p.kind = NONE_; return p; }
    static PNode mk_tuple(std::vector<PNode> v){ PNode p; p.kind = TUPLE; p.items = std::move(v); return p; }
    static PNode mk_mark()            { PNode p; p.kind = MARK_; return p; }
};

struct PickleReader {
    const uint8_t* p;
    const uint8_t* end;
    std::vector<PNode> stack;
    std::unordered_map<int, PNode> memo;
    // persistent_load resolver (supplied by caller):
    // takes the 5-tuple ('storage', StorageGlobal, key, device, numel) and
    // returns something that the BINPERSID will push onto the stack.
    std::function<PNode(const PNode&)> persistent_load;

    PickleReader(const uint8_t* data, size_t n) : p(data), end(data + n) {}

    uint8_t get_byte() {
        if (p >= end) throw std::runtime_error("pickle: unexpected EOF");
        return *p++;
    }
    uint16_t read16() { uint16_t a = get_byte(); a |= (uint16_t)get_byte() << 8; return a; }
    uint32_t read32() { uint32_t a = 0; for (int i = 0; i < 4; ++i) a |= (uint32_t)get_byte() << (i * 8); return a; }
    int64_t read_i32() { return (int32_t)read32(); }
    std::string read_line() {
        std::string s;
        while (true) { uint8_t c = get_byte(); if (c == '\n') break; s.push_back((char)c); }
        return s;
    }
    std::string read_bytes(size_t n) {
        if (p + n > end) throw std::runtime_error("pickle: short read");
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }

    // Pop items back to the last MARK and return them in order.
    std::vector<PNode> pop_to_mark() {
        std::vector<PNode> out;
        while (!stack.empty() && stack.back().kind != PNode::MARK_) {
            out.push_back(std::move(stack.back()));
            stack.pop_back();
        }
        if (stack.empty()) throw std::runtime_error("pickle: mark underflow");
        stack.pop_back();  // pop the MARK itself
        std::reverse(out.begin(), out.end());
        return out;
    }

    // Execute pickle stream and return final top-of-stack.
    PNode run() {
        while (p < end) {
            uint8_t op = get_byte();
            switch (op) {
                case PROTO:   get_byte(); break;
                case STOP:    {
                    if (stack.empty()) throw std::runtime_error("pickle: empty stack at STOP");
                    PNode r = std::move(stack.back()); stack.pop_back(); return r;
                }
                case MARK:    stack.push_back(PNode::mk_mark()); break;
                case NONE:    stack.push_back(PNode::mk_none()); break;
                case NEWTRUE: stack.push_back(PNode::mk_bool(true)); break;
                case NEWFALSE:stack.push_back(PNode::mk_bool(false)); break;
                case BININT:  stack.push_back(PNode::mk_int(read_i32())); break;
                case BININT1: stack.push_back(PNode::mk_int(get_byte())); break;
                case BININT2: {
                    int64_t v = get_byte();
                    v |= (int64_t)get_byte() << 8;
                    stack.push_back(PNode::mk_int(v));
                    break;
                }
                case LONG1: {
                    uint8_t n = get_byte();
                    int64_t v = 0;
                    int nb = (n > 8) ? 8 : (int)n;
                    for (int i = 0; i < nb; ++i) v |= (int64_t)get_byte() << (i * 8);
                    // Discard any excess bytes past 8 (extremely unlikely for shapes).
                    for (int i = nb; i < (int)n; ++i) get_byte();
                    // sign-extend (skip if n == 0 or n >= 8).
                    if (nb > 0 && nb < 8) {
                        int64_t sign = (v >> (nb * 8 - 1)) & 1;
                        if (sign) v |= (~0LL) << (nb * 8);
                    }
                    stack.push_back(PNode::mk_int(v));
                    break;
                }
                case BINSTRING: {
                    uint32_t n = read32();
                    stack.push_back(PNode::mk_str(read_bytes(n)));
                    break;
                }
                case SHORT_BINSTRING: {
                    uint8_t n = get_byte();
                    stack.push_back(PNode::mk_str(read_bytes(n)));
                    break;
                }
                case SHORT_BINUNICODE: {
                    uint8_t n = get_byte();
                    stack.push_back(PNode::mk_str(read_bytes(n)));
                    break;
                }
                case BINUNICODE: {
                    uint32_t n = read32();
                    stack.push_back(PNode::mk_str(read_bytes(n)));
                    break;
                }
                case EMPTY_TUPLE: stack.push_back(PNode::mk_tuple({})); break;
                case TUPLE1: {
                    PNode a = std::move(stack.back()); stack.pop_back();
                    stack.push_back(PNode::mk_tuple({ std::move(a) }));
                    break;
                }
                case TUPLE2: {
                    PNode b = std::move(stack.back()); stack.pop_back();
                    PNode a = std::move(stack.back()); stack.pop_back();
                    stack.push_back(PNode::mk_tuple({ std::move(a), std::move(b) }));
                    break;
                }
                case TUPLE3: {
                    PNode c = std::move(stack.back()); stack.pop_back();
                    PNode b = std::move(stack.back()); stack.pop_back();
                    PNode a = std::move(stack.back()); stack.pop_back();
                    stack.push_back(PNode::mk_tuple({ std::move(a), std::move(b), std::move(c) }));
                    break;
                }
                case TUPLE: {
                    auto items = pop_to_mark();
                    stack.push_back(PNode::mk_tuple(std::move(items)));
                    break;
                }
                case EMPTY_DICT: {
                    PNode d; d.kind = PNode::DICT; stack.push_back(std::move(d));
                    break;
                }
                case EMPTY_LIST: {
                    PNode l; l.kind = PNode::LIST; stack.push_back(std::move(l));
                    break;
                }
                case SETITEM: {
                    PNode v = std::move(stack.back()); stack.pop_back();
                    PNode k = std::move(stack.back()); stack.pop_back();
                    stack.back().kv.emplace_back(std::move(k), std::move(v));
                    break;
                }
                case SETITEMS: {
                    auto items = pop_to_mark();
                    PNode& d = stack.back();
                    for (size_t i = 0; i + 1 < items.size(); i += 2)
                        d.kv.emplace_back(std::move(items[i]), std::move(items[i + 1]));
                    break;
                }
                case APPENDS: {
                    auto items = pop_to_mark();
                    PNode& l = stack.back();
                    for (auto& x : items) l.items.push_back(std::move(x));
                    break;
                }
                case GLOBAL: {
                    std::string mod = read_line();
                    std::string cls = read_line();
                    PNode g; g.kind = PNode::GLOBAL; g.gmod = std::move(mod); g.gcls = std::move(cls);
                    stack.push_back(std::move(g));
                    break;
                }
                case REDUCE: {
                    PNode args = std::move(stack.back()); stack.pop_back();
                    PNode callable = std::move(stack.back()); stack.pop_back();
                    PNode r; r.kind = PNode::REDUCE;
                    r.items.push_back(std::move(callable));
                    r.items.push_back(std::move(args));
                    stack.push_back(std::move(r));
                    break;
                }
                case BUILD: {
                    // pop state, leave object — we don't use state.
                    stack.pop_back();
                    break;
                }
                case BINPUT: {
                    uint8_t id = get_byte();
                    memo[id] = stack.back();
                    break;
                }
                case LONG_BINPUT: {
                    uint32_t id = read32();
                    memo[id] = stack.back();
                    break;
                }
                case BINGET: {
                    uint8_t id = get_byte();
                    auto it = memo.find(id);
                    if (it == memo.end()) throw std::runtime_error("pickle: bad BINGET");
                    stack.push_back(it->second);
                    break;
                }
                case LONG_BINGET: {
                    uint32_t id = read32();
                    auto it = memo.find(id);
                    if (it == memo.end()) throw std::runtime_error("pickle: bad LONG_BINGET");
                    stack.push_back(it->second);
                    break;
                }
                case BINPERSID: {
                    PNode pid = std::move(stack.back()); stack.pop_back();
                    if (!persistent_load) throw std::runtime_error("pickle: BINPERSID without resolver");
                    stack.push_back(persistent_load(pid));
                    break;
                }
                default:
                    throw std::runtime_error("pickle: unsupported opcode 0x" +
                                             std::to_string((int)op));
            }
        }
        throw std::runtime_error("pickle: missing STOP");
    }
};

// Safe list of accepted GLOBALs. Anything else → abort.
inline bool safe_global(const std::string& mod, const std::string& cls) {
    if (mod == "torch._utils" && cls == "_rebuild_tensor_v2") return true;
    if (mod == "torch" && (
        cls == "FloatStorage" || cls == "DoubleStorage" ||
        cls == "HalfStorage"  || cls == "BFloat16Storage" ||
        cls == "LongStorage"  || cls == "IntStorage" ||
        cls == "ShortStorage" || cls == "CharStorage" ||
        cls == "ByteStorage"  || cls == "BoolStorage")) return true;
    if (mod == "collections" && cls == "OrderedDict") return true;
    return false;
}

// Build a Tensor from the parsed _rebuild_tensor_v2(args...) REDUCE node.
//   args = ( storage_ref , offset , sizes , stride , rg , hooks )
// where storage_ref is a PNode representing raw tensor bytes of that dtype:
//   storage_ref.kind == TUPLE  with items = [ dtype-marker , key , numel ]
// We carry the raw bytes via a side map from key → (dtype, bytes_ptr).
inline Tensor materialize_tensor(const PNode& reduce_node,
                                 const std::unordered_map<std::string,
                                     std::pair<c10::ScalarType, const uint8_t*>>& storage_blob) {
    if (reduce_node.kind != PNode::REDUCE || reduce_node.items.size() != 2)
        throw std::runtime_error("load_pytorch: not a REDUCE node");
    const PNode& callable = reduce_node.items[0];
    const PNode& args = reduce_node.items[1];
    if (callable.kind != PNode::GLOBAL ||
        callable.gmod != "torch._utils" || callable.gcls != "_rebuild_tensor_v2")
        throw std::runtime_error("load_pytorch: unexpected reducer");
    if (args.kind != PNode::TUPLE || args.items.size() < 4)
        throw std::runtime_error("load_pytorch: bad _rebuild_tensor_v2 args");

    // args.items[0] = persistent-resolved storage reference (a PNode STR key)
    const PNode& storage_ref = args.items[0];
    // Our persistent_load packs the key into STR and remembers dtype elsewhere.
    if (storage_ref.kind != PNode::STR)
        throw std::runtime_error("load_pytorch: storage ref is not string");
    auto it = storage_blob.find(storage_ref.s);
    if (it == storage_blob.end())
        throw std::runtime_error("load_pytorch: missing storage " + storage_ref.s);
    c10::ScalarType dtype = it->second.first;
    const uint8_t* raw = it->second.second;

    int64_t offset = args.items[1].i;
    if (args.items[2].kind != PNode::TUPLE)
        throw std::runtime_error("load_pytorch: sizes not tuple");

    std::vector<int64_t> sizes;
    for (const auto& n : args.items[2].items) sizes.push_back(n.i);

    Tensor t = at::empty(sizes, at::TensorOptions().dtype(dtype));
    size_t nbytes = t.nbytes();
    size_t off_bytes = offset * c10::elementSize(dtype);
    std::memcpy(t.data_ptr(), raw + off_bytes, nbytes);
    return t;
}

// --------------------------------------------------------------------------
// Top-level reader.
// --------------------------------------------------------------------------
inline std::unordered_map<std::string, Tensor> load_pytorch(const std::string& path) {
    ZipReader zr(path);

    // Locate data.pkl inside archive/ (name varies).
    std::string pkl_name = zr.find_suffix("/data.pkl");
    if (pkl_name.empty())
        throw std::runtime_error("load_pytorch: data.pkl not found in " + path);
    auto pkl = zr.get(pkl_name);

    // Derive archive prefix to find data/<key> entries.
    std::string prefix = pkl_name.substr(0, pkl_name.size() - std::string("data.pkl").size());

    // Persistent-load resolver:  pid = ('storage', StorageGlobal, key, device, numel).
    // We record (key → dtype) and push a STR node carrying the key; the
    // materialize_tensor() step then looks up the raw bytes via storage_blob.
    std::unordered_map<std::string, c10::ScalarType> key_to_dtype;

    PickleReader pr(pkl.first, pkl.second);
    pr.persistent_load = [&](const PNode& pid) -> PNode {
        if (pid.kind != PNode::TUPLE || pid.items.size() < 5)
            throw std::runtime_error("persistent_load: bad tuple");
        if (pid.items[0].kind != PNode::STR || pid.items[0].s != "storage")
            throw std::runtime_error("persistent_load: not a storage record");
        if (pid.items[1].kind != PNode::GLOBAL || pid.items[1].gmod != "torch")
            throw std::runtime_error("persistent_load: not a torch storage class");
        if (!safe_global(pid.items[1].gmod, pid.items[1].gcls))
            throw std::runtime_error("persistent_load: unsafe class");
        std::string key = pid.items[2].s;
        key_to_dtype[key] = storage_name_to_dtype(pid.items[1].gcls);
        return PNode::mk_str(key);
    };

    // Before running pickle, pre-validate: scan for GLOBAL opcodes and refuse
    // anything outside the safe set. We also walk the result tree afterward.
    // (We re-scan with a lightweight pass; cheap compared to raw tensor I/O.)
    {
        const uint8_t* q = pkl.first;
        const uint8_t* e = pkl.first + pkl.second;
        while (q < e) {
            uint8_t op = *q++;
            if (op == GLOBAL) {
                std::string mod, cls;
                while (q < e && *q != '\n') mod.push_back((char)*q++); if (q < e) ++q;
                while (q < e && *q != '\n') cls.push_back((char)*q++); if (q < e) ++q;
                if (!safe_global(mod, cls))
                    throw std::runtime_error("load_pytorch: unsafe class " + mod + "." + cls);
            }
        }
    }

    PNode root = pr.run();

    // Build storage_blob map: for each referenced key, fetch raw bytes.
    std::unordered_map<std::string, std::pair<c10::ScalarType, const uint8_t*>> storage_blob;
    for (const auto& kv : key_to_dtype) {
        std::string entry = prefix + "data/" + kv.first;
        auto d = zr.get(entry);
        storage_blob[kv.first] = { kv.second, d.first };
    }

    // Walk root dict → name → REDUCE-node → Tensor.
    if (root.kind != PNode::DICT)
        throw std::runtime_error("load_pytorch: root is not a dict");
    std::unordered_map<std::string, Tensor> out;
    for (const auto& kv : root.kv) {
        if (kv.first.kind != PNode::STR)
            throw std::runtime_error("load_pytorch: non-string key");
        out[kv.first.s] = materialize_tensor(kv.second, storage_blob);
    }
    return out;
}

} // namespace pt_pickle

// --------------------------------------------------------------------------
// Public API — `torch::save_pytorch` / `torch::load_pytorch`.
// --------------------------------------------------------------------------
inline bool save_pytorch(const std::unordered_map<std::string, Tensor>& state_dict,
                         const std::string& path) {
    return pt_pickle::save_pytorch(state_dict, path);
}

inline std::unordered_map<std::string, Tensor> load_pytorch(const std::string& path) {
    return pt_pickle::load_pytorch(path);
}

} // namespace torch
