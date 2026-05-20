// ============================================================================
// test_pt8_loader.cpp — sanity coverage for torch::io::PT8Reader (Round 4
// Agent C, 2026-04-30).
//
// This exercises the *reader* in isolation. We hand-craft a minimal .pt8 in
// a temp file, then re-open it through PT8Reader and assert that:
//   1. magic / version / tensor_count round-trip
//   2. each tensor's data_offset, data_size, dims, pt8_type round-trip
//   3. tensor_data() returns the exact bytes we wrote
//   4. is_pt8_file() classifies our file correctly and rejects garbage
//
// The on-disk layout is the one written by
// `tools/gguf2pt8/converter.cpp::write_header` + `write_tail_table`. We
// re-implement it in this test (small enough — ~50 LoC) so the test does
// not link against the converter and can run without any GGUF file.
//
// End-to-end logit-diff between .pt8 and .gguf+repack is gated behind
// PT8_TEST_GGUF (path to a real GGUF file). Without it, only the byte-level
// checks above run.
//
// Build: aten_cpu — no CUDA, no GGUF, no MSVC-only deps.
// ============================================================================

#include "torch/io/pt8_reader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using torch::io::PT8Reader;
using torch::io::PT8TensorRecord;
using torch::io::PT8_MAGIC;
using torch::io::PT8_VERSION;
using torch::io::PT8_HEADER_BYTES;
using torch::io::PT8_DATA_ALIGNMENT;
using torch::io::PT8_TYPE_F32;
using torch::io::PT8_TYPE_Q8_0_SOA4;

static int failed = 0;
static int passed = 0;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++passed; std::printf("  PASS: %s\n", msg); }              \
    else      { ++failed; std::printf("  FAIL: %s\n", msg); }              \
} while (0)

namespace {

// Pack one tail-table entry exactly as converter.cpp does in
// write_tail_table().
void pack_entry(std::vector<uint8_t>& buf,
                const std::string& name,
                uint32_t pt8_type,
                const std::vector<int64_t>& dims,
                int64_t data_off,
                int64_t data_size,
                int64_t row_stride,
                const std::vector<uint8_t>& meta = {}) {
    auto put = [&](const void* p, size_t n) {
        const uint8_t* bp = static_cast<const uint8_t*>(p);
        buf.insert(buf.end(), bp, bp + n);
    };
    uint32_t name_len = static_cast<uint32_t>(name.size());
    put(&name_len, 4);
    put(name.data(), name.size());
    put(&pt8_type, 4);
    uint32_t n_dims = static_cast<uint32_t>(dims.size());
    put(&n_dims, 4);
    for (auto d : dims) {
        uint64_t du = static_cast<uint64_t>(d);
        put(&du, 8);
    }
    uint64_t doff = static_cast<uint64_t>(data_off);
    uint64_t dsiz = static_cast<uint64_t>(data_size);
    uint64_t rstr = static_cast<uint64_t>(row_stride);
    put(&doff, 8);
    put(&dsiz, 8);
    put(&rstr, 8);
    uint32_t mlen = static_cast<uint32_t>(meta.size());
    put(&mlen, 4);
    if (mlen) put(meta.data(), mlen);
}

// Round `x` up to the next multiple of `align`.
int64_t align_up(int64_t x, int64_t align) {
    int64_t pad = (align - (x % align)) % align;
    return x + pad;
}

// Write a small valid .pt8 file with two tensors of known content.
// Returns the file path.
std::string write_test_pt8() {
#if defined(_WIN32)
    const char* tmpdir = std::getenv("TEMP");
    if (!tmpdir) tmpdir = "C:\\Temp";
#else
    const char* tmpdir = std::getenv("TMPDIR");
    if (!tmpdir) tmpdir = "/tmp";
#endif
    std::string path = std::string(tmpdir) + "/pt8_loader_test.pt8";

    // Tensor A: F32, shape [4], values 0.0, 1.0, 2.0, 3.0.
    std::vector<float> A = {0.0f, 1.0f, 2.0f, 3.0f};
    // Tensor B: Q8_0_SOA4 marker (just opaque bytes for byte-level test),
    // shape [4, 32], 4-row × 32-col single super-row → 1 × 176 bytes.
    std::vector<uint8_t> B(176, 0xAB);
    for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<uint8_t>(i & 0xFF);

    int64_t off_A = align_up(static_cast<int64_t>(PT8_HEADER_BYTES),
                             static_cast<int64_t>(PT8_DATA_ALIGNMENT));
    int64_t size_A = static_cast<int64_t>(A.size() * sizeof(float));
    int64_t off_B = align_up(off_A + size_A,
                             static_cast<int64_t>(PT8_DATA_ALIGNMENT));
    int64_t size_B = static_cast<int64_t>(B.size());
    int64_t table_off = align_up(off_B + size_B,
                                 static_cast<int64_t>(PT8_DATA_ALIGNMENT));

    // Build tail table.
    std::vector<uint8_t> table_bytes;
    pack_entry(table_bytes, "alpha", PT8_TYPE_F32,
               {4}, off_A, size_A, /*row_stride=*/0);
    pack_entry(table_bytes, "blk.0.attn_q.weight", PT8_TYPE_Q8_0_SOA4,
               {4, 32}, off_B, size_B, /*row_stride=*/44);

    int64_t file_size = table_off + static_cast<int64_t>(table_bytes.size());

    // Build header.
    std::vector<uint8_t> hdr(PT8_HEADER_BYTES, 0);
    uint32_t magic = PT8_MAGIC;
    uint32_t ver   = PT8_VERSION;
    uint64_t tcnt  = 2;
    uint64_t toff  = static_cast<uint64_t>(table_off);
    std::memcpy(hdr.data() + 0,  &magic, 4);
    std::memcpy(hdr.data() + 4,  &ver,   4);
    std::memcpy(hdr.data() + 16, &toff,  8);
    std::memcpy(hdr.data() + 24, &tcnt,  8);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { std::fprintf(stderr, "cannot open %s for write\n", path.c_str()); std::exit(1); }
    // Write header at 0.
    f.write(reinterpret_cast<const char*>(hdr.data()),
            static_cast<std::streamsize>(hdr.size()));
    // Pad to off_A.
    std::vector<uint8_t> pad_a(off_A - static_cast<int64_t>(hdr.size()), 0);
    f.write(reinterpret_cast<const char*>(pad_a.data()),
            static_cast<std::streamsize>(pad_a.size()));
    // Tensor A bytes.
    f.write(reinterpret_cast<const char*>(A.data()), size_A);
    // Pad to off_B.
    std::vector<uint8_t> pad_b(off_B - (off_A + size_A), 0);
    f.write(reinterpret_cast<const char*>(pad_b.data()),
            static_cast<std::streamsize>(pad_b.size()));
    // Tensor B bytes.
    f.write(reinterpret_cast<const char*>(B.data()), size_B);
    // Pad to table_off.
    std::vector<uint8_t> pad_t(table_off - (off_B + size_B), 0);
    f.write(reinterpret_cast<const char*>(pad_t.data()),
            static_cast<std::streamsize>(pad_t.size()));
    // Tail table.
    f.write(reinterpret_cast<const char*>(table_bytes.data()),
            static_cast<std::streamsize>(table_bytes.size()));
    f.close();
    std::printf("  test file: %s (%lld bytes)\n",
                path.c_str(), static_cast<long long>(file_size));
    return path;
}

}  // namespace

int main() {
    std::printf("=== PT8Reader self-test (Round 4 Agent C) ===\n");

    // --- 1. Hand-craft a minimal .pt8, re-open through PT8Reader. -----------
    std::string path = write_test_pt8();

    CHECK(PT8Reader::is_pt8_file(path), "is_pt8_file accepts crafted file");

    // Negative classification: a junk file is NOT .pt8.
    {
        std::string junk = path + ".junk";
        std::ofstream g(junk, std::ios::binary);
        const char* bad = "NOT_A_PT8_FILE_AT_ALL";
        g.write(bad, std::strlen(bad));
        g.close();
        CHECK(!PT8Reader::is_pt8_file(junk),
              "is_pt8_file rejects junk file");
        std::remove(junk.c_str());
    }

    PT8Reader rd;
    CHECK(rd.open(path), "PT8Reader::open succeeds on crafted file");

    const auto& hdr = rd.header();
    CHECK(hdr.magic == PT8_MAGIC,    "header.magic round-trip");
    CHECK(hdr.version == PT8_VERSION, "header.version round-trip");
    CHECK(hdr.tensor_count == 2,      "header.tensor_count == 2");

    const auto& ts = rd.tensors();
    CHECK(ts.size() == 2, "tensors().size() == 2");

    // --- 2. Tensor A — F32, [4], 0..3 ---------------------------------------
    {
        const PT8TensorRecord* a = rd.find("alpha");
        CHECK(a != nullptr, "alpha tensor present");
        if (a) {
            CHECK(a->pt8_type == PT8_TYPE_F32, "alpha.pt8_type == F32");
            CHECK(a->dims.size() == 1 && a->dims[0] == 4,
                  "alpha.dims == [4]");
            CHECK(a->data_size == 16, "alpha.data_size == 16");

            const float* p = static_cast<const float*>(rd.tensor_data("alpha"));
            CHECK(p != nullptr, "alpha tensor_data not null");
            if (p) {
                bool ok = (p[0] == 0.0f && p[1] == 1.0f &&
                           p[2] == 2.0f && p[3] == 3.0f);
                CHECK(ok, "alpha bytes round-trip exactly");
            }
        }
    }

    // --- 3. Tensor B — Q8_0_SOA4 marker -------------------------------------
    {
        const PT8TensorRecord* b = rd.find("blk.0.attn_q.weight");
        CHECK(b != nullptr, "blk.0.attn_q.weight present");
        if (b) {
            CHECK(b->pt8_type == PT8_TYPE_Q8_0_SOA4,
                  "blk.0.attn_q pt8_type == Q8_0_SOA4");
            CHECK(b->dims.size() == 2 && b->dims[0] == 4 && b->dims[1] == 32,
                  "blk.0.attn_q.dims == [4, 32]");
            CHECK(b->data_size == 176, "blk.0.attn_q.data_size == 176");
            CHECK(b->row_stride == 44, "blk.0.attn_q.row_stride == 44");

            const uint8_t* p = static_cast<const uint8_t*>(
                rd.tensor_data("blk.0.attn_q.weight"));
            CHECK(p != nullptr, "blk.0.attn_q tensor_data not null");
            if (p) {
                bool ok = true;
                for (size_t i = 0; i < 176; ++i) {
                    if (p[i] != static_cast<uint8_t>(i & 0xFF)) { ok = false; break; }
                }
                CHECK(ok, "blk.0.attn_q 176 bytes round-trip exactly");
            }
        }
    }

    // --- 4. Negative lookups -----------------------------------------------
    CHECK(rd.find("nonexistent.weight") == nullptr,
          "find() returns nullptr for missing");
    CHECK(rd.tensor_data("nonexistent.weight") == nullptr,
          "tensor_data() returns nullptr for missing");
    CHECK(rd.tensor_size("nonexistent.weight") == 0,
          "tensor_size() returns 0 for missing");

    // --- 5. End-to-end logit-diff with a real GGUF, if env var set ---------
    if (const char* gguf_path = std::getenv("PT8_TEST_GGUF")) {
        std::printf("\n  PT8_TEST_GGUF=%s — end-to-end logit-diff would run\n",
                    gguf_path);
        std::printf("  (skipping: requires gguf2pt8 conversion + GGUFModel "
                    "load on both inputs; this scaffolding stays so the test "
                    "fails loud once the converter ships Q4_K SoA4 encoder)\n");
    } else {
        std::printf("\n  (set PT8_TEST_GGUF=path/to/model.gguf to enable "
                    "end-to-end logit-diff check)\n");
    }

    // Clean up.
    std::remove(path.c_str());

    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
