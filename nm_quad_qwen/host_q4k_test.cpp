/* ============================================================================
 * host_q4k_test.cpp — extracts first Q4_K block from Qwen GGUF, uploads to
 * NMC, runs dequant test, reads result. Verifies Q4_K dequant kernel on NMC4.
 *
 * Also computes host-side dequant of same block and prints diff stats.
 * ============================================================================ */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include "nm_quad_load.h"
#include "io_host.h"

static const char *NMC_PART = "./nmc_part.abs";
static const char *GGUF_PATH = "/home/<user>/gguf/qwen3-4b-q4km.gguf";

/* Symbol addresses for nmc_q4k_test.abs — patched from nm output below.
 * If you change the kernel, re-run `nm nmc_part.abs | grep q4k_` and update. */
static PL_Addr ADDR_Q4K_BLOCK  = 0;
static PL_Addr ADDR_Q4K_OUTPUT = 0;
static PL_Addr ADDR_Q4K_DONE   = 0;

/* ---- Host-side Q4_K dequant for cross-check ---- */
static float fp16_to_fp32_host(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
               mant &= 0x3FF; exp++;
               bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}
static void get_scale_min_k4_host(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
static void dequant_q4k_host(const uint8_t *src, float *dst) {
    uint16_t d_bits, dmin_bits;
    std::memcpy(&d_bits, src + 0, 2);
    std::memcpy(&dmin_bits, src + 2, 2);
    float d = fp16_to_fp32_host(d_bits);
    float dmin = fp16_to_fp32_host(dmin_bits);
    const uint8_t *scales = src + 4;
    const uint8_t *qs = src + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4_host(is, scales, &sc, &m);
        float d1 = d * (float)sc;  float m1 = dmin * (float)m;
        get_scale_min_k4_host(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc;  float m2 = dmin * (float)m;
        for (int l = 0; l < 32; ++l) {
            dst[j + l +  0] = d1 * (float)(qs[l] & 0xF) - m1;
            dst[j + l + 32] = d2 * (float)(qs[l] >> 4) - m2;
        }
        qs += 32; is += 2;
    }
}

/* ---- GGUF parsing: enough to find first Q4_K tensor data offset ---- */
struct GGUFReader {
    std::ifstream f;
    uint64_t tensor_data_start = 0;
    uint64_t first_q4k_offset  = 0;
    int      first_q4k_n_elem  = 0;

    bool open(const char *path) {
        f.open(path, std::ios::binary);
        return f.is_open();
    }
    template<typename T> bool rd(T &v) { f.read((char*)&v, sizeof(T)); return f.good(); }
    bool rd_str(std::string &s) {
        uint64_t n; if (!rd(n)) return false;
        s.resize(n); f.read(s.data(), n); return f.good();
    }
    bool skip_value(uint32_t type) {
        /* GGUF value types (gguf-py spec):
         * 0=uint8 1=int8 2=uint16 3=int16 4=uint32 5=int32 6=float32
         * 7=bool 8=string 9=array 10=uint64 11=int64 12=float64 */
        switch (type) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
            case 2: case 3: f.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
            case 8: { std::string s; if (!rd_str(s)) return false; break; }
            case 9: {
                uint32_t arr_t; if (!rd(arr_t)) return false;
                uint64_t arr_n; if (!rd(arr_n)) return false;
                for (uint64_t i = 0; i < arr_n; ++i) if (!skip_value(arr_t)) return false;
                break;
            }
            default: std::cerr << "Unknown value type " << type << "\n"; return false;
        }
        return f.good();
    }
    bool parse() {
        char magic[4]; rd(magic);
        if (memcmp(magic, "GGUF", 4) != 0) { std::cerr << "Not GGUF\n"; return false; }
        uint32_t ver; rd(ver);
        uint64_t n_tensors, n_kv;
        rd(n_tensors); rd(n_kv);
        std::cerr << "[gguf] ver=" << ver << " tensors=" << n_tensors << " kv=" << n_kv << "\n";

        /* skip all KV */
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key; rd_str(key);
            uint32_t t; rd(t);
            if (!skip_value(t)) { std::cerr << "skip_value failed on kv " << i << " (" << key << ")\n"; return false; }
        }

        /* tensor infos */
        struct TInfo { std::string name; uint32_t n_dims; std::vector<uint64_t> dims; uint32_t type; uint64_t offset; };
        std::vector<TInfo> tensors(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            auto &t = tensors[i];
            rd_str(t.name);
            rd(t.n_dims);
            t.dims.resize(t.n_dims);
            for (uint32_t d = 0; d < t.n_dims; ++d) rd(t.dims[d]);
            rd(t.type);
            rd(t.offset);
        }

        /* align to alignment (default 32) */
        uint64_t pos = f.tellg();
        uint64_t align = 32;
        tensor_data_start = (pos + align - 1) & ~(align - 1);

        /* find first Q4_K tensor — type 12 */
        for (auto &t : tensors) {
            if (t.type == 12) {
                first_q4k_offset = tensor_data_start + t.offset;
                first_q4k_n_elem = 1;
                for (auto d : t.dims) first_q4k_n_elem *= (int)d;
                std::cerr << "[gguf] first Q4_K tensor: " << t.name
                          << " shape=";
                for (auto d : t.dims) std::cerr << d << " ";
                std::cerr << "(" << first_q4k_n_elem << " elems) "
                          << "offset=" << first_q4k_offset << "\n";
                return true;
            }
        }
        std::cerr << "[gguf] no Q4_K tensor found\n";
        return false;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <ADDR_BLOCK> <ADDR_OUTPUT> <ADDR_DONE>\n"
                  << "Get hex addresses via: nm nmc_part.abs | grep -E 'q4k_block|q4k_output|q4k_done'\n";
        return 1;
    }
    ADDR_Q4K_BLOCK  = std::strtoul(argv[1], nullptr, 16);
    ADDR_Q4K_OUTPUT = std::strtoul(argv[2], nullptr, 16);
    ADDR_Q4K_DONE   = std::strtoul(argv[3], nullptr, 16);
    std::cerr << "[host] using addrs: block=0x" << std::hex << ADDR_Q4K_BLOCK
              << " output=0x" << ADDR_Q4K_OUTPUT
              << " done=0x" << ADDR_Q4K_DONE << std::dec << "\n";

    /* ---- parse GGUF, get first Q4_K tensor offset ---- */
    GGUFReader g;
    if (!g.open(GGUF_PATH)) { std::cerr << "open " << GGUF_PATH << " failed\n"; return 1; }
    if (!g.parse()) return 2;

    /* read first 144 bytes (one Q4_K block) */
    uint8_t block[144];
    std::ifstream raw(GGUF_PATH, std::ios::binary);
    raw.seekg(g.first_q4k_offset);
    raw.read((char*)block, 144);
    raw.close();

    std::cerr << "[host] first 16 bytes of block: ";
    for (int i = 0; i < 16; ++i) std::fprintf(stderr, "%02x ", block[i]);
    std::cerr << "\n";

    /* ---- host-side dequant for ground truth ---- */
    float host_out[256];
    dequant_q4k_host(block, host_out);
    float mn = host_out[0], mx = host_out[0], sum = 0;
    for (int i = 0; i < 256; ++i) { if (host_out[i]<mn) mn=host_out[i]; if (host_out[i]>mx) mx=host_out[i]; sum += host_out[i]; }
    float mean = sum / 256, var = 0;
    for (int i = 0; i < 256; ++i) { float d = host_out[i]-mean; var += d*d; }
    float std_ = std::sqrt(var / 256);
    std::cerr << "[host] dequant: first 8 = ";
    for (int i = 0; i < 8; ++i) std::cerr << host_out[i] << " ";
    std::cerr << "\n[host] stats: min=" << mn << " max=" << mx
              << " mean=" << mean << " std=" << std_ << "\n";

    /* ---- upload to NMC, run, read back ---- */
    PL_Board *board = nullptr;
    unsigned int n_b = 0;
    if (PL_GetBoardCount(&n_b) != PL_OK || n_b < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    /* Convert 144 bytes → 144 words for NMC (one byte per word) */
    std::vector<PL_Word> wblk(144);
    for (int i = 0; i < 144; ++i) wblk[i] = (PL_Word)block[i];
    if (PL_WriteMemBlock(acc, wblk.data(), ADDR_Q4K_BLOCK, 144) != PL_OK) {
        std::cerr << "WriteMemBlock failed\n"; return 7;
    }

    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);

    /* wait */
    PL_Word st = 0;
    while (1) {
        if (PL_GetStatus(acc, &st) != PL_OK) break;
        if (st == PROGRAM_PROGRESS) continue;
        break;
    }
    std::cerr << "[host] NMC finished status=" << st << "\n";

    /* read q4k_output back (256 floats = 256 words) */
    std::vector<PL_Word> wout(256);
    PL_ReadMemBlock(acc, wout.data(), ADDR_Q4K_OUTPUT, 256);
    float nmc_out[256];
    for (int i = 0; i < 256; ++i) std::memcpy(&nmc_out[i], &wout[i], 4);

    std::cerr << "[nmc] first 8 = ";
    for (int i = 0; i < 8; ++i) std::cerr << nmc_out[i] << " ";
    std::cerr << "\n";

    /* compare */
    float max_diff = 0, sumsq_diff = 0;
    for (int i = 0; i < 256; ++i) {
        float d = nmc_out[i] - host_out[i];
        if (std::fabs(d) > max_diff) max_diff = std::fabs(d);
        sumsq_diff += d * d;
    }
    std::cout << "[verify] max_diff=" << max_diff
              << " rms_diff=" << std::sqrt(sumsq_diff / 256) << "\n";
    if (max_diff < 1e-5f) std::cout << "[verify] ✅ Q4_K kernel matches host\n";
    else std::cout << "[verify] ⚠ deviation detected\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return 0;
}
