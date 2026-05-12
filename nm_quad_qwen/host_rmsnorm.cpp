/* host_rmsnorm.cpp — RMSNorm test driver against host CPU reference.
 * Loads gamma из GGUF (blk.0.attn_norm.weight = 2560 fp32), random x,
 * compares NMC vs host result.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include "nm_quad_load.h"
#include "io_host.h"

static const char *NMC_PART  = "./nmc_part.abs";
static const char *GGUF_PATH = "/home/<user>/gguf/qwen3-4b-q4km.gguf";
#define K 2560
#define EPS 1.0e-6f

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
               mant &= 0x3FF; exp++;
               bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 0x1F) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

struct GGUFReader {
    std::ifstream f;
    uint64_t attn_norm_offset = 0;
    uint32_t attn_norm_type = 0;
    bool open_(const char *p) { f.open(p, std::ios::binary); return f.is_open(); }
    template<typename T> bool rd(T &v) { f.read((char*)&v, sizeof(T)); return f.good(); }
    bool rd_str(std::string &s) { uint64_t n; if (!rd(n)) return false; s.resize(n); f.read(s.data(), n); return f.good(); }
    bool skip_value(uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
            case 2: case 3: f.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
            case 8: { std::string s; if (!rd_str(s)) return false; break; }
            case 9: { uint32_t at; uint64_t an; if(!rd(at)||!rd(an)) return false;
                      for (uint64_t i = 0; i < an; ++i) if (!skip_value(at)) return false; break; }
            default: return false;
        }
        return f.good();
    }
    bool parse() {
        char m[4]; rd(m); if (memcmp(m, "GGUF", 4) != 0) return false;
        uint32_t ver; rd(ver);
        uint64_t nt, nk; rd(nt); rd(nk);
        for (uint64_t i = 0; i < nk; ++i) { std::string k; rd_str(k); uint32_t t; rd(t); if (!skip_value(t)) return false; }
        struct TI { std::string name; uint32_t nd; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
        std::vector<TI> tt(nt);
        for (uint64_t i = 0; i < nt; ++i) {
            rd_str(tt[i].name); rd(tt[i].nd);
            tt[i].dims.resize(tt[i].nd);
            for (uint32_t d = 0; d < tt[i].nd; ++d) rd(tt[i].dims[d]);
            rd(tt[i].type); rd(tt[i].off);
        }
        uint64_t pos = f.tellg();
        uint64_t aln = 32;
        uint64_t base = (pos + aln - 1) & ~(aln - 1);
        for (auto &t : tt) {
            if (t.name == "blk.0.attn_norm.weight") {
                attn_norm_offset = base + t.off;
                attn_norm_type = t.type;
                std::cerr << "[gguf] " << t.name << " offset=" << attn_norm_offset
                          << " type=" << attn_norm_type << " dim=" << t.dims[0] << "\n";
                return true;
            }
        }
        return false;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 4) { std::cerr << "Usage: " << argv[0] << " <ADDR_X> <ADDR_G> <ADDR_Y>\n"; return 1; }
    PL_Addr ADDR_X = std::strtoul(argv[1], nullptr, 16);
    PL_Addr ADDR_G = std::strtoul(argv[2], nullptr, 16);
    PL_Addr ADDR_Y = std::strtoul(argv[3], nullptr, 16);

    GGUFReader g;
    if (!g.open_(GGUF_PATH) || !g.parse()) { std::cerr << "gguf fail\n"; return 2; }

    /* Read gamma. type=0 = F32, type=1 = F16 */
    std::ifstream raw(GGUF_PATH, std::ios::binary);
    raw.seekg(g.attn_norm_offset);
    std::vector<float> gamma(K);
    if (g.attn_norm_type == 0) {
        raw.read((char*)gamma.data(), K * 4);
    } else if (g.attn_norm_type == 1) {
        std::vector<uint16_t> h(K);
        raw.read((char*)h.data(), K * 2);
        for (int i = 0; i < K; ++i) gamma[i] = fp16_to_fp32(h[i]);
    } else {
        std::cerr << "Unsupported gamma type " << g.attn_norm_type << "\n"; return 3;
    }
    std::cerr << "[host] gamma[0..3] = " << gamma[0] << " " << gamma[1] << " "
              << gamma[2] << " " << gamma[3] << "\n";

    /* Random x (deterministic seed) */
    std::vector<float> x(K);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < K; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    /* Host reference RMSNorm */
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) sum += x[i] * x[i];
    float mean = sum / (float)K;
    float inv_rms = 1.0f / std::sqrt(mean + EPS);
    std::vector<float> y_host(K);
    for (int i = 0; i < K; ++i) y_host[i] = x[i] * inv_rms * gamma[i];
    std::cerr << "[host] sum=" << sum << " mean=" << mean << " inv_rms=" << inv_rms << "\n";
    std::cerr << "[host] y_ref[0..3] = " << y_host[0] << " " << y_host[1] << " "
              << y_host[2] << " " << y_host[3] << "\n";

    /* NMC */
    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 4;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 5;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 6;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 7;

    std::vector<PL_Word> xw(K), gw(K);
    for (int i = 0; i < K; ++i) { std::memcpy(&xw[i], &x[i], 4); std::memcpy(&gw[i], &gamma[i], 4); }
    PL_WriteMemBlock(acc, xw.data(), ADDR_X, K);
    PL_WriteMemBlock(acc, gw.data(), ADDR_G, K);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> yw(K);
    PL_ReadMemBlock(acc, yw.data(), ADDR_Y, K);
    std::vector<float> y_nmc(K);
    for (int i = 0; i < K; ++i) std::memcpy(&y_nmc[i], &yw[i], 4);

    float maxd = 0, sumsq = 0;
    for (int i = 0; i < K; ++i) {
        float d = y_nmc[i] - y_host[i];
        if (std::fabs(d) > maxd) maxd = std::fabs(d);
        sumsq += d * d;
    }
    std::cout << "[verify] max_diff=" << maxd << " rms=" << std::sqrt(sumsq / K) << "\n";
    std::cout << "[perf] wall: " << ms << " ms, " << (2.0 * K) / (ms * 1e3) << " MFLOPS\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-4f ? 0 : 100;
}
