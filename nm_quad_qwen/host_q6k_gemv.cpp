/* host_q6k_gemv.cpp — verify Q6_K GEMV vs host reference */
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

#define M 32
#define K 2560
#define BLOCKS_PER_ROW (K / 256)
#define ROW_BYTES (BLOCKS_PER_ROW * 210)

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

static float q6k_block_dot_host(const uint8_t *blk, const float *x) {
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t  *sc = (const int8_t*)(blk + 192);
    uint16_t db; std::memcpy(&db, blk + 208, 2);
    float d = fp16_to_fp32(db);
    float acc = 0;
    for (int i = 0; i < 256; ++i) {
        int is = i / 16;
        int ql_idx = (i % 64) + 64 * (i / 128);
        int q_lo = (ql[ql_idx] >> (4 * ((i / 64) & 1))) & 0xF;
        int qh_idx = (i % 32) + 32 * (i / 128);
        int q_hi = (qh[qh_idx] >> (2 * ((i / 32) & 3))) & 0x3;
        float v = d * (float)sc[is] * (float)((q_lo | (q_hi << 4)) - 32);
        acc += v * x[i];
    }
    return acc;
}

struct GGUFReader {
    std::ifstream f;
    uint64_t attn_v_off = 0;
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
        for (auto &t : tt)
            if (t.name == "blk.0.attn_v.weight" && t.type == 14) { attn_v_off = base + t.off; return true; }
        return false;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 4) { std::cerr << "Usage: " << argv[0] << " <ADDR_W> <ADDR_X> <ADDR_Y>\n"; return 1; }
    PL_Addr ADDR_W = std::strtoul(argv[1], nullptr, 16);
    PL_Addr ADDR_X = std::strtoul(argv[2], nullptr, 16);
    PL_Addr ADDR_Y = std::strtoul(argv[3], nullptr, 16);

    GGUFReader g;
    if (!g.open_(GGUF_PATH) || !g.parse()) return 2;
    std::ifstream raw(GGUF_PATH, std::ios::binary);
    raw.seekg(g.attn_v_off);
    std::vector<uint8_t> W(M * ROW_BYTES);
    raw.read((char*)W.data(), W.size());

    std::vector<float> x(K);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < K; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    std::vector<float> y_ref(M);
    for (int r = 0; r < M; ++r) {
        float a = 0;
        for (int b = 0; b < BLOCKS_PER_ROW; ++b)
            a += q6k_block_dot_host(W.data() + r * ROW_BYTES + b * 210, x.data() + b * 256);
        y_ref[r] = a;
    }
    std::cerr << "[host] y_ref[0..3]=" << y_ref[0] << " " << y_ref[1] << " " << y_ref[2] << " " << y_ref[3] << "\n";

    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    std::vector<PL_Word> ww(M * ROW_BYTES);
    for (int i = 0; i < M * ROW_BYTES; ++i) ww[i] = (PL_Word)W[i];
    PL_WriteMemBlock(acc, ww.data(), ADDR_W, M * ROW_BYTES);
    std::vector<PL_Word> xw(K);
    for (int i = 0; i < K; ++i) std::memcpy(&xw[i], &x[i], 4);
    PL_WriteMemBlock(acc, xw.data(), ADDR_X, K);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> yw(M);
    PL_ReadMemBlock(acc, yw.data(), ADDR_Y, M);
    std::vector<float> y_nmc(M);
    for (int i = 0; i < M; ++i) std::memcpy(&y_nmc[i], &yw[i], 4);

    float maxd = 0;
    for (int i = 0; i < M; ++i) {
        float d = std::fabs(y_nmc[i] - y_ref[i]);
        if (d > maxd) maxd = d;
    }
    std::cout << "[verify] Q6_K GEMV max_diff=" << maxd << "\n";
    std::cout << "[perf] wall: " << ms << " ms\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-3f ? 0 : 100;
}
