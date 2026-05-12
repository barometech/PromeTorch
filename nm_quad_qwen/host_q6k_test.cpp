/* host_q6k_test.cpp — verify Q6_K dequant vs ggml-style host reference */
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

#define BLOCK_BYTES 210
#define VALS 256

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

static void dequant_q6k_host(const uint8_t *blk, float *out) {
    /* ggml-style Q6_K dequant */
    const uint8_t *ql = blk + 0;
    const uint8_t *qh = blk + 128;
    const int8_t  *sc = (const int8_t*)(blk + 192);
    uint16_t db; std::memcpy(&db, blk + 208, 2);
    float d = fp16_to_fp32(db);

    for (int i = 0; i < 256; ++i) {
        int is = i / 16;
        int ql_idx = (i % 64) + 64 * (i / 128);
        int ql_shift = 4 * ((i / 32) & 1);
        int q_lo = (ql[ql_idx] >> ql_shift) & 0xF;
        int qh_idx = (i % 32) + 32 * (i / 128);
        int qh_shift = 2 * ((i / 16) & 3);
        int q_hi = (qh[qh_idx] >> qh_shift) & 0x3;
        int q6 = q_lo | (q_hi << 4);
        out[i] = d * (float)sc[is] * (float)(q6 - 32);
    }
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
        for (auto &t : tt) {
            if (t.name == "blk.0.attn_v.weight" && t.type == 14) {
                attn_v_off = base + t.off;
                std::cerr << "[gguf] attn_v Q6_K offset=" << attn_v_off
                          << " dims=" << t.dims[0] << "x" << t.dims[1] << "\n";
                return true;
            }
        }
        return false;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " <ADDR_BLK> <ADDR_OUT>\n"; return 1; }
    PL_Addr ADDR_BLK = std::strtoul(argv[1], nullptr, 16);
    PL_Addr ADDR_OUT = std::strtoul(argv[2], nullptr, 16);

    GGUFReader g;
    if (!g.open_(GGUF_PATH) || !g.parse()) { std::cerr << "gguf fail\n"; return 2; }
    std::ifstream raw(GGUF_PATH, std::ios::binary);
    raw.seekg(g.attn_v_off);
    std::vector<uint8_t> blk(BLOCK_BYTES);
    raw.read((char*)blk.data(), BLOCK_BYTES);

    std::vector<float> y_host(VALS);
    dequant_q6k_host(blk.data(), y_host.data());
    std::cerr << "[host] first8 = ";
    for (int i = 0; i < 8; ++i) std::cerr << y_host[i] << " ";
    std::cerr << "\n";

    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    std::vector<PL_Word> bw(BLOCK_BYTES);
    for (int i = 0; i < BLOCK_BYTES; ++i) bw[i] = (PL_Word)blk[i];
    PL_WriteMemBlock(acc, bw.data(), ADDR_BLK, BLOCK_BYTES);

    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }

    std::vector<PL_Word> ow(VALS);
    PL_ReadMemBlock(acc, ow.data(), ADDR_OUT, VALS);
    std::vector<float> y_nmc(VALS);
    for (int i = 0; i < VALS; ++i) std::memcpy(&y_nmc[i], &ow[i], 4);

    float maxd = 0;
    for (int i = 0; i < VALS; ++i) {
        float d = std::fabs(y_nmc[i] - y_host[i]);
        if (d > maxd) maxd = d;
    }
    std::cout << "[verify] Q6_K dequant max_diff=" << maxd << "\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-4f ? 0 : 100;
}
