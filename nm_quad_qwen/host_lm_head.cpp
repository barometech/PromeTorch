/* host_lm_head.cpp — verify Q6_K lm_head subset на NMC4 */
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "nm_quad_load.h"
#include "io_host.h"

static const char *GGUF_PATH = "/home/<user>/gguf/qwen3-4b-q4km.gguf";
static const char *NMC_PART = "./nmc_part.abs";

#define K_DIM 2560
#define M_OUT 128
#define BLOCKS_PER_ROW (K_DIM / 256)
#define Q6K_ROW_BYTES (BLOCKS_PER_ROW * 210)

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
               mant &= 0x3FF; exp++;
               bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 0x1F) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    float r; std::memcpy(&r, &bits, 4); return r;
}

static float q6k_dot_h(const uint8_t *blk, const float *x) {
    const uint8_t *ql = blk; const uint8_t *qh = blk + 128;
    const int8_t *sc = (const int8_t*)(blk + 192);
    uint16_t db; std::memcpy(&db, blk + 208, 2); float d = fp16_to_fp32(db);
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (int i = 0; i < 256; i += 8) {
        for (int b = 0; b < 8; ++b) {
            int idx = i + b;
            int is = idx / 16;
            int q_lo = (ql[(idx%64) + 64*(idx/128)] >> (4*((idx/64)&1))) & 0xF;
            int q_hi = (qh[(idx%32) + 32*(idx/128)] >> (2*((idx/32)&3))) & 0x3;
            float v = d * (float)sc[is] * (float)((q_lo | (q_hi << 4)) - 32) * x[idx];
            switch (b) { case 0:a0+=v;break; case 1:a1+=v;break; case 2:a2+=v;break; case 3:a3+=v;break;
                         case 4:a4+=v;break; case 5:a5+=v;break; case 6:a6+=v;break; case 7:a7+=v;break; }
        }
    }
    return (a0+a1)+(a2+a3)+(a4+a5)+(a6+a7);
}

struct TI { std::string name; uint32_t nd; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };

static uint64_t parse_gguf(const char *path, const std::string &target, uint32_t &type_out) {
    std::ifstream f(path, std::ios::binary);
    char m[4]; f.read(m,4);
    if (std::memcmp(m, "GGUF", 4) != 0) return 0;
    uint32_t ver; f.read((char*)&ver, 4);
    uint64_t nt, nk; f.read((char*)&nt, 8); f.read((char*)&nk, 8);
    auto rdstr = [&](std::string &s) { uint64_t n; f.read((char*)&n, 8); s.resize(n); f.read(s.data(), n); };
    std::function<bool(uint32_t)> skip = [&](uint32_t t) -> bool {
        switch (t) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
            case 2: case 3: f.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
            case 8: { std::string s; rdstr(s); break; }
            case 9: { uint32_t at; uint64_t an; f.read((char*)&at,4); f.read((char*)&an,8);
                      for (uint64_t i=0;i<an;++i) if (!skip(at)) return false; break; }
            default: return false;
        }
        return f.good();
    };
    for (uint64_t i = 0; i < nk; ++i) { std::string k; rdstr(k); uint32_t t; f.read((char*)&t,4); if (!skip(t)) return 0; }
    std::vector<TI> tt(nt);
    uint64_t off_found = 0;
    for (uint64_t i = 0; i < nt; ++i) {
        rdstr(tt[i].name); f.read((char*)&tt[i].nd, 4);
        tt[i].dims.resize(tt[i].nd);
        for (uint32_t d = 0; d < tt[i].nd; ++d) f.read((char*)&tt[i].dims[d], 8);
        f.read((char*)&tt[i].type, 4); f.read((char*)&tt[i].off, 8);
    }
    uint64_t pos = f.tellg();
    uint64_t aln = 32;
    uint64_t base = (pos + aln - 1) & ~(aln - 1);
    for (auto &t : tt) if (t.name == target) { off_found = base + t.off; type_out = t.type; break; }
    return off_found;
}

int main(int argc, char *argv[]) {
    if (argc < 4) { std::cerr << "Usage: " << argv[0] << " A_W A_X A_LOGITS\n"; return 1; }
    PL_Addr A_W = std::strtoul(argv[1], nullptr, 16);
    PL_Addr A_X = std::strtoul(argv[2], nullptr, 16);
    PL_Addr A_L = std::strtoul(argv[3], nullptr, 16);

    uint32_t out_type = 0;
    uint64_t off_out = parse_gguf(GGUF_PATH, "token_embd.weight", out_type);
    if (!off_out) { std::cerr << "token_embd.weight not found\n"; return 2; }
    std::cerr << "[gguf] output.weight type=" << out_type << " off=" << off_out << "\n";

    std::ifstream raw(GGUF_PATH, std::ios::binary);
    std::vector<uint8_t> W(M_OUT * Q6K_ROW_BYTES);
    raw.seekg(off_out); raw.read((char*)W.data(), W.size());

    std::vector<float> x(K_DIM);
    uint32_t rng = 0x42;
    for (int i = 0; i < K_DIM; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    std::vector<float> logits_h(M_OUT);
    for (int r = 0; r < M_OUT; ++r) {
        float a = 0;
        for (int b = 0; b < BLOCKS_PER_ROW; ++b)
            a += q6k_dot_h(W.data() + r * Q6K_ROW_BYTES + b * 210, x.data() + b * 256);
        logits_h[r] = a;
    }
    std::cerr << "[host] logits[0..3]: " << logits_h[0] << " " << logits_h[1] << " " << logits_h[2] << " " << logits_h[3] << "\n";

    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    std::vector<PL_Word> w(W.size());
    for (size_t i = 0; i < W.size(); ++i) w[i] = (PL_Word)W[i];
    PL_WriteMemBlock(acc, w.data(), A_W, W.size());

    std::vector<PL_Word> xw(K_DIM);
    for (int i = 0; i < K_DIM; ++i) std::memcpy(&xw[i], &x[i], 4);
    PL_WriteMemBlock(acc, xw.data(), A_X, K_DIM);

    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }

    std::vector<PL_Word> lw(M_OUT);
    PL_ReadMemBlock(acc, lw.data(), A_L, M_OUT);
    std::vector<float> logits_n(M_OUT);
    for (int i = 0; i < M_OUT; ++i) std::memcpy(&logits_n[i], &lw[i], 4);

    float maxd = 0; int worst = -1;
    for (int i = 0; i < M_OUT; ++i) {
        float d = std::fabs(logits_n[i] - logits_h[i]);
        if (d > maxd) { maxd = d; worst = i; }
    }
    std::cout << "[verify] lm_head subset M=128 max_diff=" << maxd << " (row " << worst << ")\n";
    std::cout << "[nmc] logits[0..3]: " << logits_n[0] << " " << logits_n[1] << " " << logits_n[2] << " " << logits_n[3] << "\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-3 ? 0 : 100;
}
