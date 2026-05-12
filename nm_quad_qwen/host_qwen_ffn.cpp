/* host_qwen_ffn.cpp — FFN sub-layer verify (RMSNorm + gate + up + SiLU mul) */
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
#define K_DIM 2560
#define M 32
#define BLOCKS_PER_ROW (K_DIM / 256)
#define Q4K_ROW_BYTES (BLOCKS_PER_ROW * 144)
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
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
static float q4k_dot_h(const uint8_t *blk, const float *x) {
    uint16_t db, dmb; std::memcpy(&db, blk + 0, 2); std::memcpy(&dmb, blk + 2, 2);
    float d = fp16_to_fp32(db), dmin = fp16_to_fp32(dmb);
    const uint8_t *sc = blk + 4; const uint8_t *qs = blk + 16;
    float acc = 0; int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is, sc, &s, &m); float d1 = d * (float)s; float m1 = dmin * (float)m;
        get_scale_min_k4(is + 1, sc, &s, &m); float d2 = d * (float)s; float m2 = dmin * (float)m;
        for (int l = 0; l < 32; ++l) {
            acc += (d1 * (float)(qs[l] & 0xF) - m1) * x[j + l];
            acc += (d2 * (float)(qs[l] >> 4) - m2) * x[j + l + 32];
        }
        qs += 32; is += 2;
    }
    return acc;
}

struct GGUFReader {
    std::ifstream f;
    uint64_t off_ffn_norm=0, off_gate=0, off_up=0;
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
            if (t.name == "blk.0.ffn_norm.weight") off_ffn_norm = base + t.off;
            else if (t.name == "blk.0.ffn_gate.weight" && t.type == 12) off_gate = base + t.off;
            else if (t.name == "blk.0.ffn_up.weight" && t.type == 12) off_up = base + t.off;
        }
        return off_ffn_norm && off_gate && off_up;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 6) { std::cerr << "Usage: " << argv[0] << " X G_fn Wgate Wup MUL_OUT\n"; return 1; }
    PL_Addr A_X    = std::strtoul(argv[1], nullptr, 16);
    PL_Addr A_Gn   = std::strtoul(argv[2], nullptr, 16);
    PL_Addr A_Wg   = std::strtoul(argv[3], nullptr, 16);
    PL_Addr A_Wu   = std::strtoul(argv[4], nullptr, 16);
    PL_Addr A_Mul  = std::strtoul(argv[5], nullptr, 16);

    GGUFReader g;
    if (!g.open_(GGUF_PATH) || !g.parse()) return 2;
    std::ifstream raw(GGUF_PATH, std::ios::binary);

    raw.seekg(g.off_ffn_norm);
    std::vector<float> ffn_norm(K_DIM); raw.read((char*)ffn_norm.data(), K_DIM * 4);
    raw.seekg(g.off_gate);
    std::vector<uint8_t> Wgate(M * Q4K_ROW_BYTES); raw.read((char*)Wgate.data(), Wgate.size());
    raw.seekg(g.off_up);
    std::vector<uint8_t> Wup(M * Q4K_ROW_BYTES); raw.read((char*)Wup.data(), Wup.size());

    std::vector<float> x(K_DIM);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < K_DIM; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    /* host ref */
    float sum = 0;
    for (int i = 0; i < K_DIM; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum / (float)K_DIM + EPS);
    std::vector<float> y(K_DIM);
    for (int i = 0; i < K_DIM; ++i) y[i] = x[i] * inv_rms * ffn_norm[i];
    std::vector<float> g_ref(M), u_ref(M), mul_ref(M);
    for (int r = 0; r < M; ++r) {
        float ag = 0, au = 0;
        for (int b = 0; b < BLOCKS_PER_ROW; ++b) {
            ag += q4k_dot_h(Wgate.data() + r * Q4K_ROW_BYTES + b * 144, y.data() + b * 256);
            au += q4k_dot_h(Wup.data()   + r * Q4K_ROW_BYTES + b * 144, y.data() + b * 256);
        }
        g_ref[r] = ag; u_ref[r] = au;
        float silu = ag / (1.0f + std::exp(-ag));
        mul_ref[r] = silu * au;
    }
    std::cerr << "[host] mul_ref[0..3]=" << mul_ref[0] << " " << mul_ref[1] << " " << mul_ref[2] << " " << mul_ref[3] << "\n";

    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    auto upload = [&](const uint8_t *buf, int n, PL_Addr addr){
        std::vector<PL_Word> w(n);
        for (int i = 0; i < n; ++i) w[i] = (PL_Word)buf[i];
        PL_WriteMemBlock(acc, w.data(), addr, n);
    };
    upload(Wgate.data(), Wgate.size(), A_Wg);
    upload(Wup.data(), Wup.size(), A_Wu);
    std::vector<PL_Word> gw(K_DIM), xw(K_DIM);
    for (int i = 0; i < K_DIM; ++i) { std::memcpy(&gw[i], &ffn_norm[i], 4); std::memcpy(&xw[i], &x[i], 4); }
    PL_WriteMemBlock(acc, gw.data(), A_Gn, K_DIM);
    PL_WriteMemBlock(acc, xw.data(), A_X, K_DIM);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> mw(M);
    PL_ReadMemBlock(acc, mw.data(), A_Mul, M);
    std::vector<float> mul_n(M);
    for (int i = 0; i < M; ++i) std::memcpy(&mul_n[i], &mw[i], 4);

    float maxd = 0;
    for (int i = 0; i < M; ++i) {
        float d = std::fabs(mul_n[i] - mul_ref[i]);
        if (d > maxd) maxd = d;
    }
    std::cout << "[verify] FFN mul max_diff=" << maxd << "\n";
    std::cout << "[perf] wall: " << ms << " ms\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-4f ? 0 : 100;
}
