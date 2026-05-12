/* host_qwen_step2.cpp — verify RMSNorm + Q-proj composition vs host CPU.
 * Loads gamma (attn_norm) + first 32 rows of attn_q.weight from Qwen3-4B GGUF.
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
#define M 2560
#define BLOCKS_PER_ROW (K / 256)
#define ROW_BYTES (BLOCKS_PER_ROW * 144)
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
static float q4k_block_dot(const uint8_t *blk, const float *x) {
    uint16_t db, dmb;
    std::memcpy(&db, blk + 0, 2);
    std::memcpy(&dmb, blk + 2, 2);
    float d = fp16_to_fp32(db), dmin = fp16_to_fp32(dmb);
    const uint8_t *sc = blk + 4;
    const uint8_t *qs = blk + 16;
    float acc = 0; int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is, sc, &s, &m);
        float d1 = d * (float)s; float m1 = dmin * (float)m;
        get_scale_min_k4(is + 1, sc, &s, &m);
        float d2 = d * (float)s; float m2 = dmin * (float)m;
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
    uint64_t attn_norm_off = 0;
    uint64_t attn_q_off = 0;
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
            if (t.name == "blk.0.attn_norm.weight") attn_norm_off = base + t.off;
            else if (t.name == "blk.0.attn_q.weight" && t.type == 12) attn_q_off = base + t.off;
        }
        return attn_norm_off && attn_q_off;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 6) { std::cerr << "Usage: " << argv[0] << " <ADDR_X> <ADDR_G> <ADDR_W> <ADDR_Y> <ADDR_Q>\n"; return 1; }
    PL_Addr ADDR_X = std::strtoul(argv[1], nullptr, 16);
    PL_Addr ADDR_G = std::strtoul(argv[2], nullptr, 16);
    PL_Addr ADDR_W = std::strtoul(argv[3], nullptr, 16);
    PL_Addr ADDR_Y = std::strtoul(argv[4], nullptr, 16);
    PL_Addr ADDR_Q = std::strtoul(argv[5], nullptr, 16);

    GGUFReader g;
    if (!g.open_(GGUF_PATH) || !g.parse()) { std::cerr << "gguf fail\n"; return 2; }
    std::ifstream raw(GGUF_PATH, std::ios::binary);

    /* gamma */
    raw.seekg(g.attn_norm_off);
    std::vector<float> gamma(K);
    raw.read((char*)gamma.data(), K * 4);
    /* W (first 32 rows) */
    raw.seekg(g.attn_q_off);
    std::vector<uint8_t> W(M * ROW_BYTES);
    raw.read((char*)W.data(), M * ROW_BYTES);

    /* random x */
    std::vector<float> x(K);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < K; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    /* host reference */
    float sum = 0;
    for (int i = 0; i < K; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum / (float)K + EPS);
    std::vector<float> y(K);
    for (int i = 0; i < K; ++i) y[i] = x[i] * inv_rms * gamma[i];
    std::vector<float> q_ref(M);
    for (int r = 0; r < M; ++r) {
        float a = 0;
        for (int b = 0; b < BLOCKS_PER_ROW; ++b)
            a += q4k_block_dot(W.data() + r * ROW_BYTES + b * 144, y.data() + b * 256);
        q_ref[r] = a;
    }
    std::cerr << "[host] inv_rms=" << inv_rms << " q_ref[0..3]=" << q_ref[0] << " "
              << q_ref[1] << " " << q_ref[2] << " " << q_ref[3] << "\n";

    /* NMC */
    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    /* Write large W first, then gamma, then x (kernel reads x first) */
    std::vector<PL_Word> ww(M * ROW_BYTES);
    for (int i = 0; i < M * ROW_BYTES; ++i) ww[i] = (PL_Word)W[i];
    PL_WriteMemBlock(acc, ww.data(), ADDR_W, M * ROW_BYTES);

    std::vector<PL_Word> gw(K), xw(K);
    for (int i = 0; i < K; ++i) { std::memcpy(&gw[i], &gamma[i], 4); std::memcpy(&xw[i], &x[i], 4); }
    PL_WriteMemBlock(acc, gw.data(), ADDR_G, K);
    PL_WriteMemBlock(acc, xw.data(), ADDR_X, K);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> qw(M);
    PL_ReadMemBlock(acc, qw.data(), ADDR_Q, M);
    std::vector<float> q_nmc(M);
    for (int i = 0; i < M; ++i) std::memcpy(&q_nmc[i], &qw[i], 4);

    float maxd = 0, sumsq = 0;
    for (int i = 0; i < M; ++i) {
        float d = q_nmc[i] - q_ref[i];
        if (std::fabs(d) > maxd) maxd = std::fabs(d);
        sumsq += d * d;
    }
    std::cout << "[verify] step1 max_diff=" << maxd << " rms=" << std::sqrt(sumsq / M) << "\n";
    std::cout << "[perf] kernel wall: " << ms << " ms\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-4f ? 0 : 100;
}
