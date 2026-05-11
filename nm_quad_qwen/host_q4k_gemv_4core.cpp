/* ============================================================================
 * host_q4k_gemv_4core.cpp — Q4_K GEMV K=2560, parallelized over 4 cores of 1 cluster.
 * Each core processes M/4 = 8 rows; total M=32.
 * Measures speedup vs single-core baseline.
 *
 * Note: NM Quad has 4 clusters × 4 NMC4 cores = 16 cores total.
 * This test uses cluster 0 only. Each cluster has independent EMI ~5GB.
 * ============================================================================ */

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

#define M_PER_CORE 8        /* 32 / 4 cores */
#define M_TOTAL 32
#define K 2560
#define BLOCKS_PER_ROW (K / 256)
#define ROW_BYTES (BLOCKS_PER_ROW * 144)
#define NBYTES_W_PER_CORE (M_PER_CORE * ROW_BYTES)
#define NCORES 4

/* Host ref */
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
    float acc = 0;
    int is = 0;
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
static float q4k_row_dot(const uint8_t *row, const float *x) {
    float a = 0;
    for (int b = 0; b < BLOCKS_PER_ROW; ++b)
        a += q4k_block_dot(row + b * 144, x + b * 256);
    return a;
}

struct GGUFReader {
    std::ifstream f;
    uint64_t attn_q_offset = 0;
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
            if (t.name == "blk.0.attn_q.weight" && t.type == 12) { attn_q_offset = base + t.off; return true; }
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
    raw.seekg(g.attn_q_offset);
    std::vector<uint8_t> W(M_TOTAL * ROW_BYTES);
    raw.read((char*)W.data(), W.size());

    std::vector<float> x(K);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < K; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    std::vector<float> y_host(M_TOTAL);
    for (int r = 0; r < M_TOTAL; ++r)
        y_host[r] = q4k_row_dot(W.data() + r * ROW_BYTES, x.data());

    PL_Board *board = nullptr;
    unsigned int n_b = 0;
    if (PL_GetBoardCount(&n_b) != PL_OK || n_b < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    /* Get access to all 4 cores of cluster 0 */
    PL_Access *acc[NCORES] = {nullptr};
    for (int c = 0; c < NCORES; ++c) {
        PL_CoreNo core; core.cluster_id = 0; core.nm_id = c;
        if (PL_GetAccess(board, &core, &acc[c]) != PL_OK) return 5;
        if (PL_LoadProgramFile(acc[c], NMC_PART) != PL_OK) return 6;
    }

    /* Upload x + full W once (cluster EMI is shared by all 4 cores). */
    std::vector<PL_Word> xw(K);
    std::memcpy(xw.data(), x.data(), K * 4);
    PL_WriteMemBlock(acc[0], xw.data(), ADDR_X, K);

    int nbytes_w_full = M_TOTAL * ROW_BYTES;
    std::vector<PL_Word> ww(nbytes_w_full);
    for (int i = 0; i < nbytes_w_full; ++i) ww[i] = (PL_Word)W[i];
    PL_WriteMemBlock(acc[0], ww.data(), ADDR_W, nbytes_w_full);

    /* Launch all 4 cores */
    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc[NCORES] = {nullptr};
    for (int c = 0; c < NCORES; ++c)
        svc[c] = IO_ServiceStart(NMC_PART, acc[c], nullptr, nullptr, nullptr);

    /* Wait for all to finish */
    for (int c = 0; c < NCORES; ++c) {
        PL_Word st = 0;
        while (1) {
            if (PL_GetStatus(acc[c], &st) != PL_OK) break;
            if (st == PROGRAM_PROGRESS) continue;
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /* Read full y from shared EMI (cluster). */
    std::vector<float> y_nmc(M_TOTAL);
    std::vector<PL_Word> yw(M_TOTAL);
    PL_ReadMemBlock(acc[0], yw.data(), ADDR_Y, M_TOTAL);
    for (int i = 0; i < M_TOTAL; ++i)
        std::memcpy(&y_nmc[i], &yw[i], 4);

    /* Verify */
    float maxd = 0, sumsq = 0;
    int worst_row = -1;
    for (int i = 0; i < M_TOTAL; ++i) {
        float dd = y_nmc[i] - y_host[i];
        if (std::fabs(dd) > maxd) { maxd = std::fabs(dd); worst_row = i; }
        sumsq += dd * dd;
    }
    std::cerr << "[diag] per-row (nmc, host, diff):\n";
    for (int i = 0; i < M_TOTAL; ++i) {
        float dd = y_nmc[i] - y_host[i];
        if (std::fabs(dd) > 1e-5f || i == 0)
            std::cerr << "  row " << i << " (core " << i/M_PER_CORE << "): "
                      << y_nmc[i] << " vs " << y_host[i] << " diff=" << dd << "\n";
    }
    std::cerr << "[diag] worst row=" << worst_row << " diff=" << maxd << "\n";
    std::cout << "[verify] (32 rows) max_diff=" << maxd
              << " rms_diff=" << std::sqrt(sumsq / M_TOTAL) << "\n";
    std::cout << "[perf-4c] wall: " << ms << " ms\n";
    double macs = (double)M_TOTAL * K;
    std::cout << "[perf-4c] " << macs / (ms * 1e3) << " MMACs/sec across 4 cores\n";

    for (int c = 0; c < NCORES; ++c) {
        if (svc[c]) IO_ServiceStop(&svc[c], nullptr);
        PL_CloseAccess(acc[c]);
    }
    PL_CloseBoardDesc(board);
    return maxd < 1e-2f ? 0 : 100;
}
