/* host_q4k_gemv_tile.cpp — full Wq @ y on NMC4 4-cores via 32 tile invocations.
 * Each tile = 128 rows (4 cores × 32 rows). Full Wq M=4096 = 32 tiles.
 *
 * Wq upload once (~5.9 MW = 23.6 MB), then 32 invocations of bit-exact kernel.
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
#include <functional>

static const char *NMC_PART  = "./nmc_part.abs";
static const char *GGUF_PATH = "/home/<user>/gguf/qwen3-4b-q4km.gguf";

#define K 2560
#define BLOCKS_PER_ROW (K / 256)
#define ROW_BYTES (BLOCKS_PER_ROW * 144)
#define M_TILE 128
#define M_FULL 4096
#define N_TILES (M_FULL / M_TILE)
#define NCORES 4

/* EMI addresses from nmc_part.map */
static const PL_Addr ADDR_X     = 0x02000002UL;
static const PL_Addr ADDR_W     = 0x02000a02UL;
static const PL_Addr ADDR_Y     = 0x025a0a02UL;
static const PL_Addr ADDR_SLICE = 0x025a1a02UL;

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

/* Find attn_q.weight offset in gguf */
static uint64_t find_attn_q_offset() {
    std::ifstream f(GGUF_PATH, std::ios::binary);
    if (!f) return 0;
    auto rd_u64 = [&](){ uint64_t v; f.read((char*)&v,8); return v; };
    auto rd_u32 = [&](){ uint32_t v; f.read((char*)&v,4); return v; };
    auto rd_str = [&](){ uint64_t n=rd_u64(); std::string s(n,'\0'); f.read(&s[0],n); return s; };
    char m[4]; f.read(m,4); if (memcmp(m,"GGUF",4)!=0) return 0;
    rd_u32(); uint64_t nt=rd_u64(), nk=rd_u64();
    std::function<bool(uint32_t)> skip = [&](uint32_t t)->bool {
        switch(t){case 0:case 1:case 7:f.seekg(1,std::ios::cur);return true;
                  case 2:case 3:f.seekg(2,std::ios::cur);return true;
                  case 4:case 5:case 6:f.seekg(4,std::ios::cur);return true;
                  case 10:case 11:case 12:f.seekg(8,std::ios::cur);return true;
                  case 8:rd_str();return true;
                  case 9:{uint32_t at=rd_u32(); uint64_t an=rd_u64(); for(uint64_t i=0;i<an;++i) if(!skip(at)) return false; return true;}
                  default: return false;}
    };
    for (uint64_t i=0;i<nk;++i){rd_str(); uint32_t t=rd_u32(); if(!skip(t)) return 0;}
    uint64_t attn_off=0;
    for (uint64_t i=0;i<nt;++i){
        std::string nm=rd_str(); uint32_t nd=rd_u32();
        std::vector<uint64_t> dims(nd); for(uint32_t d=0;d<nd;++d) dims[d]=rd_u64();
        uint32_t ty=rd_u32(); uint64_t off=rd_u64();
        if (nm=="blk.0.attn_q.weight" && ty==12) attn_off=off;
    }
    uint64_t pos=f.tellg(); uint64_t aln=32;
    uint64_t base=(pos+aln-1)&~(aln-1);
    return attn_off ? (base+attn_off) : 0;
}

int main() {
    uint64_t gguf_off = find_attn_q_offset();
    if (!gguf_off) { std::cerr << "find_attn_q failed\n"; return 1; }
    std::cerr << "[gguf] attn_q.weight at 0x" << std::hex << gguf_off << std::dec << "\n";

    /* Load full Wq (M=4096 rows, K=2560) from gguf */
    std::vector<uint8_t> W(M_FULL * ROW_BYTES);
    {
        std::ifstream f(GGUF_PATH, std::ios::binary);
        f.seekg(gguf_off);
        f.read((char*)W.data(), M_FULL * ROW_BYTES);
    }
    std::cerr << "[load] " << W.size() << " bytes Wq loaded\n";

    /* Generate test x (sane Gaussian) */
    std::vector<float> x(K);
    srand(42);
    for (int i = 0; i < K; ++i) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;

    /* CPU reference */
    std::vector<float> y_host(M_FULL);
    auto t_cpu0 = std::chrono::steady_clock::now();
    for (int r = 0; r < M_FULL; ++r)
        y_host[r] = q4k_row_dot(W.data() + r * ROW_BYTES, x.data());
    auto t_cpu1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t_cpu1-t_cpu0).count();
    std::cerr << "[cpu] " << cpu_ms << " ms for full M=" << M_FULL << "\n";

    /* Init NMC */
    PL_Board *board = nullptr;
    unsigned int n_b = 0;
    if (PL_GetBoardCount(&n_b) != PL_OK || n_b < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    PL_Access *acc[NCORES] = {nullptr};
    for (int c = 0; c < NCORES; ++c) {
        PL_CoreNo core; core.cluster_id = 0; core.nm_id = c;
        if (PL_GetAccess(board, &core, &acc[c]) != PL_OK) return 5;
        if (PL_LoadProgramFile(acc[c], NMC_PART) != PL_OK) return 6;
    }

    /* Upload x + full W once */
    std::vector<PL_Word> xw(K);
    std::memcpy(xw.data(), x.data(), K*4);
    PL_WriteMemBlock(acc[0], xw.data(), ADDR_X, K);
    std::cerr << "[upload] x " << K*4 << " bytes\n";

    int W_words = M_FULL * ROW_BYTES;
    std::vector<PL_Word> ww(W_words);
    for (int i = 0; i < W_words; ++i) ww[i] = (PL_Word)W[i];
    PL_WriteMemBlock(acc[0], ww.data(), ADDR_W, W_words);
    std::cerr << "[upload] W " << W_words*4 << " bytes\n";

    /* Run N_TILES invocations, each processes 128 rows */
    auto t_nmc0 = std::chrono::steady_clock::now();
    IO_Service *svc[NCORES] = {nullptr};
    for (int c = 0; c < NCORES; ++c)
        svc[c] = IO_ServiceStart(NMC_PART, acc[c], nullptr, nullptr, nullptr);

    for (int tile = 0; tile < N_TILES; ++tile) {
        /* Write slice idx to gemv_slice EMI cell */
        PL_Word slice_w = (PL_Word)tile;
        PL_WriteMemBlock(acc[0], &slice_w, ADDR_SLICE, 1);
        /* Trigger 4 cores sequentially via re-loading program (simplest) */
        for (int c = 0; c < NCORES; ++c) {
            PL_LoadProgramFile(acc[c], NMC_PART);
            IO_Service *sv = IO_ServiceStart(NMC_PART, acc[c], nullptr, nullptr, nullptr);
            PL_Word st = 0;
            while (1) {
                if (PL_GetStatus(acc[c], &st) != PL_OK) break;
                if (st == PROGRAM_PROGRESS) continue;
                break;
            }
            if (sv) IO_ServiceStop(&sv, nullptr);
        }
    }
    auto t_nmc1 = std::chrono::steady_clock::now();
    double nmc_ms = std::chrono::duration<double,std::milli>(t_nmc1-t_nmc0).count();

    /* Read full y */
    std::vector<float> y_nmc(M_FULL);
    std::vector<PL_Word> yw(M_FULL);
    PL_ReadMemBlock(acc[0], yw.data(), ADDR_Y, M_FULL);
    for (int i = 0; i < M_FULL; ++i) std::memcpy(&y_nmc[i], &yw[i], 4);

    /* Verify */
    float maxd = 0;
    int worst = -1;
    for (int i = 0; i < M_FULL; ++i) {
        float dd = std::fabs(y_nmc[i] - y_host[i]);
        if (dd > maxd) { maxd = dd; worst = i; }
    }
    std::cout << "[verify] max_diff=" << maxd << " (row " << worst << ")\n";
    std::cout << "[perf] NMC4 full Wq (M=" << M_FULL << ") wall=" << nmc_ms << " ms\n";
    double macs = (double)M_FULL * K;
    std::cout << "[perf] " << macs / (nmc_ms*1e3) << " MMACs/sec\n";
    std::cout << "[perf] vs CPU ref " << cpu_ms << " ms (speedup " << (cpu_ms/nmc_ms) << "x)\n";

    for (int c = 0; c < NCORES; ++c) PL_CloseAccess(acc[c]);
    PL_CloseBoardDesc(board);
    return 0;
}
