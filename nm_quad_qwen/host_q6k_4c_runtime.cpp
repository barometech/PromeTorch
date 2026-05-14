/* host_q6k_4core.cpp — measure 4-core Q6_K GEMV M=32 K=2560 wall */
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include "nm_quad_load.h"
#include "io_host.h"

#define M_TOTAL 32
#define K 2560
#define BLOCKS_PER_ROW (K / 256)
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

int main(int argc, char *argv[]) {
    if (argc < 4) { std::cerr << "Usage: " << argv[0] << " A_W A_X A_Y\n"; return 1; }
    PL_Addr A_W = std::strtoul(argv[1], nullptr, 16);
    PL_Addr A_X = std::strtoul(argv[2], nullptr, 16);
    PL_Addr A_Y = std::strtoul(argv[3], nullptr, 16);

    // Load Q6_K weights for first 32 rows of token_embd.weight (Qwen3-4B tied)
    std::ifstream f("/home/<user>/gguf/qwen3-4b-q4km.gguf", std::ios::binary);
    // We know off_t for token_embd.weight from earlier scan: base+10240 → ~6000000 byte. Hardcode for proof.
    // From earlier: type=14 off=5967296
    uint64_t base_off = 5967296;
    std::vector<uint8_t> W(M_TOTAL * Q6K_ROW_BYTES);
    f.seekg(base_off); f.read((char*)W.data(), W.size());

    std::vector<float> x(K);
    uint32_t rng = 0x42;
    for (int i = 0; i < K; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    std::vector<float> y_h(M_TOTAL);
    for (int r = 0; r < M_TOTAL; ++r) {
        float a = 0;
        for (int b = 0; b < BLOCKS_PER_ROW; ++b)
            a += q6k_dot_h(W.data() + r * Q6K_ROW_BYTES + b * 210, x.data() + b * 256);
        y_h[r] = a;
    }

    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 2;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 3;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    PL_Access *acc[4] = {nullptr,nullptr,nullptr,nullptr};
    for (int c = 0; c < 4; ++c) {
        PL_CoreNo core; core.cluster_id = 0; core.nm_id = c;
        if (PL_GetAccess(board, &core, &acc[c]) != PL_OK) return 4;
        if (PL_LoadProgramFile(acc[c], "./nmc_part.abs") != PL_OK) return 5;
    }

    std::vector<PL_Word> w_buf(W.size());
    for (size_t i = 0; i < W.size(); ++i) w_buf[i] = (PL_Word)W[i];
    PL_WriteMemBlock(acc[0], w_buf.data(), A_W, W.size());

    std::vector<PL_Word> xw(K);
    for (int i = 0; i < K; ++i) std::memcpy(&xw[i], &x[i], 4);
    PL_WriteMemBlock(acc[0], xw.data(), A_X, K);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc[4];
    for (int c = 0; c < 4; ++c) {
        svc[c] = IO_ServiceStart("./nmc_part.abs", acc[c], nullptr, nullptr, nullptr);
    }
    for (int c = 0; c < 4; ++c) {
        PL_Word st = 0;
        while (1) { if (PL_GetStatus(acc[c], &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> yw(M_TOTAL);
    PL_ReadMemBlock(acc[0], yw.data(), A_Y, M_TOTAL);
    std::vector<float> y_n(M_TOTAL);
    for (int i = 0; i < M_TOTAL; ++i) std::memcpy(&y_n[i], &yw[i], 4);

    float maxd = 0;
    for (int i = 0; i < M_TOTAL; ++i) {
        float d = std::fabs(y_n[i] - y_h[i]);
        if (d > maxd) maxd = d;
    }
    std::cout << "[q6k4c] M=32 K=2560 max_diff=" << maxd << " wall=" << ms << " ms\n";

    for (int c = 0; c < 4; ++c) if (svc[c]) IO_ServiceStop(&svc[c], nullptr);
    for (int c = 0; c < 4; ++c) PL_CloseAccess(acc[c]);
    PL_CloseBoardDesc(board);
    return maxd < 1e-3 ? 0 : 100;
}
