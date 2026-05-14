/* host_rope.cpp — RoPE test driver vs host CPU reference */
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include "nm_quad_load.h"
#include "io_host.h"

static const char *NMC_PART = "./nmc_part.abs";
#define HEAD_DIM 128
#define BASE 1000000.0f

int main(int argc, char *argv[]) {
    if (argc < 5) { std::cerr << "Usage: " << argv[0] << " <ADDR_X> <ADDR_Y> <ADDR_POS> <pos>\n"; return 1; }
    PL_Addr ADDR_X   = std::strtoul(argv[1], nullptr, 16);
    PL_Addr ADDR_Y   = std::strtoul(argv[2], nullptr, 16);
    PL_Addr ADDR_POS = std::strtoul(argv[3], nullptr, 16);
    int pos = std::atoi(argv[4]);

    std::vector<float> x(HEAD_DIM);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < HEAD_DIM; ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    /* Host reference */
    std::vector<float> y_ref(HEAD_DIM);
    int half = HEAD_DIM / 2;
    for (int i = 0; i < half; ++i) {
        float ep = 2.0f * (float)i / (float)HEAD_DIM;
        float theta = 1.0f / std::pow(BASE, ep);
        float angle = (float)pos * theta;
        float c = std::cos(angle);
        float s = std::sin(angle);
        y_ref[i]        = x[i] * c - x[i + half] * s;
        y_ref[i + half] = x[i] * s + x[i + half] * c;
    }
    std::cerr << "[host] pos=" << pos << " y_ref[0..3]=" << y_ref[0] << " " << y_ref[1] << " "
              << y_ref[2] << " " << y_ref[3] << "\n";

    /* NMC */
    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 2;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 3;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 4;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 5;

    float posf = (float)pos;
    PL_Word pw;
    std::memcpy(&pw, &posf, 4);
    PL_WriteMemBlock(acc, &pw, ADDR_POS, 1);
    std::vector<PL_Word> xw(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; ++i) std::memcpy(&xw[i], &x[i], 4);
    PL_WriteMemBlock(acc, xw.data(), ADDR_X, HEAD_DIM);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> yw(HEAD_DIM);
    PL_ReadMemBlock(acc, yw.data(), ADDR_Y, HEAD_DIM);
    std::vector<float> y_nmc(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; ++i) std::memcpy(&y_nmc[i], &yw[i], 4);

    float maxd = 0;
    for (int i = 0; i < HEAD_DIM; ++i) {
        float d = std::fabs(y_nmc[i] - y_ref[i]);
        if (d > maxd) maxd = d;
    }
    std::cout << "[verify] RoPE pos=" << pos << " max_diff=" << maxd << "\n";
    std::cout << "[perf] wall: " << ms << " ms\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-5f ? 0 : 100;
}
