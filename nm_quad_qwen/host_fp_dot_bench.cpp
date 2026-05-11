/* Host driver for nmc_fp_dot_bench.c — measures NMC4 baseline FP throughput */
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include "nm_quad_load.h"
#include "io_host.h"

static const char *NMC_PART = "./nmc_part.abs";
#define M 32
#define K 2560

int main(int argc, char *argv[]) {
    if (argc < 4) { std::cerr << "Usage: " << argv[0] << " <ADDR_A> <ADDR_X> <ADDR_Y>\n"; return 1; }
    PL_Addr ADDR_A = std::strtoul(argv[1], nullptr, 16);
    PL_Addr ADDR_X = std::strtoul(argv[2], nullptr, 16);
    PL_Addr ADDR_Y = std::strtoul(argv[3], nullptr, 16);

    std::vector<float> A(M * K);
    std::vector<float> x(K);
    uint32_t rng = 0x1234567;
    for (size_t i = 0; i < A.size(); ++i) {
        rng = rng * 1103515245u + 12345u;
        A[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < x.size(); ++i) {
        rng = rng * 1103515245u + 12345u;
        x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    PL_Board *board = nullptr;
    unsigned int n_b = 0;
    if (PL_GetBoardCount(&n_b) != PL_OK || n_b < 1) return 3;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 4;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 5;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 6;

    std::vector<PL_Word> aw(M * K);
    std::memcpy(aw.data(), A.data(), M * K * 4);
    PL_WriteMemBlock(acc, aw.data(), ADDR_A, M * K);

    std::vector<PL_Word> xw(K);
    std::memcpy(xw.data(), x.data(), K * 4);
    PL_WriteMemBlock(acc, xw.data(), ADDR_X, K);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) {
        if (PL_GetStatus(acc, &st) != PL_OK) break;
        if (st == PROGRAM_PROGRESS) continue;
        break;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[perf] FP dot kernel wall: " << ms << " ms\n";
    double macs = (double)M * K;
    std::cout << "[perf] " << macs / (ms * 1e3) << " MMACs/sec @ M=" << M << " K=" << K << "\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return 0;
}
