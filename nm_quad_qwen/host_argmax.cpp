/* host_argmax.cpp — verify NMC4 argmax over random logits */
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include "nm_quad_load.h"
#include "io_host.h"

static const char *NMC_PART = "./nmc_part.abs";
#define VOCAB 151936

int main(int argc, char *argv[]) {
    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " A_LOGITS A_TOKEN\n"; return 1; }
    PL_Addr A_L = std::strtoul(argv[1], nullptr, 16);
    PL_Addr A_T = std::strtoul(argv[2], nullptr, 16);

    std::vector<float> logits(VOCAB);
    uint32_t rng = 0xABCDEF;
    for (int i = 0; i < VOCAB; ++i) {
        rng = rng * 1103515245u + 12345u;
        logits[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 20.0f - 10.0f;
    }
    int expected_idx = 0;
    float expected_max = logits[0];
    for (int i = 1; i < VOCAB; ++i) {
        if (logits[i] > expected_max) { expected_max = logits[i]; expected_idx = i; }
    }
    std::cerr << "[host] argmax token=" << expected_idx << " max_logit=" << expected_max << "\n";

    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 2;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 3;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 4;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 5;

    std::vector<PL_Word> lw(VOCAB);
    for (int i = 0; i < VOCAB; ++i) std::memcpy(&lw[i], &logits[i], 4);
    PL_WriteMemBlock(acc, lw.data(), A_L, VOCAB);

    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }

    PL_Word tw;
    PL_ReadMemBlock(acc, &tw, A_T, 1);
    int nmc_token = (int)tw;
    std::cout << "[verify] NMC argmax token=" << nmc_token << " host=" << expected_idx << " match=" << (nmc_token == expected_idx) << "\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return nmc_token == expected_idx ? 0 : 100;
}
