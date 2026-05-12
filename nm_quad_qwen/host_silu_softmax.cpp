/* host_silu_softmax.cpp — test SiLU + Softmax vs host reference */
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include "nm_quad_load.h"
#include "io_host.h"

static const char *NMC_PART = "./nmc_part.abs";
#define K_SILU 8192
#define K_SM   128

int main(int argc, char *argv[]) {
    if (argc < 5) { std::cerr << "Usage: " << argv[0] << " <ADDR_SILU_IN> <ADDR_SILU_OUT> <ADDR_SM_IN> <ADDR_SM_OUT>\n"; return 1; }
    PL_Addr ADDR_SILU_IN  = std::strtoul(argv[1], nullptr, 16);
    PL_Addr ADDR_SILU_OUT = std::strtoul(argv[2], nullptr, 16);
    PL_Addr ADDR_SM_IN    = std::strtoul(argv[3], nullptr, 16);
    PL_Addr ADDR_SM_OUT   = std::strtoul(argv[4], nullptr, 16);

    /* Random inputs */
    std::vector<float> silu_in(K_SILU), sm_in(K_SM);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < K_SILU; ++i) {
        rng = rng * 1103515245u + 12345u;
        silu_in[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 6.0f - 3.0f;
    }
    for (int i = 0; i < K_SM; ++i) {
        rng = rng * 1103515245u + 12345u;
        sm_in[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 10.0f - 5.0f;
    }

    /* Host reference */
    std::vector<float> silu_ref(K_SILU);
    for (int i = 0; i < K_SILU; ++i) silu_ref[i] = silu_in[i] / (1.0f + std::exp(-silu_in[i]));

    float mx = sm_in[0];
    for (int i = 1; i < K_SM; ++i) if (sm_in[i] > mx) mx = sm_in[i];
    float sum = 0;
    std::vector<float> sm_ref(K_SM);
    for (int i = 0; i < K_SM; ++i) { sm_ref[i] = std::exp(sm_in[i] - mx); sum += sm_ref[i]; }
    for (int i = 0; i < K_SM; ++i) sm_ref[i] /= sum;
    std::cerr << "[host] silu_ref[0..3] = " << silu_ref[0] << " " << silu_ref[1] << " "
              << silu_ref[2] << " " << silu_ref[3] << "\n";
    std::cerr << "[host] sm_ref[0..3]   = " << sm_ref[0] << " " << sm_ref[1] << " "
              << sm_ref[2] << " " << sm_ref[3] << " (sum=" << sum << ")\n";

    /* NMC */
    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 2;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 3;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 4;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 5;

    std::vector<PL_Word> sw(K_SILU), mw(K_SM);
    for (int i = 0; i < K_SILU; ++i) std::memcpy(&sw[i], &silu_in[i], 4);
    for (int i = 0; i < K_SM; ++i)   std::memcpy(&mw[i], &sm_in[i], 4);
    PL_WriteMemBlock(acc, sw.data(), ADDR_SILU_IN, K_SILU);
    PL_WriteMemBlock(acc, mw.data(), ADDR_SM_IN,   K_SM);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> sow(K_SILU), mow(K_SM);
    PL_ReadMemBlock(acc, sow.data(), ADDR_SILU_OUT, K_SILU);
    PL_ReadMemBlock(acc, mow.data(), ADDR_SM_OUT,   K_SM);
    std::vector<float> silu_nmc(K_SILU), sm_nmc(K_SM);
    for (int i = 0; i < K_SILU; ++i) std::memcpy(&silu_nmc[i], &sow[i], 4);
    for (int i = 0; i < K_SM; ++i)   std::memcpy(&sm_nmc[i], &mow[i], 4);

    float maxd_silu = 0;
    for (int i = 0; i < K_SILU; ++i) {
        float d = std::fabs(silu_nmc[i] - silu_ref[i]);
        if (d > maxd_silu) maxd_silu = d;
    }
    float maxd_sm = 0;
    for (int i = 0; i < K_SM; ++i) {
        float d = std::fabs(sm_nmc[i] - sm_ref[i]);
        if (d > maxd_sm) maxd_sm = d;
    }

    std::cout << "[verify] SiLU K=" << K_SILU << " max_diff=" << maxd_silu << "\n";
    std::cout << "[verify] Softmax K=" << K_SM << " max_diff=" << maxd_sm << "\n";
    std::cout << "[perf] kernel wall: " << ms << " ms\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return (maxd_silu < 1e-4f && maxd_sm < 1e-5f) ? 0 : 100;
}
