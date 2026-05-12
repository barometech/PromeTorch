/* host_attn.cpp — Attention head test driver */
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
#define CACHE_LEN 128

int main(int argc, char *argv[]) {
    if (argc < 5) { std::cerr << "Usage: " << argv[0] << " <ADDR_Q> <ADDR_K> <ADDR_V> <ADDR_OUT>\n"; return 1; }
    PL_Addr ADDR_Q   = std::strtoul(argv[1], nullptr, 16);
    PL_Addr ADDR_K   = std::strtoul(argv[2], nullptr, 16);
    PL_Addr ADDR_V   = std::strtoul(argv[3], nullptr, 16);
    PL_Addr ADDR_OUT = std::strtoul(argv[4], nullptr, 16);

    /* Random Q, K, V */
    std::vector<float> Q(HEAD_DIM);
    std::vector<float> K(CACHE_LEN * HEAD_DIM);
    std::vector<float> V(CACHE_LEN * HEAD_DIM);
    uint32_t rng = 0x1234567;
    for (int i = 0; i < HEAD_DIM; ++i) {
        rng = rng * 1103515245u + 12345u;
        Q[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }
    for (int i = 0; i < CACHE_LEN * HEAD_DIM; ++i) {
        rng = rng * 1103515245u + 12345u;
        K[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
        rng = rng * 1103515245u + 12345u;
        V[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f;
    }

    /* Host reference */
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    std::vector<float> scores(CACHE_LEN);
    for (int t = 0; t < CACHE_LEN; ++t) {
        float s = 0;
        for (int i = 0; i < HEAD_DIM; ++i) s += Q[i] * K[t * HEAD_DIM + i];
        scores[t] = s * scale;
    }
    float mx = scores[0];
    for (int t = 1; t < CACHE_LEN; ++t) if (scores[t] > mx) mx = scores[t];
    float sum = 0;
    for (int t = 0; t < CACHE_LEN; ++t) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
    for (int t = 0; t < CACHE_LEN; ++t) scores[t] /= sum;
    std::vector<float> out_ref(HEAD_DIM, 0);
    for (int t = 0; t < CACHE_LEN; ++t)
        for (int i = 0; i < HEAD_DIM; ++i) out_ref[i] += scores[t] * V[t * HEAD_DIM + i];
    std::cerr << "[host] out_ref[0..3] = " << out_ref[0] << " " << out_ref[1] << " "
              << out_ref[2] << " " << out_ref[3] << "\n";

    /* NMC */
    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 2;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 3;
    PL_ResetBoard(board); PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 4;
    if (PL_LoadProgramFile(acc, NMC_PART) != PL_OK) return 5;

    std::vector<PL_Word> qw(HEAD_DIM), kw(CACHE_LEN * HEAD_DIM), vw(CACHE_LEN * HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; ++i) std::memcpy(&qw[i], &Q[i], 4);
    for (int i = 0; i < CACHE_LEN * HEAD_DIM; ++i) {
        std::memcpy(&kw[i], &K[i], 4);
        std::memcpy(&vw[i], &V[i], 4);
    }
    /* Write large arrays first, then Q (kernel reads Q first → potentially stale) */
    PL_WriteMemBlock(acc, kw.data(), ADDR_K, CACHE_LEN * HEAD_DIM);
    PL_WriteMemBlock(acc, vw.data(), ADDR_V, CACHE_LEN * HEAD_DIM);
    PL_WriteMemBlock(acc, qw.data(), ADDR_Q, HEAD_DIM);

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart(NMC_PART, acc, nullptr, nullptr, nullptr);
    PL_Word st = 0;
    while (1) { if (PL_GetStatus(acc, &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<PL_Word> ow(HEAD_DIM);
    PL_ReadMemBlock(acc, ow.data(), ADDR_OUT, HEAD_DIM);
    std::vector<float> out_nmc(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; ++i) std::memcpy(&out_nmc[i], &ow[i], 4);

    float maxd = 0;
    for (int i = 0; i < HEAD_DIM; ++i) {
        float d = std::fabs(out_nmc[i] - out_ref[i]);
        if (d > maxd) maxd = d;
    }
    std::cout << "[verify] attn max_diff=" << maxd << "\n";
    std::cout << "[perf] kernel wall: " << ms << " ms (incl 3ms DMA wait)\n";
    /* MACs: HEAD_DIM*CACHE_LEN for scores + HEAD_DIM*CACHE_LEN for out = 2*HEAD_DIM*CACHE_LEN */
    double macs = (double)2 * HEAD_DIM * CACHE_LEN;
    std::cout << "[perf] " << macs / (ms * 1e3) << " MMACs/sec\n";

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return maxd < 1e-3f ? 0 : 100;
}
