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

#define M_TOTAL 2048
#define M_PER_CORE 128
#define K 2560
#define BLOCKS_PER_ROW (K / 256)
#define Q6K_ROW_BYTES (BLOCKS_PER_ROW * 210)

int main(int argc, char *argv[]) {
    if (argc < 4) { std::cerr << "Usage: " << argv[0] << " A_W A_X A_Y\n"; return 1; }
    PL_Addr A_W = std::strtoul(argv[1], nullptr, 16);
    PL_Addr A_X = std::strtoul(argv[2], nullptr, 16);
    PL_Addr A_Y = std::strtoul(argv[3], nullptr, 16);

    std::ifstream f("/home/<user>/gguf/qwen3-4b-q4km.gguf", std::ios::binary);
    std::vector<uint8_t> W(M_TOTAL * Q6K_ROW_BYTES);
    f.seekg(5967296); f.read((char*)W.data(), W.size());
    std::vector<float> x(K);
    uint32_t rng = 0x42;
    for (int i = 0; i < K; ++i) { rng = rng * 1103515245u + 12345u; x[i] = ((float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000) * 2.0f - 1.0f; }

    PL_Board *board = nullptr; unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 2;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 3;
    PL_ResetBoard(board); PL_LoadInitCode(board);

    PL_Access *acc[4][4];
    for (int cl = 0; cl < 4; ++cl) {
        for (int c = 0; c < 4; ++c) {
            PL_CoreNo core; core.cluster_id = cl; core.nm_id = c;
            if (PL_GetAccess(board, &core, &acc[cl][c]) != PL_OK) {
                std::cerr << "GetAccess fail cluster=" << cl << " core=" << c << "\n";
                return 4;
            }
            if (PL_LoadProgramFile(acc[cl][c], "./nmc_part.abs") != PL_OK) {
                std::cerr << "LoadProgramFile fail cluster=" << cl << " core=" << c << "\n";
                return 5;
            }
        }
    }
    std::cerr << "[16c] all cores loaded\n";

    // Upload W and x to ALL 4 clusters (each chip has own EMI)
    std::vector<PL_Word> w_buf(W.size());
    for (size_t i = 0; i < W.size(); ++i) w_buf[i] = (PL_Word)W[i];
    std::vector<PL_Word> xw(K);
    for (int i = 0; i < K; ++i) std::memcpy(&xw[i], &x[i], 4);
    for (int cl = 0; cl < 4; ++cl) {
        PL_WriteMemBlock(acc[cl][0], w_buf.data(), A_W, W.size());
        PL_WriteMemBlock(acc[cl][0], xw.data(), A_X, K);
    }
    std::cerr << "[16c] upload done\n";

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc[4][4];
    for (int cl = 0; cl < 4; ++cl)
        for (int c = 0; c < 4; ++c)
            svc[cl][c] = IO_ServiceStart("./nmc_part.abs", acc[cl][c], nullptr, nullptr, nullptr);
    for (int cl = 0; cl < 4; ++cl) {
        for (int c = 0; c < 4; ++c) {
            PL_Word st = 0;
            while (1) { if (PL_GetStatus(acc[cl][c], &st) != PL_OK) break; if (st == PROGRAM_PROGRESS) continue; break; }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Read y from each cluster — only same rows that this cluster computed
    std::vector<float> y_full(M_TOTAL, 0.0f);
    for (int cl = 0; cl < 4; ++cl) {
        std::vector<PL_Word> yw(M_TOTAL);
        PL_ReadMemBlock(acc[cl][0], yw.data(), A_Y, M_TOTAL);
        for (int i = cl * 8; i < (cl+1) * 8; ++i) std::memcpy(&y_full[i], &yw[i], 4);
    }

    std::cout << "[16c] M=32 K=2560 wall=" << ms << " ms\n";
    std::cout << "[16c] y_full[0..3]: " << y_full[0] << " " << y_full[1] << " " << y_full[2] << " " << y_full[3] << "\n";
    std::cout << "[16c] y_full[28..31]: " << y_full[28] << " " << y_full[29] << " " << y_full[30] << " " << y_full[31] << "\n";

    for (int cl = 0; cl < 4; ++cl)
        for (int c = 0; c < 4; ++c)
            if (svc[cl][c]) IO_ServiceStop(&svc[cl][c], nullptr);
    for (int cl = 0; cl < 4; ++cl)
        for (int c = 0; c < 4; ++c)
            PL_CloseAccess(acc[cl][c]);
    PL_CloseBoardDesc(board);
    return 0;
}
