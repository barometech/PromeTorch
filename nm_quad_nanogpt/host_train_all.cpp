/* ============================================================================
 * Host driver: loads nanogpt_train.abs into ALL 16 NMC4 cores (4 clusters × 4)
 * in parallel and waits for completion. Each core trains its own copy with a
 * different RNG seed (data-parallel ensemble).
 *
 * Build: g++ -pthread host_train_all.cpp -DNM_QUAD
 *          -I/usr/local/rc_module/board-nm_quad/include
 *          -L/usr/local/rc_module/board-nm_quad/lib
 *          -Wl,-rpath=/usr/local/rc_module/board-nm_quad/lib
 *          -lnm_quad_load -lio_host -ldl -o host_train_all
 * ============================================================================ */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include "nm_quad_load.h"
#include "io_host.h"

static const char *NMC_PART = "./nmc_part.abs";

#define N_CLUSTERS 4
#define N_CORES    4
#define N_NMC      (N_CLUSTERS * N_CORES)

int main() {
    PL_Board *board = nullptr;
    unsigned int n_boards = 0;
    if (PL_GetBoardCount(&n_boards) != PL_OK || n_boards < 1) {
        std::cerr << "ERROR: no NM_Quad board found.\n";
        return 1;
    }
    if (PL_GetBoardDesc(0, &board) != PL_OK) {
        std::cerr << "ERROR: cannot open board.\n";
        return 2;
    }
    if (PL_ResetBoard(board) != PL_OK) {
        std::cerr << "ERROR: reset board failed.\n";
        PL_CloseBoardDesc(board); return 3;
    }
    if (PL_LoadInitCode(board) != PL_OK) {
        std::cerr << "ERROR: load init code failed.\n";
        PL_CloseBoardDesc(board); return 4;
    }

    PL_Access  *access[N_CLUSTERS][N_CORES] = {};
    IO_Service *service[N_CLUSTERS][N_CORES] = {};
    bool        done[N_CLUSTERS][N_CORES] = {};

    auto t0 = std::chrono::steady_clock::now();

    int loaded = 0;
    for (int c = 0; c < N_CLUSTERS; ++c) {
        for (int n = 0; n < N_CORES; ++n) {
            PL_CoreNo core;
            core.cluster_id = c;
            core.nm_id      = n;
            if (PL_GetAccess(board, &core, &access[c][n]) != PL_OK) {
                std::cerr << "WARN: GetAccess failed for chip=" << c << " core=" << n << "\n";
                continue;
            }
            if (PL_LoadProgramFile(access[c][n], NMC_PART) != PL_OK) {
                std::cerr << "WARN: LoadProgramFile failed for chip=" << c << " core=" << n << "\n";
                PL_CloseAccess(access[c][n]);
                access[c][n] = nullptr;
                continue;
            }
            service[c][n] = IO_ServiceStart(NMC_PART, access[c][n], nullptr, nullptr, nullptr);
            ++loaded;
        }
    }
    std::cerr << "[host] Loaded " << loaded << "/" << N_NMC << " cores; waiting for completion...\n";

    int done_count = 0;
    int wait_limit = 100000000;  /* polls */
    while (done_count < loaded && wait_limit-- > 0) {
        for (int c = 0; c < N_CLUSTERS; ++c) {
            for (int n = 0; n < N_CORES; ++n) {
                if (done[c][n] || access[c][n] == nullptr) continue;
                PL_Word status = 0;
                if (PL_GetStatus(access[c][n], &status) != PL_OK) continue;
                if (status == PROGRAM_PROGRESS) continue;
                if (status == PROGRAM_FINISHED) {
                    PL_Word result = 0;
                    PL_GetResult(access[c][n], &result);
                    std::cout << "[chip=" << c << " core=" << n
                              << "] DONE result=" << result
                              << " (loss*1000)" << std::endl;
                    done[c][n] = true; ++done_count;
                } else {
                    std::cerr << "[chip=" << c << " core=" << n << "] FAILED status=" << status << "\n";
                    done[c][n] = true; ++done_count;
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[host] All cores finished in " << elapsed << " s\n";

    for (int c = 0; c < N_CLUSTERS; ++c)
        for (int n = 0; n < N_CORES; ++n) {
            if (service[c][n]) IO_ServiceStop(&service[c][n], nullptr);
            if (access[c][n])  PL_CloseAccess(access[c][n]);
        }
    PL_CloseBoardDesc(board);
    return done_count == loaded ? 0 : 100;
}
