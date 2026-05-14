/* Host ping-pong test for ncl_hostSync foundation */
#include <iostream>
#include <chrono>
#include "nm_quad_load.h"
#include "io_host.h"

int main() {
    PL_Board *board = nullptr;
    unsigned int nb = 0;
    if (PL_GetBoardCount(&nb) != PL_OK || nb < 1) return 1;
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 2;
    PL_ResetBoard(board);
    PL_LoadInitCode(board);
    PL_CoreNo core; core.cluster_id = 0; core.nm_id = 0;
    PL_Access *acc = nullptr;
    if (PL_GetAccess(board, &core, &acc) != PL_OK) return 3;
    if (PL_LoadProgramFile(acc, "./nmc_part.abs") != PL_OK) return 4;

    auto t0 = std::chrono::steady_clock::now();
    IO_Service *svc = IO_ServiceStart("./nmc_part.abs", acc, nullptr, nullptr, nullptr);

    for (int i = 0; i < 5; ++i) {
        int reply = 0;
        int rv = PL_Sync(acc, i * 100, &reply);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[host] ping " << i << " rv=" << rv << " kernel_replied=" << reply << " elapsed=" << ms << " ms\n";
    }

    if (svc) IO_ServiceStop(&svc, nullptr);
    PL_CloseAccess(acc);
    PL_CloseBoardDesc(board);
    return 0;
}
