// ============================================================================
// test_matmul_host.cpp — Host-side test: matmul on NM QUAD
// ============================================================================
// Compiles with MSVC (host x86), links nm_quad_load.lib
// Loads dispatcher on chip 0, sends 2x2 matmul, verifies result.
//
// Build:
//   cl /EHsc /I C:\Module\NM_Quad\include test_matmul_host.cpp
//      C:\Module\NM_Quad\lib\nm_quad_load.lib
//
// Run: test_matmul_host.exe

#include <iostream>
#include <cstring>
#include <cmath>
#include "nm_quad_load.h"

// Must match dispatcher_nmquad.cpp
enum Opcode : unsigned int {
    OP_NOP     = 0,
    OP_MATMUL  = 1,
    OP_ADD     = 2,
    OP_RELU    = 4,
    OP_DONE    = 0xFF
};

struct CmdBlock {
    unsigned int opcode;
    unsigned int status;
    unsigned int a_addr;
    unsigned int b_addr;
    unsigned int c_addr;
    unsigned int M, N, K;
    unsigned int alpha_bits;
    unsigned int reserved[23];
};

static const unsigned int DDR_START = 0x00340000;
static const unsigned int CMD_BLOCK_BASE = DDR_START;
static const unsigned int DATA_BASE = DDR_START + 0x200;

int main() {
    std::cout << "=== NM QUAD Matmul Test ===" << std::endl;

    // Get board
    unsigned int board_count = 0;
    if (PL_GetBoardCount(&board_count) != PL_OK || board_count == 0) {
        std::cerr << "No boards found!" << std::endl;
        return 1;
    }
    std::cout << "Boards: " << board_count << std::endl;

    PL_Board* board;
    PL_GetBoardDesc(0, &board);
    PL_ResetBoard(board);
    PL_LoadInitCode(board);
    PL_SetTimeout(10000);

    // Get access to chip 0, core 0
    PL_CoreNo core = {0, 0};
    PL_Access* access;
    PL_GetAccess(board, &core, &access);

    // Load dispatcher
    std::cout << "Loading dispatcher..." << std::endl;
    if (PL_LoadProgramFile(access, "dispatcher_nmquad.abs") != PL_OK) {
        std::cerr << "Failed to load dispatcher!" << std::endl;
        return 1;
    }
    std::cout << "Dispatcher loaded on core 0.0" << std::endl;

    // Prepare 2x2 matmul: C = A * B
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // Expected C = [[19, 22], [43, 50]]
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[4] = {0};

    unsigned int a_addr = DATA_BASE;
    unsigned int b_addr = DATA_BASE + 4;  // 4 words after A
    unsigned int c_addr = DATA_BASE + 8;  // 4 words after B

    // Upload A and B to DDR
    std::cout << "Uploading matrices..." << std::endl;
    PL_WriteMemBlock(access, (PL_Word*)A, a_addr, 4);
    PL_WriteMemBlock(access, (PL_Word*)B, b_addr, 4);

    // Write command block
    CmdBlock cmd = {};
    cmd.opcode = OP_MATMUL;
    cmd.status = 0;
    cmd.a_addr = a_addr;
    cmd.b_addr = b_addr;
    cmd.c_addr = c_addr;
    cmd.M = 2;
    cmd.N = 2;
    cmd.K = 2;

    PL_WriteMemBlock(access, (PL_Word*)&cmd, CMD_BLOCK_BASE, sizeof(CmdBlock)/sizeof(PL_Word));

    // Sync — signal dispatcher to execute
    std::cout << "Executing matmul on NM6408..." << std::endl;
    int ret = 0;
    PL_Sync(access, 1, &ret);
    std::cout << "Sync returned: " << ret << std::endl;

    // Read result
    PL_ReadMemBlock(access, (PL_Word*)C, c_addr, 4);

    std::cout << "\nResult C = A * B:" << std::endl;
    std::cout << "  [" << C[0] << ", " << C[1] << "]" << std::endl;
    std::cout << "  [" << C[2] << ", " << C[3] << "]" << std::endl;

    // Verify
    float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
    bool pass = true;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(C[i] - expected[i]) > 0.01f) {
            std::cerr << "MISMATCH at [" << i << "]: got " << C[i]
                      << ", expected " << expected[i] << std::endl;
            pass = false;
        }
    }

    if (pass) {
        std::cout << "\n*** PASS: 2x2 matmul correct! ***" << std::endl;
    } else {
        std::cout << "\n*** FAIL ***" << std::endl;
    }

    // Send shutdown
    cmd.opcode = OP_DONE;
    PL_WriteMemBlock(access, (PL_Word*)&cmd, CMD_BLOCK_BASE, sizeof(CmdBlock)/sizeof(PL_Word));
    PL_Sync(access, -1, &ret);

    // Cleanup
    PL_CloseAccess(access);
    PL_CloseBoardDesc(board);

    return pass ? 0 : 1;
}
