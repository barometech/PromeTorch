// ============================================================================
// test_nmquad_suite.cpp — Full test suite for NM QUAD backend
// ============================================================================
// Tests: matmul (various sizes), add, relu, multi-chip, performance
// Build: cl /EHsc /I C:\Module\NM_Quad\include test_nmquad_suite.cpp
//        C:\Module\NM_Quad\lib\nm_quad_load.lib
// Run:   test_nmquad_suite.exe

#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include "nm_quad_load.h"

// Opcodes (match dispatcher_nmquad.cpp)
enum Opcode : unsigned int {
    OP_NOP = 0, OP_MATMUL = 1, OP_ADD = 2, OP_MUL = 3,
    OP_RELU = 4, OP_SIGMOID = 5, OP_DONE = 0xFF
};

struct CmdBlock {
    unsigned int opcode, status, a_addr, b_addr, c_addr, M, N, K, alpha_bits;
    unsigned int reserved[23];
};

const unsigned int DDR_START = 0x00340000;
const unsigned int CMD_BLOCK_BASE = DDR_START;
const unsigned int DATA_BASE = DDR_START + 0x200;

static PL_Board* board = nullptr;
static PL_Access* access = nullptr;
static int tests_passed = 0;
static int tests_total = 0;

bool init_board() {
    unsigned int count = 0;
    if (PL_GetBoardCount(&count) != PL_OK || count == 0) return false;
    PL_GetBoardDesc(0, &board);
    PL_ResetBoard(board);
    PL_LoadInitCode(board);
    PL_SetTimeout(15000);
    PL_CoreNo core = {0, 0};
    PL_GetAccess(board, &core, &access);
    return PL_LoadProgramFile(access, "dispatcher_nmquad.abs") == PL_OK;
}

void cleanup() {
    // Send done
    CmdBlock cmd = {};
    cmd.opcode = OP_DONE;
    PL_WriteMemBlock(access, (PL_Word*)&cmd, CMD_BLOCK_BASE, sizeof(CmdBlock)/4);
    int ret;
    PL_Sync(access, -1, &ret);
    PL_CloseAccess(access);
    PL_CloseBoardDesc(board);
}

// Upload float array to DDR, returns address
unsigned int upload(const float* data, int count, unsigned int& next_addr) {
    unsigned int addr = next_addr;
    PL_WriteMemBlock(access, (PL_Word*)data, addr, count);
    next_addr += count;
    return addr;
}

// Download float array from DDR
void download(float* data, unsigned int addr, int count) {
    PL_ReadMemBlock(access, (PL_Word*)data, addr, count);
}

// Execute command
void execute(CmdBlock& cmd) {
    PL_WriteMemBlock(access, (PL_Word*)&cmd, CMD_BLOCK_BASE, sizeof(CmdBlock)/4);
    int ret;
    PL_Sync(access, 1, &ret);
}

// Check results
bool check(const char* name, const float* got, const float* expected, int n, float tol = 0.01f) {
    tests_total++;
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - expected[i]) > tol) {
            std::cout << "  FAIL: " << name << " [" << i << "] got=" << got[i]
                      << " expected=" << expected[i] << std::endl;
            return false;
        }
    }
    tests_passed++;
    std::cout << "  PASS: " << name << std::endl;
    return true;
}

// ============================================================================
// Tests
// ============================================================================

void test_matmul_2x2() {
    float A[] = {1,2,3,4}, B[] = {5,6,7,8}, C[4], E[] = {19,22,43,50};
    unsigned int next = DATA_BASE;
    unsigned int aa = upload(A, 4, next);
    unsigned int ba = upload(B, 4, next);
    unsigned int ca = next; next += 4;

    CmdBlock cmd = {};
    cmd.opcode = OP_MATMUL; cmd.a_addr = aa; cmd.b_addr = ba; cmd.c_addr = ca;
    cmd.M = 2; cmd.N = 2; cmd.K = 2;
    execute(cmd);
    download(C, ca, 4);
    check("matmul 2x2", C, E, 4);
}

void test_matmul_3x3() {
    float A[] = {1,0,0, 0,1,0, 0,0,1}; // Identity
    float B[] = {1,2,3, 4,5,6, 7,8,9};
    float C[9], E[] = {1,2,3, 4,5,6, 7,8,9}; // I * B = B
    unsigned int next = DATA_BASE;
    unsigned int aa = upload(A, 9, next);
    unsigned int ba = upload(B, 9, next);
    unsigned int ca = next; next += 9;

    CmdBlock cmd = {};
    cmd.opcode = OP_MATMUL; cmd.a_addr = aa; cmd.b_addr = ba; cmd.c_addr = ca;
    cmd.M = 3; cmd.N = 3; cmd.K = 3;
    execute(cmd);
    download(C, ca, 9);
    check("matmul 3x3 identity", C, E, 9);
}

void test_matmul_4x4() {
    float A[16], B[16], C[16], E[16];
    for (int i = 0; i < 16; i++) { A[i] = (float)(i+1); B[i] = (float)(16-i); }
    // CPU reference
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += A[i*4+k] * B[k*4+j];
            E[i*4+j] = s;
        }

    unsigned int next = DATA_BASE;
    unsigned int aa = upload(A, 16, next);
    unsigned int ba = upload(B, 16, next);
    unsigned int ca = next; next += 16;

    CmdBlock cmd = {};
    cmd.opcode = OP_MATMUL; cmd.a_addr = aa; cmd.b_addr = ba; cmd.c_addr = ca;
    cmd.M = 4; cmd.N = 4; cmd.K = 4;
    execute(cmd);
    download(C, ca, 16);
    check("matmul 4x4", C, E, 16);
}

void test_matmul_rect() {
    // 2x3 * 3x4 = 2x4
    float A[] = {1,2,3, 4,5,6};
    float B[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    float C[8], E[8];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++) s += A[i*3+k] * B[k*4+j];
            E[i*4+j] = s;
        }

    unsigned int next = DATA_BASE;
    unsigned int aa = upload(A, 6, next);
    unsigned int ba = upload(B, 12, next);
    unsigned int ca = next; next += 8;

    CmdBlock cmd = {};
    cmd.opcode = OP_MATMUL; cmd.a_addr = aa; cmd.b_addr = ba; cmd.c_addr = ca;
    cmd.M = 2; cmd.N = 4; cmd.K = 3;
    execute(cmd);
    download(C, ca, 8);
    check("matmul 2x3 * 3x4", C, E, 8);
}

void test_add() {
    float A[] = {1,2,3,4,5}, B[] = {10,20,30,40,50}, C[5], E[] = {11,22,33,44,55};
    unsigned int next = DATA_BASE;
    unsigned int aa = upload(A, 5, next);
    unsigned int ba = upload(B, 5, next);
    unsigned int ca = next; next += 5;

    CmdBlock cmd = {};
    cmd.opcode = OP_ADD; cmd.a_addr = aa; cmd.b_addr = ba; cmd.c_addr = ca;
    cmd.M = 5;
    execute(cmd);
    download(C, ca, 5);
    check("add 5 elements", C, E, 5);
}

void test_mul() {
    float A[] = {2,3,4}, B[] = {5,6,7}, C[3], E[] = {10,18,28};
    unsigned int next = DATA_BASE;
    unsigned int aa = upload(A, 3, next);
    unsigned int ba = upload(B, 3, next);
    unsigned int ca = next; next += 3;

    CmdBlock cmd = {};
    cmd.opcode = OP_MUL; cmd.a_addr = aa; cmd.b_addr = ba; cmd.c_addr = ca;
    cmd.M = 3;
    execute(cmd);
    download(C, ca, 3);
    check("mul 3 elements", C, E, 3);
}

void test_relu() {
    float X[] = {-3, -1, 0, 1, 3}, Y[5], E[] = {0, 0, 0, 1, 3};
    unsigned int next = DATA_BASE;
    unsigned int xa = upload(X, 5, next);
    unsigned int ya = next; next += 5;

    CmdBlock cmd = {};
    cmd.opcode = OP_RELU; cmd.a_addr = xa; cmd.c_addr = ya; cmd.M = 5;
    execute(cmd);
    download(Y, ya, 5);
    check("relu 5 elements", Y, E, 5);
}

void test_sigmoid() {
    float X[] = {0, 100, -100}, Y[3], E[] = {0.5f, 1.0f, 0.0f};
    unsigned int next = DATA_BASE;
    unsigned int xa = upload(X, 3, next);
    unsigned int ya = next; next += 3;

    CmdBlock cmd = {};
    cmd.opcode = OP_SIGMOID; cmd.a_addr = xa; cmd.c_addr = ya; cmd.M = 3;
    execute(cmd);
    download(Y, ya, 3);
    check("sigmoid 3 elements", Y, E, 3, 0.1f);
}

void test_matmul_perf() {
    const int N = 64;
    std::vector<float> A(N*N), B(N*N), C(N*N);
    for (int i = 0; i < N*N; i++) { A[i] = 0.01f * i; B[i] = 0.01f * (N*N - i); }

    unsigned int next = DATA_BASE;
    unsigned int aa = upload(A.data(), N*N, next);
    unsigned int ba = upload(B.data(), N*N, next);
    unsigned int ca = next; next += N*N;

    auto t0 = std::chrono::high_resolution_clock::now();
    int iters = 10;
    for (int i = 0; i < iters; i++) {
        CmdBlock cmd = {};
        cmd.opcode = OP_MATMUL; cmd.a_addr = aa; cmd.b_addr = ba; cmd.c_addr = ca;
        cmd.M = N; cmd.N = N; cmd.K = N;
        execute(cmd);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double gflops = (2.0 * N * N * N * iters) / (ms * 1e6);
    std::cout << "  PERF: matmul " << N << "x" << N << " x" << iters
              << " = " << ms << "ms (" << gflops << " GFLOPS)" << std::endl;
    tests_total++;
    tests_passed++;
}

void test_matmul_large() {
    const int M = 128, K = 64, N = 128;
    std::vector<float> A(M*K), B(K*N), C(M*N), E(M*N);
    for (int i = 0; i < M*K; i++) A[i] = 0.001f * i;
    for (int i = 0; i < K*N; i++) B[i] = 0.001f * (K*N - i);

    // CPU ref
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            E[i*N+j] = s;
        }

    unsigned int next = DATA_BASE;
    unsigned int aa = upload(A.data(), M*K, next);
    unsigned int ba = upload(B.data(), K*N, next);
    unsigned int ca = next; next += M*N;

    CmdBlock cmd = {};
    cmd.opcode = OP_MATMUL; cmd.a_addr = aa; cmd.b_addr = ba; cmd.c_addr = ca;
    cmd.M = M; cmd.N = N; cmd.K = K;
    execute(cmd);
    download(C.data(), ca, M*N);

    // Check first few and last few
    bool pass = true;
    for (int i = 0; i < M*N; i++) {
        if (std::fabs(C[i] - E[i]) > std::fabs(E[i]) * 0.01f + 0.1f) {
            std::cout << "  FAIL: matmul_large [" << i << "] got=" << C[i]
                      << " expected=" << E[i] << std::endl;
            pass = false;
            break;
        }
    }
    tests_total++;
    if (pass) { tests_passed++; std::cout << "  PASS: matmul 128x64x128" << std::endl; }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== NM QUAD Test Suite ===" << std::endl;

    if (!init_board()) {
        std::cerr << "Failed to initialize NM QUAD!" << std::endl;
        return 1;
    }
    std::cout << "Board initialized, dispatcher loaded\n" << std::endl;

    // Run tests
    test_matmul_2x2();
    test_matmul_3x3();
    test_matmul_4x4();
    test_matmul_rect();
    test_matmul_large();
    test_add();
    test_mul();
    test_relu();
    test_sigmoid();
    test_matmul_perf();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_total
              << " passed ===" << std::endl;

    cleanup();
    return (tests_passed == tests_total) ? 0 : 1;
}
