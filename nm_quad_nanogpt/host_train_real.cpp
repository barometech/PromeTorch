/* ============================================================================
 * Host driver: loads nanogpt_train_v2.abs into all 16 NMC4 cores AND first
 * uploads TinyStories text into each core's training_data[] buffer (EMI).
 *
 * Per NMC4 quirk: 1 char on NMC = 32-bit word in memory. So to send N bytes
 * of text we write N words (one byte per word, zero-extended).
 * ============================================================================ */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <chrono>

#include "nm_quad_load.h"
#include "io_host.h"

#define N_CLUSTERS  4
#define N_CORES     4
#define DATA_SIZE   (1 * 1024 * 1024)   /* bytes of source text */

/* Symbol addresses from `nmc-nm nmc_part.abs` for nanogpt_train_v2.abs.
 * These are NMC native word addresses. If you re-link the .abs, re-run
 * `nm nmc_part.abs | grep training_data` and update. */
static const PL_Addr ADDR_TRAINING_DATA = 0x0201b7eaUL;
static const PL_Addr ADDR_CORE_LOSS_OUT = 0x0201b7e8UL;

static const char *NMC_PART     = "./nmc_part.abs";
static const char *DEFAULT_DATA = "/home/<user>/nanogpt/data/tinystories_valid.txt";

int main(int argc, char *argv[]) {
    const char *data_path = (argc > 1) ? argv[1] : DEFAULT_DATA;

    std::ifstream fin(data_path, std::ios::binary);
    if (!fin) { std::cerr << "ERROR: cannot open " << data_path << "\n"; return 1; }
    fin.seekg(0, std::ios::end);
    size_t fsize = fin.tellg();
    fin.seekg(0, std::ios::beg);
    size_t take = (fsize < DATA_SIZE) ? fsize : DATA_SIZE;
    std::vector<char> bytes(DATA_SIZE, ' ');
    fin.read(bytes.data(), take);
    std::cerr << "[host] Read " << take << " bytes from " << data_path
              << " (capacity " << DATA_SIZE << ")\n";

    /* Pad with newline padding for stability */
    if (take < DATA_SIZE) {
        for (size_t i = take; i < DATA_SIZE; ++i) bytes[i] = bytes[i % take];
    }

    /* Expand to one 32-bit word per char (zero-extend). */
    std::vector<PL_Word> word_buf(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i)
        word_buf[i] = (PL_Word)((unsigned char)bytes[i]);

    /* Open board */
    PL_Board *board = nullptr;
    unsigned int n_boards = 0;
    if (PL_GetBoardCount(&n_boards) != PL_OK || n_boards < 1) {
        std::cerr << "ERROR: no NM_Quad board found.\n";
        return 1;
    }
    if (PL_GetBoardDesc(0, &board) != PL_OK) return 2;
    if (PL_ResetBoard(board) != PL_OK) { PL_CloseBoardDesc(board); return 3; }
    if (PL_LoadInitCode(board) != PL_OK) { PL_CloseBoardDesc(board); return 4; }

    PL_Access  *access[N_CLUSTERS][N_CORES] = {};
    IO_Service *service[N_CLUSTERS][N_CORES] = {};
    bool        done[N_CLUSTERS][N_CORES] = {};
    float       loss[N_CLUSTERS][N_CORES] = {};

    auto t0 = std::chrono::steady_clock::now();

    int loaded = 0;
    for (int c = 0; c < N_CLUSTERS; ++c) {
        for (int n = 0; n < N_CORES; ++n) {
            PL_CoreNo core;
            core.cluster_id = c;
            core.nm_id      = n;
            if (PL_GetAccess(board, &core, &access[c][n]) != PL_OK) {
                std::cerr << "WARN: GetAccess failed chip=" << c << " core=" << n << "\n";
                continue;
            }
            if (PL_LoadProgramFile(access[c][n], NMC_PART) != PL_OK) {
                std::cerr << "WARN: LoadProgramFile failed chip=" << c << " core=" << n << "\n";
                PL_CloseAccess(access[c][n]); access[c][n] = nullptr;
                continue;
            }
            /* Upload TinyStories text to this core's training_data[] */
            if (PL_WriteMemBlock(access[c][n], word_buf.data(),
                                 ADDR_TRAINING_DATA, (DWORD)DATA_SIZE) != PL_OK) {
                std::cerr << "WARN: WriteMemBlock failed chip=" << c << " core=" << n << "\n";
                PL_CloseAccess(access[c][n]); access[c][n] = nullptr;
                continue;
            }
            service[c][n] = IO_ServiceStart(NMC_PART, access[c][n],
                                            nullptr, nullptr, nullptr);
            ++loaded;
        }
    }
    auto t_upload = std::chrono::steady_clock::now();
    std::cerr << "[host] Loaded + uploaded data to " << loaded << "/16 cores in "
              << std::chrono::duration<double>(t_upload - t0).count() << " s\n";

    int done_count = 0;
    int wait_limit = 100000000;
    while (done_count < loaded && wait_limit-- > 0) {
        for (int c = 0; c < N_CLUSTERS; ++c)
            for (int n = 0; n < N_CORES; ++n) {
                if (done[c][n] || access[c][n] == nullptr) continue;
                PL_Word status = 0;
                if (PL_GetStatus(access[c][n], &status) != PL_OK) continue;
                if (status == PROGRAM_PROGRESS) continue;
                if (status == PROGRAM_FINISHED) {
                    PL_Word result = 0;
                    PL_GetResult(access[c][n], &result);
                    /* Read final loss from EMI */
                    PL_Word loss_word = 0;
                    PL_ReadMemBlock(access[c][n], &loss_word, ADDR_CORE_LOSS_OUT, 1);
                    float l; std::memcpy(&l, &loss_word, 4);
                    loss[c][n] = l;
                    std::cout << "[chip=" << c << " core=" << n
                              << "] DONE result=" << result
                              << " loss=" << l << std::endl;
                    done[c][n] = true; ++done_count;
                } else {
                    std::cerr << "[chip=" << c << " core=" << n
                              << "] FAILED status=" << status << "\n";
                    done[c][n] = true; ++done_count;
                }
            }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    /* Summary */
    int n_done = 0;
    float sum = 0;
    for (int c = 0; c < N_CLUSTERS; ++c)
        for (int n = 0; n < N_CORES; ++n)
            if (done[c][n] && loss[c][n] > 0) { sum += loss[c][n]; ++n_done; }
    float avg = (n_done > 0) ? (sum / (float)n_done) : 0.0f;
    std::cerr << "[host] All cores finished in " << elapsed << " s; mean_loss="
              << avg << " (over " << n_done << " cores)\n";

    for (int c = 0; c < N_CLUSTERS; ++c)
        for (int n = 0; n < N_CORES; ++n) {
            if (service[c][n]) IO_ServiceStop(&service[c][n], nullptr);
            if (access[c][n])  PL_CloseAccess(access[c][n]);
        }
    PL_CloseBoardDesc(board);
    return done_count == loaded ? 0 : 100;
}
