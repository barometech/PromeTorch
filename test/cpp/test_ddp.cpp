// ============================================================================
// test_ddp.cpp — fork-based self-test for torch::distributed DDP (POSIX TCP)
// ============================================================================
// Spawns 2 processes, each builds a [3,4] tensor with rank-specific values,
// calls all_reduce, and verifies both ranks get the same SUM.
// Exit code 0 = pass, non-zero = fail.
// On Windows this file just prints "skipped" (fork() not available).
// ============================================================================
#include "torch/distributed/ddp.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
int main() {
    std::printf("test_ddp: SKIPPED on Windows (no fork)\n");
    return 0;
}
#else

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace td = torch::distributed;

static int run_rank(int rank, int world, int port) {
    td::DDPConfig cfg;
    cfg.rank        = rank;
    cfg.world_size  = world;
    cfg.master_addr = "127.0.0.1";
    cfg.master_port = port;
    cfg.timeout_sec = 30;

    try {
        td::init_process_group(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] init failed: %s\n", rank, e.what());
        return 10;
    }

    // Build [3,4] = 12 floats. Rank 0 fills with i, rank 1 fills with i*10.
    at::Tensor t = at::zeros({3, 4});
    float* p = t.mutable_data_ptr<float>();
    for (int i = 0; i < 12; ++i) {
        p[i] = (rank == 0) ? (float)i : (float)(i * 10);
    }

    try {
        td::all_reduce(t);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] all_reduce failed: %s\n", rank, e.what());
        td::destroy_process_group();
        return 11;
    }

    // Expected: rank0 + rank1 = i + 10*i = 11*i
    int errors = 0;
    for (int i = 0; i < 12; ++i) {
        float expect = (float)(11 * i);
        if (p[i] != expect) {
            std::fprintf(stderr, "[rank %d] mismatch idx=%d got=%f want=%f\n",
                         rank, i, p[i], expect);
            ++errors;
        }
    }

    // Broadcast test: rank 0 sends t (now [0,11,22,...]) → flip rank-1's tensor
    // and verify it matches.
    if (rank == 1) {
        for (int i = 0; i < 12; ++i) p[i] = -1.0f;
    }
    try {
        td::broadcast(t, /*src_rank=*/0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] broadcast failed: %s\n", rank, e.what());
        td::destroy_process_group();
        return 12;
    }
    for (int i = 0; i < 12; ++i) {
        float expect = (float)(11 * i);
        if (p[i] != expect) {
            std::fprintf(stderr, "[rank %d] bcast mismatch idx=%d got=%f want=%f\n",
                         rank, i, p[i], expect);
            ++errors;
        }
    }

    // Barrier test
    try { td::barrier(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] barrier failed: %s\n", rank, e.what());
        td::destroy_process_group();
        return 13;
    }

    td::destroy_process_group();

    if (errors == 0) {
        std::printf("[rank %d] PASS\n", rank);
        return 0;
    }
    std::printf("[rank %d] FAIL: %d mismatches\n", rank, errors);
    return 1;
}

int main(int argc, char** argv) {
    int port = 29500 + (int)(getpid() % 1000);

    // Allow a child to be invoked directly via "--child <rank>".
    if (argc >= 4 && std::string(argv[1]) == "--child") {
        int rank  = std::atoi(argv[2]);
        int world = std::atoi(argv[3]);
        int p     = (argc >= 5) ? std::atoi(argv[4]) : port;
        return run_rank(rank, world, p);
    }

    pid_t child = fork();
    if (child < 0) { std::perror("fork"); return 2; }

    if (child == 0) {
        // Child = rank 1
        return run_rank(1, 2, port);
    }

    // Parent = rank 0
    int rc0 = run_rank(0, 2, port);

    int status = 0;
    waitpid(child, &status, 0);
    int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : 99;

    if (rc0 == 0 && rc1 == 0) {
        std::printf("test_ddp: PASS (rank0=%d rank1=%d)\n", rc0, rc1);
        return 0;
    }
    std::printf("test_ddp: FAIL (rank0=%d rank1=%d)\n", rc0, rc1);
    return 1;
}

#endif  // !_WIN32
