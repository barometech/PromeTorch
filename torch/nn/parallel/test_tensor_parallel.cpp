// ============================================================================
// test_tensor_parallel.cpp — two-process self-test for ColumnParallelLinear.
// ============================================================================
// Spawns itself with --rank=0 and --rank=1 (world_size=2). Each rank builds a
// ColumnParallelLinear(in=8, out=16, world=2, gather_output=true). Both ranks
// init from the same seed so their virtual full weight matches a single
// non-parallel Linear. Each rank then runs a fixed input through both:
//   * the parallel layer (which all-gathers the output to a [B,16] tensor)
//   * a reference Linear with the same seed
// and prints max abs diff. A diff <1e-4 means the TP plumbing is correct.
//
// Build (linux):
//   g++ -std=c++17 -O2 test_tensor_parallel.cpp -I../../.. -lc10 -laten -ltorch_autograd -o tp_test
// Then: ./tp_test  (it will fork itself).
// ============================================================================
// node.h must precede engine.h: engine.h uses Node members but only forward-
// declares the struct, so a translation unit that triggers engine.h
// instantiations before node.h gets pulled in will fail. Including node.h up
// front avoids that ordering trap.
#include "torch/csrc/autograd/node.h"
#include "torch/nn/parallel/tensor_parallel.h"
#include "torch/nn/modules/linear.h"
#include "aten/src/ATen/ATen.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace torch::nn;
using namespace torch::nn::parallel;

static int run_rank(int rank, int world) {
    TPConfig tp;
    tp.rank = rank;
    tp.world_size = world;
    tp.sync_dir = "/dev/shm/pt_tp_test";

    const int64_t B = 4, IN = 8, OUT = 16;
    const uint64_t SEED = 42424242ULL;

    // Build parallel + reference layers.
    ColumnParallelLinear cpl(IN, OUT, tp, /*gather_output=*/true,
                             /*bias=*/true, SEED);

    // Build a non-parallel reference by using a world_size=1 ColumnParallel
    // with the same seed — guaranteed identical full weight by construction.
    TPConfig solo; solo.rank = 0; solo.world_size = 1;
    solo.sync_dir = tp.sync_dir + "_ref";
    ColumnParallelLinear ref(IN, OUT, solo, /*gather_output=*/false,
                             /*bias=*/true, SEED);

    // Deterministic input: same on every rank.
    at::Tensor x = at::empty({B, IN});
    float* xd = x.mutable_data_ptr<float>();
    for (int64_t i = 0; i < B * IN; ++i) {
        xd[i] = 0.01f * (float)(i - 16);
    }

    at::Tensor y_par = cpl.forward(x);
    at::Tensor y_ref = ref.forward(x);

    if (y_par.sizes() != y_ref.sizes()) {
        std::fprintf(stderr, "[rank %d] SHAPE MISMATCH par=%lldx%lld ref=%lldx%lld\n",
                     rank, (long long)y_par.size(0), (long long)y_par.size(1),
                     (long long)y_ref.size(0), (long long)y_ref.size(1));
        return 2;
    }

    const float* a = y_par.data_ptr<float>();
    const float* b = y_ref.data_ptr<float>();
    float maxd = 0.f;
    for (int64_t i = 0; i < y_par.numel(); ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    std::printf("[rank %d] ColumnParallel gather vs reference: max|diff|=%.6g\n",
                rank, maxd);
    return (maxd < 1e-4f) ? 0 : 1;
}

int main(int argc, char** argv) {
    int rank = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--rank=", 7) == 0) rank = std::atoi(argv[i] + 7);
    }
    if (rank >= 0) return run_rank(rank, 2);

#if defined(_WIN32)
    std::fprintf(stderr, "self-spawn requires fork(); run with --rank=0 and --rank=1 manually\n");
    return 0;
#else
    pid_t pid = ::fork();
    if (pid == 0) {
        execl(argv[0], argv[0], "--rank=1", (char*)nullptr);
        _exit(127);
    }
    int rc0 = run_rank(0, 2);
    int status = 0; ::waitpid(pid, &status, 0);
    int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : 99;
    std::printf("rank0 rc=%d  rank1 rc=%d\n", rc0, rc1);
    return rc0 | rc1;
#endif
}
