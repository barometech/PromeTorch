// ============================================================================
// test_deepspeed.cpp — self-tests for DeepSpeed-style wrappers over FSDP.
// ============================================================================
// Runs:
//   * 1F1B pipeline self-test (multi-process fork; POSIX-only)
//   * CPU-offload optimizer smoke test (single-process)
//   * Hierarchical ZeRO-3 construction smoke test (single-process)
// Skipped on Windows.
// ============================================================================
#include "torch/distributed/deepspeed.h"
#include "torch/optim/sgd.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

#if defined(_WIN32)
int main() {
    std::printf("test_deepspeed: SKIPPED on Windows (no fork)\n");
    return 0;
}
#else

static int test_offload() {
    using namespace torch;
    // Tiny parameter set.
    auto p_t = at::empty({4});
    for (int i = 0; i < 4; ++i) p_t.mutable_data_ptr<float>()[i] = (float)(i + 1);
    auto p = std::make_shared<nn::Parameter>(p_t);
    p->set_grad(at::empty({4}));
    for (int i = 0; i < 4; ++i) p->grad().mutable_data_ptr<float>()[i] = 0.1f;

    std::vector<nn::Parameter*> params{p.get()};
    auto inner = std::make_shared<optim::SGD>(params, /*lr=*/0.1);
    distributed::deepspeed::OffloadOptimizer off(params, inner, "cpu");
    off.step();
    off.zero_grad();
    std::printf("test_deepspeed: offload step OK (offload_bytes=%zu)\n",
                off.offload_bytes());
    return 0;
}

static int test_hierarchy() {
    using namespace torch;
    // Build a 2-layer model, wrap in ZeRO-3 with a 1×1 hierarchy (degenerate,
    // exercises the hierarchical code path without requiring real peers).
    auto mlp = std::make_shared<nn::Sequential>();
    // Simple identity-ish layers so forward() in a single-process test works.
    // Use TinyLinear from pipeline_schedule.h helpers via anonymous build.
    distributed::ZeROStage3Config cfg;
    cfg.fsdp.rank = 0;
    cfg.fsdp.world_size = 1;
    cfg.fsdp.sync_dir = "/dev/shm/pt_ds_zero3_test";
    cfg.inner_world_size = 1;
    cfg.inter_world_size = 1;  // collapse to flat — tests construction path
    auto z = std::make_shared<distributed::ZeROStage3>(mlp, cfg);
    (void)z;
    std::printf("test_deepspeed: ZeRO-3 construction OK\n");
    return 0;
}

int main(int argc, char** argv) {
    std::string sync_dir = "/dev/shm/pt_pipe1f1b_test";
    if (argc > 1) sync_dir = argv[1];
    int rc_off = test_offload();
    int rc_pipe = torch::distributed::pipeline_1f1b_selftest_main(sync_dir);
    int rc_zero = test_hierarchy();
    int rc = rc_off | rc_pipe | rc_zero;
    if (rc == 0) std::printf("test_deepspeed: PASS\n");
    else         std::printf("test_deepspeed: FAIL (offload=%d pipe=%d zero=%d)\n",
                             rc_off, rc_pipe, rc_zero);
    return rc;
}
#endif
