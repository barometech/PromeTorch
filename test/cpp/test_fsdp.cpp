// ============================================================================
// test_fsdp.cpp — fork-based self-test for torch::distributed::FSDP
// ============================================================================
// Forks 2 worker processes that train a 4-layer MLP under FULL_SHARD, then
// compares the post-step parameters of rank 0 against an unsharded baseline.
// Pass: max abs error < 1e-4. Skipped on Windows (no fork).
// ============================================================================
#include "torch/distributed/fsdp.h"

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
int main() {
    std::printf("test_fsdp: SKIPPED on Windows (no fork)\n");
    return 0;
}
#else
int main(int argc, char** argv) {
    std::string sync_dir = "/dev/shm/pt_fsdp_test";
    if (argc > 1) sync_dir = argv[1];
    int rc = torch::distributed::fsdp_selftest_main(sync_dir);
    if (rc == 0) std::printf("test_fsdp: PASS\n");
    else         std::printf("test_fsdp: FAIL (rc=%d)\n", rc);
    return rc;
}
#endif
