// ============================================================================
// test_launcher.cpp — Self-test for torch::distributed::launch + DistributedSampler
// ============================================================================
// Build (Elbrus):
//   lcc -std=c++17 -O2 -I. examples/distributed/test_launcher.cpp \
//       -o build_mt/examples/distributed/test_launcher
//
// Expected output (order across ranks may vary):
//   hello from rank 0 of 4
//   hello from rank 1 of 4
//   hello from rank 2 of 4
//   hello from rank 3 of 4
//   [sampler] rank=0 epoch=0 num_samples=25 first=...
//   [sampler] rank=1 epoch=0 num_samples=25 first=...
//   [sampler] rank=2 epoch=0 num_samples=25 first=...
//   [sampler] rank=3 epoch=0 num_samples=25 first=...
//   PASS: 4 children exited with 0
// ============================================================================
#include "torch/data/distributed_sampler.h"
#include "torch/distributed/launcher.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

int main(int argc, char** argv) {
    int world_size = 4;
    if (argc >= 2) world_size = std::atoi(argv[1]);

    int rc = torch::distributed::launch(
        world_size,
        [](int rank, int ws) -> int {
            // 1. Print hello (per task spec).
            std::printf("hello from rank %d of %d\n", rank, ws);
            std::fflush(stdout);

            // 2. Verify env vars set by launcher.
            const char* m_addr = std::getenv("MASTER_ADDR");
            const char* m_port = std::getenv("MASTER_PORT");
            const char* r_env  = std::getenv("RANK");
            const char* w_env  = std::getenv("WORLD_SIZE");
            if (!m_addr || !m_port || !r_env || !w_env) {
                std::fprintf(stderr, "[rank %d] missing env vars\n", rank);
                return 2;
            }
            if (std::atoi(r_env) != rank || std::atoi(w_env) != ws) {
                std::fprintf(stderr, "[rank %d] env mismatch RANK=%s WORLD_SIZE=%s\n",
                             rank, r_env, w_env);
                return 3;
            }

            // 3. Exercise DistributedSampler — disjoint-shards property.
            const size_t N = 100;
            torch::data::DistributedSampler s(N, ws, rank,
                                              /*shuffle=*/true,
                                              /*seed=*/42,
                                              /*drop_last=*/false);
            s.set_epoch(0);
            auto batch = s.sample(s.num_samples());
            size_t first = batch.empty() ? 0 : batch.front();
            std::printf("[sampler] rank=%d epoch=0 num_samples=%zu first=%zu batch=%zu\n",
                        rank, s.num_samples(), first, batch.size());
            std::fflush(stdout);

            // 4. Sanity: indices for this rank are unique and < N.
            std::set<size_t> seen(batch.begin(), batch.end());
            if (seen.size() != batch.size()) {
                std::fprintf(stderr, "[rank %d] duplicate indices in shard!\n", rank);
                return 4;
            }
            for (size_t v : batch) {
                if (v >= N) {
                    std::fprintf(stderr, "[rank %d] index %zu out of range\n", rank, v);
                    return 5;
                }
            }
            return 0;
        },
        "127.0.0.1",
        29500
    );

    if (rc == 0) {
        std::printf("PASS: %d children exited with 0\n", world_size);
        return 0;
    }
    std::fprintf(stderr, "FAIL: max child rc = %d\n", rc);
    return rc;
}
