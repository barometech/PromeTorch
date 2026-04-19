// Verify DistributedSampler shards across all ranks form a partition.
#include "torch/data/distributed_sampler.h"
#include <cstdio>
#include <set>
#include <vector>

int main() {
    const size_t N = 100;
    const int    W = 4;
    int  failures = 0;

    // Two epochs, two drop_last modes.
    for (bool drop_last : {false, true}) {
        for (uint64_t epoch : {0u, 1u, 7u}) {
            std::vector<size_t> total;
            size_t expected_per_rank = 0;
            for (int rank = 0; rank < W; ++rank) {
                torch::data::DistributedSampler s(N, W, rank, true, 42, drop_last);
                s.set_epoch(epoch);
                auto b = s.sample(s.num_samples());
                if (rank == 0) expected_per_rank = s.num_samples();
                if (b.size() != expected_per_rank) {
                    std::printf("FAIL drop_last=%d epoch=%lu rank=%d size=%zu expected=%zu\n",
                                drop_last, (unsigned long)epoch, rank, b.size(), expected_per_rank);
                    failures++;
                }
                for (size_t v : b) total.push_back(v);
            }
            std::set<size_t> uniq(total.begin(), total.end());
            // With drop_last=false and N=100, W=4 -> total_size=100, exactly N unique.
            // With drop_last=true and N=100, W=4 -> total_size=100, exactly N unique.
            // (For non-divisible N, drop_last=false would have wrap-around dups
            //  and uniq.size() < total.size().)
            std::printf("drop_last=%d epoch=%lu total=%zu unique=%zu\n",
                        drop_last, (unsigned long)epoch, total.size(), uniq.size());
            if (total.size() != 100) failures++;
            if (uniq.size()  != 100) failures++;
            for (size_t v : uniq) if (v >= N) { failures++; break; }
        }
    }

    if (failures == 0) {
        std::printf("PASS: DistributedSampler partitions cleanly\n");
        return 0;
    }
    std::printf("FAIL: %d assertion(s)\n", failures);
    return 1;
}
