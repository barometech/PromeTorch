// ============================================================================
// test_distributed.cpp — Multi-device distributed training tests
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "torch/distributed/distributed.h"
#include <iostream>
#include <thread>
#include <cmath>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) \
    do { \
        tests_total++; \
        std::cout << "  [" << tests_total << "] " << name << "..."; \
        try {

#define PASS() \
            tests_passed++; \
            std::cout << " PASS" << std::endl; \
        } catch (const std::exception& e) { \
            std::cout << " FAIL: " << e.what() << std::endl; \
        } \
    } while(0)

#define CHECK(cond) \
    if (!(cond)) { throw std::runtime_error(std::string("CHECK FAILED: ") + #cond); }

#define CHECK_CLOSE(a, b, tol) \
    if (std::fabs((a) - (b)) > (tol)) { \
        throw std::runtime_error(std::string("CHECK_CLOSE FAILED: ") + \
            std::to_string(a) + " vs " + std::to_string(b)); \
    }

using namespace torch::distributed;

void test_init() {
    TEST("init / finalize")
        dist::init(4);
        CHECK(dist::is_initialized());
        CHECK(dist::world_size() == 4);
        dist::finalize();
        CHECK(!dist::is_initialized());
    PASS();
}

void test_allreduce_sum() {
    TEST("AllReduce SUM (4 ranks)")
        int W = 4;
        dist::init(W);

        // Each rank has tensor [rank, rank, rank, rank]
        std::vector<at::Tensor> tensors(W);
        for (int r = 0; r < W; ++r) {
            tensors[r] = at::full({4}, static_cast<float>(r + 1));
        }

        // Run AllReduce from each rank's thread
        std::vector<std::thread> threads;
        for (int r = 0; r < W; ++r) {
            threads.emplace_back([&tensors, r]() {
                dist::all_reduce(tensors[r], r, ReduceOp::SUM);
            });
        }
        for (auto& t : threads) t.join();

        // After SUM: all should be [1+2+3+4, ...] = [10, 10, 10, 10]
        for (int r = 0; r < W; ++r) {
            CHECK_CLOSE(tensors[r].data_ptr<float>()[0], 10.0f, 1e-5f);
        }

        dist::finalize();
    PASS();
}

void test_allreduce_avg() {
    TEST("AllReduce AVG (4 ranks)")
        int W = 4;
        dist::init(W);

        std::vector<at::Tensor> tensors(W);
        for (int r = 0; r < W; ++r) {
            tensors[r] = at::full({4}, static_cast<float>(r + 1));
        }

        std::vector<std::thread> threads;
        for (int r = 0; r < W; ++r) {
            threads.emplace_back([&tensors, r]() {
                dist::all_reduce(tensors[r], r, ReduceOp::AVG);
            });
        }
        for (auto& t : threads) t.join();

        // AVG of [1,2,3,4] = 2.5
        for (int r = 0; r < W; ++r) {
            CHECK_CLOSE(tensors[r].data_ptr<float>()[0], 2.5f, 1e-5f);
        }

        dist::finalize();
    PASS();
}

void test_broadcast() {
    TEST("Broadcast from rank 0")
        int W = 4;
        dist::init(W);

        std::vector<at::Tensor> tensors(W);
        tensors[0] = at::tensor({42.0f, 43.0f, 44.0f, 45.0f});
        for (int r = 1; r < W; ++r) {
            tensors[r] = at::zeros({4});
        }

        std::vector<std::thread> threads;
        for (int r = 0; r < W; ++r) {
            threads.emplace_back([&tensors, r]() {
                dist::broadcast(tensors[r], r, 0);
            });
        }
        for (auto& t : threads) t.join();

        // All should have [42, 43, 44, 45]
        for (int r = 0; r < W; ++r) {
            CHECK_CLOSE(tensors[r].data_ptr<float>()[0], 42.0f, 1e-5f);
            CHECK_CLOSE(tensors[r].data_ptr<float>()[3], 45.0f, 1e-5f);
        }

        dist::finalize();
    PASS();
}

void test_scatter() {
    TEST("Scatter batch across ranks")
        int W = 4;
        dist::init(W);

        // Batch of 8 items
        auto batch = at::arange(0, 8).to(c10::ScalarType::Float);
        batch = batch.view({8, 1});

        for (int r = 0; r < W; ++r) {
            auto shard = dist::scatter(batch, r);
            CHECK(shard.size(0) == 2); // 8/4 = 2 items per rank
            CHECK_CLOSE(shard.data_ptr<float>()[0], static_cast<float>(r * 2), 1e-5f);
        }

        dist::finalize();
    PASS();
}

void test_allreduce_large() {
    TEST("AllReduce with large tensor (1024 elements)")
        int W = 2;
        dist::init(W);

        std::vector<at::Tensor> tensors(W);
        tensors[0] = at::ones({1024});
        tensors[1] = at::full({1024}, 3.0f);

        std::vector<std::thread> threads;
        for (int r = 0; r < W; ++r) {
            threads.emplace_back([&tensors, r]() {
                dist::all_reduce(tensors[r], r, ReduceOp::SUM);
            });
        }
        for (auto& t : threads) t.join();

        // SUM: 1 + 3 = 4
        CHECK_CLOSE(tensors[0].data_ptr<float>()[0], 4.0f, 1e-5f);
        CHECK_CLOSE(tensors[0].data_ptr<float>()[1023], 4.0f, 1e-5f);
        CHECK_CLOSE(tensors[1].data_ptr<float>()[0], 4.0f, 1e-5f);

        dist::finalize();
    PASS();
}

void test_allreduce_max_min() {
    TEST("AllReduce MAX/MIN")
        int W = 3;
        dist::init(W);

        std::vector<at::Tensor> max_tensors(W), min_tensors(W);
        max_tensors[0] = at::tensor({1.0f, 5.0f, 3.0f});
        max_tensors[1] = at::tensor({4.0f, 2.0f, 6.0f});
        max_tensors[2] = at::tensor({7.0f, 8.0f, 1.0f});
        for (int r = 0; r < W; ++r) min_tensors[r] = max_tensors[r].clone();

        std::vector<std::thread> threads;
        for (int r = 0; r < W; ++r) {
            threads.emplace_back([&max_tensors, &min_tensors, r]() {
                dist::all_reduce(max_tensors[r], r, ReduceOp::MAX);
            });
        }
        for (auto& t : threads) t.join();

        // MAX: [7, 8, 6]
        CHECK_CLOSE(max_tensors[0].data_ptr<float>()[0], 7.0f, 1e-5f);
        CHECK_CLOSE(max_tensors[0].data_ptr<float>()[1], 8.0f, 1e-5f);
        CHECK_CLOSE(max_tensors[0].data_ptr<float>()[2], 6.0f, 1e-5f);

        dist::finalize();
    PASS();
}

int main() {
    std::cout << "=== Distributed Training Tests ===" << std::endl;
    std::cout << "Backend: shared-memory (multi-thread)" << std::endl;

    test_init();
    test_allreduce_sum();
    test_allreduce_avg();
    test_broadcast();
    test_scatter();
    test_allreduce_large();
    test_allreduce_max_min();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_total
              << " tests passed ===" << std::endl;

    return (tests_passed == tests_total) ? 0 : 1;
}
