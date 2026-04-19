// ============================================================================
// PromeTorch Data Loading Tests
// ============================================================================

#include "torch/data/data.h"
#include <gtest/gtest.h>
#include <vector>
#include <set>

using namespace torch;
using namespace torch::data;

// ============================================================================
// TensorDataset Tests
// ============================================================================

TEST(TensorDatasetTest, BasicConstruction) {
    at::Tensor inputs = at::randn({100, 10});
    at::Tensor labels = at::randint(0, 5, {100});

    TensorDataset dataset(inputs, labels);

    EXPECT_EQ(dataset.size(), 100);
    EXPECT_FALSE(dataset.empty());
}

TEST(TensorDatasetTest, GetExample) {
    at::Tensor inputs = at::arange(0, 30).view({10, 3});  // 10 samples, 3 features
    at::Tensor labels = at::arange(0, 10);  // Float tensor by default

    TensorDataset dataset(inputs, labels);

    auto example = dataset.get(5);

    // Check data shape
    EXPECT_EQ(example.data.numel(), 3);

    // Check data values: row 5 should be [15, 16, 17]
    float* data = example.data.mutable_data_ptr<float>();
    EXPECT_NEAR(data[0], 15.0f, 1e-5f);
    EXPECT_NEAR(data[1], 16.0f, 1e-5f);
    EXPECT_NEAR(data[2], 17.0f, 1e-5f);

    // Check label (arange returns float by default)
    float* label = example.target.mutable_data_ptr<float>();
    EXPECT_NEAR(label[0], 5.0f, 1e-5f);
}

TEST(TensorDatasetTest, WithoutTargets) {
    at::Tensor inputs = at::randn({50, 5});
    TensorDataset dataset(inputs);

    EXPECT_EQ(dataset.size(), 50);

    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_FALSE(example.target.defined());
}

// ============================================================================
// Sampler Tests
// ============================================================================

TEST(SequentialSamplerTest, ProducesCorrectSequence) {
    SequentialSampler sampler(5);

    std::vector<size_t> indices;
    for (size_t idx : sampler) {
        indices.push_back(idx);
    }

    EXPECT_EQ(indices.size(), 5);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);
    EXPECT_EQ(indices[2], 2);
    EXPECT_EQ(indices[3], 3);
    EXPECT_EQ(indices[4], 4);
}

TEST(SequentialSamplerTest, Reset) {
    SequentialSampler sampler(3);

    // First pass
    std::vector<size_t> pass1;
    for (size_t idx : sampler) {
        pass1.push_back(idx);
    }

    // Second pass after reset
    std::vector<size_t> pass2;
    for (size_t idx : sampler) {
        pass2.push_back(idx);
    }

    EXPECT_EQ(pass1, pass2);
}

TEST(RandomSamplerTest, ProducesAllIndices) {
    RandomSampler sampler(10, false, std::nullopt, 42);

    std::set<size_t> indices;
    for (size_t idx : sampler) {
        indices.insert(idx);
    }

    // Should produce all indices exactly once (no replacement)
    EXPECT_EQ(indices.size(), 10);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_TRUE(indices.count(i) > 0);
    }
}

TEST(RandomSamplerTest, WithReplacement) {
    RandomSampler sampler(5, true, 20, 42);  // 20 samples with replacement

    size_t count = 0;
    for (size_t idx : sampler) {
        EXPECT_LT(idx, 5);  // All indices should be valid
        ++count;
    }

    EXPECT_EQ(count, 20);
}

TEST(RandomSamplerTest, Reproducible) {
    RandomSampler sampler1(10, false, std::nullopt, 12345);
    RandomSampler sampler2(10, false, std::nullopt, 12345);

    std::vector<size_t> seq1, seq2;
    for (size_t idx : sampler1) seq1.push_back(idx);
    for (size_t idx : sampler2) seq2.push_back(idx);

    EXPECT_EQ(seq1, seq2);
}

TEST(SubsetRandomSamplerTest, SamplesFromSubset) {
    std::vector<size_t> subset = {2, 5, 7, 9};
    SubsetRandomSampler sampler(subset, 42);

    std::set<size_t> sampled;
    for (size_t idx : sampler) {
        sampled.insert(idx);
    }

    EXPECT_EQ(sampled.size(), 4);
    for (size_t idx : subset) {
        EXPECT_TRUE(sampled.count(idx) > 0);
    }
}

TEST(BatchSamplerTest, CreatesBatches) {
    auto base_sampler = std::make_unique<SequentialSampler>(10);
    BatchSampler batch_sampler(std::move(base_sampler), 3, false);

    std::vector<std::vector<size_t>> batches;
    for (const auto& batch : batch_sampler) {
        batches.push_back(batch);
    }

    // With 10 items, batch_size=3, drop_last=false: 4 batches (3,3,3,1)
    EXPECT_EQ(batches.size(), 4);
    EXPECT_EQ(batches[0].size(), 3);
    EXPECT_EQ(batches[1].size(), 3);
    EXPECT_EQ(batches[2].size(), 3);
    EXPECT_EQ(batches[3].size(), 1);
}

TEST(BatchSamplerTest, DropLast) {
    auto base_sampler = std::make_unique<SequentialSampler>(10);
    BatchSampler batch_sampler(std::move(base_sampler), 3, true);  // drop_last=true

    std::vector<std::vector<size_t>> batches;
    for (const auto& batch : batch_sampler) {
        batches.push_back(batch);
    }

    // With drop_last=true: only 3 full batches
    EXPECT_EQ(batches.size(), 3);
    for (const auto& batch : batches) {
        EXPECT_EQ(batch.size(), 3);
    }
}

// ============================================================================
// DataLoader Tests
// ============================================================================

TEST(DataLoaderTest, BasicIteration) {
    at::Tensor inputs = at::randn({100, 5});
    at::Tensor labels = at::randint(0, 10, {100});
    TensorDataset dataset(inputs, labels);

    auto loader = make_data_loader(dataset, 10, false);  // batch_size=10, no shuffle

    size_t batch_count = 0;
    size_t total_samples = 0;

    for (auto& batch : loader) {
        EXPECT_EQ(batch.data.size(0), 10);  // batch size
        EXPECT_EQ(batch.data.size(1), 5);   // features
        EXPECT_EQ(batch.target.size(0), 10);
        batch_count++;
        total_samples += batch.size;
    }

    EXPECT_EQ(batch_count, 10);  // 100 / 10 = 10 batches
    EXPECT_EQ(total_samples, 100);
}

TEST(DataLoaderTest, WithShuffle) {
    at::Tensor inputs = at::arange(0, 20).view({20, 1});
    at::Tensor labels = at::arange(0, 20);
    TensorDataset dataset(inputs, labels);

    auto loader = make_data_loader(
        dataset,
        DataLoaderOptions().batch_size_(5).shuffle_(true).seed_(42)
    );

    std::vector<int64_t> all_labels;
    for (auto& batch : loader) {
        const int64_t* ptr = batch.target.data_ptr<int64_t>();
        for (size_t i = 0; i < batch.size; ++i) {
            all_labels.push_back(ptr[i]);
        }
    }

    // Should have all 20 labels
    EXPECT_EQ(all_labels.size(), 20);

    // Should be shuffled (not in order 0,1,2,...)
    bool is_ordered = true;
    for (size_t i = 0; i < all_labels.size(); ++i) {
        if (all_labels[i] != static_cast<int64_t>(i)) {
            is_ordered = false;
            break;
        }
    }
    EXPECT_FALSE(is_ordered);  // Should be shuffled
}

TEST(DataLoaderTest, DropLast) {
    at::Tensor inputs = at::randn({25, 3});
    at::Tensor labels = at::randint(0, 5, {25});
    TensorDataset dataset(inputs, labels);

    auto loader = make_data_loader(
        dataset,
        DataLoaderOptions().batch_size_(10).drop_last_(true)
    );

    size_t batch_count = 0;
    for (auto& batch : loader) {
        EXPECT_EQ(batch.size, 10);  // All batches should be full
        batch_count++;
    }

    // 25 / 10 = 2 full batches (last 5 dropped)
    EXPECT_EQ(batch_count, 2);
}

TEST(DataLoaderTest, Size) {
    at::Tensor inputs = at::randn({100, 5});
    at::Tensor labels = at::randint(0, 10, {100});
    TensorDataset dataset(inputs, labels);

    auto loader1 = make_data_loader(dataset, DataLoaderOptions().batch_size_(10));
    EXPECT_EQ(loader1.size(), 10);  // 100 / 10 = 10

    auto loader2 = make_data_loader(dataset, DataLoaderOptions().batch_size_(30));
    EXPECT_EQ(loader2.size(), 4);  // ceil(100 / 30) = 4

    auto loader3 = make_data_loader(dataset, DataLoaderOptions().batch_size_(30).drop_last_(true));
    EXPECT_EQ(loader3.size(), 3);  // 100 / 30 = 3
}

TEST(DataLoaderTest, MultipleEpochs) {
    at::Tensor inputs = at::randn({20, 3});
    at::Tensor labels = at::randint(0, 5, {20});
    TensorDataset dataset(inputs, labels);

    auto loader = make_data_loader(dataset, 5, false);

    for (int epoch = 0; epoch < 3; ++epoch) {
        size_t batch_count = 0;
        for (auto& batch : loader) {
            (void)batch;
            batch_count++;
        }
        EXPECT_EQ(batch_count, 4);
    }
}

// ============================================================================
// Dataset Utilities Tests
// ============================================================================

TEST(ConcatDatasetTest, ConcatenateTwoDatasets) {
    at::Tensor inputs1 = at::full({10, 2}, 1.0f);
    at::Tensor labels1 = at::full({10}, 0);
    auto ds1 = std::make_shared<TensorDataset>(inputs1, labels1);

    at::Tensor inputs2 = at::full({5, 2}, 2.0f);
    at::Tensor labels2 = at::full({5}, 1);
    auto ds2 = std::make_shared<TensorDataset>(inputs2, labels2);

    ConcatDataset<Tensor, Tensor> concat_ds({ds1, ds2});

    EXPECT_EQ(concat_ds.size(), 15);

    // First dataset items
    auto ex0 = concat_ds.get(0);
    EXPECT_NEAR(ex0.data.data_ptr<float>()[0], 1.0f, 1e-5f);

    // Second dataset items
    auto ex10 = concat_ds.get(10);
    EXPECT_NEAR(ex10.data.data_ptr<float>()[0], 2.0f, 1e-5f);
}

TEST(SubsetDatasetTest, CreatesSubset) {
    at::Tensor inputs = at::arange(0, 10).view({10, 1});
    at::Tensor labels = at::arange(0, 10);
    auto full_ds = std::make_shared<TensorDataset>(inputs, labels);

    std::vector<size_t> indices = {2, 5, 7};
    SubsetDataset<Tensor, Tensor> subset(full_ds, indices);

    EXPECT_EQ(subset.size(), 3);

    auto ex0 = subset.get(0);
    EXPECT_NEAR(ex0.data.data_ptr<float>()[0], 2.0f, 1e-5f);

    auto ex1 = subset.get(1);
    EXPECT_NEAR(ex1.data.data_ptr<float>()[0], 5.0f, 1e-5f);

    auto ex2 = subset.get(2);
    EXPECT_NEAR(ex2.data.data_ptr<float>()[0], 7.0f, 1e-5f);
}

// ============================================================================
// Transform Tests
// ============================================================================

TEST(TransformTest, Normalize) {
    Example<Tensor, Tensor> example(
        at::full({3}, 10.0f),
        at::full({1}, 0)
    );

    Normalize normalize(5.0, 2.0);  // (x - 5) / 2
    auto transformed = normalize(example);

    // (10 - 5) / 2 = 2.5
    float* data = transformed.data.mutable_data_ptr<float>();
    EXPECT_NEAR(data[0], 2.5f, 1e-5f);
    EXPECT_NEAR(data[1], 2.5f, 1e-5f);
    EXPECT_NEAR(data[2], 2.5f, 1e-5f);
}

TEST(TransformTest, Compose) {
    Example<Tensor, Tensor> example(
        at::full({2, 3}, 10.0f),
        at::full({1}, 0)
    );

    auto transform = compose(
        Flatten(),
        Normalize(5.0, 2.0)
    );

    auto transformed = transform(example);

    // Should be flattened to 6 elements
    EXPECT_EQ(transformed.data.numel(), 6);

    // And normalized: (10 - 5) / 2 = 2.5
    float* data = transformed.data.mutable_data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(data[i], 2.5f, 1e-5f);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(IntegrationTest, TrainingLoopSimulation) {
    // Create synthetic dataset
    at::Tensor X = at::randn({64, 10});
    at::Tensor y = at::randint(0, 2, {64});
    TensorDataset dataset(X, y);

    // Create dataloader
    auto loader = make_data_loader(
        dataset,
        DataLoaderOptions().batch_size_(16).shuffle_(true)
    );

    // Simulate training loop
    int num_epochs = 2;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        size_t batch_idx = 0;
        for (auto& batch : loader) {
            // Verify batch shapes
            EXPECT_EQ(batch.data.dim(), 2);
            EXPECT_EQ(batch.data.size(1), 10);
            EXPECT_LE(batch.size, 16);

            batch_idx++;
        }
        EXPECT_EQ(batch_idx, 4);  // 64 / 16 = 4 batches
    }
}

// ============================================================================
// Multi-worker Prefetch Tests (num_workers > 0)
// ============================================================================
// Compare a synchronous DataLoader against a 4-worker prefetching DataLoader
// over the same synthetic dataset. Both must yield the same total number of
// batches and the same total number of items, regardless of arrival order.

TEST(DataLoaderMultiWorker, SameTotalsAsSingleThreaded) {
    const int64_t N = 256;
    at::Tensor X = at::arange(0, N).view({N, 1});  // each sample is its own index
    TensorDataset dataset(X);

    // Synchronous reference run.
    auto loader_sync = make_data_loader(
        dataset,
        DataLoaderOptions().batch_size_(8).shuffle_(false).drop_last_(false)
    );

    size_t batches_sync = 0;
    size_t items_sync = 0;
    std::set<int64_t> seen_sync;
    for (auto& batch : loader_sync) {
        ++batches_sync;
        items_sync += batch.size;
        float* p = batch.data.mutable_data_ptr<float>();
        for (size_t i = 0; i < batch.size; ++i) {
            seen_sync.insert(static_cast<int64_t>(p[i]));
        }
    }

    // Multi-worker run with the same dataset/options + workers.
    auto loader_mt = make_data_loader(
        dataset,
        DataLoaderOptions().batch_size_(8).shuffle_(false).drop_last_(false)
            .num_workers_(4).prefetch_factor_(2)
    );

    size_t batches_mt = 0;
    size_t items_mt = 0;
    std::set<int64_t> seen_mt;
    for (auto& batch : loader_mt) {
        ++batches_mt;
        items_mt += batch.size;
        float* p = batch.data.mutable_data_ptr<float>();
        for (size_t i = 0; i < batch.size; ++i) {
            seen_mt.insert(static_cast<int64_t>(p[i]));
        }
    }

    EXPECT_EQ(batches_mt, batches_sync);
    EXPECT_EQ(items_mt, items_sync);
    EXPECT_EQ(items_mt, static_cast<size_t>(N));
    EXPECT_EQ(seen_mt, seen_sync);  // every index covered exactly once
}

TEST(DataLoaderMultiWorker, MultiEpochAndDropLast) {
    const int64_t N = 100;
    at::Tensor X = at::arange(0, N).view({N, 1});
    TensorDataset dataset(X);

    auto loader = make_data_loader(
        dataset,
        DataLoaderOptions().batch_size_(16).shuffle_(true).drop_last_(true)
            .num_workers_(2).prefetch_factor_(3).seed_(42)
    );

    // drop_last=true => 100/16 = 6 full batches.
    for (int epoch = 0; epoch < 3; ++epoch) {
        size_t batches = 0;
        size_t items = 0;
        for (auto& batch : loader) {
            ++batches;
            items += batch.size;
            EXPECT_EQ(batch.size, 16u);
        }
        EXPECT_EQ(batches, 6u);
        EXPECT_EQ(items, 96u);
    }
}
