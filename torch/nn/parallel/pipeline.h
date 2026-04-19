#pragma once

// ============================================================================
// Pipeline Parallel — GPipe-style pipeline parallelism for Sequential models.
// ============================================================================
//
// Splits a Sequential into `num_stages` logical stages. Each stage runs in its
// own std::thread. The input minibatch is sliced along dim 0 into `chunks`
// micro-batches and pushed through the pipeline. Outputs from the last stage
// are collected and concatenated back along dim 0.
//
// This implementation targets CPU-only execution (Elbrus / LCC, MSVC, GCC).
// It uses std::thread, std::queue, std::mutex, std::condition_variable.
// No CUDA, no platform-specific primitives.
//
// PLATFORM NOTE — Elbrus E2K + EML BLAS:
// On Elbrus the EML library's cblas_sgemm SIGILLs when called from a non-main
// pthread (documented in MEMORY.md → feedback_eml_pthread_sigill.md). Pipeline
// stages run on std::thread workers, so any layer that internally calls
// EML sgemm (Linear, Conv via gemm) will SIGILL on E2K. Stages that only use
// pthread-safe ops (activations, normalizations, RMSNorm, attention without
// EML) work correctly. On x86_64 / Windows / non-EML BLAS this constraint
// does not apply.
//
// BACKWARD IS NOT YET IMPLEMENTED.
// -----------------------------------------------------------------------------
// True GPipe backward (1F1B / BPPB scheduling) requires:
//   * caching activations per micro-batch per stage,
//   * coordinated reverse pipeline schedule,
//   * gradient routing across stage threads.
// For now, Pipeline is a forward-only inference / forward-pass primitive.
// If you need backward, run the underlying Sequential directly on a single
// device — the autograd engine will work as usual. Pipeline backward is a
// follow-up task (see TECHNICAL_SPECIFICATION.md, Phase 18+).
//
// Usage:
//     auto seq = std::make_shared<nn::Sequential>(...);
//     auto pipe = std::make_shared<nn::parallel::Pipeline>(seq, /*stages=*/2,
//                                                          /*chunks=*/4);
//     Tensor y = pipe->forward(x);   // x.size(0) % chunks == 0
//
// ============================================================================

#include "torch/nn/modules/container.h"
#include "torch/nn/parameter.h"
#include "aten/src/ATen/ATen.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace torch {
namespace nn {
namespace parallel {

using at::Tensor;

// ----------------------------------------------------------------------------
// ThreadSafeQueue — minimal SPSC/MPSC queue used between pipeline stages.
// Supports a "close" sentinel so consumer threads can drain & exit cleanly.
// ----------------------------------------------------------------------------
template <typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            q_.push(std::move(value));
        }
        cv_.notify_one();
    }

    // Returns true with the value, false if queue is closed and empty.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        std::queue<T> empty;
        std::swap(q_, empty);
        closed_ = false;
    }

private:
    std::queue<T> q_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool closed_ = false;
};

// ----------------------------------------------------------------------------
// Pipeline — GPipe-style forward pipeline over a Sequential.
// ----------------------------------------------------------------------------
class Pipeline {
public:
    // Split `model`'s layers across `num_stages` logical stages.
    // Partition strategy: stage i gets layers [i*N/S, (i+1)*N/S), where
    //   N = model->size(), S = num_stages.
    // `chunks` controls micro-batch count for pipeline parallelism (GPipe).
    Pipeline(std::shared_ptr<Sequential> model, int num_stages, int chunks = 4)
        : num_stages_(num_stages)
        , chunks_(chunks)
    {
        if (!model) {
            throw std::invalid_argument("Pipeline: model must not be null");
        }
        if (num_stages <= 0) {
            throw std::invalid_argument("Pipeline: num_stages must be > 0");
        }
        if (chunks <= 0) {
            throw std::invalid_argument("Pipeline: chunks must be > 0");
        }
        const int64_t N = static_cast<int64_t>(model->size());
        if (N == 0) {
            throw std::invalid_argument(
                "Pipeline: model has no layers to partition");
        }
        if (num_stages > N) {
            throw std::invalid_argument(
                "Pipeline: num_stages exceeds number of layers");
        }

        stages_.reserve(num_stages_);
        for (int i = 0; i < num_stages_; ++i) {
            int64_t lo = (i * N) / num_stages_;
            int64_t hi = ((i + 1) * N) / num_stages_;
            auto stage = std::make_shared<Sequential>();
            for (int64_t j = lo; j < hi; ++j) {
                stage->push_back((*model)[static_cast<size_t>(j)]);
            }
            stages_.push_back(stage);
        }
    }

    ~Pipeline() = default;

    // Forward pass.
    //
    // Splits `input` along dim 0 into `chunks` micro-batches, feeds them
    // sequentially into stage 0, runs each stage in its own std::thread, and
    // collects micro-batch outputs from the last stage. Returns the
    // concatenation along dim 0 (same shape as `input`'s dim 0 batch).
    //
    // Constraints:
    //   * input must have ndim >= 1
    //   * input.size(0) % chunks == 0
    Tensor forward(const Tensor& input) {
        if (!input.defined()) {
            throw std::invalid_argument("Pipeline::forward: undefined input");
        }
        const int64_t batch = input.size(0);
        if (batch % chunks_ != 0) {
            throw std::invalid_argument(
                "Pipeline::forward: batch size must be divisible by chunks");
        }
        const int64_t micro = batch / chunks_;

        // Per-stage input queues (queues_[s] feeds stage s).
        // queues_[num_stages_] collects final outputs from the last stage.
        std::vector<ThreadSafeQueue<Tensor>> queues(num_stages_ + 1);

        // Worker function for stage s: pull from queues[s], run all layers,
        // push result to queues[s+1]. Exits when its input queue is closed.
        auto worker = [&queues, this](int s) {
            auto& in_q  = queues[s];
            auto& out_q = queues[s + 1];
            auto& stage_mod = stages_[s];
            Tensor mb;
            while (in_q.pop(mb)) {
                Tensor y = stage_mod->forward(mb);
                out_q.push(std::move(y));
            }
            out_q.close();   // propagate close to downstream stage
        };

        // Spawn worker threads for every stage.
        std::vector<std::thread> threads;
        threads.reserve(num_stages_);
        for (int s = 0; s < num_stages_; ++s) {
            threads.emplace_back(worker, s);
        }

        // Producer: split input into micro-batches and feed stage 0.
        for (int c = 0; c < chunks_; ++c) {
            Tensor mb = input.narrow(0, c * micro, micro).contiguous();
            queues[0].push(std::move(mb));
        }
        // Signal end-of-stream to the head of the pipeline; close cascades.
        queues[0].close();

        // Consumer: collect outputs from the tail in order.
        std::vector<Tensor> out_chunks;
        out_chunks.reserve(static_cast<size_t>(chunks_));
        Tensor out;
        while (queues[num_stages_].pop(out)) {
            out_chunks.push_back(std::move(out));
        }

        // Join workers.
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        if (static_cast<int>(out_chunks.size()) != chunks_) {
            throw std::runtime_error(
                "Pipeline::forward: produced fewer chunks than expected");
        }
        return at::cat(out_chunks, /*dim=*/0);
    }

    // Concatenate parameters from every stage. Used to wire an optimizer to
    // the entire pipeline. Returns raw Parameter* (matches the convention used
    // by torch::optim::SGD / Adam constructors).
    std::vector<Parameter*> parameters() const {
        std::vector<Parameter*> all;
        for (const auto& stage : stages_) {
            auto p = stage->parameters(/*recurse=*/true);
            all.insert(all.end(), p.begin(), p.end());
        }
        return all;
    }

    int num_stages() const { return num_stages_; }
    int chunks()     const { return chunks_; }
    const std::vector<std::shared_ptr<Sequential>>& stages() const {
        return stages_;
    }

    // Backward is intentionally not provided — see header comment.
    // Calling the underlying Sequential directly will exercise autograd.

private:
    std::vector<std::shared_ptr<Sequential>> stages_;
    int num_stages_;
    int chunks_;
};

} // namespace parallel
} // namespace nn
} // namespace torch
