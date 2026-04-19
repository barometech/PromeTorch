// ============================================================================
// test_ddp_no_sync.cpp — verify DDP no_sync()/DDPNoSyncGuard behaviour
// ============================================================================
// PromeTorch's DDP wrappers (torch::distributed::DistributedDataParallel in
// both torch/distributed/distributed.h and torch/distributed/ddp.h) have a
// MANUAL gradient-sync API: the user calls finish_gradient_synchronization()
// (or sync_gradients() / allreduce_grads()) AFTER backward and BEFORE
// optimizer.step(). There is no autograd post-backward hook in this codebase
// (engine.h has no hook system; see also the CPU-only Local-SGD path used on
// Elbrus that doesn't go through this DDP at all).
//
// This test verifies that wrapping the manual sync calls in a DDPNoSyncGuard
// (or with ddp.no_sync(): in Python) makes them no-ops, and that the previous
// flag value is properly restored on guard destruction (incl. exception path).
//
// To keep this single-process and runnable on Windows, we install a minimal
// "counting" ProcessGroup subclass that records every all_reduce call. We then
// make assertions about whether finish_gradient_synchronization() actually
// dispatches to all_reduce based on the no_sync flag.
//
// TODO(multi-process): A second test that fork()s 2 ranks, accumulates grads
// with no_sync() across N-1 micro-batches and verifies the final gradient on
// the syncing step equals the average of both ranks' summed local grads.
// That requires fork() (POSIX only) — see test/cpp/test_ddp.cpp for the
// existing POSIX 2-rank pattern.
//
// Build: link against aten_cpu + torch (no CUDA needed).
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "torch/distributed/distributed.h"
#include "torch/nn/module.h"
#include "torch/nn/parameter.h"

#include <cstdio>
#include <memory>
#include <stdexcept>

using at::Tensor;
using torch::distributed::DistributedDataParallel;
using torch::distributed::DDPNoSyncGuard;
using torch::distributed::ProcessGroup;
using torch::distributed::ReduceOp;
using torch::distributed::BackendType;

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++passed; std::printf("  PASS: %s\n", msg); }              \
    else      { ++failed; std::printf("  FAIL: %s\n", msg); }              \
} while (0)

// ---------------------------------------------------------------------------
// CountingPG — a ProcessGroup that records each collective call without doing
// any actual reduction. Lets the test observe whether DDP dispatched a sync.
// ---------------------------------------------------------------------------
class CountingPG : public ProcessGroup {
public:
    CountingPG() : ProcessGroup(/*rank=*/0, /*world_size=*/2,
                                BackendType::SHARED_MEMORY) {}

    int allreduce_calls = 0;
    int broadcast_calls = 0;

    void all_reduce(at::Tensor& /*t*/, ReduceOp /*op*/ = ReduceOp::SUM) override {
        ++allreduce_calls;
    }
    void broadcast(at::Tensor& /*t*/, int /*src*/ = 0) override {
        ++broadcast_calls;
    }
    void barrier() override {}
};

// ---------------------------------------------------------------------------
// TinyModule — one Parameter so DDP has something to iterate.
// ---------------------------------------------------------------------------
class TinyModule : public torch::nn::Module {
public:
    TinyModule() : torch::nn::Module("TinyModule") {
        register_parameter("w", torch::nn::Parameter(at::ones({4})));
    }
    Tensor forward(const Tensor& x) override { return x; }
};

// Install a fake gradient on every parameter so finish_gradient_synchronization
// doesn't skip the param via the `meta->grad_` null check.
static void inject_fake_grads(torch::nn::Module& m) {
    auto params = m.parameters(/*recurse=*/true);
    for (auto* p : params) {
        if (!p) continue;
        Tensor& t = p->data();
        if (!t.defined()) continue;
        auto* meta = torch::autograd::ensure_autograd_meta_impl(t);
        // Allocate a contiguous zeros grad of the same shape as the param.
        Tensor g = at::zeros(t.sizes());
        meta->grad_ = g.getIntrusivePtr();
    }
}

int main() {
    std::printf("=== DDP no_sync() self-test ===\n");

    // ------------------------------------------------------------------------
    // 1. Default: require_grad_sync()==true, sync calls hit the PG.
    // ------------------------------------------------------------------------
    {
        auto pg     = std::make_shared<CountingPG>();
        auto module = std::make_shared<TinyModule>();
        auto ddp    = std::make_shared<DistributedDataParallel>(
            module, pg, /*broadcast_parameters=*/false);
        inject_fake_grads(*module);

        CHECK(ddp->require_grad_sync() == true,
              "default require_grad_sync() is true (back-compat)");

        int before = pg->allreduce_calls;
        ddp->finish_gradient_synchronization();
        int after = pg->allreduce_calls;
        CHECK(after == before + 1,
              "finish_gradient_synchronization() runs 1 AllReduce per param "
              "with sync enabled (1 param)");

        before = pg->allreduce_calls;
        ddp->sync_gradients();
        after = pg->allreduce_calls;
        CHECK(after == before + 1,
              "sync_gradients() runs 1 AllReduce per param with sync enabled");
    }

    // ------------------------------------------------------------------------
    // 2. Inside DDPNoSyncGuard: sync calls are no-ops, flag restores on exit.
    // ------------------------------------------------------------------------
    {
        auto pg     = std::make_shared<CountingPG>();
        auto module = std::make_shared<TinyModule>();
        auto ddp    = std::make_shared<DistributedDataParallel>(
            module, pg, /*broadcast_parameters=*/false);
        inject_fake_grads(*module);

        int before = pg->allreduce_calls;
        {
            DDPNoSyncGuard g(*ddp);
            CHECK(ddp->require_grad_sync() == false,
                  "guard sets require_grad_sync=false on enter");
            ddp->finish_gradient_synchronization();   // expected: no-op
            ddp->sync_gradients();                    // expected: no-op
        }
        int after = pg->allreduce_calls;
        CHECK(after == before,
              "no AllReduce dispatched while DDPNoSyncGuard alive");
        CHECK(ddp->require_grad_sync() == true,
              "guard restored require_grad_sync=true on destruction");

        // After the guard ends, sync should work again.
        ddp->finish_gradient_synchronization();
        CHECK(pg->allreduce_calls == before + 1,
              "post-guard finish_gradient_synchronization() syncs as before");
    }

    // ------------------------------------------------------------------------
    // 3. Nesting: nested guards correctly save/restore the prior value.
    // ------------------------------------------------------------------------
    {
        auto pg     = std::make_shared<CountingPG>();
        auto module = std::make_shared<TinyModule>();
        auto ddp    = std::make_shared<DistributedDataParallel>(
            module, pg, /*broadcast_parameters=*/false);
        inject_fake_grads(*module);

        ddp->set_require_grad_sync(false);   // start with sync OFF
        {
            DDPNoSyncGuard g1(*ddp);
            CHECK(ddp->require_grad_sync() == false, "outer guard: sync false");
            {
                DDPNoSyncGuard g2(*ddp);
                CHECK(ddp->require_grad_sync() == false, "inner guard: sync false");
            }
            // Inner guard restored to outer's value (which was already false).
            CHECK(ddp->require_grad_sync() == false,
                  "after inner guard exits, prev value (false) restored");
        }
        // Outer guard restored to the pre-guard value (false, set above).
        CHECK(ddp->require_grad_sync() == false,
              "after outer guard exits, original value (false) restored");

        ddp->set_require_grad_sync(true);    // restore for cleanup
    }

    // ------------------------------------------------------------------------
    // 4. Exception safety: if the user throws inside the guard scope, the
    //    guard still restores the flag (RAII). Important for Python exceptions.
    // ------------------------------------------------------------------------
    {
        auto pg     = std::make_shared<CountingPG>();
        auto module = std::make_shared<TinyModule>();
        auto ddp    = std::make_shared<DistributedDataParallel>(
            module, pg, /*broadcast_parameters=*/false);
        inject_fake_grads(*module);

        bool caught = false;
        try {
            DDPNoSyncGuard g(*ddp);
            CHECK(ddp->require_grad_sync() == false, "pre-throw: sync false");
            throw std::runtime_error("simulated user error inside no_sync()");
        } catch (const std::runtime_error&) {
            caught = true;
        }
        CHECK(caught, "exception propagated out of guard scope");
        CHECK(ddp->require_grad_sync() == true,
              "guard restored sync=true even when scope unwound by exception");
    }

    // ------------------------------------------------------------------------
    // 5. Gradient-accumulation usage pattern: N-1 no_sync() backwards then 1
    //    syncing backward. Only the final call should hit the PG.
    // ------------------------------------------------------------------------
    {
        const int N = 4;
        auto pg     = std::make_shared<CountingPG>();
        auto module = std::make_shared<TinyModule>();
        auto ddp    = std::make_shared<DistributedDataParallel>(
            module, pg, /*broadcast_parameters=*/false);
        inject_fake_grads(*module);

        int before = pg->allreduce_calls;
        for (int mb = 0; mb < N; ++mb) {
            // First N-1 micro-batches: suppress sync via guard.
            // Final micro-batch: run finish_gradient_synchronization() outside
            // the guard so the locally-accumulated grad is averaged.
            if (mb < N - 1) {
                DDPNoSyncGuard g(*ddp);
                ddp->finish_gradient_synchronization();   // no-op
            } else {
                ddp->finish_gradient_synchronization();   // 1 AllReduce
            }
        }
        int after = pg->allreduce_calls;
        CHECK(after - before == 1,
              "N=4 micro-batches with no_sync() across first 3 → exactly 1 "
              "AllReduce instead of 4 (saves N-1)");
    }

    std::printf("=== Result: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
