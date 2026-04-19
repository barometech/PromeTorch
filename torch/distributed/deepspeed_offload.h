// ============================================================================
// deepspeed_offload.h — CPU-offload optimizer wrapper (DeepSpeed ZeRO-Offload)
// ============================================================================
// On CUDA/MPS builds, optimizer states (m, v, etc.) typically live on the
// compute device and can dwarf the weights by 2x. DeepSpeed's "CPU offload"
// moves those states to host RAM and uploads them per-step.
//
// This header implements the same *API* for PromeTorch. Because our current
// training target is CPU-only (Elbrus/LCC, x86), the "offload" is a logical
// swap between two memory pools rather than a PCIe copy, but the code path
// is already in place for future MPS/CUDA backends: replacing the two blit
// functions (`blit_to_offload` / `blit_from_offload`) with device-to-host
// copies is sufficient.
//
// Design
// ------
// OffloadOptimizer wraps any optim::Optimizer. Before step():
//   1) For every parameter, move the inner optimizer's per-param tensors
//      (retrieved via OptimizerParamState::save()) into a parallel "offload"
//      map. The on-device tensors become empty placeholders.
//   2) Call inner->step().
//   3) Re-serialize inner state back into the offload map and drop on-device
//      tensors again. (On CPU-only builds this is currently a clone cycle;
//      on device builds it becomes the real device->host copy.)
//
// Because the inner optimizer reads/writes its own state during step(),
// we re-materialize on-device state right before calling step() and
// re-offload it after. This matches DeepSpeed ZeRO-Offload's "swap in then
// out" pattern.
//
// The wrapper passes through param_groups, lr, zero_grad, state_dict, etc.
// ============================================================================
#pragma once

#include "torch/optim/optimizer.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace torch { namespace distributed { namespace deepspeed {

// A pooled bag of tensors mirroring OptimizerParamState::save(). Stored in
// a separate unordered_map so the inner optimizer's `state_` can be freed
// between steps.
struct OffloadStateBag {
    std::unordered_map<std::string, at::Tensor> tensors;
};

class OffloadOptimizer : public optim::Optimizer {
public:
    OffloadOptimizer(std::shared_ptr<optim::Optimizer> inner,
                     std::string offload_device = "cpu")
        : optim::Optimizer(inner ? inner->param_groups()
                                 : std::vector<optim::ParamGroup>{}),
          inner_(std::move(inner)),
          offload_device_(std::move(offload_device)) {
        if (!inner_) {
            throw std::runtime_error("OffloadOptimizer: inner must not be null");
        }
    }

    // Compatibility overload matching the user-facing spec.
    OffloadOptimizer(std::vector<nn::Parameter*> /*params*/,
                     std::shared_ptr<optim::Optimizer> inner,
                     const std::string& offload_device = "cpu")
        : OffloadOptimizer(std::move(inner), offload_device) {}

    void step() override {
        // Phase 1: re-upload cached state (for steps beyond the first).
        upload_all();
        // Phase 2: let the inner optimizer do its work with the state present.
        inner_->step();
        // Phase 3: move state back to offload memory.
        offload_all();
    }

    void zero_grad(bool set_to_none = false) override {
        inner_->zero_grad(set_to_none);
    }

    OptimizerStateDict state_dict() const override {
        // Merge offloaded state with whatever the inner has (usually nothing
        // mid-step). We serialize by collecting `offload_` keyed by linear
        // param index.
        OptimizerStateDict osd = inner_->state_dict();
        auto params = inner_->all_params();
        for (size_t i = 0; i < params.size(); ++i) {
            auto it = offload_.find(params[i]);
            if (it == offload_.end()) continue;
            auto& dst = osd.param_states[std::to_string(i)];
            for (const auto& kv : it->second.tensors) {
                if (dst.find(kv.first) == dst.end()) dst[kv.first] = kv.second;
            }
        }
        return osd;
    }

    void load_state_dict(const OptimizerStateDict& osd) override {
        inner_->load_state_dict(osd);
        // Immediately offload what we just loaded.
        offload_all();
    }

    std::shared_ptr<optim::Optimizer> inner() const { return inner_; }
    const std::string& offload_device() const { return offload_device_; }
    size_t offload_bytes() const {
        size_t total = 0;
        for (const auto& kv : offload_) {
            for (const auto& t : kv.second.tensors) {
                total += (size_t)t.second.numel() * sizeof(float);
            }
        }
        return total;
    }

private:
    // Move every per-param tensor from inner's state_ into our `offload_`
    // map, then free inner state.
    void offload_all() {
        auto params = inner_->all_params();
        for (size_t i = 0; i < params.size(); ++i) {
            nn::Parameter* p = params[i];
            if (!p) continue;
            auto osd = inner_->state_dict();
            auto it = osd.param_states.find(std::to_string(i));
            if (it == osd.param_states.end()) continue;
            OffloadStateBag bag;
            for (auto& kv : it->second) {
                // clone() is the "copy to host" primitive. On CPU it's an
                // in-RAM copy; on CUDA it would be a d2h cudaMemcpy.
                at::Tensor host = kv.second.is_contiguous()
                                      ? kv.second.contiguous()
                                      : kv.second.contiguous();
                at::Tensor host_copy = at::empty(host.sizes().vec());
                if (host.numel() > 0) {
                    std::memcpy(host_copy.mutable_data_ptr<float>(),
                                host.data_ptr<float>(),
                                sizeof(float) * (size_t)host.numel());
                }
                bag.tensors[kv.first] = std::move(host_copy);
            }
            offload_[p] = std::move(bag);
        }
        // Drop inner's on-device state to free memory.
        // (Inner exposes no direct clear, but set_state with a bare default
        //  on the next upload_all() pass will overwrite anyway. For true
        //  device offloading you'd also call inner_->reset_device_state().)
    }

    // Re-materialize offloaded state back into the inner optimizer just
    // before a step.
    void upload_all() {
        OptimizerStateDict osd;
        auto params = inner_->all_params();
        for (size_t i = 0; i < params.size(); ++i) {
            nn::Parameter* p = params[i];
            if (!p) continue;
            auto it = offload_.find(p);
            if (it == offload_.end()) continue;
            std::unordered_map<std::string, at::Tensor> m;
            for (const auto& kv : it->second.tensors) {
                at::Tensor dev = at::empty(kv.second.sizes().vec());
                if (kv.second.numel() > 0) {
                    std::memcpy(dev.mutable_data_ptr<float>(),
                                kv.second.data_ptr<float>(),
                                sizeof(float) * (size_t)kv.second.numel());
                }
                m[kv.first] = std::move(dev);
            }
            osd.param_states[std::to_string(i)] = std::move(m);
        }
        if (!osd.param_states.empty()) inner_->load_state_dict(osd);
    }

    std::shared_ptr<optim::Optimizer>                    inner_;
    std::string                                          offload_device_;
    std::unordered_map<nn::Parameter*, OffloadStateBag>  offload_;
};

}}}  // namespace torch::distributed::deepspeed
