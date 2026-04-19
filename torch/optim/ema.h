#pragma once

#include "torch/nn/parameter.h"
#include "aten/src/ATen/ATen.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <stdexcept>

namespace torch {
namespace optim {

using at::Tensor;
using nn::Parameter;

// ============================================================================
// EMA - Exponential Moving Average of model parameters
// ============================================================================
// Maintains a shadow copy of each parameter: shadow = decay * shadow +
// (1 - decay) * current_weight. Used to produce a smoother weight set for
// evaluation; call apply_shadow() to swap shadow into the live weights (backup
// stashed), and restore() to put the originals back for continued training.
//
// Usage:
//   EMA ema(model.parameters(), 0.999f);
//   for (batch in loader) {
//       optimizer.zero_grad(); loss.backward(); optimizer.step();
//       ema.update();
//   }
//   // For evaluation:
//   ema.apply_shadow();
//   evaluate(model);
//   ema.restore();

class EMA {
public:
    EMA(std::vector<std::shared_ptr<Parameter>> params, float decay = 0.999f)
        : params_(std::move(params)), decay_(decay) {
        if (decay_ < 0.0f || decay_ >= 1.0f) {
            throw std::runtime_error("EMA: decay must be in [0, 1)");
        }
        // Initialize shadow with detached clones of current parameter values.
        for (auto& pp : params_) {
            if (!pp || !pp->defined()) continue;
            Parameter* key = pp.get();
            shadow_[key] = pp->data().detach().clone();
        }
    }

    // Raw-pointer overload for code that already has vector<Parameter*>.
    EMA(const std::vector<Parameter*>& raw_params, float decay = 0.999f)
        : decay_(decay) {
        if (decay_ < 0.0f || decay_ >= 1.0f) {
            throw std::runtime_error("EMA: decay must be in [0, 1)");
        }
        params_.reserve(raw_params.size());
        for (auto* p : raw_params) {
            if (!p || !p->defined()) continue;
            // Wrap into shared_ptr with no-op deleter (not owning).
            params_.emplace_back(p, [](Parameter*){});
            shadow_[p] = p->data().detach().clone();
        }
    }

    // Update shadow weights from the live parameters.
    //   shadow = decay * shadow + (1 - decay) * weight
    void update() {
        const float one_minus = 1.0f - decay_;
        for (auto& pp : params_) {
            if (!pp || !pp->defined()) continue;
            Parameter* key = pp.get();
            auto it = shadow_.find(key);
            if (it == shadow_.end()) {
                shadow_[key] = pp->data().detach().clone();
                continue;
            }
            Tensor& sh = it->second;
            const Tensor& w = pp->data();
            int64_t n = sh.numel();
            if (n != w.numel()) {
                // Shape changed (e.g. resized param) – reinit shadow.
                sh = w.detach().clone();
                continue;
            }
            // In-place EMA update on raw float* (CPU path).
            float* sp = sh.mutable_data_ptr<float>();
            const float* wp = w.data_ptr<float>();
            for (int64_t i = 0; i < n; ++i) {
                sp[i] = decay_ * sp[i] + one_minus * wp[i];
            }
        }
    }

    // Swap live weights into backup and install shadow weights for eval.
    void apply_shadow() {
        backup_.clear();
        for (auto& pp : params_) {
            if (!pp || !pp->defined()) continue;
            Parameter* key = pp.get();
            auto it = shadow_.find(key);
            if (it == shadow_.end()) continue;
            // Stash original.
            backup_[key] = pp->data().detach().clone();
            // Copy shadow into live weights.
            pp->data().copy_(it->second);
        }
    }

    // Restore the originals saved by apply_shadow().
    void restore() {
        for (auto& pp : params_) {
            if (!pp || !pp->defined()) continue;
            Parameter* key = pp.get();
            auto it = backup_.find(key);
            if (it == backup_.end()) continue;
            pp->data().copy_(it->second);
        }
        backup_.clear();
    }

    float decay() const { return decay_; }
    void set_decay(float d) {
        if (d < 0.0f || d >= 1.0f) throw std::runtime_error("EMA: decay must be in [0, 1)");
        decay_ = d;
    }

    // Expose shadow for inspection / serialization.
    const std::unordered_map<Parameter*, Tensor>& shadow() const { return shadow_; }

private:
    std::vector<std::shared_ptr<Parameter>> params_;
    std::unordered_map<Parameter*, Tensor> shadow_;
    std::unordered_map<Parameter*, Tensor> backup_;
    float decay_;
};

} // namespace optim
} // namespace torch
