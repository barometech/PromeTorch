#pragma once

#include "torch/optim/optimizer.h"
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
#include <limits>

namespace torch {
namespace optim {

// ============================================================================
// LRScheduler - Base class for learning rate schedulers
// ============================================================================
// Learning rate schedulers adjust the learning rate during training according
// to some schedule. This can help achieve better convergence.
//
// Usage:
//   auto optimizer = SGD(model.parameters(), 0.1);
//   auto scheduler = StepLR(optimizer, /*step_size=*/30, /*gamma=*/0.1);
//
//   for (epoch = 0; epoch < num_epochs; epoch++) {
//       train_one_epoch();
//       scheduler.step();
//   }

class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer)
        : optimizer_(&optimizer), last_epoch_(-1) {
        // Store initial learning rates
        for (const auto& group : optimizer.param_groups()) {
            base_lrs_.push_back(group.lr);
        }
    }

    virtual ~LRScheduler() = default;

    // Advance to next epoch and update learning rates for ALL param groups.
    virtual void step(int64_t epoch = -1) {
        if (epoch == -1) {
            last_epoch_++;
        } else {
            last_epoch_ = epoch;
        }

        // Compute new learning rates
        std::vector<double> new_lrs = get_lr();

        // Apply to optimizer
        auto& groups = optimizer_->param_groups();
        for (size_t i = 0; i < groups.size() && i < new_lrs.size(); ++i) {
            groups[i].lr = new_lrs[i];
        }
    }

    // Per-group step: advance the schedule and update only ONE param group's lr.
    // Other groups keep their current lrs untouched. Useful for SequentialLR-style
    // chains where different groups follow different schedules.
    //
    // Note: last_epoch_ is still advanced (the underlying schedule is stateful and
    // shared); only the write-back is restricted to the chosen group. A separate
    // method name avoids ambiguity with step(int64_t epoch).
    virtual void step_group(size_t group_idx, int64_t epoch = -1) {
        if (epoch == -1) {
            last_epoch_++;
        } else {
            last_epoch_ = epoch;
        }
        std::vector<double> new_lrs = get_lr();
        auto& groups = optimizer_->param_groups();
        if (group_idx < groups.size() && group_idx < new_lrs.size()) {
            groups[group_idx].lr = new_lrs[group_idx];
        }
    }

    // Get current learning rates for each parameter group
    virtual std::vector<double> get_lr() = 0;

    // Get last epoch
    int64_t last_epoch() const { return last_epoch_; }

    // Get base learning rates
    const std::vector<double>& base_lrs() const { return base_lrs_; }

    // Get last learning rate (first group)
    double get_last_lr() const {
        if (optimizer_->param_groups().empty()) return 0.0;
        return optimizer_->param_groups()[0].lr;
    }

protected:
    Optimizer* optimizer_;
    int64_t last_epoch_;
    std::vector<double> base_lrs_;
};

// ============================================================================
// StepLR - Decay learning rate by gamma every step_size epochs
// ============================================================================
// lr = base_lr * gamma^(epoch // step_size)

class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, int64_t step_size, double gamma = 0.1)
        : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma) {}

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());

        double factor = std::pow(gamma_, last_epoch_ / step_size_);
        for (double base_lr : base_lrs_) {
            lrs.push_back(base_lr * factor);
        }
        return lrs;
    }

private:
    int64_t step_size_;
    double gamma_;
};

// ============================================================================
// MultiStepLR - Decay learning rate at specific milestones
// ============================================================================
// lr = base_lr * gamma^(number of milestones passed)

class MultiStepLR : public LRScheduler {
public:
    MultiStepLR(Optimizer& optimizer, std::vector<int64_t> milestones, double gamma = 0.1)
        : LRScheduler(optimizer), milestones_(std::move(milestones)), gamma_(gamma) {
        // Sort milestones
        std::sort(milestones_.begin(), milestones_.end());
    }

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());

        // Count how many milestones we've passed
        int64_t milestone_count = 0;
        for (int64_t milestone : milestones_) {
            if (last_epoch_ >= milestone) {
                milestone_count++;
            }
        }

        double factor = std::pow(gamma_, milestone_count);
        for (double base_lr : base_lrs_) {
            lrs.push_back(base_lr * factor);
        }
        return lrs;
    }

private:
    std::vector<int64_t> milestones_;
    double gamma_;
};

// ============================================================================
// ExponentialLR - Decay learning rate exponentially every epoch
// ============================================================================
// lr = base_lr * gamma^epoch

class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer& optimizer, double gamma)
        : LRScheduler(optimizer), gamma_(gamma) {}

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());

        double factor = std::pow(gamma_, last_epoch_);
        for (double base_lr : base_lrs_) {
            lrs.push_back(base_lr * factor);
        }
        return lrs;
    }

private:
    double gamma_;
};

// ============================================================================
// CosineAnnealingLR - Cosine annealing schedule
// ============================================================================
// lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int64_t T_max, double eta_min = 0.0)
        : LRScheduler(optimizer), T_max_(T_max), eta_min_(eta_min) {}

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());

        double cos_factor = (1.0 + std::cos(M_PI * last_epoch_ / T_max_)) / 2.0;
        for (double base_lr : base_lrs_) {
            double lr = eta_min_ + (base_lr - eta_min_) * cos_factor;
            lrs.push_back(lr);
        }
        return lrs;
    }

private:
    int64_t T_max_;
    double eta_min_;
};

// ============================================================================
// LinearLR - Linear warmup/decay schedule
// ============================================================================
// Multiplies LR by a factor that changes linearly from start_factor to end_factor

class LinearLR : public LRScheduler {
public:
    LinearLR(Optimizer& optimizer, double start_factor = 1.0/3.0,
             double end_factor = 1.0, int64_t total_iters = 5)
        : LRScheduler(optimizer)
        , start_factor_(start_factor)
        , end_factor_(end_factor)
        , total_iters_(total_iters) {}

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());

        double factor;
        if (last_epoch_ >= total_iters_) {
            factor = end_factor_;
        } else {
            // Linear interpolation
            double progress = static_cast<double>(last_epoch_) / total_iters_;
            factor = start_factor_ + progress * (end_factor_ - start_factor_);
        }

        for (double base_lr : base_lrs_) {
            lrs.push_back(base_lr * factor);
        }
        return lrs;
    }

private:
    double start_factor_;
    double end_factor_;
    int64_t total_iters_;
};

// ============================================================================
// ConstantLR - Multiply LR by a constant factor
// ============================================================================
// lr = base_lr * factor for total_iters epochs, then base_lr

class ConstantLR : public LRScheduler {
public:
    ConstantLR(Optimizer& optimizer, double factor = 1.0/3.0, int64_t total_iters = 5)
        : LRScheduler(optimizer), factor_(factor), total_iters_(total_iters) {}

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());

        double mult = (last_epoch_ < total_iters_) ? factor_ : 1.0;
        for (double base_lr : base_lrs_) {
            lrs.push_back(base_lr * mult);
        }
        return lrs;
    }

private:
    double factor_;
    int64_t total_iters_;
};

// ============================================================================
// ReduceLROnPlateau - Reduce LR when a metric has stopped improving
// ============================================================================
// Reduces learning rate when the monitored metric stops improving.

class ReduceLROnPlateau {
public:
    enum class Mode { Min, Max };

    ReduceLROnPlateau(
        Optimizer& optimizer,
        Mode mode = Mode::Min,
        double factor = 0.1,
        int64_t patience = 10,
        double threshold = 1e-4,
        double min_lr = 0.0
    )
        : optimizer_(&optimizer)
        , mode_(mode)
        , factor_(factor)
        , patience_(patience)
        , threshold_(threshold)
        , min_lr_(min_lr)
        , num_bad_epochs_(0)
        , best_(mode == Mode::Min ? std::numeric_limits<double>::infinity()
                                   : -std::numeric_limits<double>::infinity()) {}

    // Get last learning rate (first group)
    double get_last_lr() const {
        if (optimizer_->param_groups().empty()) return 0.0;
        return optimizer_->param_groups()[0].lr;
    }

    // Call this with the metric value after each epoch
    void step(double metric) {
        bool is_better;
        if (mode_ == Mode::Min) {
            is_better = metric < best_ - threshold_;
        } else {
            is_better = metric > best_ + threshold_;
        }

        if (is_better) {
            best_ = metric;
            num_bad_epochs_ = 0;
        } else {
            num_bad_epochs_++;

            if (num_bad_epochs_ > patience_) {
                // Reduce learning rate
                for (auto& group : optimizer_->param_groups()) {
                    double new_lr = std::max(group.lr * factor_, min_lr_);
                    group.lr = new_lr;
                }
                num_bad_epochs_ = 0;
            }
        }
    }

private:
    Optimizer* optimizer_;
    Mode mode_;
    double factor_;
    int64_t patience_;
    double threshold_;
    double min_lr_;
    int64_t num_bad_epochs_;
    double best_;
};

// ============================================================================
// WarmupCosineAnnealingLR - Cosine annealing with warmup
// ============================================================================

class WarmupCosineAnnealingLR : public LRScheduler {
public:
    WarmupCosineAnnealingLR(
        Optimizer& optimizer,
        int64_t warmup_epochs,
        int64_t total_epochs,
        double eta_min = 0.0
    )
        : LRScheduler(optimizer)
        , warmup_epochs_(warmup_epochs)
        , total_epochs_(total_epochs)
        , eta_min_(eta_min) {}

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());

        for (double base_lr : base_lrs_) {
            double lr;
            if (last_epoch_ < warmup_epochs_) {
                // Linear warmup
                lr = base_lr * (last_epoch_ + 1) / warmup_epochs_;
            } else {
                // Cosine annealing
                int64_t cosine_epoch = last_epoch_ - warmup_epochs_;
                int64_t cosine_total = total_epochs_ - warmup_epochs_;
                double cos_factor = (1.0 + std::cos(M_PI * cosine_epoch / cosine_total)) / 2.0;
                lr = eta_min_ + (base_lr - eta_min_) * cos_factor;
            }
            lrs.push_back(lr);
        }
        return lrs;
    }

private:
    int64_t warmup_epochs_;
    int64_t total_epochs_;
    double eta_min_;
};

// ============================================================================
// OneCycleLR - 1cycle learning rate policy
// ============================================================================
// Implements the 1cycle learning rate policy from "Super-Convergence"
// by Leslie Smith (2018).

class OneCycleLR {
public:
    OneCycleLR(
        Optimizer& optimizer,
        double max_lr,
        int64_t total_steps,
        double pct_start = 0.3,
        double div_factor = 25.0,
        double final_div_factor = 1e4
    )
        : optimizer_(&optimizer)
        , max_lr_(max_lr)
        , total_steps_(total_steps)
        , pct_start_(pct_start)
        , initial_lr_(max_lr / div_factor)
        , min_lr_(initial_lr_ / final_div_factor)
        , step_count_(0)
    {
        // Set initial learning rate
        for (auto& group : optimizer_->param_groups()) {
            group.lr = initial_lr_;
        }
    }

    void step() {
        step_count_++;

        double lr;
        int64_t warmup_steps = static_cast<int64_t>(total_steps_ * pct_start_);

        if (step_count_ <= warmup_steps) {
            // Phase 1: Linear warmup from initial_lr to max_lr
            double progress = static_cast<double>(step_count_) / warmup_steps;
            lr = initial_lr_ + progress * (max_lr_ - initial_lr_);
        } else {
            // Phase 2: Cosine annealing from max_lr to min_lr
            int64_t anneal_step = step_count_ - warmup_steps;
            int64_t anneal_total = total_steps_ - warmup_steps;
            double cos_factor = (1.0 + std::cos(M_PI * anneal_step / anneal_total)) / 2.0;
            lr = min_lr_ + (max_lr_ - min_lr_) * cos_factor;
        }

        for (auto& group : optimizer_->param_groups()) {
            group.lr = lr;
        }
    }

    double get_last_lr() const {
        if (optimizer_->param_groups().empty()) return 0.0;
        return optimizer_->param_groups()[0].lr;
    }

private:
    Optimizer* optimizer_;
    double max_lr_;
    int64_t total_steps_;
    double pct_start_;
    double initial_lr_;
    double min_lr_;
    int64_t step_count_;
};

// ============================================================================
// CosineAnnealingWarmRestarts - SGDR schedule (Loshchilov & Hutter 2017)
// ============================================================================
// Starts with cycle of length T_0. After each cycle finishes, the next cycle
// length is multiplied by T_mult. Within a cycle, LR anneals from base_lr
// down to eta_min via cosine.
//   lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * T_cur / T_i)) / 2

class CosineAnnealingWarmRestarts : public LRScheduler {
public:
    CosineAnnealingWarmRestarts(
        Optimizer& optimizer,
        int64_t T_0,
        int64_t T_mult = 1,
        double eta_min = 0.0
    )
        : LRScheduler(optimizer)
        , T_0_(T_0)
        , T_mult_(T_mult)
        , eta_min_(eta_min)
        , T_i_(T_0)
        , T_cur_(0) {
        if (T_0_ <= 0) throw std::runtime_error("T_0 must be positive");
        if (T_mult_ < 1) throw std::runtime_error("T_mult must be >= 1");
    }

    // Override step() to track T_cur / T_i across restarts
    void step(int64_t epoch = -1) override {
        if (epoch == -1) {
            last_epoch_++;
            T_cur_++;
            if (T_cur_ >= T_i_) {
                T_cur_ = 0;
                T_i_ *= T_mult_;
            }
        } else {
            last_epoch_ = epoch;
            // Recompute T_cur / T_i from scratch for the given absolute epoch
            int64_t rem = epoch;
            int64_t T_i = T_0_;
            while (rem >= T_i) {
                rem -= T_i;
                T_i *= T_mult_;
            }
            T_cur_ = rem;
            T_i_ = T_i;
        }

        auto lrs = get_lr();
        auto& groups = optimizer_->param_groups();
        for (size_t i = 0; i < groups.size() && i < lrs.size(); ++i) {
            groups[i].lr = lrs[i];
        }
    }

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());
        double cos_factor = (1.0 + std::cos(M_PI * T_cur_ / static_cast<double>(T_i_))) / 2.0;
        for (double base_lr : base_lrs_) {
            lrs.push_back(eta_min_ + (base_lr - eta_min_) * cos_factor);
        }
        return lrs;
    }

private:
    int64_t T_0_;
    int64_t T_mult_;
    double eta_min_;
    int64_t T_i_;     // current cycle length
    int64_t T_cur_;   // position within current cycle
};

// ============================================================================
// CyclicLR - Triangular cycle between base_lr and max_lr
// ============================================================================
// Modes:
//   Triangular       : symmetric up/down each cycle
//   Triangular2      : like Triangular, but amplitude halves each cycle
//   ExpRange         : amplitude scaled by gamma^iteration
// step_size_up   - iterations in increasing half of cycle
// step_size_down - iterations in decreasing half (defaults to step_size_up)

class CyclicLR : public LRScheduler {
public:
    enum class Mode { Triangular, Triangular2, ExpRange };

    CyclicLR(
        Optimizer& optimizer,
        double base_lr,
        double max_lr,
        int64_t step_size_up = 2000,
        int64_t step_size_down = -1,
        Mode mode = Mode::Triangular,
        double gamma = 1.0
    )
        : LRScheduler(optimizer)
        , base_lr_(base_lr)
        , max_lr_(max_lr)
        , step_size_up_(step_size_up)
        , step_size_down_(step_size_down < 0 ? step_size_up : step_size_down)
        , mode_(mode)
        , gamma_(gamma) {
        // Override base_lrs_ with user-supplied base_lr
        for (auto& g : optimizer.param_groups()) {
            g.lr = base_lr_;
        }
        for (auto& bl : base_lrs_) bl = base_lr_;
    }

    std::vector<double> get_lr() override {
        int64_t total_size = step_size_up_ + step_size_down_;
        int64_t cycle = last_epoch_ / total_size;
        int64_t pos = last_epoch_ - cycle * total_size;

        double x;
        if (pos < step_size_up_) {
            x = static_cast<double>(pos) / step_size_up_;
        } else {
            x = 1.0 - static_cast<double>(pos - step_size_up_) / step_size_down_;
        }

        double amplitude = max_lr_ - base_lr_;
        double scale = 1.0;
        if (mode_ == Mode::Triangular2) {
            scale = 1.0 / std::pow(2.0, static_cast<double>(cycle));
        } else if (mode_ == Mode::ExpRange) {
            scale = std::pow(gamma_, static_cast<double>(last_epoch_));
        }

        double lr = base_lr_ + amplitude * x * scale;
        return std::vector<double>(base_lrs_.size(), lr);
    }

private:
    double base_lr_;
    double max_lr_;
    int64_t step_size_up_;
    int64_t step_size_down_;
    Mode mode_;
    double gamma_;
};

// ============================================================================
// PolynomialLR - Polynomial decay
// ============================================================================
// lr = initial_lr * (1 - step / max_step)^power  (clamped at 0 after max_step)

class PolynomialLR : public LRScheduler {
public:
    PolynomialLR(Optimizer& optimizer, int64_t total_iters, double power = 1.0)
        : LRScheduler(optimizer), total_iters_(total_iters), power_(power) {}

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());
        double factor;
        if (last_epoch_ >= total_iters_) {
            factor = 0.0;
        } else {
            double frac = 1.0 - static_cast<double>(last_epoch_) / total_iters_;
            factor = std::pow(frac, power_);
        }
        for (double base_lr : base_lrs_) lrs.push_back(base_lr * factor);
        return lrs;
    }

private:
    int64_t total_iters_;
    double power_;
};

// ============================================================================
// LambdaLR - User-supplied function of step
// ============================================================================
// lr = base_lr * lambda_fn(last_epoch)
// If a per-group lambda vector is supplied, it must match param_groups().size().

class LambdaLR : public LRScheduler {
public:
    using LambdaFn = std::function<double(int64_t)>;

    LambdaLR(Optimizer& optimizer, LambdaFn fn)
        : LRScheduler(optimizer) {
        fns_.resize(optimizer.param_groups().size(), std::move(fn));
    }

    LambdaLR(Optimizer& optimizer, std::vector<LambdaFn> fns)
        : LRScheduler(optimizer), fns_(std::move(fns)) {
        if (fns_.size() != optimizer.param_groups().size()) {
            throw std::runtime_error(
                "LambdaLR: number of lambdas must match number of param_groups");
        }
    }

    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());
        for (size_t i = 0; i < base_lrs_.size(); ++i) {
            lrs.push_back(base_lrs_[i] * fns_[i](last_epoch_));
        }
        return lrs;
    }

private:
    std::vector<LambdaFn> fns_;
};

// ============================================================================
// MultiplicativeLR - Multiply current LR by lambda_fn(step) each step
// ============================================================================
// lr_{t} = lr_{t-1} * lambda_fn(last_epoch)   (uses current group lr, not base)

class MultiplicativeLR : public LRScheduler {
public:
    using LambdaFn = std::function<double(int64_t)>;

    MultiplicativeLR(Optimizer& optimizer, LambdaFn fn)
        : LRScheduler(optimizer) {
        fns_.resize(optimizer.param_groups().size(), std::move(fn));
    }

    MultiplicativeLR(Optimizer& optimizer, std::vector<LambdaFn> fns)
        : LRScheduler(optimizer), fns_(std::move(fns)) {
        if (fns_.size() != optimizer.param_groups().size()) {
            throw std::runtime_error(
                "MultiplicativeLR: lambda count must match param_groups");
        }
    }

    // Multiplicative update is stateful: apply factor to current lr directly.
    void step(int64_t epoch = -1) override {
        if (epoch == -1) {
            last_epoch_++;
        } else {
            last_epoch_ = epoch;
        }
        auto& groups = optimizer_->param_groups();
        if (last_epoch_ == 0) {
            // At epoch 0, lr stays at base_lr (no multiplication applied).
            return;
        }
        for (size_t i = 0; i < groups.size() && i < fns_.size(); ++i) {
            groups[i].lr *= fns_[i](last_epoch_);
        }
    }

    // get_lr here is not used internally (step overridden) but provided for API symmetry
    std::vector<double> get_lr() override {
        std::vector<double> lrs;
        lrs.reserve(optimizer_->param_groups().size());
        for (const auto& g : optimizer_->param_groups()) lrs.push_back(g.lr);
        return lrs;
    }

private:
    std::vector<LambdaFn> fns_;
};

// ============================================================================
// SequentialLR - Chain multiple schedulers back-to-back
// ============================================================================
// Runs scheduler[0] for milestones[0] steps, then scheduler[1] for
// (milestones[1] - milestones[0]) steps, etc. milestones[i] is the step
// index at which the (i+1)-th scheduler takes over.

class SequentialLR {
public:
    SequentialLR(
        Optimizer& optimizer,
        std::vector<std::shared_ptr<LRScheduler>> schedulers,
        std::vector<int64_t> milestones
    )
        : optimizer_(&optimizer)
        , schedulers_(std::move(schedulers))
        , milestones_(std::move(milestones))
        , step_count_(-1) {
        if (schedulers_.empty()) {
            throw std::runtime_error("SequentialLR: schedulers must not be empty");
        }
        if (milestones_.size() + 1 != schedulers_.size()) {
            throw std::runtime_error(
                "SequentialLR: len(milestones) must equal len(schedulers) - 1");
        }
        for (size_t i = 1; i < milestones_.size(); ++i) {
            if (milestones_[i] <= milestones_[i-1]) {
                throw std::runtime_error("SequentialLR: milestones must be strictly increasing");
            }
        }
    }

    void step() {
        step_count_++;
        // Find the active scheduler: first i such that step_count_ < milestones_[i]
        size_t active = schedulers_.size() - 1;
        for (size_t i = 0; i < milestones_.size(); ++i) {
            if (step_count_ < milestones_[i]) {
                active = i;
                break;
            }
        }
        schedulers_[active]->step();
    }

    double get_last_lr() const {
        if (optimizer_->param_groups().empty()) return 0.0;
        return optimizer_->param_groups()[0].lr;
    }

private:
    Optimizer* optimizer_;
    std::vector<std::shared_ptr<LRScheduler>> schedulers_;
    std::vector<int64_t> milestones_;
    int64_t step_count_;
};

// ============================================================================
// ChainedScheduler - Apply several schedulers in parallel
// ============================================================================
// After each step(), the final LR is the product of effects from each
// component scheduler. Each scheduler.step() is called independently; the
// resulting LR of the first param_group is combined multiplicatively by
// tracking the ratio each scheduler wants vs its own base_lr.

class ChainedScheduler {
public:
    explicit ChainedScheduler(
        Optimizer& optimizer,
        std::vector<std::shared_ptr<LRScheduler>> schedulers
    )
        : optimizer_(&optimizer), schedulers_(std::move(schedulers)) {
        if (schedulers_.empty()) {
            throw std::runtime_error("ChainedScheduler: schedulers must not be empty");
        }
        // Capture the shared base lrs (all schedulers must share the same optimizer).
        for (const auto& g : optimizer_->param_groups()) {
            base_lrs_.push_back(g.lr);
        }
    }

    void step() {
        // Compute each scheduler's desired multiplier vs its base and multiply.
        std::vector<double> combined(base_lrs_.size(), 1.0);
        for (auto& sch : schedulers_) {
            auto desired = sch->get_lr();
            // Advance the sub-scheduler's epoch counter without directly
            // writing back to the optimizer (we'll write the combined value below).
            // We still need to advance last_epoch_ inside the sub-scheduler; the
            // simplest way is to call its step() and then overwrite the LR after.
            sch->step();
            auto post = sch->get_lr();
            const auto& base = sch->base_lrs();
            for (size_t i = 0; i < combined.size() && i < post.size() && i < base.size(); ++i) {
                double denom = (base[i] != 0.0) ? base[i] : 1.0;
                combined[i] *= post[i] / denom;
                (void)desired;
            }
        }
        auto& groups = optimizer_->param_groups();
        for (size_t i = 0; i < groups.size() && i < combined.size(); ++i) {
            groups[i].lr = base_lrs_[i] * combined[i];
        }
    }

    double get_last_lr() const {
        if (optimizer_->param_groups().empty()) return 0.0;
        return optimizer_->param_groups()[0].lr;
    }

private:
    Optimizer* optimizer_;
    std::vector<std::shared_ptr<LRScheduler>> schedulers_;
    std::vector<double> base_lrs_;
};

} // namespace optim
} // namespace torch
