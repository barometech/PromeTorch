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

    // Advance to next epoch and update learning rates
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

} // namespace optim
} // namespace torch
