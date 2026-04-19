#pragma once

// ============================================================================
// PromeTorch Trainer — Lightning-style training wrapper
// ============================================================================
// Eliminates training-loop boilerplate. Subclass LightningModule, implement
// training_step / configure_optimizer, then call Trainer(cfg).fit(model, loader).
//
// Compiles CPU-only on Elbrus LCC: header-only, only STL + existing PromeTorch.
//
// Usage:
//   class MyModel : public torch::trainer::LightningModule {
//   public:
//       Tensor training_step(const Tensor& batch, int) override { ... }
//       std::shared_ptr<optim::Optimizer> configure_optimizer() override { ... }
//   };
//
//   TrainerConfig cfg; cfg.max_epochs = 10;
//   Trainer trainer(cfg);
//   trainer.fit(model, train_loader, &val_loader);

#include "torch/nn/module.h"
#include "torch/nn/nn.h"                  // clip_grad_norm_
#include "torch/optim/optimizer.h"
#include "torch/serialization.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/engine.h"   // autograd::backward

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace torch {
namespace trainer {

using at::Tensor;

// ============================================================================
// LightningModule — base class users subclass
// ============================================================================
// Override the three hooks below. Only training_step + configure_optimizer
// are required; validation_step has a default no-op implementation.

class LightningModule : public nn::Module {
public:
    LightningModule() : nn::Module("LightningModule") {}
    explicit LightningModule(const std::string& name) : nn::Module(name) {}

    // Compute and return the loss for one training batch.
    // The Trainer will call backward() on the returned tensor.
    virtual Tensor training_step(const Tensor& batch, int batch_idx) = 0;

    // Optional: compute a per-batch validation metric (loss, accuracy, ...).
    // Return an undefined tensor to skip / signal "no metric".
    virtual Tensor validation_step(const Tensor& batch, int batch_idx) {
        (void)batch; (void)batch_idx;
        return Tensor();
    }

    // Build and return the optimizer to use for training.
    virtual std::shared_ptr<optim::Optimizer> configure_optimizer() = 0;

    // Optional lifecycle hooks (Lightning parity).
    virtual void on_train_epoch_start(int /*epoch*/) {}
    virtual void on_train_epoch_end(int /*epoch*/, double /*avg_loss*/) {}
    virtual void on_validation_end(int /*epoch*/, double /*avg_val_loss*/) {}
};

// ============================================================================
// TrainerConfig
// ============================================================================

struct TrainerConfig {
    int max_epochs              = 10;
    int log_every_n_steps       = 50;
    int val_check_interval      = 0;     // 0 = once per epoch
    std::string checkpoint_dir  = "./checkpoints";
    int save_every_n_epochs     = 1;
    bool enable_progress_bar    = true;
    double gradient_clip_val    = 0.0;   // 0 = no clip (L2 norm clip)
    int accumulate_grad_batches = 1;     // backward N times, step once
};

// ============================================================================
// Internal helpers
// ============================================================================

namespace detail {

inline double tensor_to_double(const Tensor& t) {
    if (!t.defined() || t.numel() == 0) return 0.0;
    Tensor c = t.contiguous();
    return static_cast<double>(c.data_ptr<float>()[0]);
}

inline std::string format_loss(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4) << v;
    return ss.str();
}

inline std::string format_speed(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << v;
    return ss.str();
}

// Best-effort mkdir without <filesystem> (LCC may not ship it cleanly).
// On POSIX uses mkdir(2); on Windows uses _mkdir. Failure is silent — the
// later save() will surface a real error if the directory truly cannot be
// created.
inline void ensure_dir(const std::string& path) {
    if (path.empty()) return;
#if defined(_WIN32)
    std::system((std::string("mkdir \"") + path + "\" 2>nul").c_str());
#else
    std::system((std::string("mkdir -p \"") + path + "\" 2>/dev/null").c_str());
#endif
}

} // namespace detail

// ============================================================================
// Trainer
// ============================================================================
// Note: DataLoader in PromeTorch is a class template parameterised on the
// dataset type, so fit/test are templated on the loader type. This keeps the
// public surface identical to the spec (DataLoader& / DataLoader*) while
// supporting any concrete instantiation the user passes in.

class Trainer {
public:
    explicit Trainer(TrainerConfig cfg) : cfg_(std::move(cfg)) {
        if (cfg_.accumulate_grad_batches < 1) cfg_.accumulate_grad_batches = 1;
        if (cfg_.log_every_n_steps < 1)       cfg_.log_every_n_steps = 1;
        if (cfg_.save_every_n_epochs < 0)     cfg_.save_every_n_epochs = 0;
    }

    // -----------------------------------------------------------------------
    // fit — main training entry point
    // -----------------------------------------------------------------------
    template<typename TrainLoader, typename ValLoader = TrainLoader>
    void fit(LightningModule& module,
             TrainLoader& train_loader,
             ValLoader* val_loader = nullptr) {
        auto optimizer = module.configure_optimizer();
        if (!optimizer) {
            throw std::runtime_error("Trainer::fit: configure_optimizer() returned nullptr");
        }

        detail::ensure_dir(cfg_.checkpoint_dir);

        const int total_batches = static_cast<int>(train_loader.size());
        global_step_ = 0;

        for (int epoch = 1; epoch <= cfg_.max_epochs; ++epoch) {
            module.train(true);
            module.on_train_epoch_start(epoch);

            double epoch_loss_sum = 0.0;
            int    epoch_loss_n   = 0;
            int    batch_idx      = 0;

            auto epoch_start = std::chrono::steady_clock::now();
            auto win_start   = epoch_start;
            int  win_steps   = 0;

            optimizer->zero_grad();

            for (auto it = train_loader.begin(); it != train_loader.end(); ++it, ++batch_idx) {
                const auto& batch = *it;

                // training_step: most user batches expose a `.data` field
                // (DataLoader::Batch). Pass that as the tensor argument.
                Tensor loss = module.training_step(batch.data, batch_idx);

                if (!loss.defined()) {
                    throw std::runtime_error("training_step returned an undefined tensor");
                }

                // Scale loss when accumulating so the effective gradient is
                // the average over the accumulation window.
                if (cfg_.accumulate_grad_batches > 1) {
                    loss.mul_(at::Scalar(1.0 / cfg_.accumulate_grad_batches));
                }

                torch::autograd::backward({loss});

                const double loss_val = detail::tensor_to_double(loss)
                                        * (cfg_.accumulate_grad_batches > 1
                                           ? cfg_.accumulate_grad_batches : 1);
                epoch_loss_sum += loss_val;
                epoch_loss_n   += 1;

                // Step the optimizer at accumulation boundaries.
                const bool do_step =
                    ((batch_idx + 1) % cfg_.accumulate_grad_batches == 0) ||
                    (batch_idx + 1 == total_batches);

                if (do_step) {
                    if (cfg_.gradient_clip_val > 0.0) {
                        torch::nn::clip_grad_norm_(module, cfg_.gradient_clip_val);
                    }
                    optimizer->step();
                    optimizer->zero_grad();
                    ++global_step_;
                    ++win_steps;

                    // Periodic validation triggered by training-step count.
                    if (val_loader && cfg_.val_check_interval > 0 &&
                        (global_step_ % cfg_.val_check_interval == 0)) {
                        run_validation(module, *val_loader, epoch);
                        module.train(true);
                    }
                }

                // Progress / logging.
                if (cfg_.enable_progress_bar &&
                    ((batch_idx + 1) % cfg_.log_every_n_steps == 0 ||
                     batch_idx + 1 == total_batches)) {
                    auto now = std::chrono::steady_clock::now();
                    double dt = std::chrono::duration<double>(now - win_start).count();
                    double sps = (dt > 0) ? (win_steps / dt) : 0.0;
                    win_start = now;
                    win_steps = 0;

                    std::cout << "\repoch " << epoch << "/" << cfg_.max_epochs
                              << " step " << (batch_idx + 1) << "/" << total_batches
                              << " loss " << detail::format_loss(loss_val)
                              << " (" << detail::format_speed(sps) << " step/s)"
                              << "        " << std::flush;
                }
            }

            // End-of-epoch newline so progress bar doesn't stomp the next line.
            if (cfg_.enable_progress_bar) {
                std::cout << std::endl;
            }

            const double avg = epoch_loss_n ? epoch_loss_sum / epoch_loss_n : 0.0;
            auto epoch_end = std::chrono::steady_clock::now();
            double epoch_dt = std::chrono::duration<double>(epoch_end - epoch_start).count();

            std::cout << "[epoch " << epoch << "] avg_loss=" << detail::format_loss(avg)
                      << " time=" << detail::format_speed(epoch_dt) << "s"
                      << " steps=" << epoch_loss_n
                      << std::endl;

            module.on_train_epoch_end(epoch, avg);

            // Per-epoch validation when no step-interval is set.
            if (val_loader && cfg_.val_check_interval == 0) {
                run_validation(module, *val_loader, epoch);
            }

            // Checkpoint.
            if (cfg_.save_every_n_epochs > 0 &&
                (epoch % cfg_.save_every_n_epochs == 0)) {
                std::string path = cfg_.checkpoint_dir + "/epoch_" +
                                   std::to_string(epoch) + ".ptor";
                save_checkpoint(module, path);
                std::cout << "[epoch " << epoch << "] saved " << path << std::endl;
            }
        }
    }

    // -----------------------------------------------------------------------
    // test — evaluate over a held-out loader
    // -----------------------------------------------------------------------
    template<typename TestLoader>
    void test(LightningModule& module, TestLoader& test_loader) {
        module.eval();
        double sum = 0.0;
        int    n   = 0;
        int    idx = 0;
        auto t0 = std::chrono::steady_clock::now();

        for (auto it = test_loader.begin(); it != test_loader.end(); ++it, ++idx) {
            Tensor m = module.validation_step((*it).data, idx);
            if (m.defined()) {
                sum += detail::tensor_to_double(m);
                ++n;
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        double avg = n ? sum / n : 0.0;
        std::cout << "[test] avg=" << detail::format_loss(avg)
                  << " batches=" << n
                  << " time=" << detail::format_speed(dt) << "s" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Checkpointing
    // -----------------------------------------------------------------------
    void save_checkpoint(LightningModule& module, const std::string& path) {
        auto sd = module.state_dict();
        torch::save_state_dict(sd, path);
    }

    bool load_checkpoint(LightningModule& module, const std::string& path) {
        try {
            auto sd = torch::load_state_dict(path);
            module.load_state_dict(sd, /*strict=*/false);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "load_checkpoint(" << path << ") failed: "
                      << e.what() << std::endl;
            return false;
        }
    }

    int global_step() const { return global_step_; }
    const TrainerConfig& config() const { return cfg_; }

private:
    template<typename ValLoader>
    void run_validation(LightningModule& module, ValLoader& val_loader, int epoch) {
        module.eval();
        double sum = 0.0;
        int    n   = 0;
        int    idx = 0;

        for (auto it = val_loader.begin(); it != val_loader.end(); ++it, ++idx) {
            Tensor m = module.validation_step((*it).data, idx);
            if (m.defined()) {
                sum += detail::tensor_to_double(m);
                ++n;
            }
        }

        double avg = n ? sum / n : 0.0;
        std::cout << "[epoch " << epoch << "][val] avg=" << detail::format_loss(avg)
                  << " batches=" << n << std::endl;
        module.on_validation_end(epoch, avg);
    }

    TrainerConfig cfg_;
    int           global_step_ = 0;
};

} // namespace trainer
} // namespace torch
