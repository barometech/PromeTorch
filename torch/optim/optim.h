#pragma once

// ============================================================================
// PromeTorch Optimizer Module
// ============================================================================
// This is the main header for the torch::optim namespace.
// It provides optimizers and learning rate schedulers for training neural networks.
//
// Optimizers:
//   - SGD: Stochastic Gradient Descent with momentum and Nesterov
//   - Adam: Adaptive Moment Estimation
//   - AdamW: Adam with decoupled weight decay
//   - RMSprop: Root Mean Square Propagation
//
// Learning Rate Schedulers:
//   - StepLR: Decay by gamma every step_size epochs
//   - MultiStepLR: Decay at specific milestones
//   - ExponentialLR: Exponential decay every epoch
//   - CosineAnnealingLR: Cosine annealing schedule
//   - LinearLR: Linear warmup/decay
//   - ConstantLR: Constant factor for a number of epochs
//   - ReduceLROnPlateau: Reduce when metric stops improving
//   - WarmupCosineAnnealingLR: Warmup + cosine annealing
//   - OneCycleLR: 1cycle learning rate policy
//
// Usage:
//   #include "torch/optim/optim.h"
//
//   // Create optimizer
//   auto optimizer = torch::optim::Adam(model.parameters(), 0.001);
//
//   // Training loop
//   for (int epoch = 0; epoch < num_epochs; epoch++) {
//       for (auto& batch : dataloader) {
//           optimizer.zero_grad();
//           auto loss = model.forward(batch);
//           loss.backward();
//           optimizer.step();
//       }
//   }
//
// With learning rate scheduler:
//   auto optimizer = torch::optim::SGD(model.parameters(), 0.1);
//   auto scheduler = torch::optim::CosineAnnealingLR(optimizer, num_epochs);
//
//   for (int epoch = 0; epoch < num_epochs; epoch++) {
//       train_one_epoch(model, optimizer);
//       scheduler.step();
//   }
// ============================================================================

#include "torch/optim/optimizer.h"
#include "torch/optim/sgd.h"
#include "torch/optim/adam.h"
#include "torch/optim/rmsprop.h"
#include "torch/optim/lr_scheduler.h"

namespace torch {
namespace optim {

// ============================================================================
// Convenient factory functions
// ============================================================================

// Create SGD optimizer with common options
inline SGD make_sgd(std::vector<Parameter*> params, double lr = 0.01, double momentum = 0.0) {
    SGDOptions opts(lr);
    opts.momentum_(momentum);
    return SGD(std::move(params), opts);
}

// Create Adam optimizer with common options
inline Adam make_adam(std::vector<Parameter*> params, double lr = 0.001,
                      double beta1 = 0.9, double beta2 = 0.999) {
    AdamOptions opts(lr);
    opts.betas(beta1, beta2);
    return Adam(std::move(params), opts);
}

// Create AdamW optimizer with common options
inline AdamW make_adamw(std::vector<Parameter*> params, double lr = 0.001,
                        double weight_decay = 0.01) {
    AdamWOptions opts(lr);
    opts.weight_decay_(weight_decay);
    return AdamW(std::move(params), opts);
}

// Create RMSprop optimizer with common options
inline RMSprop make_rmsprop(std::vector<Parameter*> params, double lr = 0.01,
                            double alpha = 0.99) {
    RMSpropOptions opts(lr);
    opts.alpha_(alpha);
    return RMSprop(std::move(params), opts);
}

} // namespace optim
} // namespace torch
