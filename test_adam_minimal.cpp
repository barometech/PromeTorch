// Minimal Adam test to diagnose the issue
#include "aten/src/ATen/ATen.h"
#include "torch/optim/adam.h"
#include "torch/nn/parameter.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace torch::optim;
using namespace torch::nn;
using at::Tensor;

// Simple quadratic loss: L = sum(w^2) / 2
// Gradient: dL/dw = w
// Optimal: w = 0

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Minimal Adam Test ===" << std::endl;

    // Create a simple parameter tensor
    Tensor w_data = at::ones({10});  // Start at w = [1, 1, ..., 1]
    Parameter w(w_data);

    // Create Adam optimizer
    AdamOptions opts(0.001);  // lr = 0.001
    std::vector<Parameter*> params = {&w};
    Adam optimizer(params, opts);

    std::cout << "Initial weight[0]: " << w.data().data_ptr<float>()[0] << std::endl;
    std::cout << "lr=" << opts.lr << ", beta1=" << opts.beta1 << ", beta2=" << opts.beta2 << std::endl;
    std::cout << std::endl;

    // Manual gradient computation: grad = w (for L = w^2/2)
    for (int step = 1; step <= 20; ++step) {
        // Zero grad
        if (w.grad().defined()) {
            w.grad().zero_();
        }

        // Compute loss: L = sum(w^2) / 2
        Tensor loss = w.data().mul(w.data()).sum().mul(at::Scalar(0.5f));
        float loss_val = loss.data_ptr<float>()[0];

        // Manual gradient: grad = w
        Tensor grad = w.data().clone();
        w.set_grad(grad);

        // Print before step
        float w0_before = w.data().data_ptr<float>()[0];
        float g0 = grad.data_ptr<float>()[0];

        // Take optimizer step
        optimizer.step();

        // Print after step
        float w0_after = w.data().data_ptr<float>()[0];
        float delta = w0_after - w0_before;

        std::cout << "Step " << std::setw(2) << step
                  << ": loss=" << std::setw(10) << loss_val
                  << ", w[0]=" << std::setw(10) << w0_before
                  << ", grad[0]=" << std::setw(10) << g0
                  << ", delta=" << std::setw(12) << delta
                  << ", new_w[0]=" << std::setw(10) << w0_after
                  << std::endl;

        // Check for explosion
        if (std::abs(w0_after) > 100 || std::isnan(w0_after)) {
            std::cout << "*** EXPLOSION DETECTED! ***" << std::endl;
            break;
        }
    }

    std::cout << std::endl;
    std::cout << "Expected behavior: w should decrease towards 0" << std::endl;
    std::cout << "Adam update: w = w - lr * m_hat / (sqrt(v_hat) + eps)" << std::endl;
    std::cout << "For w>0, grad>0, so update should be negative (decreasing w)" << std::endl;

    return 0;
}
