// Minimal test to verify gradient direction for Linear + CrossEntropy
// If gradient direction is wrong, this test will fail

#include "torch/nn/nn.h"
#include "torch/csrc/autograd/autograd.h"
#include <iostream>
#include <cmath>

using namespace torch;
using namespace torch::nn;

int main() {
    std::cout << "=== Gradient Direction Test ===" << std::endl;

    // Create simple linear model: 784 -> 10
    auto fc = std::make_shared<Linear>(784, 10);
    CrossEntropyLoss criterion;

    // Fixed input and target (no randomness)
    at::Tensor input = at::zeros({1, 784});
    at::Tensor target = at::zeros({1});

    // Set input to simple pattern
    float* in_ptr = input.mutable_data_ptr<float>();
    for (int i = 0; i < 784; ++i) {
        in_ptr[i] = (i % 10) / 10.0f;  // 0.0, 0.1, 0.2, ... 0.9, 0.0, ...
    }

    // Target class = 5
    target.mutable_data_ptr<float>()[0] = 5.0f;

    // Forward
    at::Tensor logits = fc->forward(input);
    at::Tensor loss = criterion.forward(logits, target);
    float loss0 = loss.data_ptr<float>()[0];

    std::cout << "Initial loss: " << loss0 << std::endl;

    // Print logits
    std::cout << "Logits: ";
    const float* log_ptr = logits.data_ptr<float>();
    for (int i = 0; i < 10; ++i) {
        std::cout << log_ptr[i] << " ";
    }
    std::cout << std::endl;

    // Backward
    torch::autograd::backward({loss});

    // Get weight gradient
    auto* weight_param = fc->parameters()[0];
    at::Tensor grad = weight_param->grad();

    if (!grad.defined()) {
        std::cout << "ERROR: Gradient not defined!" << std::endl;
        return 1;
    }

    std::cout << "Gradient shape: [" << grad.size(0) << ", " << grad.size(1) << "]" << std::endl;

    // Print some gradients
    const float* g_ptr = grad.data_ptr<float>();
    std::cout << "Sample gradients (first 10 weights):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  grad[0," << i << "] = " << g_ptr[i] << std::endl;
    }

    // Test: Apply gradient update and check if loss decreases
    std::cout << "\n=== Testing gradient step ===" << std::endl;

    float lr = 0.001f;
    at::Tensor weight = weight_param->data();
    float* w_ptr = weight.mutable_data_ptr<float>();

    // Manual update: w = w - lr * grad
    for (int64_t i = 0; i < weight.numel(); ++i) {
        w_ptr[i] -= lr * g_ptr[i];
    }

    // Forward again (clear any autograd state first)
    fc->zero_grad();
    at::Tensor logits2 = fc->forward(input);
    at::Tensor loss2 = criterion.forward(logits2, target);
    float loss1 = loss2.data_ptr<float>()[0];

    std::cout << "Loss after gradient step: " << loss1 << std::endl;
    std::cout << "Loss change: " << (loss1 - loss0) << std::endl;

    if (loss1 < loss0) {
        std::cout << "\n*** PASS: Loss decreased (gradient direction is correct) ***" << std::endl;
    } else {
        std::cout << "\n*** FAIL: Loss increased (gradient direction is WRONG!) ***" << std::endl;

        // Additional diagnostics
        std::cout << "\nDiagnostics:" << std::endl;
        std::cout << "If loss increased, gradient might have wrong sign." << std::endl;
        std::cout << "Expected: w_new = w - lr * grad should decrease loss" << std::endl;
        std::cout << "But: loss(w_new) > loss(w)" << std::endl;

        // Test opposite direction
        std::cout << "\nTesting OPPOSITE direction (w = w + lr * grad):" << std::endl;
        for (int64_t i = 0; i < weight.numel(); ++i) {
            w_ptr[i] += 2 * lr * g_ptr[i];  // Undo and go opposite
        }
        fc->zero_grad();
        at::Tensor logits3 = fc->forward(input);
        at::Tensor loss3 = criterion.forward(logits3, target);
        float loss2_val = loss3.data_ptr<float>()[0];

        std::cout << "Loss with opposite direction: " << loss2_val << std::endl;
        if (loss2_val < loss0) {
            std::cout << "*** IMPORTANT: Opposite direction DECREASES loss! ***" << std::endl;
            std::cout << "This confirms gradient sign is INVERTED somewhere!" << std::endl;
        }
    }

    return 0;
}
