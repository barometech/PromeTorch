// Minimal test to debug backward pass
#include "torch/nn/nn.h"
#include "torch/csrc/autograd/autograd.h"
#include <iostream>

using namespace torch;
using namespace torch::nn;

int main() {
    std::cout << "=== Testing Backward Debug ===\n";

    // Create simple linear layer
    auto fc = std::make_shared<Linear>(10, 5);

    // Create input (no gradient needed for input)
    at::Tensor input = at::randn({2, 10});

    // Forward pass
    std::cout << "Forward pass...\n";
    at::Tensor output = fc->forward(input);
    std::cout << "Output shape: [" << output.size(0) << ", " << output.size(1) << "]\n";

    // Simple loss: sum of squares
    std::cout << "Computing loss...\n";
    at::Tensor loss = output.pow(2.0f).sum();
    float loss_val = loss.data_ptr<float>()[0];
    std::cout << "Loss value: " << loss_val << "\n";

    // Check if loss has autograd metadata
    std::cout << "\n=== Checking autograd metadata ===\n";
    std::cout << "loss.defined(): " << loss.defined() << "\n";
    std::cout << "loss.requires_grad(): " << loss.requires_grad() << "\n";

    auto* meta = torch::autograd::get_autograd_meta(loss);
    std::cout << "loss meta: " << (meta ? "yes" : "null") << "\n";
    if (meta) {
        std::cout << "  meta->grad_fn: " << (meta->grad_fn ? meta->grad_fn->name() : "null") << "\n";
        std::cout << "  meta->is_leaf: " << meta->is_leaf_ << "\n";
    }

    // Check weight autograd
    auto* weight = fc->get_parameter("weight");
    std::cout << "\nweight.requires_grad(): " << weight->data().requires_grad() << "\n";
    auto* w_meta = torch::autograd::get_autograd_meta(weight->data());
    std::cout << "weight meta: " << (w_meta ? "yes" : "null") << "\n";
    if (w_meta) {
        std::cout << "  meta->grad_fn: " << (w_meta->grad_fn ? w_meta->grad_fn->name() : "null") << "\n";
        std::cout << "  meta->is_leaf: " << w_meta->is_leaf_ << "\n";
    }

    // Try backward
    std::cout << "\n=== Calling backward ===\n";
    torch::autograd::backward({loss});

    // Check gradients
    std::cout << "\n=== Checking gradients ===\n";
    at::Tensor grad = weight->grad();
    std::cout << "weight.grad().defined(): " << grad.defined() << "\n";
    if (grad.defined()) {
        float grad_sum = 0;
        const float* g = grad.data_ptr<float>();
        for (int64_t i = 0; i < grad.numel(); ++i) {
            grad_sum += g[i];
        }
        std::cout << "Gradient sum: " << grad_sum << "\n";
    }

    std::cout << "\n=== Test Complete ===\n";
    return 0;
}
