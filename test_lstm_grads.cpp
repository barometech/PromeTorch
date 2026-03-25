// Quick test: does the LSTM cell get gradients?
#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/rnn.h"
#include "torch/nn/modules/loss.h"

#include <cstdio>

int main() {
    using namespace torch;
    using namespace torch::nn;
    using at::Tensor;

    printf("=== LSTM Gradient Debug Test ===\n\n");

    // Create LSTM cell (input=1, hidden=4) + Linear(4,2)
    auto cell = std::make_shared<LSTMCellImpl>(1, 4);
    auto fc = std::make_shared<Linear>(4, 2);
    CrossEntropyLoss criterion;

    // Single sample: input [1, 3, 1] (batch=1, seq_len=3, features=1)
    Tensor x = at::empty({1, 3, 1});
    float* xp = x.mutable_data_ptr<float>();
    xp[0] = 0.5f; xp[1] = -0.3f; xp[2] = 0.8f;

    // Target: class 1 (positive sum)
    Tensor target = at::empty({1});
    target.mutable_data_ptr<float>()[0] = 1.0f;

    // Forward pass
    Tensor h = at::zeros({1, 4});
    Tensor c = at::zeros({1, 4});

    printf("Input x: [%.2f, %.2f, %.2f]\n", xp[0], xp[1], xp[2]);

    for (int t = 0; t < 3; ++t) {
        Tensor xt = autograd::select_autograd(x, 1, t); // [1, 1]
        printf("  t=%d: xt shape=[%lld,%lld] requires_grad=%d\n",
               t, xt.size(0), xt.size(1), xt.requires_grad());

        auto [h_new, c_new] = cell->forward_lstm(xt, h, c);
        printf("  t=%d: h_new requires_grad=%d, c_new requires_grad=%d\n",
               t, h_new.requires_grad(), c_new.requires_grad());
        h = h_new;
        c = c_new;
    }

    printf("\nFinal h requires_grad=%d\n", h.requires_grad());

    Tensor logits = fc->forward(h);
    printf("Logits shape=[%lld,%lld] requires_grad=%d\n",
           logits.size(0), logits.size(1), logits.requires_grad());

    Tensor loss = criterion.forward(logits, target);
    printf("Loss = %.6f requires_grad=%d\n\n",
           loss.data_ptr<float>()[0], loss.requires_grad());

    // Backward
    autograd::backward({loss});

    // Check gradients on LSTM parameters
    printf("LSTM cell parameters:\n");
    auto params = cell->parameters();
    for (auto* p : params) {
        Tensor g = p->grad();
        if (g.defined()) {
            float grad_norm = 0;
            const float* gp = g.data_ptr<float>();
            for (int64_t i = 0; i < g.numel(); ++i)
                grad_norm += gp[i] * gp[i];
            grad_norm = std::sqrt(grad_norm);
            printf("  param [%lld", p->data().size(0));
            if (p->data().dim() > 1) printf("x%lld", p->data().size(1));
            printf("] grad_norm = %.6g\n", grad_norm);
        } else {
            printf("  param [%lld", p->data().size(0));
            if (p->data().dim() > 1) printf("x%lld", p->data().size(1));
            printf("] NO GRADIENT!\n");
        }
    }

    printf("\nFC parameters:\n");
    auto fc_params = fc->parameters();
    for (auto* p : fc_params) {
        Tensor g = p->grad();
        if (g.defined()) {
            float grad_norm = 0;
            const float* gp = g.data_ptr<float>();
            for (int64_t i = 0; i < g.numel(); ++i)
                grad_norm += gp[i] * gp[i];
            grad_norm = std::sqrt(grad_norm);
            printf("  param [%lld", p->data().size(0));
            if (p->data().dim() > 1) printf("x%lld", p->data().size(1));
            printf("] grad_norm = %.6g\n", grad_norm);
        } else {
            printf("  param [%lld", p->data().size(0));
            if (p->data().dim() > 1) printf("x%lld", p->data().size(1));
            printf("] NO GRADIENT!\n");
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}
