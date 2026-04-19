// ============================================================================
// QAT self-test — train a small Linear classifier with Quantization-Aware
// Training, then verify accuracy is within 2% of the float baseline.
// CPU-only, intended to compile and run on Elbrus.
// ----------------------------------------------------------------------------
// Synthetic "MNIST 0-vs-1" task: 1024 random samples of 16-D feature vectors,
// labelled 0 if mean<0 and 1 otherwise.  Trains a single Linear(16->2)
// classifier with CrossEntropy + SGD for both float and QAT models, then
// reports test accuracy and max output drift.
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/quantization/qat.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using at::Tensor;
using torch::nn::Linear;
using torch::nn::CrossEntropyLoss;
using torch::optim::SGD;
using torch::quantization::QuantizedLinear;
using torch::quantization::FakeQuantize;
using torch::quantization::int8_matmul;
using torch::quantization::fake_quantize_qdq;

// ---------- synthetic dataset -------------------------------------------------
// NOTE: CrossEntropyLoss's fused CPU path reads targets as float32 — we follow
// that ABI here.  Labels are integer-valued floats.
static void make_dataset(int N, int D, Tensor& X, Tensor& Y) {
    X = at::empty({N, D});
    Y = at::empty({N});                  // float32 — matches CE fused path
    float* xp = X.mutable_data_ptr<float>();
    float* yp = Y.mutable_data_ptr<float>();
    std::srand(7);
    for (int i = 0; i < N; ++i) {
        float mean = 0.f;
        for (int j = 0; j < D; ++j) {
            float v = ((float)std::rand() / RAND_MAX) - 0.5f;
            xp[i * D + j] = v;
            mean += v;
        }
        yp[i] = (mean > 0.f) ? 1.0f : 0.0f;
    }
}

// ---------- accuracy ----------------------------------------------------------
static float accuracy(Tensor& logits, Tensor& y) {
    const int64_t N = logits.size(0);
    const int64_t C = logits.size(1);
    const float* lp = logits.data_ptr<float>();
    const float* yp = y.data_ptr<float>();
    int correct = 0;
    for (int64_t i = 0; i < N; ++i) {
        int best = 0;
        float bv = lp[i * C + 0];
        for (int64_t k = 1; k < C; ++k) {
            if (lp[i * C + k] > bv) { bv = lp[i * C + k]; best = (int)k; }
        }
        if (best == static_cast<int>(yp[i])) correct++;
    }
    return 100.f * correct / N;
}

// ---------- training loop -----------------------------------------------------
template <typename ModelT>
static void train(std::shared_ptr<ModelT> m, Tensor& X, Tensor& Y,
                  int epochs, float lr) {
    auto crit = std::make_shared<CrossEntropyLoss>();
    SGD opt(m->parameters(), lr);
    for (int e = 0; e < epochs; ++e) {
        m->train();
        opt.zero_grad();
        // IMPORTANT: X must require grad so the autograd graph actually builds
        // (CrossEntropy only wires backward when inputs require grad).
        Tensor xg = X;
        xg.set_requires_grad(true);
        Tensor logits = m->forward(xg);
        Tensor loss   = crit->forward(logits, Y);
        torch::autograd::backward({loss});
        opt.step();
        if (e == 0 || (e + 1) % 20 == 0) {
            std::printf("  epoch %3d  loss = %.4f  train-acc = %.1f%%\n",
                        e + 1, loss.data_ptr<float>()[0],
                        accuracy(logits, Y));
        }
    }
}

int main() {
    constexpr int N = 1024, D = 16, C = 2;

    Tensor X, Y;  make_dataset(N, D, X, Y);

    // ------------------------------------------------------------------ Float
    std::printf("=== Float baseline ===\n");
    auto fmodel = std::make_shared<Linear>(D, C);
    train(fmodel, X, Y, 400, 0.2f);
    fmodel->eval();
    Tensor flogits = fmodel->forward(X);
    float facc = accuracy(flogits, Y);
    std::printf("Float accuracy: %.2f%%\n", facc);

    // -------------------------------------------------------------------- QAT
    std::printf("\n=== Quantization-Aware Training ===\n");
    auto qmodel = QuantizedLinear::from_linear(fmodel);
    train(qmodel, X, Y, 100, 0.1f);   // QAT fine-tune
    torch::quantization::convert(*qmodel);
    qmodel->eval();
    Tensor qlogits = qmodel->forward(X);
    float qacc = accuracy(qlogits, Y);
    std::printf("QAT accuracy:   %.2f%%\n", qacc);

    // ---- Drift -------------------------------------------------------------
    const float* a = flogits.data_ptr<float>();
    const float* b = qlogits.data_ptr<float>();
    float maxd = 0.f;
    for (int i = 0; i < N * C; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    std::printf("Max logit drift: %.4f\n", maxd);

    // ---- Verdict -----------------------------------------------------------
    bool ok_acc   = (facc >= 90.f) && (qacc >= 90.f);
    bool ok_close = std::fabs(facc - qacc) < 2.0f;
    std::printf("\nResult: float=%.1f%%  qat=%.1f%%  drift=%.3f  =>  %s\n",
                facc, qacc, maxd, (ok_acc && ok_close) ? "PASS" : "FAIL");

    // ---- Smoke test int8_matmul -------------------------------------------
    std::printf("\n=== int8_matmul smoke test ===\n");
    Tensor a8 = at::empty({4, 8}, at::TensorOptions().dtype(c10::ScalarType::Char));
    Tensor b8 = at::empty({8, 3}, at::TensorOptions().dtype(c10::ScalarType::Char));
    int8_t* ap = a8.mutable_data_ptr<int8_t>();
    int8_t* bp = b8.mutable_data_ptr<int8_t>();
    for (int i = 0; i < 32; ++i) ap[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < 24; ++i) bp[i] = (int8_t)((i % 5) - 2);
    Tensor mm = int8_matmul(a8, 0.01f, b8, 0.02f);
    std::printf("int8_matmul[0,0]=%.6f  shape=[%lld,%lld]\n",
                mm.data_ptr<float>()[0],
                (long long)mm.size(0), (long long)mm.size(1));

    return (ok_acc && ok_close) ? 0 : 1;
}
