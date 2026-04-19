// Minimal ONNX export smoke test.
//
// Builds a small MLP + Conv2d model, exports it to /tmp/test_{mlp,cnn}.onnx,
// and prints file sizes. Verify with Python:
//   python3 -c "import onnx; m=onnx.load('/tmp/test_mlp.onnx'); onnx.checker.check_model(m); print('OK:', [n.op_type for n in m.graph.node])"

#include "torch/onnx/export.h"
#include "torch/nn/nn.h"
#include "aten/src/ATen/core/TensorFactory.h"

#include <cstdio>
#include <fstream>
#include <memory>

namespace nn = torch::nn;
using at::Tensor;

static long file_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return -1;
    return static_cast<long>(f.tellg());
}

int main() {
    // --- Test 1: self-test --------------------------------------------------
    bool st = torch::onnx::onnx_self_test("/tmp/test_self.onnx");
    std::printf("self_test: %s (file size=%ld)\n",
                st ? "OK" : "FAIL",
                file_size("/tmp/test_self.onnx"));

    // --- Test 2: MLP --------------------------------------------------------
    auto mlp = std::make_shared<nn::Sequential>();
    mlp->push_back("fc1", std::make_shared<nn::Linear>(32, 64));
    mlp->push_back("relu1", std::make_shared<nn::ReLU>());
    mlp->push_back("fc2", std::make_shared<nn::Linear>(64, 32));
    mlp->push_back("sigmoid", std::make_shared<nn::Sigmoid>());
    mlp->push_back("fc3", std::make_shared<nn::Linear>(32, 10));
    mlp->push_back("softmax", std::make_shared<nn::Softmax>(-1));

    Tensor x_mlp = at::zeros({1, 32});
    bool mlp_ok = torch::onnx::export_model(*mlp, x_mlp, "/tmp/test_mlp.onnx");
    std::printf("mlp export: %s (file size=%ld)\n",
                mlp_ok ? "OK" : "FAIL",
                file_size("/tmp/test_mlp.onnx"));

    // --- Test 3: small CNN --------------------------------------------------
    auto cnn = std::make_shared<nn::Sequential>();
    cnn->push_back("conv1", std::make_shared<nn::Conv2d>(3, 8, 3, 1, 1));
    cnn->push_back("bn1",   std::make_shared<nn::BatchNorm2d>(8));
    cnn->push_back("relu1", std::make_shared<nn::ReLU>());
    cnn->push_back("pool1", std::make_shared<nn::MaxPool2d>(2));
    cnn->push_back("conv2", std::make_shared<nn::Conv2d>(8, 16, 3, 1, 1));
    cnn->push_back("tanh",  std::make_shared<nn::Tanh>());
    cnn->push_back("pool2", std::make_shared<nn::AvgPool2d>(2));

    Tensor x_cnn = at::zeros({1, 3, 16, 16});
    bool cnn_ok = torch::onnx::export_model(*cnn, x_cnn, "/tmp/test_cnn.onnx");
    std::printf("cnn export: %s (file size=%ld)\n",
                cnn_ok ? "OK" : "FAIL",
                file_size("/tmp/test_cnn.onnx"));

    return (st && mlp_ok && cnn_ok) ? 0 : 1;
}
