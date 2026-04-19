// Self-test for torch::mlir::export_mlir
//
// Builds a 3-layer MLP, exports to MLIR text, and verifies:
//   - Output begins with "module {"
//   - Contains exactly 3 linalg.matmul ops (one per Linear layer)
//   - Contains a func.func @forward declaration
//   - Contains expected element-wise ops (relu)
//
// Compile (CPU-only, Elbrus-friendly):
//   g++ -std=c++17 -I.. examples/mlir/test_mlir_export.cpp <torch/aten objects> -o test_mlir_export

#include "torch/mlir/export.h"
#include "torch/nn/modules/container.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/activation.h"

#include <iostream>
#include <string>

static int count_substr(const std::string& s, const std::string& needle) {
    if (needle.empty()) return 0;
    int n = 0;
    size_t p = 0;
    while ((p = s.find(needle, p)) != std::string::npos) { ++n; p += needle.size(); }
    return n;
}

int main() {
    using namespace torch;

    // 3-layer MLP: 784 -> 128 -> 64 -> 10
    nn::Sequential model({
        std::shared_ptr<nn::Module>(new nn::Linear(784, 128)),
        std::shared_ptr<nn::Module>(new nn::ReLU()),
        std::shared_ptr<nn::Module>(new nn::Linear(128, 64)),
        std::shared_ptr<nn::Module>(new nn::ReLU()),
        std::shared_ptr<nn::Module>(new nn::Linear(64, 10)),
    });

    std::string mlir_text = mlir::export_mlir(model, {1, 784}, "");

    // --- Assertions ---
    bool ok = true;

    auto require = [&](bool cond, const std::string& msg) {
        if (!cond) { std::cerr << "FAIL: " << msg << "\n"; ok = false; }
        else        { std::cout << "PASS: " << msg << "\n"; }
    };

    require(mlir_text.compare(0, 3, "// ") == 0 ||
            mlir_text.find("module {") != std::string::npos,
            "starts with module/comment header");
    require(mlir_text.find("module {") != std::string::npos,
            "contains 'module {'");
    require(mlir_text.find("func.func @forward") != std::string::npos,
            "contains 'func.func @forward'");

    int matmuls = count_substr(mlir_text, "linalg.matmul");
    require(matmuls == 3,
            std::string("exactly 3 linalg.matmul ops (got ") +
                std::to_string(matmuls) + ")");

    // ReLU produces arith.maximumf inside a linalg.generic.
    int relus = count_substr(mlir_text, "arith.maximumf");
    require(relus >= 2, std::string("at least 2 ReLU maxf ops (got ") +
                            std::to_string(relus) + ")");

    require(mlir_text.find("return ") != std::string::npos,
            "contains return statement");
    require(mlir_text.find("tensor<1x10xf32>") != std::string::npos,
            "output shape is 1x10xf32");

    std::cout << "\n--- MLIR snippet (first 1500 chars) ---\n";
    std::cout << mlir_text.substr(0, 1500) << "\n";
    std::cout << "--- end snippet ---\n";
    std::cout << "Total MLIR length: " << mlir_text.size() << " bytes\n";

    return ok ? 0 : 1;
}
