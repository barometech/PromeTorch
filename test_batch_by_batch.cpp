// Detailed batch-by-batch test to compare with PyTorch
// Prints every value at every step for debugging

#include "torch/nn/nn.h"
#include "torch/csrc/autograd/autograd.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <vector>
#include <cstdint>

using namespace torch;
using namespace torch::nn;

// Load MNIST data (simplified - first N samples)
std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& path, int max_samples = 1000) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << std::endl;
        return {};
    }

    // Read header
    uint32_t magic, num, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    // Byte swap (big-endian)
    magic = ((magic >> 24) & 0xff) | ((magic >> 8) & 0xff00) |
            ((magic << 8) & 0xff0000) | ((magic << 24) & 0xff000000);
    num = ((num >> 24) & 0xff) | ((num >> 8) & 0xff00) |
          ((num << 8) & 0xff0000) | ((num << 24) & 0xff000000);
    rows = ((rows >> 24) & 0xff) | ((rows >> 8) & 0xff00) |
           ((rows << 8) & 0xff0000) | ((rows << 24) & 0xff000000);
    cols = ((cols >> 24) & 0xff) | ((cols >> 8) & 0xff00) |
           ((cols << 8) & 0xff0000) | ((cols << 24) & 0xff000000);

    int n = std::min((int)num, max_samples);
    std::vector<std::vector<uint8_t>> images(n);

    for (int i = 0; i < n; ++i) {
        images[i].resize(rows * cols);
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }

    return images;
}

std::vector<uint8_t> load_mnist_labels(const std::string& path, int max_samples = 1000) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << std::endl;
        return {};
    }

    uint32_t magic, num;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num), 4);

    num = ((num >> 24) & 0xff) | ((num >> 8) & 0xff00) |
          ((num << 8) & 0xff0000) | ((num << 24) & 0xff000000);

    int n = std::min((int)num, max_samples);
    std::vector<uint8_t> labels(n);
    file.read(reinterpret_cast<char*>(labels.data()), n);

    return labels;
}

// Helper to print tensor stats
void print_tensor_stats(const std::string& name, const at::Tensor& t) {
    const float* data = t.data_ptr<float>();
    int64_t n = t.numel();

    float sum = 0, sum_sq = 0, min_val = data[0], max_val = data[0];
    for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
        sum_sq += data[i] * data[i];
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    float mean = sum / n;
    float norm = std::sqrt(sum_sq);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  " << name << ": sum=" << sum << " mean=" << mean
              << " norm=" << norm << " min=" << min_val << " max=" << max_val << std::endl;
}

// Print first few elements
void print_first_elements(const std::string& name, const at::Tensor& t, int count = 5) {
    const float* data = t.data_ptr<float>();
    std::cout << "  " << name << " first " << count << ": [";
    for (int i = 0; i < count && i < t.numel(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(6) << data[i];
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "=== Batch-by-Batch Comparison Test ===" << std::endl;
    std::cout << "This output should be compared with pytorch_batch_by_batch.py" << std::endl;
    std::cout << std::endl;

    // Load small subset of MNIST
    std::string data_dir = "C:/Users/paper/Desktop/promethorch/data/mnist/";
    auto images = load_mnist_images(data_dir + "train-images-idx3-ubyte", 320);  // 10 batches of 32
    auto labels = load_mnist_labels(data_dir + "train-labels-idx1-ubyte", 320);

    if (images.empty() || labels.empty()) {
        std::cerr << "Failed to load MNIST data" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << images.size() << " samples" << std::endl;

    // Create simple linear model: 784 -> 10
    auto fc = std::make_shared<Linear>(784, 10);
    CrossEntropyLoss criterion;

    // Get parameter pointers
    Parameter* weight_param = fc->get_parameter("weight");
    Parameter* bias_param = fc->get_parameter("bias");

    // Initialize weights to specific values for reproducibility
    // Use same LCG as Python script
    {
        float* w_data = weight_param->data().mutable_data_ptr<float>();
        float* b_data = bias_param->data().mutable_data_ptr<float>();

        float scale = 1.0f / std::sqrt(784.0f);
        uint32_t seed = 42;
        for (int i = 0; i < 784 * 10; ++i) {
            seed = seed * 1103515245 + 12345;
            float r = ((seed >> 16) & 0x7fff) / 32767.0f;
            w_data[i] = (r * 2.0f - 1.0f) * scale;
        }
        for (int i = 0; i < 10; ++i) {
            b_data[i] = 0.0f;
        }
    }

    std::cout << "\n=== Initial Weights ===" << std::endl;
    print_tensor_stats("fc.weight", weight_param->data());
    print_first_elements("fc.weight", weight_param->data());
    print_tensor_stats("fc.bias", bias_param->data());

    // Training parameters
    const int batch_size = 32;
    const float lr = 0.01f;
    const int num_batches = 10;

    std::cout << "\n=== Training Config ===" << std::endl;
    std::cout << "batch_size=" << batch_size << " lr=" << lr << " num_batches=" << num_batches << std::endl;

    // Training loop
    for (int batch = 0; batch < num_batches; ++batch) {
        std::cout << "\n========== BATCH " << batch << " ==========" << std::endl;

        // Create batch
        at::Tensor input = at::zeros({batch_size, 784});
        at::Tensor target = at::zeros({batch_size});

        float* in_ptr = input.mutable_data_ptr<float>();
        float* tgt_ptr = target.mutable_data_ptr<float>();

        for (int i = 0; i < batch_size; ++i) {
            int idx = batch * batch_size + i;
            tgt_ptr[i] = static_cast<float>(labels[idx]);
            for (int j = 0; j < 784; ++j) {
                float pixel = images[idx][j] / 255.0f;
                in_ptr[i * 784 + j] = (pixel - 0.1307f) / 0.3081f;
            }
        }

        std::cout << "\n--- Input ---" << std::endl;
        print_tensor_stats("input", input);
        print_first_elements("target", target);

        // Forward
        at::Tensor logits = fc->forward(input);

        std::cout << "\n--- Logits ---" << std::endl;
        print_tensor_stats("logits", logits);
        print_first_elements("logits[0]", logits.select(0, 0));

        // Loss
        at::Tensor loss = criterion.forward(logits, target);
        float loss_val = loss.data_ptr<float>()[0];

        std::cout << "\n--- Loss ---" << std::endl;
        std::cout << "  loss = " << std::fixed << std::setprecision(6) << loss_val << std::endl;

        // Accuracy
        int correct = 0;
        for (int i = 0; i < batch_size; ++i) {
            const float* row = logits.data_ptr<float>() + i * 10;
            int pred = 0;
            for (int j = 1; j < 10; ++j) {
                if (row[j] > row[pred]) pred = j;
            }
            if (pred == static_cast<int>(tgt_ptr[i])) correct++;
        }
        std::cout << "  accuracy = " << (100.0f * correct / batch_size) << "%" << std::endl;

        // Backward
        std::cout << "\n--- Before Backward ---" << std::endl;
        std::cout << "  fc.weight.grad defined: " << (weight_param->grad().defined() ? "yes" : "no") << std::endl;

        fc->zero_grad();
        torch::autograd::backward({loss});

        std::cout << "\n--- After Backward ---" << std::endl;

        if (weight_param->grad().defined()) {
            print_tensor_stats("fc.weight.grad", weight_param->grad());
            print_first_elements("fc.weight.grad", weight_param->grad());
        } else {
            std::cout << "  ERROR: fc.weight.grad not defined!" << std::endl;
        }

        if (bias_param->grad().defined()) {
            print_tensor_stats("fc.bias.grad", bias_param->grad());
        }

        // Manual SGD step
        std::cout << "\n--- Before SGD Step ---" << std::endl;
        print_tensor_stats("fc.weight", weight_param->data());

        {
            at::Tensor w = weight_param->data();
            at::Tensor g = weight_param->grad();
            float* w_ptr = w.mutable_data_ptr<float>();
            const float* g_ptr = g.data_ptr<float>();

            for (int64_t i = 0; i < w.numel(); ++i) {
                w_ptr[i] -= lr * g_ptr[i];
            }
        }

        {
            at::Tensor b = bias_param->data();
            at::Tensor g = bias_param->grad();
            float* b_ptr = b.mutable_data_ptr<float>();
            const float* g_ptr = g.data_ptr<float>();

            for (int64_t i = 0; i < b.numel(); ++i) {
                b_ptr[i] -= lr * g_ptr[i];
            }
        }

        std::cout << "\n--- After SGD Step ---" << std::endl;
        print_tensor_stats("fc.weight", weight_param->data());
        print_first_elements("fc.weight", weight_param->data());

        // Clear autograd graph
        torch::autograd::clear_grad_fn(loss);
        torch::autograd::clear_grad_fn(logits);
    }

    std::cout << "\n=== Final Weights ===" << std::endl;
    print_tensor_stats("fc.weight", weight_param->data());
    print_tensor_stats("fc.bias", bias_param->data());

    // Final evaluation
    std::cout << "\n=== Final Evaluation ===" << std::endl;
    int total_correct = 0;
    float total_loss = 0;

    for (int batch = 0; batch < num_batches; ++batch) {
        at::Tensor input = at::zeros({batch_size, 784});
        at::Tensor target = at::zeros({batch_size});

        float* in_ptr = input.mutable_data_ptr<float>();
        float* tgt_ptr = target.mutable_data_ptr<float>();

        for (int i = 0; i < batch_size; ++i) {
            int idx = batch * batch_size + i;
            tgt_ptr[i] = static_cast<float>(labels[idx]);
            for (int j = 0; j < 784; ++j) {
                float pixel = images[idx][j] / 255.0f;
                in_ptr[i * 784 + j] = (pixel - 0.1307f) / 0.3081f;
            }
        }

        at::Tensor logits = fc->forward(input);
        at::Tensor loss = criterion.forward(logits, target);
        total_loss += loss.data_ptr<float>()[0];

        for (int i = 0; i < batch_size; ++i) {
            const float* row = logits.data_ptr<float>() + i * 10;
            int pred = 0;
            for (int j = 1; j < 10; ++j) {
                if (row[j] > row[pred]) pred = j;
            }
            if (pred == static_cast<int>(tgt_ptr[i])) total_correct++;
        }
    }

    std::cout << "Final Loss: " << (total_loss / num_batches) << std::endl;
    std::cout << "Final Accuracy: " << (100.0f * total_correct / (num_batches * batch_size)) << "%" << std::endl;

    return 0;
}
