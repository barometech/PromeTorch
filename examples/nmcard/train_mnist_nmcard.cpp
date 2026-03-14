// ============================================================================
// MNIST Training on NM Card Mini (Emulator or Hardware)
// ============================================================================
// MLP model trained on the NMCard backend. Same architecture as CPU MNIST,
// but all tensors live on device "nmcard" and ops go through emulator or hardware.
// Usage: train_mnist_nmcard.exe --data . --epochs 3 --device nmcard
//        train_mnist_nmcard.exe --device nmcard --hardware --dispatcher path/to/dispatcher.abs
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "aten/src/ATen/nmcard/NMCardDispatch.h"
#include "c10/nmcard/NMCardAllocator.h"
#include "aten/src/ATen/nmcard/NMCardEmulator.h"
#include "aten/src/ATen/nmcard/NMCardHardware.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#ifdef _MSC_VER
#include <stdlib.h>
#define bswap32(x) _byteswap_ulong(x)
#else
#define bswap32(x) __builtin_bswap32(x)
#endif

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

// MNIST normalization constants
constexpr float MNIST_MEAN = 0.1307f;
constexpr float MNIST_STD = 0.3081f;

// Device mode
static bool g_use_nmcard = true;

inline at::Tensor to_device(const at::Tensor& t) {
    if (g_use_nmcard) {
        return at::to_nmcard(t);
    }
    return t;
}

inline at::Tensor to_cpu(const at::Tensor& t) {
    if (t.is_nmcard()) {
        return at::nmcard_to_cpu(t);
    }
    return t;
}

// ============================================================================
// MNIST Data Loading (IDX format)
// ============================================================================

std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << std::endl;
        return {};
    }

    int32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = bswap32(magic);
    num_images = bswap32(num_images);
    rows = bswap32(rows);
    cols = bswap32(cols);

    std::cout << "MNIST images: " << num_images << " x " << rows << "x" << cols << std::endl;

    std::vector<std::vector<uint8_t>> images(num_images);
    for (int i = 0; i < num_images; ++i) {
        images[i].resize(rows * cols);
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }

    return images;
}

std::vector<uint8_t> load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << std::endl;
        return {};
    }

    int32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = bswap32(magic);
    num_labels = bswap32(num_labels);

    std::cout << "MNIST labels: " << num_labels << std::endl;

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}

// ============================================================================
// MLP Model
// ============================================================================

class MNISTMLP : public Module {
public:
    MNISTMLP() : Module("MNISTMLP") {
        // 784 -> 256 -> 128 -> 10 (smaller than CPU version for speed)
        fc1 = std::make_shared<Linear>(784, 256);
        fc2 = std::make_shared<Linear>(256, 128);
        fc3 = std::make_shared<Linear>(128, 10);

        relu1 = std::make_shared<ReLU>();
        relu2 = std::make_shared<ReLU>();

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    Tensor forward(const Tensor& x) override {
        Tensor h = fc1->forward(x);
        h = relu1->forward(h);
        h = fc2->forward(h);
        h = relu2->forward(h);
        h = fc3->forward(h);
        return h;
    }

private:
    std::shared_ptr<Linear> fc1, fc2, fc3;
    std::shared_ptr<ReLU> relu1, relu2;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_dir = ".";
    std::string device_str = "nmcard";
    std::string dispatcher_path = "";
    int64_t batch_size = 64;
    int64_t epochs = 3;
    float lr = 0.001f;
    bool fixed_point = false;
    bool use_hardware = false;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoll(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoll(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        } else if (arg == "--fixed_point") {
            fixed_point = true;
        } else if (arg == "--hardware") {
            use_hardware = true;
        } else if (arg == "--dispatcher" && i + 1 < argc) {
            dispatcher_path = argv[++i];
        }
    }

    // Set device
    if (device_str == "cpu") {
        g_use_nmcard = false;
        std::cout << "Using CPU" << std::endl;
    } else {
        g_use_nmcard = true;
        // Register NMCard allocator (both DLL-internal and local registry)
        c10::nmcard::register_nmcard_allocator();
        c10::nmcard::register_nmcard_allocator_local();

        if (use_hardware) {
            // Try to initialize real NM Card Mini hardware
            auto& hw = at::nmcard::NMCardHardware::get();
            if (hw.init(dispatcher_path)) {
                std::cout << "Using NM Card Mini HARDWARE" << std::endl;
            } else {
                std::cout << "Hardware init failed, falling back to emulator" << std::endl;
            }
        }

        if (!at::nmcard::NMCardHardware::get().is_available()) {
            auto& emu = at::nmcard::NMCardEmulator::get();
            if (fixed_point) {
                emu.set_fixed_point(true);
                std::cout << "Using NMCard Emulator (Q16.16 fixed-point)" << std::endl;
            } else {
                std::cout << "Using NMCard Emulator (float32)" << std::endl;
            }
            std::cout << "Virtual NMC4 cores: " << emu.num_cores() << std::endl;
        }
    }

    // Load data
    std::cout << "Loading MNIST from: " << data_dir << std::endl;
    auto train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    auto train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    auto test_images = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    auto test_labels = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    if (train_images.empty()) {
        std::cerr << "Failed to load MNIST data. Download from http://yann.lecun.com/exdb/mnist/" << std::endl;
        return 1;
    }

    int64_t n_train = train_images.size();
    int64_t n_test = test_images.size();

    // Create model
    auto model = std::make_shared<MNISTMLP>();
    std::cout << "MLP Model: 784 -> 256 -> 128 -> 10" << std::endl;

    // Move model parameters to NMCard if needed
    if (g_use_nmcard) {
        c10::Device nmc_dev("nmcard:0");
        model->to(nmc_dev);
        std::cout << "Model moved to NMCard" << std::endl;
    }

    // Optimizer
    SGDOptions sgd_opts(lr);
    SGD optimizer(model->parameters(), sgd_opts);
    std::cout << "SGD optimizer (lr=" << lr << ")" << std::endl;

    // Loss
    CrossEntropyLoss criterion;

    // Training loop
    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "\n=== Training on NMCard Emulator ===" << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << std::endl;
    std::cout << "Training samples: " << n_train << ", Test samples: " << n_test << std::endl;
    std::cout << std::endl;

    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        model->train();

        // Shuffle indices
        std::vector<int64_t> indices(n_train);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        float epoch_loss = 0.0f;
        int64_t correct = 0;
        int64_t total = 0;
        int64_t batches_processed = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int64_t batch_start = 0; batch_start < n_train; batch_start += batch_size) {
            int64_t batch_end = std::min(batch_start + batch_size, n_train);
            int64_t B = batch_end - batch_start;

            // Create batch tensors on CPU first
            Tensor inputs = at::empty({B, 784});
            Tensor targets = at::empty({B});
            float* in_ptr = inputs.mutable_data_ptr<float>();
            float* tgt_ptr = targets.mutable_data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int64_t idx = indices[batch_start + i];
                tgt_ptr[i] = static_cast<float>(train_labels[idx]);

                for (int j = 0; j < 784; ++j) {
                    float pixel = train_images[idx][j] / 255.0f;
                    in_ptr[i * 784 + j] = (pixel - MNIST_MEAN) / MNIST_STD;
                }
            }

            // Move to device
            inputs = to_device(inputs);
            targets = to_device(targets);

            // Forward
            optimizer.zero_grad();
            Tensor logits = model->forward(inputs);
            Tensor loss = criterion.forward(logits, targets);

            // Backward
            torch::autograd::backward({loss});

            // Optimizer step
            optimizer.step();

            // Stats
            Tensor loss_cpu = to_cpu(loss);
            float batch_loss = loss_cpu.data_ptr<float>()[0];
            epoch_loss += batch_loss * B;

            // Accuracy
            Tensor logits_cpu = to_cpu(logits);
            Tensor targets_cpu = to_cpu(targets);
            const float* log_ptr = logits_cpu.data_ptr<float>();
            const float* tgt_cpu_ptr = targets_cpu.data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int pred = 0;
                float max_val = log_ptr[i * 10];
                for (int c = 1; c < 10; ++c) {
                    if (log_ptr[i * 10 + c] > max_val) {
                        max_val = log_ptr[i * 10 + c];
                        pred = c;
                    }
                }
                if (pred == static_cast<int>(tgt_cpu_ptr[i])) {
                    correct++;
                }
                total++;
            }

            batches_processed++;
            if (batches_processed % 100 == 0) {
                std::cout << "  Batch " << batches_processed << "/" << (n_train / batch_size)
                          << " loss=" << (epoch_loss / total)
                          << " acc=" << (100.0f * correct / total) << "%" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cout << "Epoch " << epoch << "/" << epochs
                  << " - Loss: " << (epoch_loss / n_train)
                  << " - Train Acc: " << (100.0f * correct / total) << "%"
                  << " - Time: " << elapsed << "ms" << std::endl;

        // Test evaluation
        model->eval();
        int64_t test_correct = 0;

        for (int64_t batch_start = 0; batch_start < n_test; batch_start += batch_size) {
            int64_t batch_end = std::min(batch_start + batch_size, n_test);
            int64_t B = batch_end - batch_start;

            Tensor inputs = at::empty({B, 784});
            float* in_ptr = inputs.mutable_data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int64_t idx = batch_start + i;
                for (int j = 0; j < 784; ++j) {
                    float pixel = test_images[idx][j] / 255.0f;
                    in_ptr[i * 784 + j] = (pixel - MNIST_MEAN) / MNIST_STD;
                }
            }

            inputs = to_device(inputs);
            Tensor logits = model->forward(inputs);
            Tensor logits_cpu = to_cpu(logits);
            const float* log_ptr = logits_cpu.data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int pred = 0;
                float max_val = log_ptr[i * 10];
                for (int c = 1; c < 10; ++c) {
                    if (log_ptr[i * 10 + c] > max_val) {
                        max_val = log_ptr[i * 10 + c];
                        pred = c;
                    }
                }
                if (pred == test_labels[batch_start + i]) {
                    test_correct++;
                }
            }
        }

        std::cout << "  Test Accuracy: " << (100.0f * test_correct / n_test) << "%" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== Training Complete ===" << std::endl;
    return 0;
}
