// ============================================================================
// ResNet-20 on CIFAR-10 — reference PromeTorch training example
// ============================================================================
// Architecture: torch::vision::models::resnet20 (He et al. 2015, CIFAR variant)
// Target: 91%+ test accuracy on CIFAR-10.
//
// Data: CIFAR-10 binary format, layout per-file:
//   data_batch_1.bin ... data_batch_5.bin   (train, 50 000 images total)
//   test_batch.bin                          (test,  10 000 images)
// Each file is 10000 records of 3073 bytes:
//     byte 0:        label (0..9)
//     bytes 1..1024: red plane   (32x32)
//     bytes 1025..2048: green plane
//     bytes 2049..3072: blue plane
// Total size per file = 10000 * 3073 = 30_730_000 bytes.
//
// If the dataset is missing, this program prints instructions and exits 0.
// Download URL (195 MB tarball):
//   https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
// Unpack into:  data/cifar-10-batches-bin/
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/optim/lr_scheduler.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/vision/resnet.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;
using at::Tensor;

// ----- global device -------------------------------------------------------
static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

inline Tensor to_device(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) return at::to_cuda(t);
#endif
    return t;
}

inline Tensor move_to_cpu(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) return at::to_cpu(t);
#endif
    return t;
}

// ============================================================================
// CIFAR-10 Loader
// ============================================================================

struct CifarDataset {
    // Raw uint8 pixels stored CHW (3, 32, 32) per image, flattened row-major.
    std::vector<uint8_t> images;   // size = N * 3 * 32 * 32
    std::vector<uint8_t> labels;   // size = N
    int64_t n = 0;

    static constexpr int64_t CHW = 3 * 32 * 32;   // 3072
};

// Read one .bin file (10000 records of 3073 bytes) and append to ds.
// Returns true on success.
static bool append_cifar_bin(const std::string& path, CifarDataset& ds) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "  [cifar] cannot open: " << path << "\n";
        return false;
    }
    constexpr int64_t kPerRecord = 1 + CifarDataset::CHW;   // 3073
    constexpr int64_t kRecords = 10000;

    std::vector<uint8_t> buf(kPerRecord);
    for (int64_t i = 0; i < kRecords; ++i) {
        f.read(reinterpret_cast<char*>(buf.data()), kPerRecord);
        if (f.gcount() != kPerRecord) {
            std::cerr << "  [cifar] short read in " << path << " at record " << i << "\n";
            return false;
        }
        ds.labels.push_back(buf[0]);
        ds.images.insert(ds.images.end(), buf.begin() + 1, buf.end());
    }
    ds.n += kRecords;
    return true;
}

static bool load_cifar10(const std::string& dir,
                         CifarDataset& train, CifarDataset& test) {
    bool ok = true;
    for (int i = 1; i <= 5; ++i) {
        std::string p = dir + "/data_batch_" + std::to_string(i) + ".bin";
        ok = append_cifar_bin(p, train) && ok;
    }
    ok = append_cifar_bin(dir + "/test_batch.bin", test) && ok;
    return ok;
}

static void print_missing_data_instructions(const std::string& dir) {
    std::cerr << "\n"
              << "================================================================\n"
              << " CIFAR-10 data not found at: " << dir << "\n"
              << "================================================================\n"
              << "\n"
              << " Please download the binary version of CIFAR-10:\n"
              << "   https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz\n"
              << "\n"
              << " Unpack so you have:\n"
              << "   " << dir << "/data_batch_1.bin\n"
              << "   " << dir << "/data_batch_2.bin\n"
              << "   ...\n"
              << "   " << dir << "/data_batch_5.bin\n"
              << "   " << dir << "/test_batch.bin\n"
              << "\n"
              << " Unix one-liner:\n"
              << "   mkdir -p data && cd data && \\\n"
              << "     curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz && \\\n"
              << "     tar xf cifar-10-binary.tar.gz\n"
              << "\n"
              << " Then re-run:  train_resnet --data " << dir << "\n"
              << "================================================================\n";
}

// ============================================================================
// Batch construction + augmentation (CPU)
// ============================================================================
// We keep the augmentation inline instead of going through torchvision-style
// transform objects, so that (a) the hot path is a single tight loop and
// (b) we don't pay the cost of creating Tensor objects per sample.

static const float kCifarMean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float kCifarStd [3] = {0.2470f, 0.2435f, 0.2616f};

struct BatchBuilder {
    // Padded image buffer (3, 40, 40) for random crop, reused across calls.
    std::vector<float> pad_plane;  // size = 3 * 40 * 40
    std::mt19937 rng;

    BatchBuilder() : pad_plane(3 * 40 * 40), rng(std::random_device{}()) {}

    // Fill `out` (shape [B, 3, 32, 32]) and `labels` (shape [B]) for a batch.
    // If `train` is true, apply random crop(32, pad=4) + horizontal flip.
    void build(const CifarDataset& ds,
               const std::vector<int64_t>& idxs, int64_t start, int64_t end,
               float* out, float* labels_out, bool train) {
        const int64_t B = end - start;
        constexpr int64_t IMG = 32;
        constexpr int64_t PAD = 4;
        constexpr int64_t PADDED = IMG + 2 * PAD;   // 40

        std::uniform_int_distribution<int64_t> crop_h(0, 2 * PAD);
        std::uniform_int_distribution<int64_t> crop_w(0, 2 * PAD);
        std::uniform_real_distribution<float> flip(0.0f, 1.0f);

        for (int64_t b = 0; b < B; ++b) {
            const int64_t idx = idxs[start + b];
            const uint8_t* src = ds.images.data() + idx * CifarDataset::CHW;

            labels_out[b] = static_cast<float>(ds.labels[idx]);

            int64_t top = PAD, left = PAD;
            bool do_flip = false;
            if (train) {
                top = crop_h(rng);
                left = crop_w(rng);
                do_flip = (flip(rng) < 0.5f);
            }

            // Normalize + (optional) pad into pad_plane, then crop out a
            // 32x32 window. In inference path we just do the normalize step
            // directly — skip the pad buffer entirely.
            float* dst_base = out + b * 3 * IMG * IMG;

            if (!train) {
                for (int64_t c = 0; c < 3; ++c) {
                    const float m = kCifarMean[c];
                    const float inv_s = 1.0f / kCifarStd[c];
                    const uint8_t* sp = src + c * IMG * IMG;
                    float* dp = dst_base + c * IMG * IMG;
                    for (int64_t i = 0; i < IMG * IMG; ++i) {
                        dp[i] = (sp[i] * (1.0f / 255.0f) - m) * inv_s;
                    }
                }
                continue;
            }

            // Train path: zero-fill pad buffer, copy+normalize image into
            // its centre, then random-crop (and flip on the fly).
            std::fill(pad_plane.begin(), pad_plane.end(), 0.0f);
            for (int64_t c = 0; c < 3; ++c) {
                const float m = kCifarMean[c];
                const float inv_s = 1.0f / kCifarStd[c];
                const uint8_t* sp = src + c * IMG * IMG;
                float* dp_channel = pad_plane.data() + c * PADDED * PADDED;
                for (int64_t y = 0; y < IMG; ++y) {
                    for (int64_t x = 0; x < IMG; ++x) {
                        float v = (sp[y * IMG + x] * (1.0f / 255.0f) - m) * inv_s;
                        dp_channel[(y + PAD) * PADDED + (x + PAD)] = v;
                    }
                }
            }

            for (int64_t c = 0; c < 3; ++c) {
                const float* pc = pad_plane.data() + c * PADDED * PADDED;
                float* dp = dst_base + c * IMG * IMG;
                for (int64_t y = 0; y < IMG; ++y) {
                    for (int64_t x = 0; x < IMG; ++x) {
                        int64_t sx = do_flip ? (IMG - 1 - x) : x;
                        dp[y * IMG + x] = pc[(top + y) * PADDED + (left + sx)];
                    }
                }
            }
        }
    }
};

// ============================================================================
// Evaluation
// ============================================================================

static float evaluate(Module& model, const CifarDataset& ds, int64_t batch_size = 128) {
    torch::autograd::NoGradGuard no_grad;
    model.eval();

    const int64_t N = ds.n;
    int64_t correct = 0;
    BatchBuilder bb;

    std::vector<int64_t> idxs(N);
    std::iota(idxs.begin(), idxs.end(), 0);

    for (int64_t start = 0; start < N; start += batch_size) {
        int64_t end = std::min(start + batch_size, N);
        int64_t B = end - start;
        Tensor inputs = at::empty({B, 3, 32, 32});
        Tensor labels = at::empty({B});
        bb.build(ds, idxs, start, end,
                 inputs.mutable_data_ptr<float>(),
                 labels.mutable_data_ptr<float>(), /*train=*/false);
        inputs = to_device(inputs);
        Tensor logits = model.forward(inputs);
        Tensor preds = move_to_cpu(logits).argmax(1);
        const int64_t* p = preds.data_ptr<int64_t>();
        const float* lbl = labels.data_ptr<float>();
        for (int64_t b = 0; b < B; ++b) {
            if (p[b] == static_cast<int64_t>(lbl[b])) ++correct;
        }
    }
    return 100.0f * correct / static_cast<float>(N);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_dir = "data/cifar-10-batches-bin";
    std::string device_str = "cuda";
    int64_t batch_size = 128;
    int64_t epochs = 150;
    float lr = 0.1f;
    float momentum = 0.9f;
    float weight_decay = 5e-4f;
    int64_t log_interval = 100;
    // Smoke-test mode: if --smoke is passed, we run only a handful of epochs
    // and report early progress instead of full training.
    int64_t smoke_epochs = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int more) { return i + more < argc; };
        if      (a == "--data"         && need(1)) data_dir     = argv[++i];
        else if (a == "--device"       && need(1)) device_str   = argv[++i];
        else if (a == "--batch_size"   && need(1)) batch_size   = std::stoll(argv[++i]);
        else if (a == "--epochs"       && need(1)) epochs       = std::stoll(argv[++i]);
        else if (a == "--lr"           && need(1)) lr           = std::stof (argv[++i]);
        else if (a == "--momentum"     && need(1)) momentum     = std::stof (argv[++i]);
        else if (a == "--weight_decay" && need(1)) weight_decay = std::stof (argv[++i]);
        else if (a == "--smoke"        && need(1)) smoke_epochs = std::stoll(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout <<
                "Usage: train_resnet [--data DIR] [--device cpu|cuda]\n"
                "                    [--batch_size N] [--epochs N] [--lr X]\n"
                "                    [--momentum X] [--weight_decay X] [--smoke K]\n";
            return 0;
        }
    }

    // Device selection
    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Device: CUDA\n";
#else
        std::cerr << "CUDA not compiled in — falling back to CPU\n";
#endif
    } else {
        std::cout << "Device: CPU\n";
    }

    // --- Load data --------------------------------------------------------
    CifarDataset train, test;
    std::cout << "Loading CIFAR-10 from: " << data_dir << std::endl;
    if (!load_cifar10(data_dir, train, test) || train.n != 50000 || test.n != 10000) {
        print_missing_data_instructions(data_dir);
        return 0;   // clean exit, no error
    }
    std::cout << "  train: " << train.n << " images, test: " << test.n << " images\n";

    // --- Build model ------------------------------------------------------
    auto model = torch::vision::models::resnet20(/*num_classes=*/10);
    int64_t n_params = 0;
    for (auto* p : model->parameters()) n_params += p->data().numel();
    std::cout << "Model: ResNet-20  (" << n_params << " params)\n";

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        model->to(g_device);
        std::cout << "Model moved to CUDA\n";
    }
#endif

    // --- Optimizer + schedule --------------------------------------------
    SGDOptions opts(lr);
    opts.momentum_(momentum).weight_decay_(weight_decay).nesterov_(false);
    SGD optimizer(model->parameters(), opts);

    // Multi-step decay: ×0.1 at epochs 80 and 120.
    MultiStepLR scheduler(optimizer, std::vector<int64_t>{80, 120}, /*gamma=*/0.1);

    CrossEntropyLoss criterion;

    // --- Training loop ----------------------------------------------------
    std::random_device rd;
    std::mt19937 shuffler(rd());
    BatchBuilder bb;

    int64_t total_epochs = (smoke_epochs > 0) ? smoke_epochs : epochs;
    std::cout << "\n=== Training (" << total_epochs << " epochs, bs=" << batch_size
              << ", lr=" << lr << ", momentum=" << momentum
              << ", wd=" << weight_decay << ") ===\n";

    float best_test_acc = 0.0f;
    auto train_start = std::chrono::steady_clock::now();

    for (int64_t epoch = 1; epoch <= total_epochs; ++epoch) {
        model->train();

        std::vector<int64_t> idxs(train.n);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::shuffle(idxs.begin(), idxs.end(), shuffler);

        auto epoch_start = std::chrono::steady_clock::now();
        float run_loss = 0.0f;
        int64_t run_correct = 0;
        int64_t run_seen = 0;
        int64_t step = 0;

        for (int64_t s = 0; s < train.n; s += batch_size) {
            int64_t e = std::min(s + batch_size, train.n);
            int64_t B = e - s;
            Tensor inputs = at::empty({B, 3, 32, 32});
            Tensor targets = at::empty({B});
            bb.build(train, idxs, s, e,
                     inputs.mutable_data_ptr<float>(),
                     targets.mutable_data_ptr<float>(), /*train=*/true);
            inputs = to_device(inputs);
            targets = to_device(targets);

            optimizer.zero_grad();
            Tensor logits = model->forward(inputs);
            Tensor loss = criterion.forward(logits, targets);

            torch::autograd::backward({loss});
            optimizer.step();

            // -------- running stats (cheap: argmax on device, copy once) ----
            Tensor loss_cpu = move_to_cpu(loss);
            run_loss += loss_cpu.data_ptr<float>()[0] * B;

            Tensor preds = move_to_cpu(logits).argmax(1);
            Tensor tgt_cpu = move_to_cpu(targets);
            const int64_t* p = preds.data_ptr<int64_t>();
            const float* t = tgt_cpu.data_ptr<float>();
            for (int64_t b = 0; b < B; ++b) {
                if (p[b] == static_cast<int64_t>(t[b])) ++run_correct;
            }
            run_seen += B;
            ++step;

            if (step % log_interval == 0) {
                auto now = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now - epoch_start).count();
                std::cout << "  epoch " << epoch << " step " << step
                          << " loss " << (run_loss / run_seen)
                          << " acc " << (100.0f * run_correct / run_seen) << "%"
                          << " img/s " << static_cast<int>(run_seen / std::max(sec, 1e-6))
                          << "\n";
            }
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();

        float train_loss = run_loss / static_cast<float>(train.n);
        float train_acc  = 100.0f * run_correct / static_cast<float>(train.n);
        float test_acc   = evaluate(*model, test, /*batch_size=*/256);
        if (test_acc > best_test_acc) best_test_acc = test_acc;

        std::cout << "[epoch " << epoch << "/" << total_epochs << "]"
                  << "  lr=" << optimizer.get_lr()
                  << "  train_loss=" << train_loss
                  << "  train_acc=" << train_acc << "%"
                  << "  test_acc=" << test_acc << "%"
                  << "  best=" << best_test_acc << "%"
                  << "  (" << static_cast<int>(epoch_sec) << "s)\n";
        std::cout.flush();

        scheduler.step();
    }

    auto train_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(train_end - train_start).count();
    std::cout << "\n=== Training complete in " << static_cast<int>(total_sec)
              << "s.  best test_acc = " << best_test_acc << "% ===\n";
    return 0;
}
