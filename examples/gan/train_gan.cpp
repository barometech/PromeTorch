// ============================================================================
// DCGAN on MNIST — canonical training example for PromeTorch.
// ============================================================================
// Generator: z[100] → Linear → [128,7,7] → ConvT(→64,14,14) BN ReLU
//                               → ConvT(→1,28,28) Tanh
// Discriminator: [1,28,28] → Conv(→64,14,14) LReLU
//                         → Conv(→128,7,7) BN LReLU
//                         → Flatten → Linear(1)   (logits; BCEWithLogits)
//
// Trained with non-saturating G loss (G: max log D(G(z))).
// Images are normalized to [-1, 1] to match Tanh output range.
//
// Run:
//   train_gan --device cuda --data data/mnist --epochs 25 --batch_size 64
// Samples are written as 4x4 PPM grids every 5 epochs.
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <stdlib.h>
#define bswap32(x) _byteswap_ulong(x)
#else
#define bswap32(x) __builtin_bswap32(x)
#endif

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

inline at::Tensor to_device(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) {
        return at::to_cuda(t);
    }
#endif
    return t;
}

inline at::Tensor move_to_cpu(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) return at::to_cpu(t);
#endif
    return t;
}

// ---------------------------------------------------------------------------
// MNIST Loading
// ---------------------------------------------------------------------------

std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open: " << path << std::endl; return {}; }
    int32_t magic, num, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num),   4);
    file.read(reinterpret_cast<char*>(&rows),  4);
    file.read(reinterpret_cast<char*>(&cols),  4);
    num  = bswap32(num);
    rows = bswap32(rows);
    cols = bswap32(cols);
    std::vector<std::vector<uint8_t>> images(num);
    for (int i = 0; i < num; ++i) {
        images[i].resize(rows * cols);
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }
    std::cout << "MNIST images: " << num << " x " << rows << "x" << cols << std::endl;
    return images;
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------
// z[B,100] -> FC -> [B,128,7,7]
//   -> ConvT 128->64 k4 s2 p1 -> [B,64,14,14]  BN  ReLU
//   -> ConvT 64->1   k4 s2 p1 -> [B,1,28,28]   Tanh

class GeneratorNet : public Module {
public:
    GeneratorNet() : Module("GeneratorNet") {
        fc     = std::make_shared<Linear>(100, 128 * 7 * 7);
        deconv1 = std::make_shared<ConvTranspose2d>(128, 64, 4, 2, 1);
        bn1     = std::make_shared<BatchNorm2d>(64);
        deconv2 = std::make_shared<ConvTranspose2d>(64, 1, 4, 2, 1);

        register_module("fc",      fc);
        register_module("deconv1", deconv1);
        register_module("bn1",     bn1);
        register_module("deconv2", deconv2);
    }

    Tensor forward(const Tensor& z) override {
        int64_t B = z.size(0);
        Tensor h = fc->forward(z);
        h = torch::autograd::relu_autograd(h);
        h = torch::autograd::reshape_autograd(h, {B, 128, 7, 7});

        h = deconv1->forward(h);
        h = bn1->forward(h);
        h = torch::autograd::relu_autograd(h);

        h = deconv2->forward(h);
        h = torch::autograd::tanh_autograd(h);
        return h;
    }

private:
    std::shared_ptr<Linear>          fc;
    std::shared_ptr<ConvTranspose2d> deconv1, deconv2;
    std::shared_ptr<BatchNorm2d>     bn1;
};

// ---------------------------------------------------------------------------
// Discriminator
// ---------------------------------------------------------------------------
// [B,1,28,28] -> Conv 1->64   k4 s2 p1 -> [B,64,14,14]  LReLU 0.2
//             -> Conv 64->128 k4 s2 p1 -> [B,128,7,7]   BN LReLU 0.2
//             -> Flatten -> Linear(128*7*7, 1)  (logits)

class Discriminator : public Module {
public:
    Discriminator() : Module("Discriminator") {
        conv1 = std::make_shared<Conv2d>(1, 64, 4, 2, 1);
        conv2 = std::make_shared<Conv2d>(64, 128, 4, 2, 1);
        bn2   = std::make_shared<BatchNorm2d>(128);
        fc    = std::make_shared<Linear>(128 * 7 * 7, 1);

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("bn2",   bn2);
        register_module("fc",    fc);
    }

    Tensor forward(const Tensor& x) override {
        int64_t B = x.size(0);
        Tensor h = conv1->forward(x);
        h = torch::autograd::leaky_relu_autograd(h, 0.2);

        h = conv2->forward(h);
        h = bn2->forward(h);
        h = torch::autograd::leaky_relu_autograd(h, 0.2);

        h = torch::autograd::reshape_autograd(h, {B, 128 * 7 * 7});
        h = fc->forward(h);   // [B, 1]  logits
        return h;
    }

private:
    std::shared_ptr<Conv2d>      conv1, conv2;
    std::shared_ptr<BatchNorm2d> bn2;
    std::shared_ptr<Linear>      fc;
};

// ---------------------------------------------------------------------------
// BCE-with-logits loss built from autograd-aware primitives.
//
//   L = mean(softplus(x) - x * y)
//     = mean(log(1 + exp(x)) - x*y)
//
//   For y=1: softplus(x) - x = -log(sigmoid(x))
//   For y=0: softplus(x)     = -log(1 - sigmoid(x))
// ---------------------------------------------------------------------------

Tensor bce_with_logits(const Tensor& logits, const Tensor& targets) {
    Tensor sp = torch::autograd::softplus_autograd(logits, 1.0, 20.0);
    Tensor xy = torch::autograd::mul_autograd(logits, targets);
    Tensor per = torch::autograd::sub_autograd(sp, xy);
    return torch::autograd::mean_autograd(per);
}

// ---------------------------------------------------------------------------
// Save 4x4 grid of generated images as a PPM file.
// Input samples are in [-1, 1] (tanh range); rescale to [0, 255].
// ---------------------------------------------------------------------------

void save_grid_ppm(const Tensor& samples_cpu, const std::string& path) {
    // samples_cpu: [16, 1, 28, 28]
    const int n = 16, cols = 4, rows = 4, H = 28, W = 28;
    const int GW = cols * W, GH = rows * H;
    std::vector<uint8_t> img(GW * GH, 0);
    const float* data = samples_cpu.data_ptr<float>();
    for (int k = 0; k < n; ++k) {
        int gr = (k / cols) * H;
        int gc = (k % cols) * W;
        const float* s = data + k * H * W;
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                float v = s[i * W + j];
                v = 0.5f * (v + 1.0f);         // [-1,1] -> [0,1]
                v = std::max(0.0f, std::min(1.0f, v));
                img[(gr + i) * GW + (gc + j)] = static_cast<uint8_t>(v * 255.0f);
            }
        }
    }
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << GW << " " << GH << "\n255\n";
    for (int p : img) {
        uint8_t c = static_cast<uint8_t>(p);
        f.put(c); f.put(c); f.put(c);
    }
    std::cout << "  Saved " << path << std::endl;
}

// ---------------------------------------------------------------------------
// Build a MNIST batch normalized to [-1, 1] (for Tanh generator output range).
// ---------------------------------------------------------------------------

static Tensor make_real_batch(const std::vector<std::vector<uint8_t>>& images,
                              const std::vector<int64_t>& idx,
                              int64_t start, int64_t B) {
    Tensor out = at::empty({B, 1, 28, 28});
    float* op = out.mutable_data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
        const auto& src = images[idx[start + b]];
        for (int i = 0; i < 28 * 28; ++i) {
            op[b * 28 * 28 + i] = (src[i] / 255.0f) * 2.0f - 1.0f;
        }
    }
    return out;
}

static Tensor make_noise(int64_t B, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Tensor z = at::empty({B, 100});
    float* zp = z.mutable_data_ptr<float>();
    for (int64_t i = 0; i < B * 100; ++i) zp[i] = nd(rng);
    return z;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string data_dir   = "data/mnist";
    std::string device_str = "cpu";
    std::string out_dir    = ".";
    int64_t batch_size     = 64;
    int64_t epochs         = 25;
    double  lr             = 2e-4;
    double  beta1          = 0.5;
    int64_t sample_every   = 5;
    int64_t max_batches    = 0;  // 0 = full epoch

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--data"          && i + 1 < argc) data_dir   = argv[++i];
        else if (a == "--device"        && i + 1 < argc) device_str = argv[++i];
        else if (a == "--out"           && i + 1 < argc) out_dir    = argv[++i];
        else if (a == "--batch_size"    && i + 1 < argc) batch_size = std::stoll(argv[++i]);
        else if (a == "--epochs"        && i + 1 < argc) epochs     = std::stoll(argv[++i]);
        else if (a == "--lr"            && i + 1 < argc) lr         = std::stod(argv[++i]);
        else if (a == "--beta1"         && i + 1 < argc) beta1      = std::stod(argv[++i]);
        else if (a == "--sample_every"  && i + 1 < argc) sample_every = std::stoll(argv[++i]);
        else if (a == "--max_batches"   && i + 1 < argc) max_batches  = std::stoll(argv[++i]);
    }

    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Using CUDA" << std::endl;
#else
        std::cerr << "CUDA not compiled; falling back to CPU" << std::endl;
#endif
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    // Load data
    std::cout << "Loading MNIST from: " << data_dir << std::endl;
    auto images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    if (images.empty()) {
        std::cerr << "Failed to load MNIST training images" << std::endl;
        return 1;
    }
    int64_t N = static_cast<int64_t>(images.size());

    // Build models
    auto G = std::make_shared<GeneratorNet>();
    auto D = std::make_shared<Discriminator>();
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) { G->to(g_device); D->to(g_device); }
#endif

    std::cout << "G params: " << count_parameters(*G) << std::endl;
    std::cout << "D params: " << count_parameters(*D) << std::endl;

    // Optimizers (Adam, beta1=0.5 per DCGAN paper)
    AdamOptions gopt(lr); gopt.betas(beta1, 0.999);
    AdamOptions dopt(lr); dopt.betas(beta1, 0.999);
    Adam g_optim(G->parameters(), gopt);
    Adam d_optim(D->parameters(), dopt);

    std::mt19937 rng(1234);

    // Fixed noise for consistent sample evolution
    Tensor fixed_z_cpu = make_noise(16, rng);

    std::cout << "\n=== DCGAN Training ===\n"
              << "Epochs: " << epochs << ", BS: " << batch_size
              << ", lr: " << lr << ", beta1: " << beta1 << "\n";

    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        G->train(); D->train();

        std::vector<int64_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);

        int64_t nb = N / batch_size;
        if (max_batches > 0 && max_batches < nb) nb = max_batches;

        double d_loss_sum = 0.0, g_loss_sum = 0.0;
        int64_t count = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int64_t b = 0; b < nb; ++b) {
            int64_t start = b * batch_size;

            // ----- Discriminator step -----
            D->zero_grad();

            Tensor real = make_real_batch(images, idx, start, batch_size);
            real = to_device(real);

            // Labels on CPU first, then device
            Tensor ones  = at::empty({batch_size, 1});
            Tensor zeros = at::empty({batch_size, 1});
            {
                float* o = ones.mutable_data_ptr<float>();
                float* z = zeros.mutable_data_ptr<float>();
                for (int64_t i = 0; i < batch_size; ++i) { o[i] = 1.0f; z[i] = 0.0f; }
            }
            ones = to_device(ones); zeros = to_device(zeros);

            // Real loss
            Tensor d_real_logits = D->forward(real);
            Tensor d_real_loss   = bce_with_logits(d_real_logits, ones);

            // Fake loss (detach G: we don't want grads to flow into G here)
            Tensor z_noise;
            {
                Tensor z_cpu = make_noise(batch_size, rng);
                z_noise = to_device(z_cpu);
            }
            Tensor fake;
            {
                // Generate fakes WITHOUT recording autograd graph — saves memory
                // and avoids updating G during D step.
                torch::autograd::NoGradGuard ng;
                fake = G->forward(z_noise);
            }
            Tensor d_fake_logits = D->forward(fake);
            Tensor d_fake_loss   = bce_with_logits(d_fake_logits, zeros);

            Tensor d_loss = torch::autograd::add_autograd(d_real_loss, d_fake_loss);
            torch::autograd::backward({d_loss});
            d_optim.step();
            d_loss_sum += static_cast<double>(move_to_cpu(d_loss).data_ptr<float>()[0]);

            // ----- Generator step (non-saturating) -----
            G->zero_grad();

            Tensor z_g_cpu = make_noise(batch_size, rng);
            Tensor z_g = to_device(z_g_cpu);

            Tensor fake_g  = G->forward(z_g);       // requires_grad through G
            Tensor d_fake2 = D->forward(fake_g);    // D is also on graph but we zero its grad above
            Tensor g_loss  = bce_with_logits(d_fake2, ones);  // want D to call these "real"

            // IMPORTANT: D was NOT zero_grad'd just now — it was zero_grad'd before
            // its own backward, then step'd. We need to clear D's grads so g_loss.backward
            // doesn't accumulate garbage into them (they would be immediately overwritten
            // next iteration anyway, but clearing keeps semantics clean and avoids
            // interference with gradient clipping / diagnostics).
            D->zero_grad();

            torch::autograd::backward({g_loss});
            g_optim.step();
            g_loss_sum += static_cast<double>(move_to_cpu(g_loss).data_ptr<float>()[0]);

            ++count;

            if ((b + 1) % 50 == 0) {
                double dl = d_loss_sum / count;
                double gl = g_loss_sum / count;
                std::cout << "  epoch " << epoch
                          << " batch " << (b + 1) << "/" << nb
                          << "  D_loss=" << dl
                          << "  G_loss=" << gl << std::endl;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoch " << epoch << " | "
                  << "D_loss=" << (d_loss_sum / std::max<int64_t>(1, count))
                  << "  G_loss=" << (g_loss_sum / std::max<int64_t>(1, count))
                  << "  time=" << secs << "s" << std::endl;

        // Sampling
        if (epoch % sample_every == 0 || epoch == epochs) {
            G->eval();
            torch::autograd::NoGradGuard ng;
            Tensor z_dev = to_device(fixed_z_cpu);
            Tensor samples = G->forward(z_dev);
            Tensor samples_cpu = move_to_cpu(samples);
            std::string p = out_dir + "/gan_samples_epoch_" + std::to_string(epoch) + ".ppm";
            save_grid_ppm(samples_cpu, p);
        }
    }

    std::cout << "Done." << std::endl;
    return 0;
}
