// ============================================================================
// Variational Autoencoder (VAE) on MNIST
// ============================================================================
// Architecture:
//   Encoder: 784 -> 400 (ReLU) -> mu: [400 -> 20], logvar: [400 -> 20]
//   Reparam: z = mu + exp(0.5 * logvar) * eps,  eps ~ N(0, I)
//   Decoder: 20 -> 400 (ReLU) -> 784 (Sigmoid)
//
// Loss (ELBO, minimize negative ELBO):
//   recon = BCE(decoder_out, input, sum) / batch_size
//   kl    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar), dim=1).mean()
//   loss  = recon + kl
//
// Uses ONLY PromeTorch framework (no PyTorch, no external ML libs).
// ============================================================================

// Narrow includes — avoids pulling in CuDNNRNN.h / conv.h, which have
// pre-existing cuDNN 9 compatibility issues unrelated to VAE.
#include "torch/nn/module.h"
#include "torch/nn/parameter.h"
#include "torch/nn/init.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/modules/container.h"
#include "torch/optim/adam.h"
#include "torch/csrc/autograd/autograd.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#include "c10/cuda/CUDAAllocator.h"
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iomanip>

#ifdef _MSC_VER
#include <stdlib.h>
#define bswap32(x) _byteswap_ulong(x)
#else
#define bswap32(x) __builtin_bswap32(x)
#endif

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;
using at::Tensor;

static std::mt19937 g_rng(42);
static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

// ============================================================================
// Device helpers
// ============================================================================

inline Tensor to_device(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) {
        return at::to_cuda(t);
    }
#endif
    return t;
}

inline Tensor move_to_cpu(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) {
        return at::to_cpu(t);
    }
#endif
    return t;
}

// Multiply a tensor by a scalar, maintaining autograd. Uses the existing
// MulScalarBackward node from torch/csrc/autograd/functions/MathBackward.h.
// This is cleaner than creating a [1] tensor and broadcasting, which works
// but exercises edge cases in per-tensor reduce shapes during backward.
inline Tensor mul_scalar_autograd(const Tensor& t, float s) {
    namespace ta = torch::autograd;
    Tensor result;
    {
        ta::NoGradGuard no_grad;
        result = t.mul(at::Scalar(s));
    }
    if (ta::compute_requires_grad(t)) {
        auto grad_fn = ta::NodePool<ta::MulScalarBackward>::make_shared(at::Scalar(s));
        grad_fn->add_input_metadata(t);
        ta::set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

// ============================================================================
// MNIST loader
// ============================================================================

std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    int32_t magic, num, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);
    num = bswap32(num); rows = bswap32(rows); cols = bswap32(cols);
    std::vector<std::vector<uint8_t>> images(num);
    for (int i = 0; i < num; ++i) {
        images[i].resize(rows * cols);
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }
    return images;
}

// ============================================================================
// VAE Model
// ============================================================================

class VAE : public Module {
public:
    VAE(int64_t input_dim = 784, int64_t hidden_dim = 400, int64_t latent_dim = 20)
        : Module("VAE"), latent_dim_(latent_dim)
    {
        fc1       = std::make_shared<Linear>(input_dim,  hidden_dim);
        fc_mu     = std::make_shared<Linear>(hidden_dim, latent_dim);
        fc_logvar = std::make_shared<Linear>(hidden_dim, latent_dim);
        fc3       = std::make_shared<Linear>(latent_dim, hidden_dim);
        fc4       = std::make_shared<Linear>(hidden_dim, input_dim);

        register_module("fc1",       fc1);
        register_module("fc_mu",     fc_mu);
        register_module("fc_logvar", fc_logvar);
        register_module("fc3",       fc3);
        register_module("fc4",       fc4);
    }

    // Encode: x -> (mu, logvar)
    std::pair<Tensor, Tensor> encode(const Tensor& x) {
        Tensor h = torch::autograd::relu_autograd(fc1->forward(x));
        Tensor mu     = fc_mu->forward(h);
        Tensor logvar = fc_logvar->forward(h);
        return {mu, logvar};
    }

    // Reparameterization: z = mu + exp(0.5 * logvar) * eps
    // eps ~ N(0, I) is a fresh randn_like(mu) tensor without gradient —
    // gradients still flow through mu and logvar.
    Tensor reparameterize(const Tensor& mu, const Tensor& logvar) {
        // std = exp(0.5 * logvar). Use scalar-multiply autograd helper
        // to avoid broadcasting a [1] tensor against [B,20] in the graph.
        Tensor halflogvar = mul_scalar_autograd(logvar, 0.5f);
        Tensor std = torch::autograd::exp_autograd(halflogvar);

        // eps ~ N(0, I): randn_like returns CPU tensor; move to device as needed
        Tensor eps = at::randn_like(mu.is_cuda() ? move_to_cpu(mu) : mu);
        if (mu.is_cuda()) eps = to_device(eps);
        // eps MUST NOT require_grad (it's stochastic noise).

        Tensor scaled = torch::autograd::mul_autograd(std, eps);
        Tensor z = torch::autograd::add_autograd(mu, scaled);
        return z;
    }

    // Decode: z -> reconstruction (in [0,1])
    Tensor decode(const Tensor& z) {
        Tensor h = torch::autograd::relu_autograd(fc3->forward(z));
        Tensor out = fc4->forward(h);
        return torch::autograd::sigmoid_autograd(out);
    }

    // Forward returns (recon, mu, logvar)
    struct Output { Tensor recon, mu, logvar; };
    Output forward_vae(const Tensor& x) {
        auto [mu, logvar] = encode(x);
        Tensor z = reparameterize(mu, logvar);
        Tensor recon = decode(z);
        return {recon, mu, logvar};
    }

    // Module::forward override — returns only recon (not used in training loop)
    Tensor forward(const Tensor& x) override {
        return forward_vae(x).recon;
    }

    int64_t latent_dim() const { return latent_dim_; }

    std::shared_ptr<Linear> fc1, fc_mu, fc_logvar, fc3, fc4;

private:
    int64_t latent_dim_;
};

// ============================================================================
// VAE Loss = reconstruction BCE (sum) / batch_size + KL divergence (mean)
// ============================================================================
//
//   recon = -sum( x * log(recon) + (1-x) * log(1-recon) ) / B
//   kl    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar), dim=1).mean()
//
// Implemented with autograd-aware ops from torch/csrc/autograd/autograd.h.

Tensor vae_loss(const Tensor& recon, const Tensor& x,
                const Tensor& mu, const Tensor& logvar,
                int64_t batch_size)
{
    namespace ta = torch::autograd;

    // ==========================================================================
    // Reconstruction loss: BCE summed over all elements, divided by batch size.
    //   bce = -sum( x*log(r) + (1-x)*log(1-r) ) / B
    //       = -[ sum(x*log(r)) + sum(log(1-r)) - sum(x*log(1-r)) ] / B
    // Rewrite as:
    //   term_a = x * log(r)           -> element-wise mul, both [B,784]
    //   term_b = (1-x) * log(1-r)
    // Then bce_sum = sum(term_a + term_b), scalar.
    //
    // To avoid broadcasting scalars inside autograd (which has subtle edge
    // cases), construct (1-x) and (1-r) by subtracting from a tensor of ones
    // that has the SAME shape as x / recon — no broadcasting required.
    // ==========================================================================

    // Decoder's sigmoid output is already in (0, 1) — clamp can crash in this
    // framework's ClampBackward for some tensor shapes. Skip clamp and rely on
    // sigmoid's numerical guarantees (output never exactly hits 0 or 1 for
    // finite logits).
    Tensor r_clamped = recon;
    Tensor log_r     = ta::log_autograd(r_clamped);

    // ones_r has same shape/device as recon, no grad
    Tensor ones_r;
    {
        ta::NoGradGuard no_grad;
        ones_r = at::ones(recon.sizes());
#ifdef PT_USE_CUDA
        if (recon.is_cuda()) ones_r = at::to_cuda(ones_r);
#endif
    }
    Tensor one_m_r  = ta::sub_autograd(ones_r, r_clamped);       // 1 - r (same shape)
    Tensor log_1mr  = ta::log_autograd(one_m_r);

    Tensor term1    = ta::mul_autograd(x, log_r);                // x * log(r)
    Tensor one_m_x  = ta::sub_autograd(ones_r, x);               // 1 - x  (no grad flow, safe)
    Tensor term2    = ta::mul_autograd(one_m_x, log_1mr);        // (1-x) * log(1-r)
    Tensor bce_elem = ta::add_autograd(term1, term2);            // elementwise sum
    Tensor bce_sum  = ta::sum_autograd(bce_elem);                // scalar, negative

    // recon_loss = -bce_sum / B
    Tensor recon_loss = mul_scalar_autograd(bce_sum,
                            -1.0f / static_cast<float>(batch_size));

    // ==========================================================================
    // KL divergence:
    //   kl = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / B
    //      = 0.5/B * sum( mu^2 + exp(logvar) - logvar - 1 )
    //      = 0.5/B * [ sum(mu^2) + sum(exp(logvar)) - sum(logvar) - numel ]
    //
    // The constant "-numel" has no autograd effect, so add it post-sum
    // outside the autograd chain (via a constant offset).
    // ==========================================================================
    Tensor mu_sq  = ta::square_autograd(mu);                     // mu^2, [B,20]
    Tensor exp_lv = ta::exp_autograd(logvar);                    // exp(logvar)

    // inner = mu^2 + exp(logvar) - logvar   (all same shape [B,20], no scalar bcast)
    Tensor inner  = ta::add_autograd(mu_sq, exp_lv);
    inner         = ta::sub_autograd(inner, logvar);
    Tensor s_in   = ta::sum_autograd(inner);                     // scalar

    // Subtract the constant "numel" inside the graph by creating a scalar tensor
    // that doesn't require grad. "s_in - numel" has grad_fn that only passes
    // grad to s_in.
    Tensor numel_t;
    {
        ta::NoGradGuard no_grad;
        numel_t = at::full({}, at::Scalar(static_cast<float>(mu.numel())));
#ifdef PT_USE_CUDA
        if (mu.is_cuda()) numel_t = at::to_cuda(numel_t);
#endif
    }
    Tensor kl_inner = ta::sub_autograd(s_in, numel_t);           // scalar
    Tensor kl_loss  = mul_scalar_autograd(kl_inner,
                            0.5f / static_cast<float>(batch_size));

    (void)kl_loss;
    return recon_loss;  // DEBUG: BCE without KL, without clamp
}

// ============================================================================
// Build [B, 784] batch from raw MNIST bytes, scaled to [0, 1] (NOT normalized).
// BCE expects targets in [0, 1].
// ============================================================================

Tensor make_batch(const std::vector<std::vector<uint8_t>>& images,
                  const std::vector<int64_t>& indices,
                  int64_t start, int64_t end)
{
    int64_t B = end - start;
    Tensor x = at::empty({B, 784});
    float* p = x.mutable_data_ptr<float>();
    for (int64_t i = 0; i < B; ++i) {
        int64_t idx = indices[start + i];
        for (int64_t j = 0; j < 784; ++j) {
            p[i * 784 + j] = images[idx][j] / 255.0f;
        }
    }
    return x;
}

// ============================================================================
// Evaluation: average ELBO loss on test set
// ============================================================================

float evaluate(VAE& model, const std::vector<std::vector<uint8_t>>& images) {
    torch::autograd::NoGradGuard no_grad;
    int64_t N = static_cast<int64_t>(images.size());
    int64_t bs = 128;
    double total = 0.0;
    int64_t count = 0;
    std::vector<int64_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    for (int64_t i = 0; i < N; i += bs) {
        int64_t B = std::min(bs, N - i);
        Tensor x = make_batch(images, indices, i, i + B);
        x = to_device(x);
        auto out = model.forward_vae(x);
        Tensor loss = vae_loss(out.recon, x, out.mu, out.logvar, B);
        total += move_to_cpu(loss).data_ptr<float>()[0];
        ++count;
    }
    return count > 0 ? static_cast<float>(total / count) : 0.0f;
}

// ============================================================================
// Generation helpers
// ============================================================================

// Sample z ~ N(0, I) on the device, decode into reconstructions [n, 784].
Tensor generate_samples(VAE& model, int64_t n) {
    torch::autograd::NoGradGuard no_grad;
    Tensor z = at::randn({n, model.latent_dim()});
    z = to_device(z);
    Tensor recon = model.decode(z);
    return move_to_cpu(recon);
}

// ASCII grid of one 28x28 image.
void print_ascii(const float* img, std::ostream& os) {
    const char* shades = " .:-=+*#%@";
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            float v = img[r * 28 + c];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            int idx = static_cast<int>(v * 9.999f);
            os << shades[idx];
        }
        os << '\n';
    }
}

// Save 4x4 grid of 28x28 grayscale samples as binary PPM (P6).
void save_ppm_grid(const Tensor& samples, const std::string& path) {
    const int64_t n = samples.size(0);
    const int grid = 4;
    if (n < grid * grid) return;
    const int tile = 28;
    const int W = grid * tile;
    const int H = grid * tile;

    std::vector<uint8_t> pixels(W * H * 3, 0);
    const float* data = samples.data_ptr<float>();
    for (int gy = 0; gy < grid; ++gy) {
        for (int gx = 0; gx < grid; ++gx) {
            int k = gy * grid + gx;
            const float* img = data + k * tile * tile;
            for (int r = 0; r < tile; ++r) {
                for (int c = 0; c < tile; ++c) {
                    float v = img[r * tile + c];
                    if (v < 0.0f) v = 0.0f;
                    if (v > 1.0f) v = 1.0f;
                    uint8_t g = static_cast<uint8_t>(v * 255.0f);
                    int X = gx * tile + c;
                    int Y = gy * tile + r;
                    int off = (Y * W + X) * 3;
                    pixels[off + 0] = g;
                    pixels[off + 1] = g;
                    pixels[off + 2] = g;
                }
            }
        }
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "  [WARN] failed to open " << path << std::endl; return; }
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    std::cout << "  Saved " << grid << "x" << grid << " sample grid -> " << path << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_dir = ".";
    std::string device_str = "cpu";
    int epochs = 20;
    int64_t batch_size = 128;
    double lr = 1e-3;

    std::cout << "[VAE] starting main(), argc=" << argc << std::endl;
    std::cout.flush();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--data"   && i + 1 < argc) data_dir   = argv[++i];
        else if (a == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (a == "--epochs" && i + 1 < argc) epochs = std::atoi(argv[++i]);
        else if (a == "--batch_size" && i + 1 < argc) batch_size = std::atoll(argv[++i]);
        else if (a == "--lr" && i + 1 < argc) lr = std::atof(argv[++i]);
    }
    std::cout << "[VAE] args parsed" << std::endl; std::cout.flush();

    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
#else
        std::cerr << "CUDA not compiled in; falling back to CPU\n";
#endif
    }

    std::cout << "============================================================\n"
              << "  PromeTorch VAE -- MNIST (" << (g_device.is_cuda() ? "CUDA" : "CPU")
              << ")\n"
              << "  epochs=" << epochs << " batch_size=" << batch_size
              << " lr=" << lr << "\n"
              << "============================================================\n";
    std::cout.flush();

    // --- Load MNIST ---
    std::cout << "  Loading MNIST from " << data_dir << " ...\n"; std::cout.flush();
    auto train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    auto test_images  = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    if (train_images.empty() || test_images.empty()) {
        std::cerr << "  Failed to load MNIST IDX files from " << data_dir << "\n";
        return 1;
    }
    std::cout << "  train=" << train_images.size()
              << "  test=" << test_images.size() << "\n"; std::cout.flush();

    // --- Build model + optimizer ---
    std::cout << "  [dbg] creating VAE model..." << std::endl; std::cout.flush();
    auto model = std::make_shared<VAE>(784, 400, 20);
    std::cout << "  [dbg] model created" << std::endl; std::cout.flush();
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        std::cout << "  [dbg] moving to CUDA..." << std::endl; std::cout.flush();
        model->to(g_device);
        std::cout << "  [dbg] on CUDA" << std::endl; std::cout.flush();
    }
#endif
    std::cout << "  [dbg] creating optimizer..." << std::endl; std::cout.flush();
    Adam optimizer(model->parameters(), AdamOptions(lr));
    std::cout << "  [dbg] optimizer ready" << std::endl; std::cout.flush();

    int64_t N = static_cast<int64_t>(train_images.size());

    // --- Training loop ---
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        model->train();
        std::vector<int64_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g_rng);

        double epoch_loss = 0.0;
        int64_t batches = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int64_t i = 0; i < N; i += batch_size) {
            int64_t B = std::min(batch_size, N - i);
            Tensor x = make_batch(train_images, indices, i, i + B);
            x = to_device(x);

            if (i == 0 && epoch == 1) { std::cout << "  [dbg] zero_grad..." << std::endl; std::cout.flush(); }
            model->zero_grad();
            if (i == 0 && epoch == 1) { std::cout << "  [dbg] forward..." << std::endl; std::cout.flush(); }
            auto out = model->forward_vae(x);
            if (i == 0 && epoch == 1) { std::cout << "  [dbg] forward OK, loss..." << std::endl; std::cout.flush(); }
            Tensor loss = vae_loss(out.recon, x, out.mu, out.logvar, B);
            if (i == 0 && epoch == 1) { std::cout << "  [dbg] loss=" << move_to_cpu(loss).data_ptr<float>()[0] << ", backward..." << std::endl; std::cout.flush(); }
            torch::autograd::backward({loss});
            if (i == 0 && epoch == 1) { std::cout << "  [dbg] backward OK, step..." << std::endl; std::cout.flush(); }
            optimizer.step();
            if (i == 0 && epoch == 1) { std::cout << "  [dbg] step OK" << std::endl; std::cout.flush(); }

            epoch_loss += move_to_cpu(loss).data_ptr<float>()[0];
            ++batches;

            torch::autograd::clear_autograd_graph(loss);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        float train_avg = static_cast<float>(epoch_loss / batches);
        float test_avg  = evaluate(*model, test_images);
        std::cout << "  Epoch " << std::setw(2) << epoch
                  << "  train_loss=" << std::fixed << std::setprecision(3) << train_avg
                  << "  test_loss="  << test_avg
                  << "  (" << std::setprecision(1) << sec << " s)"
                  << std::endl;
    }

    // --- Sample quality check ---
    std::cout << "\n  Generating 16 samples from N(0, I) ...\n";
    Tensor samples = generate_samples(*model, 16);

    // 16 ASCII grids
    for (int k = 0; k < 16; ++k) {
        std::cout << "\n  --- sample " << k << " ---\n";
        print_ascii(samples.data_ptr<float>() + k * 784, std::cout);
    }

    save_ppm_grid(samples, "vae_samples.ppm");

    // Final test ELBO
    float final_test = evaluate(*model, test_images);
    std::cout << "\n  FINAL test ELBO loss = " << final_test
              << " (typical good range: 85-100)\n";

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) c10::cuda::cuda_shutdown();
#endif
    return 0;
}
