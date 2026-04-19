// ============================================================================
// Vision Transformer (ViT) Training - MNIST Classification
// ============================================================================
// Image patches as tokens, transformer encoder, classification head
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"
#include "torch/nn/modules/attention.h"
#include "torch/nn/modules/transformer.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "../common/profiler.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include <fstream>
#include <iostream>
#include <vector>
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

static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

inline at::Tensor to_device(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) return at::to_cuda(t);
#endif
    return t;
}

inline at::Tensor move_to_cpu(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) return at::to_cpu(t);
#endif
    return t;
}

// ============================================================================
// MNIST Loading
// ============================================================================

std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        LOG_ERROR("Cannot open: " + path);
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

    LOG_INFO("MNIST images: " + std::to_string(num_images) + " x " +
             std::to_string(rows) + "x" + std::to_string(cols));

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
        LOG_ERROR("Cannot open: " + path);
        return {};
    }

    int32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = bswap32(magic);
    num_labels = bswap32(num_labels);

    LOG_INFO("MNIST labels: " + std::to_string(num_labels));

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}

// ============================================================================
// Patch Embedding Layer
// ============================================================================
//
// Extracts non-overlapping patches from image via reshape+permute (device-
// agnostic), then projects to embed_dim.
// For x: [B, C, H, W] with patch size P and H=W=S:
//   reshape  -> [B, C, H/P, P, W/P, P]
//   permute  -> [B, H/P, W/P, C, P, P]  (dims 0,2,4,1,3,5)
//   reshape  -> [B, N, D]  where N=(H/P)*(W/P), D=C*P*P
//   linear   -> [B, N, embed_dim]
// ============================================================================

class PatchEmbedding : public Module {
public:
    PatchEmbedding(int64_t img_size, int64_t patch_size, int64_t in_channels, int64_t embed_dim)
        : Module("PatchEmbedding"),
          img_size_(img_size), patch_size_(patch_size),
          in_channels_(in_channels), embed_dim_(embed_dim) {

        num_patches_ = (img_size / patch_size) * (img_size / patch_size);
        patch_dim_ = in_channels * patch_size * patch_size;

        LOG_INFO("PatchEmbedding: img=" + std::to_string(img_size) +
                 ", patch=" + std::to_string(patch_size) +
                 ", num_patches=" + std::to_string(num_patches_) +
                 ", patch_dim=" + std::to_string(patch_dim_));

        // Linear projection of flattened patches
        proj = std::make_shared<Linear>(patch_dim_, embed_dim);
        register_module("proj", proj);
    }

    Tensor forward(const Tensor& x) override {
        // x: [B, C, H, W] -> patches -> [B, num_patches, embed_dim]
        LOG_FORWARD("PatchEmbedding");
        LOG_TENSOR("input", x);

        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t C = sizes[1];
        int64_t H = sizes[2];
        int64_t W = sizes[3];

        int64_t P = patch_size_;
        int64_t Hp = H / P;
        int64_t Wp = W / P;

        // Fold spatial dims into patches via reshape+permute (device-agnostic).
        // [B,C,H,W] -> [B,C,Hp,P,Wp,P] -> [B,Hp,Wp,C,P,P] -> [B, Hp*Wp, C*P*P]
        Tensor folded = x.reshape({B, C, Hp, P, Wp, P});
        Tensor perm = folded.permute({0, 2, 4, 1, 3, 5});
        Tensor patches = perm.reshape({B, num_patches_, patch_dim_});

        // Project patches through Linear. Linear handles 3D via internal reshape.
        // patches has no grad_fn (native reshape/permute) — OK, the Linear forward
        // wires grad_fn to the learnable W. AFTER Linear we need reshape_autograd
        // to keep the autograd chain alive back to proj.weight.
        Tensor flat = patches.reshape({B * num_patches_, patch_dim_});
        Tensor embedded = proj->forward(flat);
        embedded = torch::autograd::reshape_autograd(embedded, {B, num_patches_, embed_dim_});

        LOG_TENSOR("output", embedded);
        return embedded;
    }

    int64_t num_patches() const { return num_patches_; }

private:
    int64_t img_size_, patch_size_, in_channels_, embed_dim_;
    int64_t num_patches_, patch_dim_;
    std::shared_ptr<Linear> proj;
};

// ============================================================================
// Vision Transformer
// ============================================================================

class ViT : public Module {
public:
    ViT(int64_t img_size, int64_t patch_size, int64_t in_channels,
        int64_t embed_dim, int64_t n_heads, int64_t n_layers, int64_t num_classes)
        : Module("ViT"), embed_dim_(embed_dim) {

        LOG_INFO("Creating ViT: img=" + std::to_string(img_size) +
                 ", patch=" + std::to_string(patch_size) +
                 ", embed=" + std::to_string(embed_dim) +
                 ", heads=" + std::to_string(n_heads) +
                 ", layers=" + std::to_string(n_layers));

        // Patch embedding
        patch_embed = std::make_shared<PatchEmbedding>(img_size, patch_size, in_channels, embed_dim);
        register_module("patch_embed", patch_embed);

        int64_t num_patches = patch_embed->num_patches();
        int64_t seq_len = num_patches + 1;  // +1 for CLS

        // CLS token (learnable Parameter — must be registered for grad update)
        Tensor cls_init = at::randn({1, 1, embed_dim}).mul(at::Scalar(0.02f));
        register_parameter("cls_token", Parameter(cls_init));

        // Sinusoidal positional encoding (non-learnable buffer, broadcasts on add)
        // Shape [1, seq_len, embed_dim]
        Tensor pe = at::zeros({1, seq_len, embed_dim});
        float* pe_data = pe.mutable_data_ptr<float>();
        for (int64_t pos = 0; pos < seq_len; ++pos) {
            for (int64_t i = 0; i < embed_dim; i += 2) {
                double div_term = std::exp(-static_cast<double>(i) * std::log(10000.0) / embed_dim);
                pe_data[pos * embed_dim + i] = static_cast<float>(std::sin(pos * div_term));
                if (i + 1 < embed_dim) {
                    pe_data[pos * embed_dim + i + 1] = static_cast<float>(std::cos(pos * div_term));
                }
            }
        }
        register_buffer("pos_embed", Buffer(pe));

        // Transformer encoder layers — batch_first=true since we feed [B, L, E]
        for (int64_t i = 0; i < n_layers; ++i) {
            auto layer = std::make_shared<TransformerEncoderLayer>(
                embed_dim, n_heads, embed_dim * 4, 0.1f,
                "gelu", 1e-5, /*batch_first=*/true, /*norm_first=*/true);
            encoder_layers.push_back(layer);
            register_module("encoder_" + std::to_string(i), layer);
        }

        // Classification head
        norm = std::make_shared<LayerNorm>(std::vector<int64_t>{embed_dim});
        head = std::make_shared<Linear>(embed_dim, num_classes);

        register_module("norm", norm);
        register_module("head", head);
    }

    Tensor forward(const Tensor& x) override {
        PROFILE_SCOPE("ViT forward");
        LOG_TENSOR("input", x);

        auto sizes = x.sizes();
        int64_t B = sizes[0];

        // Patch embedding
        Tensor patches = patch_embed->forward(x);  // [B, num_patches, embed_dim]
        LOG_TENSOR("patches", patches);

        // Prepend CLS token. Broadcast learnable [1,1,E] to [B,1,E] via repeat
        // across batch dim using cat (autograd-aware so grad flows to cls_token).
        Parameter* cls_param = get_parameter("cls_token");
        Tensor cls = cls_param->data();  // [1, 1, E]
        std::vector<Tensor> cls_copies;
        cls_copies.reserve(B);
        for (int64_t b = 0; b < B; ++b) {
            cls_copies.push_back(cls);
        }
        Tensor cls_batch = torch::autograd::cat_autograd(cls_copies, 0);  // [B, 1, E]

        // Concatenate: [B, 1+num_patches, embed]
        Tensor tokens = torch::autograd::cat_autograd({cls_batch, patches}, 1);
        LOG_TENSOR("tokens_with_cls", tokens);

        // Add sinusoidal positional encoding (broadcasts [1, L, E] + [B, L, E])
        Buffer* pe_buf = get_buffer("pos_embed");
        Tensor pe = pe_buf->data();
        tokens = tokens.add(pe);

        // Transformer encoder
        Tensor h = tokens;
        for (size_t i = 0; i < encoder_layers.size(); ++i) {
            LOG_FORWARD("TransformerEncoder[" + std::to_string(i) + "]");
            h = encoder_layers[i]->forward(h);
        }
        LOG_TENSOR("encoder_out", h);

        // Extract CLS token output: h[:, 0, :]. Use select_autograd so gradient
        // from the classification head flows back through the encoder stack.
        Tensor cls_out = torch::autograd::select_autograd(h, 1, 0);  // [B, embed_dim]
        LOG_TENSOR("cls_out", cls_out);

        // Layer norm + classification head
        cls_out = norm->forward(cls_out);
        Tensor logits = head->forward(cls_out);

        LOG_TENSOR("logits", logits);
        return logits;
    }

private:
    int64_t embed_dim_;
    std::shared_ptr<PatchEmbedding> patch_embed;
    std::vector<std::shared_ptr<TransformerEncoderLayer>> encoder_layers;
    std::shared_ptr<LayerNorm> norm;
    std::shared_ptr<Linear> head;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_dir = "data/mnist";
    std::string device_str = "cpu";
    int64_t batch_size = 64;
    int64_t epochs = 5;
    int64_t patch_size = 7;     // 28/7 = 4x4 = 16 patches
    int64_t embed_dim = 64;
    int64_t n_heads = 4;
    int64_t n_layers = 4;
    float lr = 0.001f;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--batch_size" && i + 1 < argc) batch_size = std::stoll(argv[++i]);
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoll(argv[++i]);
        else if (arg == "--patch_size" && i + 1 < argc) patch_size = std::stoll(argv[++i]);
        else if (arg == "--embed_dim" && i + 1 < argc) embed_dim = std::stoll(argv[++i]);
        else if (arg == "--n_heads" && i + 1 < argc) n_heads = std::stoll(argv[++i]);
        else if (arg == "--n_layers" && i + 1 < argc) n_layers = std::stoll(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--verbose" || arg == "-v") verbose = true;
    }

    if (verbose) {
        profiler::Logger::instance().set_level(profiler::LogLevel::TRACE);
    } else {
        profiler::Logger::instance().set_level(profiler::LogLevel::INFO);
    }

    LOG_INFO("=== Vision Transformer (ViT) Training ===");

    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        LOG_INFO("Using CUDA");
        LOG_MEMORY("Initial");
#else
        LOG_WARN("CUDA not available, using CPU");
#endif
    } else {
        LOG_INFO("Using CPU");
    }

    // Load MNIST
    LOG_INFO("Loading MNIST from: " + data_dir);
    auto train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    auto train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    auto test_images = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    auto test_labels = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    if (train_images.empty()) {
        LOG_ERROR("Failed to load MNIST");
        return 1;
    }

    int64_t n_train = train_images.size();
    int64_t n_test = test_images.size();

    // Create ViT model
    auto model = std::make_shared<ViT>(
        28, patch_size, 1, embed_dim, n_heads, n_layers, 10);
    LOG_MEMORY("After model creation");
    LOG_INFO("Num parameters: " + std::to_string(model->num_parameters()));

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        PROFILE_SCOPE("Model to CUDA");
        model->to(g_device);
        LOG_MEMORY("After model.to(CUDA)");
    }
#endif

    AdamOptions opts(lr);
    Adam optimizer(model->parameters(), opts);
    CrossEntropyLoss criterion;

    std::random_device rd;
    std::mt19937 gen(rd());

    LOG_INFO("Training: epochs=" + std::to_string(epochs) +
             ", batch=" + std::to_string(batch_size));
    LOG_INFO("Model: patch=" + std::to_string(patch_size) +
             ", embed=" + std::to_string(embed_dim) +
             ", heads=" + std::to_string(n_heads) +
             ", layers=" + std::to_string(n_layers));

    profiler::Stats loss_stats, time_stats;

    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        model->train();
        profiler::Timer epoch_timer;
        epoch_timer.start();

        std::vector<int64_t> indices(n_train);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        float epoch_loss = 0;
        int64_t correct = 0, total = 0;

        for (int64_t batch_start = 0; batch_start < n_train; batch_start += batch_size) {
            int64_t batch_end = std::min(batch_start + batch_size, n_train);
            int64_t B = batch_end - batch_start;

            // [B, 1, 28, 28]
            Tensor inputs = at::empty({B, 1, 28, 28});
            Tensor targets = at::empty({B});

            float* in_ptr = inputs.mutable_data_ptr<float>();
            float* tgt_ptr = targets.mutable_data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int64_t idx = indices[batch_start + i];
                tgt_ptr[i] = static_cast<float>(train_labels[idx]);
                for (int j = 0; j < 784; ++j) {
                    // Standard MNIST normalization
                    in_ptr[i * 784 + j] = (train_images[idx][j] / 255.0f - 0.1307f) / 0.3081f;
                }
            }

            inputs = to_device(inputs);
            targets = to_device(targets);

            optimizer.zero_grad();

            Tensor logits;
            {
                PROFILE_SCOPE("Forward");
                logits = model->forward(inputs);
            }

            Tensor loss = criterion.forward(logits, targets);

            {
                PROFILE_SCOPE("Backward");
                torch::autograd::backward({loss});
            }

            optimizer.step();

            Tensor loss_cpu = move_to_cpu(loss);
            epoch_loss += loss_cpu.data_ptr<float>()[0] * B;

            // Accuracy
            Tensor logits_cpu = move_to_cpu(logits);
            Tensor targets_cpu = move_to_cpu(targets);
            const float* log_ptr = logits_cpu.data_ptr<float>();
            const float* tgt_ptr2 = targets_cpu.data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int pred = 0;
                float max_val = log_ptr[i * 10];
                for (int c = 1; c < 10; ++c) {
                    if (log_ptr[i * 10 + c] > max_val) {
                        max_val = log_ptr[i * 10 + c];
                        pred = c;
                    }
                }
                if (pred == static_cast<int>(tgt_ptr2[i])) correct++;
                total++;
            }
        }

        double epoch_time = epoch_timer.stop();
        time_stats.add(epoch_time);
        loss_stats.add(epoch_loss / n_train);

        LOG_INFO("Epoch " + std::to_string(epoch) + "/" + std::to_string(epochs) +
                 " | Loss: " + std::to_string(epoch_loss / n_train) +
                 " | Train Acc: " + std::to_string(100.0f * correct / total) + "%" +
                 " | Time: " + std::to_string(epoch_time) + "ms");
        LOG_MEMORY("Epoch " + std::to_string(epoch));

        // Test
        model->eval();
        int64_t test_correct = 0;

        {
            torch::autograd::NoGradGuard no_grad;
            for (int64_t batch_start = 0; batch_start < n_test; batch_start += batch_size) {
                int64_t batch_end = std::min(batch_start + batch_size, n_test);
                int64_t B = batch_end - batch_start;

                Tensor inputs = at::empty({B, 1, 28, 28});
                float* in_ptr = inputs.mutable_data_ptr<float>();

                for (int64_t i = 0; i < B; ++i) {
                    int64_t idx = batch_start + i;
                    for (int j = 0; j < 784; ++j) {
                        in_ptr[i * 784 + j] = (test_images[idx][j] / 255.0f - 0.1307f) / 0.3081f;
                    }
                }

                inputs = to_device(inputs);
                Tensor logits = model->forward(inputs);
                Tensor logits_cpu = move_to_cpu(logits);
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
                    if (pred == test_labels[batch_start + i]) test_correct++;
                }
            }
        }

        LOG_INFO("Test Acc: " + std::to_string(100.0f * test_correct / n_test) + "%");
    }

    LOG_INFO("\n=== Training Statistics ===");
    loss_stats.print("Loss");
    time_stats.print("Epoch time (ms)");

    LOG_INFO("=== ViT Training Complete ===");
    return 0;
}
