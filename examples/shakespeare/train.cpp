// ============================================================================
// Shakespeare Character-Level Language Model Training
// ============================================================================
// Trains a small decoder-only Transformer on Shakespeare text for
// character-level text generation.
//
// Usage:
//   shakespeare_train [--device cpu|cuda] [--epochs N] [--batch_size B]
//                     [--lr LR] [--block_size S] [--data PATH]
//
// Fixes (vs original):
//   - Single CrossEntropyLoss forward + torch::autograd::backward({loss}).
//     No more manual "inject gradient into logits.set_grad()" hack.
//   - optimizer.zero_grad() moved BEFORE forward (standard order).
//   - Logits [S, B, V] reshaped to [S*B, V], targets [S, B] reshaped to [S*B]
//     to match CrossEntropyLoss expected shape.
//   - Optional --device cuda: model.to(cuda) and inputs moved to cuda.

#include "model.h"
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "c10/core/Device.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>

using namespace shakespeare;
using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

// Default Shakespeare sample (from Hamlet) — tiny fallback if no file provided
const char* DEFAULT_TEXT = R"(
HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take Arms against a Sea of troubles,
And by opposing end them: to die, to sleep;
No more; and by a sleep, to say we end
The heart-ache, and the thousand natural shocks
That Flesh is heir to? 'Tis a consummation
Devoutly to be wished. To die, to sleep,
To sleep, perchance to Dream; aye, there's the rub.
)";

struct Args {
    std::string device = "cpu";
    int64_t epochs = 10;
    int64_t batch_size = 32;
    double lr = 3e-4;
    int64_t block_size = 64;
    int64_t max_iters = 0;       // 0 = derived from epochs
    int64_t log_every = 25;
    int64_t gen_tokens = 200;
    std::string data_path;       // empty -> use data/tiny_shakespeare.txt if exists
};

static void print_usage() {
    std::cout << "Usage: shakespeare_train [options]\n"
              << "  --device cpu|cuda    (default cpu)\n"
              << "  --epochs N           (default 10)\n"
              << "  --batch_size B       (default 32)\n"
              << "  --lr LR              (default 3e-4)\n"
              << "  --block_size S       (default 64)\n"
              << "  --data PATH          (default data/tiny_shakespeare.txt)\n"
              << "  --gen_tokens N       (default 200)\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(1);
            }
            return std::string(argv[++i]);
        };
        if (k == "--device") a.device = next("--device");
        else if (k == "--epochs") a.epochs = std::stoll(next("--epochs"));
        else if (k == "--batch_size") a.batch_size = std::stoll(next("--batch_size"));
        else if (k == "--lr") a.lr = std::stod(next("--lr"));
        else if (k == "--block_size") a.block_size = std::stoll(next("--block_size"));
        else if (k == "--data") a.data_path = next("--data");
        else if (k == "--gen_tokens") a.gen_tokens = std::stoll(next("--gen_tokens"));
        else if (k == "--help" || k == "-h") { print_usage(); std::exit(0); }
        else if (!k.empty() && k[0] != '-' && a.data_path.empty()) {
            // Positional arg: data path
            a.data_path = k;
        } else {
            std::cerr << "Unknown arg: " << k << std::endl;
            print_usage();
            std::exit(1);
        }
    }
    return a;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "PromeTorch Shakespeare Language Model" << std::endl;
    std::cout << "========================================" << std::endl;

    // Resolve data path: explicit -> default tiny_shakespeare.txt -> embedded
    std::string text;
    std::string path = args.data_path;
    if (path.empty()) {
        // Try default locations
        const char* candidates[] = {
            "data/tiny_shakespeare.txt",
            "../data/tiny_shakespeare.txt",
            "../../data/tiny_shakespeare.txt",
            "../../../data/tiny_shakespeare.txt",
        };
        for (const char* c : candidates) {
            std::ifstream f(c);
            if (f.is_open()) { path = c; break; }
        }
    }
    if (!path.empty()) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            text = buffer.str();
            std::cout << "Loaded text from: " << path << std::endl;
        } else {
            std::cerr << "Could not open file: " << path
                      << " — falling back to embedded sample." << std::endl;
        }
    }
    if (text.empty()) {
        text = DEFAULT_TEXT;
        std::cout << "Using built-in Shakespeare sample" << std::endl;
    }

    std::cout << "Text length: " << text.size() << " characters" << std::endl;

    // Build tokenizer
    CharTokenizer tokenizer;
    tokenizer.build_vocab(text);
    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << " characters" << std::endl;

    std::vector<int64_t> tokens = tokenizer.encode(text);
    std::cout << "Total tokens: " << tokens.size() << std::endl;

    // Device
    bool use_cuda = (args.device == "cuda");
#ifndef PT_USE_CUDA
    if (use_cuda) {
        std::cerr << "Build has no CUDA support; falling back to CPU." << std::endl;
        use_cuda = false;
    }
#endif
    std::cout << "Device: " << (use_cuda ? "cuda" : "cpu") << std::endl;

    // Model hyperparameters
    int64_t vocab_size = tokenizer.vocab_size();
    int64_t d_model = 128;
    int64_t nhead = 4;
    int64_t num_layers = 3;
    int64_t dim_feedforward = 256;
    double dropout = 0.1;
    int64_t block_size = args.block_size;
    int64_t batch_size = args.batch_size;

    if (static_cast<int64_t>(tokens.size()) < block_size + 1) {
        std::cerr << "Text too short for block_size=" << block_size << std::endl;
        return 1;
    }

    std::cout << "\nModel configuration:" << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  nhead: " << nhead << std::endl;
    std::cout << "  num_layers: " << num_layers << std::endl;
    std::cout << "  dim_feedforward: " << dim_feedforward << std::endl;
    std::cout << "  block_size: " << block_size << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  epochs: " << args.epochs << std::endl;
    std::cout << "  lr: " << args.lr << std::endl;

    auto model = std::make_shared<TransformerLM>(
        vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, block_size
    );

    std::cout << "\nModel parameters: " << count_parameters(*model) << std::endl;

#ifdef PT_USE_CUDA
    if (use_cuda) {
        model->to(c10::Device(c10::DeviceType::CUDA, 0));
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    // Dataset
    TextDataset dataset(tokens, block_size);
    std::cout << "Training examples: " << dataset.size() << std::endl;

    // Derive iters per epoch so loss measurement scales with dataset size
    int64_t iters_per_epoch = std::max<int64_t>(
        1, static_cast<int64_t>(dataset.size()) / batch_size);
    // Cap epoch size to avoid absurd runtimes on huge shakespeare corpus
    iters_per_epoch = std::min<int64_t>(iters_per_epoch, 500);
    int64_t total_iters = iters_per_epoch * args.epochs;
    std::cout << "Iters/epoch: " << iters_per_epoch
              << ", total iters: " << total_iters << std::endl;

    // Optimizer
    Adam optimizer(model->parameters(), args.lr);

    // Loss
    CrossEntropyLoss criterion;

    std::cout << "\nStarting training..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int64_t epoch = 0; epoch < args.epochs; ++epoch) {
        model->train();

        double epoch_loss_sum = 0.0;
        int64_t epoch_loss_count = 0;

        for (int64_t it = 0; it < iters_per_epoch; ++it) {
            int64_t global_it = epoch * iters_per_epoch + it;

            // Get batch: input [S, B], target [S, B] (float-encoded indices)
            auto [input, target] = dataset.get_batch(batch_size, gen);

#ifdef PT_USE_CUDA
            if (use_cuda) {
                input = at::to_cuda(input);
                target = at::to_cuda(target);
            }
#endif

            // Standard order: zero grads BEFORE forward
            optimizer.zero_grad();

            // Forward: [S, B] -> [S, B, V]
            Tensor logits = model->forward(input);

            // Reshape for CrossEntropyLoss: [N, V] and [N]
            int64_t S = logits.size(0);
            int64_t B = logits.size(1);
            int64_t V = logits.size(2);
            Tensor logits_flat  = logits.reshape({S * B, V});
            Tensor target_flat  = target.reshape({S * B});

            Tensor loss = criterion.forward(logits_flat, target_flat);

            float loss_val = 0.0f;
            {
                Tensor loss_cpu = loss;
#ifdef PT_USE_CUDA
                if (loss_cpu.is_cuda()) loss_cpu = at::to_cpu(loss_cpu);
#endif
                loss_val = loss_cpu.data_ptr<float>()[0];
            }
            epoch_loss_sum += loss_val;
            epoch_loss_count++;

            // Backward through full autograd graph (this is the fix — no more
            // manual set_grad(grad) on logits)
            torch::autograd::backward({loss});

            // Gradient clipping (stabilizes training)
            torch::nn::clip_grad_norm_(*model, 1.0);

            optimizer.step();

            if (global_it % args.log_every == 0 || global_it == total_iters - 1) {
                auto now = std::chrono::high_resolution_clock::now();
                auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                    now - start_time).count();
                std::cout << "Epoch " << std::setw(2) << (epoch + 1)
                          << "/" << args.epochs
                          << " | Iter " << std::setw(5) << global_it
                          << " | Loss: " << std::fixed << std::setprecision(4) << loss_val
                          << " | Time: " << secs << "s" << std::endl;
            }
        }

        double avg = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : 0.0;
        std::cout << ">>> Epoch " << (epoch + 1) << " avg loss: "
                  << std::fixed << std::setprecision(4) << avg << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_secs = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    std::cout << "\nTraining completed in " << total_secs << " seconds" << std::endl;

    // Generate sample text
    std::cout << "\n========================================" << std::endl;
    std::cout << "Generated Text:" << std::endl;
    std::cout << "========================================" << std::endl;

    // Move model back to CPU for generation (generate() uses CPU-style data_ptr reads)
#ifdef PT_USE_CUDA
    if (use_cuda) {
        model->to(c10::Device(c10::DeviceType::CPU, 0));
    }
#endif

    std::string prompt = "HAMLET:";
    std::vector<int64_t> prompt_tokens = tokenizer.encode(prompt);

    model->eval();
    std::vector<int64_t> generated = model->generate(
        prompt_tokens,
        args.gen_tokens,
        0.8,
        true
    );

    std::string generated_text = tokenizer.decode(generated);
    std::cout << generated_text << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Done!" << std::endl;

    return 0;
}
