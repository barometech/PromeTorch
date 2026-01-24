// ============================================================================
// Transformer Training Example - Sequence Classification
// ============================================================================
// Simple Transformer encoder for text classification
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
#include <chrono>
#include <random>
#include <unordered_map>
#include <sstream>

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

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
    if (t.is_cuda()) {
        return at::to_cpu(t);
    }
#endif
    return t;
}

// ============================================================================
// Simple Transformer Classifier
// ============================================================================

class TransformerClassifier : public Module {
public:
    TransformerClassifier(int64_t vocab_size, int64_t d_model, int64_t n_heads,
                          int64_t n_layers, int64_t num_classes, int64_t max_len = 512)
        : Module("TransformerClassifier"), d_model_(d_model) {

        // Token embedding
        token_embed = std::make_shared<Embedding>(vocab_size, d_model);

        // Positional encoding (learnable)
        pos_embed = std::make_shared<Embedding>(max_len, d_model);

        // Transformer encoder layers
        for (int64_t i = 0; i < n_layers; ++i) {
            auto layer = std::make_shared<TransformerEncoderLayer>(
                d_model, n_heads, d_model * 4, 0.1f);
            encoder_layers.push_back(layer);
            register_module("encoder_" + std::to_string(i), layer);
        }

        // Classification head
        fc = std::make_shared<Linear>(d_model, num_classes);
        dropout = std::make_shared<Dropout>(0.1f);

        register_module("token_embed", token_embed);
        register_module("pos_embed", pos_embed);
        register_module("fc", fc);
    }

    Tensor forward(const Tensor& x) override {
        // x: [B, T] token indices
        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        // Get embeddings
        Tensor tok_emb = token_embed->forward(x);  // [B, T, d_model]

        // Create position indices [0, 1, 2, ..., T-1]
        Tensor positions = at::empty({1, T});
        float* pos_ptr = positions.mutable_data_ptr<float>();
        for (int64_t t = 0; t < T; ++t) {
            pos_ptr[t] = static_cast<float>(t);
        }
        positions = to_device(positions);

        Tensor pos_emb = pos_embed->forward(positions);  // [1, T, d_model]

        // Combine embeddings
        Tensor h = tok_emb.add(pos_emb);
        h = dropout->forward(h);

        // Pass through encoder layers
        for (auto& layer : encoder_layers) {
            h = layer->forward(h);
        }

        // Global average pooling over sequence
        // h: [B, T, d_model] -> mean over T -> [B, d_model]
        h = h.mean(1);  // Mean over time dimension

        // Classification
        Tensor logits = fc->forward(h);  // [B, num_classes]

        return logits;
    }

private:
    int64_t d_model_;
    std::shared_ptr<Embedding> token_embed;
    std::shared_ptr<Embedding> pos_embed;
    std::vector<std::shared_ptr<TransformerEncoderLayer>> encoder_layers;
    std::shared_ptr<Linear> fc;
    std::shared_ptr<Dropout> dropout;
};

// ============================================================================
// Synthetic Sentiment Dataset
// ============================================================================
// Generates simple positive/negative sentences

struct SentimentSample {
    std::vector<int64_t> tokens;
    int64_t label;  // 0 = negative, 1 = positive
};

std::vector<std::string> positive_words = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "happy", "joy", "best", "beautiful", "perfect", "awesome"
};

std::vector<std::string> negative_words = {
    "bad", "terrible", "awful", "horrible", "hate", "worst", "ugly",
    "sad", "angry", "poor", "wrong", "fail", "disaster", "boring"
};

std::vector<std::string> neutral_words = {
    "the", "a", "is", "it", "this", "that", "movie", "book", "food",
    "day", "time", "thing", "place", "very", "really", "so", "and"
};

class SentimentDataset {
public:
    SentimentDataset(int64_t num_samples, int64_t max_len, uint32_t seed = 42)
        : max_len_(max_len) {

        // Build vocabulary
        vocab_["<pad>"] = 0;
        vocab_["<unk>"] = 1;

        for (const auto& w : positive_words) {
            if (vocab_.find(w) == vocab_.end()) {
                vocab_[w] = vocab_.size();
            }
        }
        for (const auto& w : negative_words) {
            if (vocab_.find(w) == vocab_.end()) {
                vocab_[w] = vocab_.size();
            }
        }
        for (const auto& w : neutral_words) {
            if (vocab_.find(w) == vocab_.end()) {
                vocab_[w] = vocab_.size();
            }
        }

        vocab_size_ = vocab_.size();
        std::cout << "Vocabulary size: " << vocab_size_ << std::endl;

        // Generate samples
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> len_dist(5, max_len);
        std::uniform_int_distribution<int> pos_dist(0, positive_words.size() - 1);
        std::uniform_int_distribution<int> neg_dist(0, negative_words.size() - 1);
        std::uniform_int_distribution<int> neu_dist(0, neutral_words.size() - 1);
        std::uniform_real_distribution<float> coin(0, 1);

        for (int64_t i = 0; i < num_samples; ++i) {
            SentimentSample sample;
            int64_t label = (i % 2);  // Alternate positive/negative
            sample.label = label;

            int64_t sent_len = len_dist(gen);
            sample.tokens.resize(sent_len);

            for (int64_t t = 0; t < sent_len; ++t) {
                float r = coin(gen);
                std::string word;

                if (r < 0.3f) {
                    // Insert sentiment word
                    if (label == 1) {
                        word = positive_words[pos_dist(gen)];
                    } else {
                        word = negative_words[neg_dist(gen)];
                    }
                } else {
                    word = neutral_words[neu_dist(gen)];
                }

                sample.tokens[t] = vocab_[word];
            }

            samples_.push_back(sample);
        }
    }

    const SentimentSample& get(int64_t idx) const {
        return samples_[idx];
    }

    int64_t size() const { return samples_.size(); }
    int64_t vocab_size() const { return vocab_size_; }
    int64_t max_len() const { return max_len_; }

private:
    std::unordered_map<std::string, int64_t> vocab_;
    std::vector<SentimentSample> samples_;
    int64_t vocab_size_;
    int64_t max_len_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string device_str = "cpu";
    int64_t d_model = 64;
    int64_t n_heads = 4;
    int64_t n_layers = 2;
    int64_t batch_size = 32;
    int64_t max_len = 32;
    int64_t num_samples = 5000;
    int64_t epochs = 10;
    float lr = 0.001f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg == "--d_model" && i + 1 < argc) {
            d_model = std::stoll(argv[++i]);
        } else if (arg == "--n_heads" && i + 1 < argc) {
            n_heads = std::stoll(argv[++i]);
        } else if (arg == "--n_layers" && i + 1 < argc) {
            n_layers = std::stoll(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoll(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoll(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        }
    }

    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Using CUDA" << std::endl;
#else
        std::cerr << "CUDA not available, using CPU" << std::endl;
#endif
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    // Create dataset
    std::cout << "Creating sentiment dataset..." << std::endl;
    SentimentDataset train_data(num_samples, max_len, 42);
    SentimentDataset test_data(num_samples / 5, max_len, 123);

    int64_t vocab_size = train_data.vocab_size();

    // Create model
    auto model = std::make_shared<TransformerClassifier>(
        vocab_size, d_model, n_heads, n_layers, 2, max_len);

    std::cout << "Transformer Classifier created" << std::endl;
    std::cout << "  d_model: " << d_model << ", heads: " << n_heads
              << ", layers: " << n_layers << std::endl;

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        model->to(g_device);
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    // Optimizer
    AdamOptions opts(lr);
    Adam optimizer(model->parameters(), opts);

    // Loss
    CrossEntropyLoss criterion;

    // Training
    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "\n=== Transformer Training ===" << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch: " << batch_size << std::endl;
    std::cout << "Train samples: " << train_data.size()
              << ", Test samples: " << test_data.size() << std::endl;

    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        model->train();

        std::vector<int64_t> indices(train_data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        float epoch_loss = 0.0f;
        int64_t correct = 0;
        int64_t total = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int64_t batch_start = 0; batch_start < (int64_t)train_data.size(); batch_start += batch_size) {
            int64_t batch_end = std::min(batch_start + batch_size, (int64_t)train_data.size());
            int64_t B = batch_end - batch_start;

            // Create batch tensors
            Tensor inputs = at::zeros({B, max_len});  // Pad with 0
            Tensor targets = at::empty({B});

            float* in_ptr = inputs.mutable_data_ptr<float>();
            float* tgt_ptr = targets.mutable_data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int64_t idx = indices[batch_start + i];
                const auto& sample = train_data.get(idx);

                tgt_ptr[i] = static_cast<float>(sample.label);

                int64_t seq_len = std::min((int64_t)sample.tokens.size(), max_len);
                for (int64_t t = 0; t < seq_len; ++t) {
                    in_ptr[i * max_len + t] = static_cast<float>(sample.tokens[t]);
                }
            }

            inputs = to_device(inputs);
            targets = to_device(targets);

            optimizer.zero_grad();
            Tensor logits = model->forward(inputs);  // [B, 2]
            Tensor loss = criterion.forward(logits, targets);

            torch::autograd::backward({loss});
            optimizer.step();

            // Stats
            Tensor loss_cpu = move_to_cpu(loss);
            epoch_loss += loss_cpu.data_ptr<float>()[0] * B;

            Tensor logits_cpu = move_to_cpu(logits);
            Tensor targets_cpu = move_to_cpu(targets);
            const float* log_ptr = logits_cpu.data_ptr<float>();
            const float* tgt_cpu_ptr = targets_cpu.data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int pred = (log_ptr[i * 2 + 1] > log_ptr[i * 2]) ? 1 : 0;
                if (pred == static_cast<int>(tgt_cpu_ptr[i])) {
                    correct++;
                }
                total++;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cout << "Epoch " << epoch << "/" << epochs
                  << " Loss: " << (epoch_loss / train_data.size())
                  << " Acc: " << (100.0f * correct / total) << "%"
                  << " Time: " << elapsed << "ms" << std::endl;

        // Test evaluation
        model->eval();
        int64_t test_correct = 0;

        for (int64_t batch_start = 0; batch_start < (int64_t)test_data.size(); batch_start += batch_size) {
            int64_t batch_end = std::min(batch_start + batch_size, (int64_t)test_data.size());
            int64_t B = batch_end - batch_start;

            Tensor inputs = at::zeros({B, max_len});
            float* in_ptr = inputs.mutable_data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int64_t idx = batch_start + i;
                const auto& sample = test_data.get(idx);

                int64_t seq_len = std::min((int64_t)sample.tokens.size(), max_len);
                for (int64_t t = 0; t < seq_len; ++t) {
                    in_ptr[i * max_len + t] = static_cast<float>(sample.tokens[t]);
                }
            }

            inputs = to_device(inputs);
            Tensor logits = model->forward(inputs);
            Tensor logits_cpu = move_to_cpu(logits);
            const float* log_ptr = logits_cpu.data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int pred = (log_ptr[i * 2 + 1] > log_ptr[i * 2]) ? 1 : 0;
                if (pred == test_data.get(batch_start + i).label) {
                    test_correct++;
                }
            }
        }

        std::cout << "  Test Acc: " << (100.0f * test_correct / test_data.size()) << "%" << std::endl;
    }

    std::cout << "\n=== Transformer Training Complete ===" << std::endl;
    return 0;
}
