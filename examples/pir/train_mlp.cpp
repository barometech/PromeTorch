// ============================================================================
// Simple MLP Language Model Training Example
// ============================================================================
// A basic proof-of-concept showing gradient descent working in PromeTorch
//
// Usage:
//   ./train_mlp <text_file> [options]
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <random>
#include <map>
#include <set>

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

// ============================================================================
// Character-Level Tokenizer
// ============================================================================

class CharTokenizer {
public:
    void build_vocab(const std::string& text) {
        std::set<char> unique_chars(text.begin(), text.end());
        int idx = 0;
        for (char c : unique_chars) {
            char_to_idx_[c] = idx;
            idx_to_char_[idx] = c;
            idx++;
        }
        vocab_size_ = static_cast<int64_t>(unique_chars.size());
        std::cout << "Vocabulary size: " << vocab_size_ << std::endl;
    }

    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> tokens;
        for (char c : text) {
            auto it = char_to_idx_.find(c);
            if (it != char_to_idx_.end()) tokens.push_back(it->second);
        }
        return tokens;
    }

    std::string decode(const std::vector<int64_t>& tokens) const {
        std::string text;
        for (int64_t idx : tokens) {
            auto it = idx_to_char_.find(idx);
            if (it != idx_to_char_.end()) text += it->second;
        }
        return text;
    }

    int64_t vocab_size() const { return vocab_size_; }

private:
    std::map<char, int> char_to_idx_;
    std::map<int, char> idx_to_char_;
    int64_t vocab_size_ = 0;
};

// ============================================================================
// Simple MLP Language Model (uses only autograd-tracked ops)
// ============================================================================

class SimpleLM : public Module {
public:
    SimpleLM(int64_t vocab_size, int64_t emb_dim, int64_t hidden_dim, int64_t context_len)
        : Module("SimpleLM")
        , vocab_size_(vocab_size)
        , emb_dim_(emb_dim)
        , context_len_(context_len)
    {
        // Embedding table
        emb_ = std::make_shared<Embedding>(vocab_size, emb_dim);
        register_module("emb", emb_);

        // MLP layers
        fc1_ = std::make_shared<Linear>(emb_dim * context_len, hidden_dim);
        fc2_ = std::make_shared<Linear>(hidden_dim, hidden_dim);
        fc3_ = std::make_shared<Linear>(hidden_dim, vocab_size);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc3", fc3_);

        std::cout << "SimpleLM initialized with " << count_params() << " parameters" << std::endl;
    }

    int64_t count_params() const {
        int64_t count = 0;
        for (auto* param : const_cast<SimpleLM*>(this)->parameters()) {
            count += param->data().numel();
        }
        return count;
    }

    // Forward: input is [batch, context_len] token indices as float
    // Returns logits [batch, vocab_size]
    Tensor forward(const Tensor& input) override {
        auto sizes = input.sizes().vec();
        int64_t B = sizes[0];

        // Embed all tokens
        Tensor flat = input.reshape({-1});
        Tensor embedded = emb_->forward(flat);
        embedded = embedded.reshape({B, context_len_ * emb_dim_});

        // MLP
        Tensor h = fc1_->forward(embedded);
        h = torch::autograd::relu_autograd(h);
        h = fc2_->forward(h);
        h = torch::autograd::relu_autograd(h);
        Tensor logits = fc3_->forward(h);

        return logits;
    }

    std::pair<Tensor, Tensor> forward_with_loss(const Tensor& input, const Tensor& target) {
        Tensor logits = forward(input);
        Tensor loss = cross_entropy_loss(logits, target);
        return {logits, loss};
    }

private:
    Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
        int64_t B = logits.size(0);
        int64_t V = logits.size(1);

        const float* logits_data = logits.data_ptr<float>();
        const float* targets_data = targets.data_ptr<float>();

        Tensor softmax = at::empty({B, V});
        float* softmax_data = softmax.mutable_data_ptr<float>();

        float total_loss = 0.0f;
        int64_t count = 0;

        for (int64_t i = 0; i < B; ++i) {
            int64_t offset = i * V;
            int64_t target_idx = static_cast<int64_t>(targets_data[i]);

            float max_logit = logits_data[offset];
            for (int64_t v = 1; v < V; ++v) {
                max_logit = std::max(max_logit, logits_data[offset + v]);
            }

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                float exp_val = std::exp(logits_data[offset + v] - max_logit);
                softmax_data[offset + v] = exp_val;
                sum_exp += exp_val;
            }

            for (int64_t v = 0; v < V; ++v) {
                softmax_data[offset + v] /= sum_exp;
            }

            if (target_idx >= 0 && target_idx < V) {
                total_loss -= std::log(softmax_data[offset + target_idx] + 1e-10f);
                count++;
            }
        }

        Tensor loss = at::empty({});
        loss.mutable_data_ptr<float>()[0] = count > 0 ? total_loss / count : 0.0f;

        if (logits.requires_grad()) {
            loss.set_requires_grad(true);
            auto backward_fn = std::make_shared<torch::autograd::CrossEntropyBackward>(
                softmax, targets, -100, V, count
            );
            backward_fn->add_input_metadata(logits);
            auto* meta = torch::autograd::ensure_autograd_meta_impl(loss);
            meta->grad_fn = backward_fn;
            meta->output_nr_ = 0;
            meta->is_leaf_ = false;
        }

        return loss;
    }

    int64_t vocab_size_;
    int64_t emb_dim_;
    int64_t context_len_;
    std::shared_ptr<Embedding> emb_;
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<Linear> fc3_;
};

// ============================================================================
// Data Loader
// ============================================================================

class TextDataLoader {
public:
    TextDataLoader(const std::vector<int64_t>& data, int64_t batch_size, int64_t context_len)
        : data_(data), batch_size_(batch_size), context_len_(context_len)
    {
        std::random_device rd;
        gen_ = std::mt19937(rd());
        dist_ = std::uniform_int_distribution<int64_t>(0, data_.size() - context_len_ - 1);
    }

    std::pair<Tensor, Tensor> get_batch() {
        Tensor inputs = at::empty({batch_size_, context_len_});
        Tensor targets = at::empty({batch_size_});

        float* in_data = inputs.mutable_data_ptr<float>();
        float* tgt_data = targets.mutable_data_ptr<float>();

        for (int64_t b = 0; b < batch_size_; ++b) {
            int64_t start = dist_(gen_);
            for (int64_t t = 0; t < context_len_; ++t) {
                in_data[b * context_len_ + t] = static_cast<float>(data_[start + t]);
            }
            tgt_data[b] = static_cast<float>(data_[start + context_len_]);
        }

        return {inputs, targets};
    }

private:
    std::vector<int64_t> data_;
    int64_t batch_size_;
    int64_t context_len_;
    std::mt19937 gen_;
    std::uniform_int_distribution<int64_t> dist_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string text_file = "";
    int64_t batch_size = 32;
    int64_t context_len = 16;
    int64_t emb_dim = 32;
    int64_t hidden_dim = 128;
    int64_t num_iterations = 500;
    int64_t log_interval = 50;
    float learning_rate = 0.01f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch_size" && i + 1 < argc) batch_size = std::stoll(argv[++i]);
        else if (arg == "--context" && i + 1 < argc) context_len = std::stoll(argv[++i]);
        else if (arg == "--emb" && i + 1 < argc) emb_dim = std::stoll(argv[++i]);
        else if (arg == "--hidden" && i + 1 < argc) hidden_dim = std::stoll(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) num_iterations = std::stoll(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) learning_rate = std::stof(argv[++i]);
        else if (arg == "--log" && i + 1 < argc) log_interval = std::stoll(argv[++i]);
        else if (arg[0] != '-') text_file = arg;
    }

    if (text_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " <text_file> [options]" << std::endl;
        return 1;
    }

    std::ifstream file(text_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << text_file << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    std::cout << "Text length: " << text.size() << " characters" << std::endl;

    CharTokenizer tokenizer;
    tokenizer.build_vocab(text);
    std::vector<int64_t> data = tokenizer.encode(text);

    SimpleLM model(tokenizer.vocab_size(), emb_dim, hidden_dim, context_len);
    TextDataLoader train_loader(data, batch_size, context_len);

    SGDOptions opts(learning_rate);
    SGD optimizer(model.parameters(), opts);

    std::cout << "\n=== Training Simple LM ===" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    float running_loss = 0.0f;
    int loss_count = 0;

    for (int64_t iter = 1; iter <= num_iterations; ++iter) {
        auto [inputs, targets] = train_loader.get_batch();

        optimizer.zero_grad();
        auto [logits, loss] = model.forward_with_loss(inputs, targets);

        float loss_val = loss.data_ptr<float>()[0];
        running_loss += loss_val;
        loss_count++;

        torch::autograd::backward({loss});
        optimizer.step();

        if (iter % log_interval == 0) {
            float avg_loss = running_loss / loss_count;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            std::cout << "Iter " << iter << "/" << num_iterations
                      << " | Loss: " << avg_loss
                      << " | Time: " << elapsed << "s" << std::endl;
            running_loss = 0.0f;
            loss_count = 0;
        }
    }

    std::cout << "\n=== Generating Text ===" << std::endl;

    std::vector<int64_t> context(context_len, 0);
    for (int64_t i = 0; i < context_len && i < static_cast<int64_t>(data.size()); ++i) {
        context[i] = data[i];
    }

    std::string generated = tokenizer.decode(context);
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < 200; ++i) {
        Tensor input = at::empty({1, context_len});
        float* input_data = input.mutable_data_ptr<float>();
        for (int64_t j = 0; j < context_len; ++j) {
            input_data[j] = static_cast<float>(context[j]);
        }

        Tensor logits = model.forward(input);
        const float* logits_data = logits.data_ptr<float>();

        // Apply temperature and sample
        std::vector<float> probs(tokenizer.vocab_size());
        float max_val = *std::max_element(logits_data, logits_data + tokenizer.vocab_size());
        float sum = 0.0f;
        for (int64_t v = 0; v < tokenizer.vocab_size(); ++v) {
            probs[v] = std::exp((logits_data[v] - max_val) / 0.8f);
            sum += probs[v];
        }
        for (auto& p : probs) p /= sum;

        std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
        int64_t next_token = dist(gen);

        generated += tokenizer.decode({next_token});

        for (int64_t j = 0; j < context_len - 1; ++j) {
            context[j] = context[j + 1];
        }
        context[context_len - 1] = next_token;
    }

    std::cout << "Generated:\n" << std::string(40, '-') << std::endl;
    std::cout << generated << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    return 0;
}
