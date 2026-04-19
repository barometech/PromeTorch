// ============================================================================
// Transformer Training Example - Sequence Classification (Sentiment)
// ============================================================================
// Simple Transformer encoder for text classification on an IMDb-style synthetic
// dataset. Demonstrates real learning on a non-trivial binary task.
//
// Bugs fixed vs. previous version:
//   1. Off-by-one in vocab construction: `vocab_[w] = vocab_.size()` reads the
//      size AFTER operator[] has already inserted the key, so every word was
//      assigned an index one too large, and the last word reached
//      `num_embeddings` (OUT OF RANGE). Fixed by caching the size first.
//   2. Explicit pad/unk handling and safe fallback for unknown words in the
//      real-text path.
//   3. Robust dataset loader that falls back to synthetic data if SST-2/IMDb
//      files are not present (stdlib TSV parsing — no external libs).
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
#include <algorithm>
#include <numeric>
#include <cctype>

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

        // Token embedding (padding_idx=0 keeps pad vectors at zero)
        token_embed = std::make_shared<Embedding>(vocab_size, d_model, /*padding_idx=*/0);

        // Positional encoding (learnable)
        pos_embed = std::make_shared<Embedding>(max_len, d_model);

        // Transformer encoder layers (batch_first=true since we feed [B, T, D])
        for (int64_t i = 0; i < n_layers; ++i) {
            auto layer = std::make_shared<TransformerEncoderLayer>(
                d_model, n_heads, d_model * 4, 0.1,
                "relu", 1e-5, /*batch_first=*/true, /*norm_first=*/false);
            encoder_layers.push_back(layer);
            register_module("encoder_" + std::to_string(i), layer);
        }

        // Classification head
        fc = std::make_shared<Linear>(d_model, num_classes);
        dropout = std::make_shared<Dropout>(0.1f);

        register_module("token_embed", token_embed);
        register_module("pos_embed", pos_embed);
        register_module("fc", fc);
        register_module("dropout", dropout);
    }

    Tensor forward(const Tensor& x) override {
        // x: [B, T] float-encoded token indices (framework-wide convention:
        // Embedding reads indices from data_ptr<float>() and static_casts to int)
        auto sizes = x.sizes();
        int64_t B = sizes[0];
        int64_t T = sizes[1];

        // Get embeddings
        Tensor tok_emb = token_embed->forward(x);  // [B, T, d_model]

        // Create position indices [0, 1, ..., T-1] (on CPU, then move)
        Tensor positions = at::empty({1, T});
        float* pos_ptr = positions.mutable_data_ptr<float>();
        for (int64_t t = 0; t < T; ++t) {
            pos_ptr[t] = static_cast<float>(t);
        }
        positions = to_device(positions);

        Tensor pos_emb = pos_embed->forward(positions);  // [1, T, d_model]

        // Combine embeddings (broadcasts over batch)
        Tensor h = tok_emb.add(pos_emb);
        h = dropout->forward(h);

        // Pass through encoder layers
        for (auto& layer : encoder_layers) {
            h = layer->forward(h);
        }

        // Global average pooling over sequence: [B, T, D] -> [B, D]
        h = h.mean(1);

        // Classification head
        return fc->forward(h);
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
// Text helpers
// ============================================================================

static std::string to_lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Whitespace + punctuation tokenizer
static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc)) {
            cur.push_back(static_cast<char>(std::tolower(uc)));
        } else {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

struct TextSample {
    std::vector<int64_t> tokens;
    int64_t label;
};

// ============================================================================
// Synthetic sentiment dataset (fallback)
// ============================================================================
// FIX: the original version had an off-by-one bug in vocab construction:
//     vocab_[w] = vocab_.size();
// operator[] inserts the key first (size becomes N+1), then size() returns N+1
// and we assign N+1 -> out-of-range at the last word. The fix is to cache
// `vocab_.size()` before the insertion.

static const std::vector<std::string> POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "happy", "joy", "best", "beautiful", "perfect", "awesome",
    "brilliant", "delightful", "superb", "outstanding"
};
static const std::vector<std::string> NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "worst", "ugly",
    "sad", "angry", "poor", "wrong", "fail", "disaster", "boring",
    "dreadful", "miserable", "pathetic"
};
static const std::vector<std::string> NEUTRAL_WORDS = {
    "the", "a", "is", "it", "this", "that", "movie", "book", "food",
    "day", "time", "thing", "place", "very", "really", "so", "and",
    "i", "was", "felt", "seemed", "quite"
};

class SentimentDataset {
public:
    SentimentDataset(int64_t num_samples, int64_t max_len, uint32_t seed = 42)
        : max_len_(max_len) {

        auto add_word = [&](const std::string& w) {
            if (vocab_.find(w) == vocab_.end()) {
                int64_t next_idx = static_cast<int64_t>(vocab_.size());  // cache before insert
                vocab_[w] = next_idx;
            }
        };

        // Reserve special tokens first: pad=0, unk=1
        add_word("<pad>");
        add_word("<unk>");
        for (const auto& w : POSITIVE_WORDS) add_word(w);
        for (const auto& w : NEGATIVE_WORDS) add_word(w);
        for (const auto& w : NEUTRAL_WORDS) add_word(w);

        vocab_size_ = static_cast<int64_t>(vocab_.size());

        // Sanity check: every assigned index must be strictly less than vocab_size_.
        for (const auto& kv : vocab_) {
            if (kv.second < 0 || kv.second >= vocab_size_) {
                throw std::runtime_error(
                    "SentimentDataset: vocab index " + std::to_string(kv.second) +
                    " out of range for size " + std::to_string(vocab_size_));
            }
        }

        std::cout << "Synthetic vocab size: " << vocab_size_
                  << " (max idx = " << (vocab_size_ - 1) << ")" << std::endl;

        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> len_dist(5, static_cast<int>(max_len));
        std::uniform_int_distribution<int> pos_dist(0, static_cast<int>(POSITIVE_WORDS.size()) - 1);
        std::uniform_int_distribution<int> neg_dist(0, static_cast<int>(NEGATIVE_WORDS.size()) - 1);
        std::uniform_int_distribution<int> neu_dist(0, static_cast<int>(NEUTRAL_WORDS.size()) - 1);
        std::uniform_real_distribution<float> coin(0.f, 1.f);

        for (int64_t i = 0; i < num_samples; ++i) {
            TextSample sample;
            sample.label = (i % 2);  // Alternate pos/neg
            int64_t sent_len = len_dist(gen);
            sample.tokens.resize(sent_len);

            for (int64_t t = 0; t < sent_len; ++t) {
                float r = coin(gen);
                std::string word;
                if (r < 0.3f) {
                    word = (sample.label == 1)
                        ? POSITIVE_WORDS[pos_dist(gen)]
                        : NEGATIVE_WORDS[neg_dist(gen)];
                } else {
                    word = NEUTRAL_WORDS[neu_dist(gen)];
                }
                sample.tokens[t] = vocab_[word];
            }
            samples_.push_back(std::move(sample));
        }
    }

    const TextSample& get(int64_t idx) const { return samples_[idx]; }
    int64_t size() const { return static_cast<int64_t>(samples_.size()); }
    int64_t vocab_size() const { return vocab_size_; }
    int64_t max_len() const { return max_len_; }

private:
    std::unordered_map<std::string, int64_t> vocab_;
    std::vector<TextSample> samples_;
    int64_t vocab_size_ = 0;
    int64_t max_len_;
};

// ============================================================================
// Real-text dataset loader (SST-2 / IMDb-style TSV)
// ============================================================================
// Expected format (per line):  "<label>\t<text>" or  "<text>\t<label>"
// where label is "0"/"1" or "neg"/"pos". Blank/malformed lines are skipped.
// Vocab is built from the top-K most frequent training tokens; reserved:
//   0 = <pad>, 1 = <unk>

struct RealDataset {
    std::vector<TextSample> train;
    std::vector<TextSample> val;
    std::unordered_map<std::string, int64_t> vocab;
    int64_t vocab_size = 0;
    int64_t max_len = 0;
};

static bool parse_line(const std::string& line, std::string& text_out, int& label_out) {
    // Try tab-separated forms. Accept either "<label>\t<text>" or "<text>\t<label>".
    auto tab = line.find('\t');
    if (tab == std::string::npos) return false;
    std::string a = line.substr(0, tab);
    std::string b = line.substr(tab + 1);
    auto is_label = [](const std::string& s, int& out) {
        std::string t = to_lower(s);
        // strip whitespace
        size_t start = t.find_first_not_of(" \t\r\n");
        size_t end = t.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return false;
        t = t.substr(start, end - start + 1);
        if (t == "0" || t == "neg" || t == "negative") { out = 0; return true; }
        if (t == "1" || t == "pos" || t == "positive") { out = 1; return true; }
        return false;
    };
    int lbl = -1;
    if (is_label(a, lbl)) { label_out = lbl; text_out = b; return true; }
    if (is_label(b, lbl)) { label_out = lbl; text_out = a; return true; }
    return false;
}

static std::vector<std::pair<std::string, int>> load_tsv(const std::string& path) {
    std::vector<std::pair<std::string, int>> out;
    std::ifstream f(path);
    if (!f.good()) return out;
    std::string line;
    // Skip potential header (first line containing non-numeric label cell)
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            std::string txt; int lbl;
            if (!parse_line(line, txt, lbl)) continue;  // header
            out.emplace_back(std::move(txt), lbl);
        } else {
            std::string txt; int lbl;
            if (parse_line(line, txt, lbl)) out.emplace_back(std::move(txt), lbl);
        }
    }
    return out;
}

static bool load_real_dataset(const std::string& data_dir,
                              int64_t max_len,
                              int64_t max_vocab,
                              RealDataset& out) {
    // Try a few common filename patterns
    std::vector<std::string> train_candidates = {
        data_dir + "/train.tsv",
        data_dir + "/sst2_train.tsv",
        data_dir + "/imdb_train.tsv"
    };
    std::vector<std::string> val_candidates = {
        data_dir + "/dev.tsv",
        data_dir + "/val.tsv",
        data_dir + "/test.tsv",
        data_dir + "/sst2_dev.tsv",
        data_dir + "/imdb_test.tsv"
    };

    std::vector<std::pair<std::string,int>> train_raw, val_raw;
    for (const auto& p : train_candidates) {
        train_raw = load_tsv(p);
        if (!train_raw.empty()) { std::cout << "Loaded train from " << p
                                            << " (" << train_raw.size() << " rows)\n"; break; }
    }
    if (train_raw.empty()) return false;
    for (const auto& p : val_candidates) {
        val_raw = load_tsv(p);
        if (!val_raw.empty()) { std::cout << "Loaded val from   " << p
                                          << " (" << val_raw.size() << " rows)\n"; break; }
    }
    if (val_raw.empty()) {
        // Carve off last 10% as val
        int64_t v = std::max<int64_t>(1, static_cast<int64_t>(train_raw.size()) / 10);
        val_raw.assign(train_raw.end() - v, train_raw.end());
        train_raw.erase(train_raw.end() - v, train_raw.end());
        std::cout << "No val file; using last 10% of train as val ("
                  << val_raw.size() << " rows)\n";
    }

    // Build vocab from training tokens, keep top-K by frequency
    std::unordered_map<std::string, int64_t> freq;
    std::vector<std::vector<std::string>> train_toks(train_raw.size());
    for (size_t i = 0; i < train_raw.size(); ++i) {
        train_toks[i] = tokenize(train_raw[i].first);
        for (const auto& t : train_toks[i]) freq[t]++;
    }
    std::vector<std::pair<std::string, int64_t>> freq_vec(freq.begin(), freq.end());
    std::sort(freq_vec.begin(), freq_vec.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    out.vocab.clear();
    out.vocab["<pad>"] = 0;
    out.vocab["<unk>"] = 1;
    int64_t limit = std::min<int64_t>(max_vocab - 2, static_cast<int64_t>(freq_vec.size()));
    for (int64_t i = 0; i < limit; ++i) {
        int64_t next_idx = static_cast<int64_t>(out.vocab.size());  // cache before insert
        out.vocab[freq_vec[i].first] = next_idx;
    }
    out.vocab_size = static_cast<int64_t>(out.vocab.size());
    out.max_len = max_len;

    auto encode = [&](const std::vector<std::string>& toks, std::vector<int64_t>& ids) {
        ids.clear();
        ids.reserve(std::min<size_t>(toks.size(), static_cast<size_t>(max_len)));
        for (const auto& t : toks) {
            if (static_cast<int64_t>(ids.size()) >= max_len) break;
            auto it = out.vocab.find(t);
            ids.push_back(it == out.vocab.end() ? 1 /*unk*/ : it->second);
        }
    };

    out.train.resize(train_raw.size());
    for (size_t i = 0; i < train_raw.size(); ++i) {
        encode(train_toks[i], out.train[i].tokens);
        out.train[i].label = train_raw[i].second;
    }
    out.val.resize(val_raw.size());
    for (size_t i = 0; i < val_raw.size(); ++i) {
        auto toks = tokenize(val_raw[i].first);
        encode(toks, out.val[i].tokens);
        out.val[i].label = val_raw[i].second;
    }

    std::cout << "Real vocab size: " << out.vocab_size
              << " (max idx = " << (out.vocab_size - 1) << ")\n";
    return true;
}

// ============================================================================
// Batch builder — fills [B, T] float tensor with token ids (pad = 0)
// ============================================================================

static void fill_batch(const std::vector<TextSample>& samples,
                       const std::vector<int64_t>& order,
                       int64_t batch_start, int64_t B, int64_t max_len,
                       Tensor& inputs, Tensor& targets, int64_t vocab_size) {
    float* in_ptr = inputs.mutable_data_ptr<float>();
    float* tgt_ptr = targets.mutable_data_ptr<float>();
    // zeros() already gave us zero-padding; just overwrite valid positions.
    for (int64_t i = 0; i < B; ++i) {
        int64_t idx = order[batch_start + i];
        const auto& s = samples[idx];
        tgt_ptr[i] = static_cast<float>(s.label);
        int64_t L = std::min<int64_t>(static_cast<int64_t>(s.tokens.size()), max_len);
        for (int64_t t = 0; t < L; ++t) {
            int64_t tok = s.tokens[t];
            if (tok < 0 || tok >= vocab_size) tok = 1;  // fall back to <unk>
            in_ptr[i * max_len + t] = static_cast<float>(tok);
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string device_str = "cpu";
    std::string data_dir = "";        // e.g. "data/sst2"
    int64_t d_model = 64;
    int64_t n_heads = 4;
    int64_t n_layers = 2;
    int64_t batch_size = 32;
    int64_t max_len = 64;
    int64_t num_samples = 5000;
    int64_t max_vocab = 10000;
    int64_t epochs = 5;
    float lr = 0.001f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--data_dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--d_model" && i + 1 < argc) d_model = std::stoll(argv[++i]);
        else if (arg == "--n_heads" && i + 1 < argc) n_heads = std::stoll(argv[++i]);
        else if (arg == "--n_layers" && i + 1 < argc) n_layers = std::stoll(argv[++i]);
        else if (arg == "--batch_size" && i + 1 < argc) batch_size = std::stoll(argv[++i]);
        else if (arg == "--max_len" && i + 1 < argc) max_len = std::stoll(argv[++i]);
        else if (arg == "--num_samples" && i + 1 < argc) num_samples = std::stoll(argv[++i]);
        else if (arg == "--max_vocab" && i + 1 < argc) max_vocab = std::stoll(argv[++i]);
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoll(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
    }

    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Using CUDA" << std::endl;
#else
        std::cerr << "CUDA not built; falling back to CPU" << std::endl;
#endif
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    // -------- Load data --------
    std::vector<TextSample> train_samples, val_samples;
    int64_t vocab_size = 0;

    RealDataset real;
    bool use_real = !data_dir.empty() && load_real_dataset(data_dir, max_len, max_vocab, real);
    if (use_real) {
        train_samples = std::move(real.train);
        val_samples   = std::move(real.val);
        vocab_size    = real.vocab_size;
        std::cout << "Dataset: REAL (" << train_samples.size() << " train / "
                  << val_samples.size() << " val)\n";
    } else {
        if (!data_dir.empty()) {
            std::cout << "WARNING: real dataset not found in '" << data_dir
                      << "', falling back to synthetic.\n";
        }
        SentimentDataset tr(num_samples, max_len, 42);
        SentimentDataset va(num_samples / 5, max_len, 123);
        // Shared vocab: regenerate val with same seed mapping isn't needed because
        // both constructors insert identical special + word lists in the same order.
        vocab_size = tr.vocab_size();
        for (int64_t i = 0; i < tr.size(); ++i) train_samples.push_back(tr.get(i));
        for (int64_t i = 0; i < va.size(); ++i) val_samples.push_back(va.get(i));
        std::cout << "Dataset: SYNTHETIC (" << train_samples.size() << " train / "
                  << val_samples.size() << " val)" << std::endl;
    }

    // Safety: verify no sample references an OOB id
    for (const auto& s : train_samples) {
        for (auto t : s.tokens) {
            if (t < 0 || t >= vocab_size) {
                throw std::runtime_error(
                    "train sample has OOB token " + std::to_string(t) +
                    " vs vocab " + std::to_string(vocab_size));
            }
        }
    }

    // -------- Model --------
    auto model = std::make_shared<TransformerClassifier>(
        vocab_size, d_model, n_heads, n_layers, 2, max_len);

    std::cout << "TransformerClassifier: d_model=" << d_model
              << " heads=" << n_heads << " layers=" << n_layers
              << " vocab=" << vocab_size << " max_len=" << max_len << std::endl;

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        model->to(g_device);
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    AdamOptions opts(lr);
    Adam optimizer(model->parameters(), opts);
    CrossEntropyLoss criterion;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "\n=== Training ===\n";
    std::cout << "Epochs=" << epochs << " batch=" << batch_size
              << " lr=" << lr << "\n";

    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        model->train();

        std::vector<int64_t> indices(train_samples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        float epoch_loss = 0.f;
        int64_t correct = 0, total = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int64_t bs = 0; bs < (int64_t)train_samples.size(); bs += batch_size) {
            int64_t be = std::min<int64_t>(bs + batch_size, train_samples.size());
            int64_t B = be - bs;

            Tensor inputs  = at::zeros({B, max_len});  // float, pad=0
            Tensor targets = at::empty({B});
            fill_batch(train_samples, indices, bs, B, max_len, inputs, targets, vocab_size);

            inputs  = to_device(inputs);
            targets = to_device(targets);

            optimizer.zero_grad();
            Tensor logits = model->forward(inputs);
            Tensor loss   = criterion.forward(logits, targets);
            torch::autograd::backward({loss});
            optimizer.step();

            Tensor loss_cpu = move_to_cpu(loss);
            epoch_loss += loss_cpu.data_ptr<float>()[0] * B;

            Tensor logits_cpu  = move_to_cpu(logits);
            Tensor targets_cpu = move_to_cpu(targets);
            const float* lp = logits_cpu.data_ptr<float>();
            const float* tp = targets_cpu.data_ptr<float>();
            for (int64_t i = 0; i < B; ++i) {
                int pred = (lp[i * 2 + 1] > lp[i * 2]) ? 1 : 0;
                if (pred == static_cast<int>(tp[i])) correct++;
                total++;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::cout << "Epoch " << epoch << "/" << epochs
                  << "  train_loss=" << (epoch_loss / train_samples.size())
                  << "  train_acc=" << (100.f * correct / total) << "%"
                  << "  time=" << ms << "ms";

        // ----- Validation -----
        model->eval();
        int64_t vcorrect = 0;
        std::vector<int64_t> vorder(val_samples.size());
        std::iota(vorder.begin(), vorder.end(), 0);
        for (int64_t bs = 0; bs < (int64_t)val_samples.size(); bs += batch_size) {
            int64_t be = std::min<int64_t>(bs + batch_size, val_samples.size());
            int64_t B = be - bs;
            Tensor inputs  = at::zeros({B, max_len});
            Tensor targets = at::empty({B});
            fill_batch(val_samples, vorder, bs, B, max_len, inputs, targets, vocab_size);
            inputs = to_device(inputs);
            Tensor logits = model->forward(inputs);
            Tensor logits_cpu = move_to_cpu(logits);
            const float* lp = logits_cpu.data_ptr<float>();
            for (int64_t i = 0; i < B; ++i) {
                int pred = (lp[i * 2 + 1] > lp[i * 2]) ? 1 : 0;
                if (pred == val_samples[vorder[bs + i]].label) vcorrect++;
            }
        }
        std::cout << "  val_acc=" << (100.f * vcorrect / val_samples.size()) << "%\n";
    }

    std::cout << "\n=== Done ===\n";
    return 0;
}
