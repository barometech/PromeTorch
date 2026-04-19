// torch/text/vocab.h
// torchtext-compatible Vocab for PromeTorch.
//
// Provides:
//   - Vocab: bidirectional int <-> string mapping with O(1) insert/lookup
//     backed by std::unordered_map + std::vector.
//   - Static factories:
//       * from_iterator(range)              - build vocab from a token iterable
//       * from_counter(counts, min_freq)    - build vocab from {token: count}
//                                              with a min-frequency threshold
//       * from_pretrained_bpe(path)         - load HF/GPT-2 style vocab.json
//   - Special tokens: <unk> <pad> <bos> <eos> (unk always present).
//   - encode(text) -> std::vector<int64_t>: whitespace-split + lookup.
//
// CPU-only, header-only, stdlib-only (LCC / Elbrus friendly).
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace text {

class Vocab {
public:
    // Default special tokens, inserted at fixed positions if flag is set.
    static constexpr const char* kUnk = "<unk>";
    static constexpr const char* kPad = "<pad>";
    static constexpr const char* kBos = "<bos>";
    static constexpr const char* kEos = "<eos>";

    Vocab() { insert_default_specials_(); }

    // Construct from an already ordered list of tokens. Specials are prepended
    // only if not already present.
    explicit Vocab(const std::vector<std::string>& tokens,
                   bool add_specials = true) {
        if (add_specials) insert_default_specials_();
        for (const auto& t : tokens) add_token(t);
    }

    // -- mutation -----------------------------------------------------------
    int64_t add_token(const std::string& token) {
        auto it = stoi_.find(token);
        if (it != stoi_.end()) return it->second;
        int64_t idx = static_cast<int64_t>(itos_.size());
        itos_.push_back(token);
        stoi_.emplace(token, idx);
        return idx;
    }

    void set_default_index(int64_t idx) { default_idx_ = idx; }

    // -- lookup -------------------------------------------------------------
    int64_t operator[](const std::string& token) const {
        auto it = stoi_.find(token);
        if (it != stoi_.end()) return it->second;
        return default_idx_;
    }

    bool contains(const std::string& token) const {
        return stoi_.find(token) != stoi_.end();
    }

    const std::string& lookup_token(int64_t idx) const {
        if (idx < 0 || static_cast<size_t>(idx) >= itos_.size())
            throw std::out_of_range("Vocab: index out of range");
        return itos_[static_cast<size_t>(idx)];
    }

    size_t size() const { return itos_.size(); }
    const std::vector<std::string>& tokens() const { return itos_; }

    int64_t unk_id() const { return default_idx_; }
    int64_t pad_id() const { return id_or_neg_(kPad); }
    int64_t bos_id() const { return id_or_neg_(kBos); }
    int64_t eos_id() const { return id_or_neg_(kEos); }

    // -- encoding / decoding -----------------------------------------------
    // Whitespace-split encode. For subword tokenizers use BPETokenizer etc.
    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> ids;
        std::istringstream iss(text);
        std::string tok;
        while (iss >> tok) ids.push_back((*this)[tok]);
        return ids;
    }

    std::vector<int64_t> encode(const std::vector<std::string>& tokens) const {
        std::vector<int64_t> ids;
        ids.reserve(tokens.size());
        for (const auto& t : tokens) ids.push_back((*this)[t]);
        return ids;
    }

    std::string decode(const std::vector<int64_t>& ids,
                       const std::string& sep = " ") const {
        std::string out;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) out += sep;
            out += lookup_token(ids[i]);
        }
        return out;
    }

    // -- factories ----------------------------------------------------------
    template <typename Range>
    static Vocab from_iterator(const Range& range, bool add_specials = true) {
        Vocab v;
        if (!add_specials) v = Vocab::without_specials_();
        for (const auto& tok : range) v.add_token(tok);
        return v;
    }

    static Vocab from_counter(
            const std::unordered_map<std::string, int64_t>& counts,
            int64_t min_freq = 1,
            bool add_specials = true) {
        // Sort by (-count, token) for deterministic ordering.
        std::vector<std::pair<std::string, int64_t>> sorted;
        sorted.reserve(counts.size());
        for (const auto& kv : counts)
            if (kv.second >= min_freq) sorted.push_back(kv);
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
            });
        Vocab v;
        if (!add_specials) v = Vocab::without_specials_();
        for (const auto& kv : sorted) v.add_token(kv.first);
        return v;
    }

    // Parse a very small HF/GPT-2 vocab.json: a flat object
    //   {"tok1": 0, "tok2": 1, ...}
    // No escapes besides \" and \\ are handled — enough for BPE vocabs that
    // use the GPT-2 byte alphabet.
    static Vocab from_pretrained_bpe(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Vocab: cannot open " + path);
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        std::vector<std::pair<std::string, int64_t>> entries;
        size_t i = 0;
        const size_t n = text.size();
        while (i < n) {
            while (i < n && text[i] != '"') ++i;
            if (i >= n) break;
            ++i;  // opening quote of key
            std::string key;
            while (i < n && text[i] != '"') {
                if (text[i] == '\\' && i + 1 < n) {
                    key.push_back(text[i + 1]);
                    i += 2;
                } else {
                    key.push_back(text[i++]);
                }
            }
            if (i < n) ++i;  // closing quote
            while (i < n && text[i] != ':') ++i;
            if (i >= n) break;
            ++i;
            while (i < n && (text[i] == ' ' || text[i] == '\t')) ++i;
            std::string num;
            while (i < n && (std::isdigit(static_cast<unsigned char>(text[i])) ||
                             text[i] == '-'))
                num.push_back(text[i++]);
            if (num.empty()) continue;
            int64_t idx = std::strtoll(num.c_str(), nullptr, 10);
            entries.emplace_back(std::move(key), idx);
        }
        std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        Vocab v = Vocab::without_specials_();
        // Use indices exactly as listed in the file. Pad with <unkX> if there
        // are gaps so that id == file-index.
        for (const auto& kv : entries) {
            while (static_cast<int64_t>(v.size()) < kv.second)
                v.add_token("<unused" + std::to_string(v.size()) + ">");
            v.add_token(kv.first);
        }
        // Make sure <unk> exists for default_idx_.
        if (!v.contains(kUnk)) v.default_idx_ = v.add_token(kUnk);
        else v.default_idx_ = v[kUnk];
        return v;
    }

private:
    std::vector<std::string> itos_;
    std::unordered_map<std::string, int64_t> stoi_;
    int64_t default_idx_ = 0;

    void insert_default_specials_() {
        default_idx_ = add_token(kUnk);
        add_token(kPad);
        add_token(kBos);
        add_token(kEos);
    }

    static Vocab without_specials_() {
        Vocab v;
        v.itos_.clear();
        v.stoi_.clear();
        v.default_idx_ = -1;
        return v;
    }

    int64_t id_or_neg_(const std::string& s) const {
        auto it = stoi_.find(s);
        return it == stoi_.end() ? -1 : it->second;
    }
};

}  // namespace text
}  // namespace torch
