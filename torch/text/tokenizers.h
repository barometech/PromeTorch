// torch/text/tokenizers.h
// torchtext-compatible tokenizers for PromeTorch.
//
// Interface:
//   ITokenizer — abstract base: encode(text)->ids, decode(ids)->text,
//                tokenize(text)->tokens (string pieces).
//
// Implementations:
//   BasicTokenizer     — whitespace + punctuation split, optional lowercase.
//   BPETokenizer       — byte-level BPE a la GPT-2. Loads merges.txt
//                        (pairs "a b" one per line) and a vocab.json
//                        (via torch::text::Vocab::from_pretrained_bpe).
//                        Greedy rank-based merge identical in semantics
//                        to Hugging Face tokenizers.
//   WordPieceTokenizer — BERT-style greedy longest-match with '##' prefix
//                        for continuation pieces.
//   CharTokenizer      — one token per Unicode codepoint (UTF-8 decoded).
//
// CPU-only, header-only, stdlib-only. Designed to compile cleanly under
// Elbrus LCC — no external deps, no exceptions across module boundaries.
#pragma once

#include "torch/text/vocab.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace text {

// ===========================================================================
// ITokenizer — common interface
// ===========================================================================
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<std::string> tokenize(const std::string& text) const = 0;
    virtual std::vector<int64_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int64_t>& ids) const = 0;
};

// ===========================================================================
// BasicTokenizer — whitespace + punctuation, optional lowercasing.
// ===========================================================================
class BasicTokenizer : public ITokenizer {
public:
    BasicTokenizer(Vocab vocab, bool lowercase = true, bool split_on_punct = true)
        : vocab_(std::move(vocab)), lowercase_(lowercase),
          split_on_punct_(split_on_punct) {}

    std::vector<std::string> tokenize(const std::string& text) const override {
        std::vector<std::string> out;
        std::string cur;
        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (std::isspace(c)) {
                if (!cur.empty()) { out.push_back(std::move(cur)); cur.clear(); }
                continue;
            }
            char ch = lowercase_
                ? static_cast<char>(std::tolower(c))
                : static_cast<char>(c);
            if (split_on_punct_ && is_punct_(c)) {
                if (!cur.empty()) { out.push_back(std::move(cur)); cur.clear(); }
                out.emplace_back(1, ch);
            } else {
                cur.push_back(ch);
            }
        }
        if (!cur.empty()) out.push_back(std::move(cur));
        return out;
    }

    std::vector<int64_t> encode(const std::string& text) const override {
        return vocab_.encode(tokenize(text));
    }

    std::string decode(const std::vector<int64_t>& ids) const override {
        return vocab_.decode(ids);
    }

    const Vocab& vocab() const { return vocab_; }

private:
    Vocab vocab_;
    bool lowercase_;
    bool split_on_punct_;

    static bool is_punct_(unsigned char c) {
        return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
               (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
    }
};

// ===========================================================================
// BPETokenizer — byte-level BPE (GPT-2 / HF compatible, greedy rank merge).
// ===========================================================================
class BPETokenizer : public ITokenizer {
public:
    // Construct from already-loaded vocab + merge list (low index = high prio).
    BPETokenizer(Vocab vocab,
                 std::vector<std::pair<std::string, std::string>> merges)
        : vocab_(std::move(vocab)) {
        for (size_t i = 0; i < merges.size(); ++i)
            merge_ranks_[pair_key_(merges[i].first, merges[i].second)] =
                static_cast<int64_t>(i);
    }

    // Convenience: load both files from disk.
    static BPETokenizer from_files(const std::string& vocab_json,
                                   const std::string& merges_txt) {
        Vocab v = Vocab::from_pretrained_bpe(vocab_json);
        auto merges = load_merges(merges_txt);
        return BPETokenizer(std::move(v), std::move(merges));
    }

    // Parse merges.txt: skip blank lines and lines starting with '#'.
    // Each remaining line contains exactly two whitespace-separated pieces.
    static std::vector<std::pair<std::string, std::string>> load_merges(
            const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("BPE: cannot open " + path);
        std::vector<std::pair<std::string, std::string>> out;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string a, b;
            if (iss >> a >> b) out.emplace_back(std::move(a), std::move(b));
        }
        return out;
    }

    std::vector<std::string> tokenize(const std::string& text) const override {
        std::vector<std::string> out;
        // Pre-tokenize on whitespace; each word is BPE'd independently.
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            auto pieces = bpe_(word);
            out.insert(out.end(),
                       std::make_move_iterator(pieces.begin()),
                       std::make_move_iterator(pieces.end()));
        }
        return out;
    }

    std::vector<int64_t> encode(const std::string& text) const override {
        return vocab_.encode(tokenize(text));
    }

    std::string decode(const std::vector<int64_t>& ids) const override {
        // Simple concat — caller decides whether to strip GPT-2 byte alphabet.
        std::string out;
        for (int64_t id : ids) out += vocab_.lookup_token(id);
        return out;
    }

    const Vocab& vocab() const { return vocab_; }

private:
    Vocab vocab_;
    std::unordered_map<std::string, int64_t> merge_ranks_;

    static std::string pair_key_(const std::string& a, const std::string& b) {
        // 0x01 is not a valid UTF-8 continuation, safe as a separator here.
        std::string k;
        k.reserve(a.size() + b.size() + 1);
        k.append(a); k.push_back('\x01'); k.append(b);
        return k;
    }

    // Greedy BPE on a single pre-tokenized word. Start with character-level
    // split; repeatedly merge the lowest-rank adjacent pair until none apply.
    std::vector<std::string> bpe_(const std::string& word) const {
        std::vector<std::string> pieces;
        pieces.reserve(word.size());
        for (char c : word) pieces.emplace_back(1, c);
        if (pieces.size() < 2) return pieces;

        while (true) {
            int64_t best_rank = std::numeric_limits<int64_t>::max();
            size_t best_i = 0;
            bool found = false;
            for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                auto it = merge_ranks_.find(pair_key_(pieces[i], pieces[i + 1]));
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                    found = true;
                }
            }
            if (!found) break;
            pieces[best_i] = pieces[best_i] + pieces[best_i + 1];
            pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
            if (pieces.size() < 2) break;
        }
        return pieces;
    }
};

// ===========================================================================
// WordPieceTokenizer — BERT-style with '##' continuation prefix.
// ===========================================================================
class WordPieceTokenizer : public ITokenizer {
public:
    WordPieceTokenizer(Vocab vocab,
                       std::string unk_token = "<unk>",
                       int max_input_chars_per_word = 100,
                       std::string continuation_prefix = "##")
        : vocab_(std::move(vocab)), unk_(std::move(unk_token)),
          max_chars_(max_input_chars_per_word),
          cont_(std::move(continuation_prefix)) {}

    std::vector<std::string> tokenize(const std::string& text) const override {
        std::vector<std::string> out;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            if (static_cast<int>(word.size()) > max_chars_) {
                out.push_back(unk_);
                continue;
            }
            bool bad = false;
            std::vector<std::string> sub;
            size_t start = 0;
            while (start < word.size()) {
                size_t end = word.size();
                std::string found;
                while (end > start) {
                    std::string piece = word.substr(start, end - start);
                    if (start > 0) piece = cont_ + piece;
                    if (vocab_.contains(piece)) { found = piece; break; }
                    --end;
                }
                if (found.empty()) { bad = true; break; }
                sub.push_back(std::move(found));
                start = end;
            }
            if (bad) out.push_back(unk_);
            else out.insert(out.end(),
                            std::make_move_iterator(sub.begin()),
                            std::make_move_iterator(sub.end()));
        }
        return out;
    }

    std::vector<int64_t> encode(const std::string& text) const override {
        return vocab_.encode(tokenize(text));
    }

    std::string decode(const std::vector<int64_t>& ids) const override {
        // Merge continuation pieces: "un" + "##afford" + "##able" -> "unaffordable".
        std::string out;
        for (size_t i = 0; i < ids.size(); ++i) {
            const std::string& t = vocab_.lookup_token(ids[i]);
            if (t.rfind(cont_, 0) == 0) out.append(t.substr(cont_.size()));
            else { if (!out.empty()) out.push_back(' '); out.append(t); }
        }
        return out;
    }

    const Vocab& vocab() const { return vocab_; }

private:
    Vocab vocab_;
    std::string unk_;
    int max_chars_;
    std::string cont_;
};

// ===========================================================================
// CharTokenizer — Unicode codepoint level.
// ===========================================================================
class CharTokenizer : public ITokenizer {
public:
    explicit CharTokenizer(Vocab vocab) : vocab_(std::move(vocab)) {}

    std::vector<std::string> tokenize(const std::string& text) const override {
        std::vector<std::string> out;
        size_t i = 0;
        while (i < text.size()) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            size_t len = 1;
            if ((c & 0x80) == 0)        len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            if (i + len > text.size()) len = text.size() - i;
            out.emplace_back(text.substr(i, len));
            i += len;
        }
        return out;
    }

    std::vector<int64_t> encode(const std::string& text) const override {
        return vocab_.encode(tokenize(text));
    }

    std::string decode(const std::vector<int64_t>& ids) const override {
        std::string out;
        for (int64_t id : ids) out += vocab_.lookup_token(id);
        return out;
    }

    const Vocab& vocab() const { return vocab_; }

private:
    Vocab vocab_;
};

}  // namespace text
}  // namespace torch
