// torch/text/text.h
// torchtext-compatible aggregator for PromeTorch.
//
// Bundles:
//   - torch/text/vocab.h       (Vocab, special tokens, factories)
//   - torch/text/tokenizers.h  (BasicTokenizer, BPETokenizer,
//                               WordPieceTokenizer, CharTokenizer)
//
// And provides in this header:
//   * Datasets:
//       TextFileDataset  - one example per line of a text file
//       CSVDataset       - comma-separated, optional header row
//       JSONLDataset     - one JSON object per line, flat key->string extraction
//   * Collators:
//       pad_sequence(batch, padding_value, batch_first)
//       pack_padded_sequence helpers (lengths, mask)
//
// CPU-only, header-only, stdlib-only — compiles under Elbrus LCC.
#pragma once

#include "torch/text/tokenizers.h"
#include "torch/text/vocab.h"

#include "torch/data/dataset.h"
#include "aten/src/ATen/ATen.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace text {

using at::Tensor;
using torch::data::Dataset;
using torch::data::Example;

// ===========================================================================
// TextFileDataset — one line per sample. Returns a std::string as the
// data field; targets are empty (Tensor()) so it composes with MapDataset.
// ===========================================================================
class TextFileDataset : public Dataset<std::string, Tensor> {
public:
    explicit TextFileDataset(const std::string& path,
                             bool skip_empty_lines = true) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("TextFileDataset: cannot open " + path);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (skip_empty_lines && line.empty()) continue;
            lines_.push_back(std::move(line));
        }
    }

    ExampleType get(size_t index) override {
        if (index >= lines_.size())
            throw std::out_of_range("TextFileDataset: index out of range");
        return ExampleType(lines_[index], Tensor());
    }

    size_t size() const override { return lines_.size(); }

    const std::vector<std::string>& lines() const { return lines_; }

private:
    std::vector<std::string> lines_;
};

// ===========================================================================
// CSVDataset — minimal CSV reader. Each row -> vector<string>. The caller
// picks which columns are inputs/labels via column indices.
// No quoted-field support (stdlib-only, Elbrus LCC friendly) — adequate for
// well-formed TSV/CSV produced by standard dataset tooling.
// ===========================================================================
class CSVDataset : public Dataset<std::vector<std::string>, Tensor> {
public:
    explicit CSVDataset(const std::string& path,
                        char delim = ',',
                        bool has_header = true) : delim_(delim) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("CSVDataset: cannot open " + path);
        std::string line;
        bool first = true;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            auto row = split_(line);
            if (first && has_header) { header_ = std::move(row); first = false; continue; }
            first = false;
            rows_.push_back(std::move(row));
        }
    }

    ExampleType get(size_t index) override {
        if (index >= rows_.size())
            throw std::out_of_range("CSVDataset: index out of range");
        return ExampleType(rows_[index], Tensor());
    }

    size_t size() const override { return rows_.size(); }

    const std::vector<std::string>& header() const { return header_; }
    const std::vector<std::vector<std::string>>& rows() const { return rows_; }

private:
    char delim_;
    std::vector<std::string> header_;
    std::vector<std::vector<std::string>> rows_;

    std::vector<std::string> split_(const std::string& line) const {
        std::vector<std::string> out;
        std::string cur;
        for (char c : line) {
            if (c == delim_) { out.push_back(std::move(cur)); cur.clear(); }
            else cur.push_back(c);
        }
        out.push_back(std::move(cur));
        return out;
    }
};

// ===========================================================================
// JSONLDataset — one JSON object per line. Tiny parser supporting flat
// objects with string values: { "text": "hello", "label": "pos" }.
// Numeric / nested values are kept as their raw JSON string form.
// ===========================================================================
class JSONLDataset : public Dataset<std::unordered_map<std::string, std::string>, Tensor> {
public:
    explicit JSONLDataset(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("JSONLDataset: cannot open " + path);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            records_.push_back(parse_flat_(line));
        }
    }

    ExampleType get(size_t index) override {
        if (index >= records_.size())
            throw std::out_of_range("JSONLDataset: index out of range");
        return ExampleType(records_[index], Tensor());
    }

    size_t size() const override { return records_.size(); }

    const std::vector<std::unordered_map<std::string, std::string>>& records() const {
        return records_;
    }

private:
    std::vector<std::unordered_map<std::string, std::string>> records_;

    static std::string parse_string_(const std::string& s, size_t& i) {
        // Assumes s[i] == '"'.
        std::string out;
        ++i;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char n = s[i + 1];
                switch (n) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case '"': out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    default:  out.push_back(n);   break;
                }
                i += 2;
            } else {
                out.push_back(s[i++]);
            }
        }
        if (i < s.size()) ++i;  // skip closing "
        return out;
    }

    static std::string parse_raw_value_(const std::string& s, size_t& i) {
        std::string out;
        while (i < s.size() && s[i] != ',' && s[i] != '}') out.push_back(s[i++]);
        // trim trailing whitespace
        while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back())))
            out.pop_back();
        return out;
    }

    static std::unordered_map<std::string, std::string> parse_flat_(const std::string& line) {
        std::unordered_map<std::string, std::string> out;
        size_t i = 0;
        while (i < line.size() && line[i] != '{') ++i;
        if (i >= line.size()) return out;
        ++i;
        while (i < line.size()) {
            while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
            if (i < line.size() && line[i] == '}') break;
            if (i >= line.size() || line[i] != '"') break;
            std::string key = parse_string_(line, i);
            while (i < line.size() && line[i] != ':') ++i;
            if (i >= line.size()) break;
            ++i;
            while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
            std::string val;
            if (i < line.size() && line[i] == '"') val = parse_string_(line, i);
            else val = parse_raw_value_(line, i);
            out.emplace(std::move(key), std::move(val));
            while (i < line.size() && (line[i] == ',' ||
                    std::isspace(static_cast<unsigned char>(line[i])))) ++i;
        }
        return out;
    }
};

// ===========================================================================
// Collators
// ===========================================================================

// Pad a batch of variable-length 1-D id sequences into a single 2-D tensor.
// - batch_first=true  -> shape [B, T_max]
// - batch_first=false -> shape [T_max, B]
// padding_value is written into all padded positions.
inline Tensor pad_sequence(const std::vector<std::vector<int64_t>>& batch,
                           int64_t padding_value = 0,
                           bool batch_first = true) {
    if (batch.empty())
        return at::empty({0, 0}, at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t B = static_cast<int64_t>(batch.size());
    int64_t T = 0;
    for (const auto& s : batch)
        T = std::max<int64_t>(T, static_cast<int64_t>(s.size()));

    std::vector<int64_t> shape = batch_first
        ? std::vector<int64_t>{B, T}
        : std::vector<int64_t>{T, B};
    Tensor out = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* p = out.mutable_data_ptr<int64_t>();
    // Fill with padding first — cheap and avoids special-casing.
    for (int64_t i = 0, n = B * T; i < n; ++i) p[i] = padding_value;
    if (batch_first) {
        for (int64_t i = 0; i < B; ++i) {
            const auto& s = batch[static_cast<size_t>(i)];
            for (int64_t t = 0; t < static_cast<int64_t>(s.size()); ++t)
                p[i * T + t] = s[static_cast<size_t>(t)];
        }
    } else {
        for (int64_t i = 0; i < B; ++i) {
            const auto& s = batch[static_cast<size_t>(i)];
            for (int64_t t = 0; t < static_cast<int64_t>(s.size()); ++t)
                p[t * B + i] = s[static_cast<size_t>(t)];
        }
    }
    return out;
}

// Return the length of each sequence as an int64 [B] tensor.
inline Tensor sequence_lengths(const std::vector<std::vector<int64_t>>& batch) {
    int64_t B = static_cast<int64_t>(batch.size());
    Tensor out = at::empty({B}, at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* p = out.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < B; ++i)
        p[i] = static_cast<int64_t>(batch[static_cast<size_t>(i)].size());
    return out;
}

// Build a [B, T] boolean-valued int64 mask: 1 where a real token lives,
// 0 where a pad token does.
inline Tensor padding_mask(const std::vector<std::vector<int64_t>>& batch,
                           bool batch_first = true) {
    if (batch.empty())
        return at::empty({0, 0}, at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t B = static_cast<int64_t>(batch.size());
    int64_t T = 0;
    for (const auto& s : batch)
        T = std::max<int64_t>(T, static_cast<int64_t>(s.size()));
    std::vector<int64_t> shape = batch_first
        ? std::vector<int64_t>{B, T}
        : std::vector<int64_t>{T, B};
    Tensor out = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* p = out.mutable_data_ptr<int64_t>();
    for (int64_t i = 0, n = B * T; i < n; ++i) p[i] = 0;
    if (batch_first) {
        for (int64_t i = 0; i < B; ++i) {
            int64_t L = static_cast<int64_t>(batch[static_cast<size_t>(i)].size());
            for (int64_t t = 0; t < L; ++t) p[i * T + t] = 1;
        }
    } else {
        for (int64_t i = 0; i < B; ++i) {
            int64_t L = static_cast<int64_t>(batch[static_cast<size_t>(i)].size());
            for (int64_t t = 0; t < L; ++t) p[t * B + i] = 1;
        }
    }
    return out;
}

// Bundle returned by pack_padded_sequence — mirrors torch.nn.utils.rnn.
struct PackedSequence {
    Tensor data;      // padded batch: [B, T_max] or [T_max, B]
    Tensor lengths;   // int64 [B]
    Tensor mask;      // int64 [B, T_max] or [T_max, B]
    bool batch_first; // layout flag for consumers
};

inline PackedSequence pack_padded_sequence(
        const std::vector<std::vector<int64_t>>& batch,
        int64_t padding_value = 0,
        bool batch_first = true) {
    PackedSequence out;
    out.data = pad_sequence(batch, padding_value, batch_first);
    out.lengths = sequence_lengths(batch);
    out.mask = padding_mask(batch, batch_first);
    out.batch_first = batch_first;
    return out;
}

}  // namespace text
}  // namespace torch
