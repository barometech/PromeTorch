#pragma once

#include "torch/io/gguf_loader.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <regex>

namespace torch {
namespace io {

// ============================================================================
// BPE Tokenizer loaded from GGUF metadata
//
// Supports:
//   - SentencePiece BPE (Llama, Gemma, Qwen)
//   - GPT-2 style BPE
// Uses tokenizer.ggml.tokens, tokenizer.ggml.scores, tokenizer.ggml.merges
// ============================================================================

class GGUFTokenizer {
public:
    std::vector<std::string> vocab;                           // token_id → string
    std::vector<float> scores;                                // token_id → score
    std::unordered_map<std::string, int32_t> token_to_id;     // string → token_id
    std::vector<std::pair<std::string, std::string>> merges;  // BPE merge pairs

    int32_t bos_id = 1;
    int32_t eos_id = 2;
    int32_t pad_id = -1;
    int32_t unk_id = 0;
    bool add_bos = true;
    std::string model_type;  // "llama", "gpt2", etc.

    // ========================================================================
    // Load from GGUF metadata
    // ========================================================================

    void load(const gguf::GGUFReader& reader) {
        model_type = reader.get_string("tokenizer.ggml.model", "llama");

        // Load vocab tokens
        if (reader.has_metadata("tokenizer.ggml.tokens")) {
            const auto& tokens_arr = reader.get_metadata("tokenizer.ggml.tokens").as_array();
            vocab.reserve(tokens_arr.size());
            for (const auto& t : tokens_arr) {
                vocab.push_back(t.str);
            }
        }

        // Build reverse mapping
        for (size_t i = 0; i < vocab.size(); ++i) {
            token_to_id[vocab[i]] = static_cast<int32_t>(i);
        }

        // Load scores
        if (reader.has_metadata("tokenizer.ggml.scores")) {
            const auto& scores_arr = reader.get_metadata("tokenizer.ggml.scores").as_array();
            scores.reserve(scores_arr.size());
            for (const auto& s : scores_arr) {
                scores.push_back(s.as_float());
            }
        }

        // Load BPE merges
        if (reader.has_metadata("tokenizer.ggml.merges")) {
            const auto& merges_arr = reader.get_metadata("tokenizer.ggml.merges").as_array();
            merges.reserve(merges_arr.size());
            for (const auto& m : merges_arr) {
                std::string merge_str = m.str;
                size_t space = merge_str.find(' ');
                if (space != std::string::npos) {
                    merges.push_back({
                        merge_str.substr(0, space),
                        merge_str.substr(space + 1)
                    });
                }
            }
        }

        // Special token IDs
        bos_id = static_cast<int32_t>(reader.get_int("tokenizer.ggml.bos_token_id", 1));
        eos_id = static_cast<int32_t>(reader.get_int("tokenizer.ggml.eos_token_id", 2));
        pad_id = static_cast<int32_t>(reader.get_int("tokenizer.ggml.padding_token_id", -1));
        unk_id = static_cast<int32_t>(reader.get_int("tokenizer.ggml.unknown_token_id", 0));

        // Whether to add BOS token
        if (reader.has_metadata("tokenizer.ggml.add_bos_token")) {
            add_bos = reader.get_metadata("tokenizer.ggml.add_bos_token").as_bool();
        }

        // Build merge rank map for fast lookup
        for (size_t i = 0; i < merges.size(); ++i) {
            merge_rank_[merges[i].first + " " + merges[i].second] = static_cast<int32_t>(i);
        }

        std::cout << "[Tokenizer] Loaded: model=" << model_type
                  << ", vocab=" << vocab.size()
                  << ", merges=" << merges.size()
                  << ", bos=" << bos_id << ", eos=" << eos_id << std::endl;
    }

    // ========================================================================
    // Encode text → token IDs
    // ========================================================================

    std::vector<int32_t> encode(const std::string& text, bool use_bos = true) const {
        std::vector<int32_t> tokens;

        if (use_bos && add_bos && bos_id >= 0) {
            tokens.push_back(bos_id);
        }

        if (text.empty()) return tokens;

        if (!merges.empty()) {
            // BPE encoding
            encode_bpe(text, tokens);
        } else {
            // Greedy longest-match encoding using scores
            encode_greedy(text, tokens);
        }

        return tokens;
    }

    // ========================================================================
    // Decode token IDs → text
    // ========================================================================

    std::string decode(const std::vector<int32_t>& tokens, bool skip_special = true) const {
        std::string result;

        for (int32_t id : tokens) {
            if (skip_special && (id == bos_id || id == eos_id || id == pad_id)) {
                continue;
            }
            if (id >= 0 && id < static_cast<int32_t>(vocab.size())) {
                std::string token = vocab[id];
                result += decode_token(token);
            }
        }

        return result;
    }

    std::string decode_token(int32_t id) const {
        if (id >= 0 && id < static_cast<int32_t>(vocab.size())) {
            return decode_token(vocab[id]);
        }
        return "";
    }

    int32_t vocab_size() const {
        return static_cast<int32_t>(vocab.size());
    }

private:
    std::unordered_map<std::string, int32_t> merge_rank_;

    // ========================================================================
    // BPE encoding with merges
    // ========================================================================

    void encode_bpe(const std::string& text, std::vector<int32_t>& tokens) const {
        bool use_sentencepiece = (model_type != "gpt2");

        if (use_sentencepiece) {
            // SentencePiece: prepend ▁ to words, then BPE the whole thing
            std::string processed;
            bool at_start = true;
            for (size_t i = 0; i < text.size(); ++i) {
                if (text[i] == ' ' || text[i] == '\n' || text[i] == '\t') {
                    at_start = true;
                    if (text[i] == '\n') {
                        processed += text[i];
                    }
                    continue;
                }
                if (at_start) {
                    processed += "\xe2\x96\x81";  // ▁
                    at_start = false;
                }
                processed += text[i];
            }
            if (!processed.empty()) {
                encode_bpe_piece(processed, tokens);
            }
        } else {
            // GPT-2 style: pre-tokenize into words (with preceding space),
            // then BPE each word independently
            std::vector<std::string> words;
            std::string current;
            for (size_t i = 0; i < text.size(); ++i) {
                char c = text[i];
                bool is_letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                 (c & 0x80);  // UTF-8 continuation
                bool is_digit = (c >= '0' && c <= '9');
                bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');

                if (is_space) {
                    if (!current.empty()) {
                        words.push_back(current);
                        current.clear();
                    }
                    // Space becomes part of next word (GPT-2 convention)
                    current += c;
                } else if (is_letter || is_digit) {
                    // Check if switching from letter to digit or vice versa
                    if (!current.empty()) {
                        char prev = current.back();
                        bool prev_letter = (prev >= 'a' && prev <= 'z') || (prev >= 'A' && prev <= 'Z');
                        bool prev_digit = (prev >= '0' && prev <= '9');
                        bool prev_space = (prev == ' ' || prev == '\n' || prev == '\t');
                        if ((is_letter && prev_digit) || (is_digit && prev_letter) ||
                            (!is_space && prev_space && current.size() > 1)) {
                            // Don't split: space + word stays together
                        }
                    }
                    current += c;
                } else {
                    // Punctuation: separate token
                    if (!current.empty()) {
                        words.push_back(current);
                        current.clear();
                    }
                    words.push_back(std::string(1, c));
                }
            }
            if (!current.empty()) {
                words.push_back(current);
            }

            // BPE encode each word (convert spaces to Ġ for GPT-2 vocab)
            for (const auto& word : words) {
                // GPT-2 uses Ġ (U+0120) to represent spaces in vocab
                std::string gpt2_word;
                for (char c : word) {
                    if (c == ' ') {
                        gpt2_word += "\xc4\xa0";  // Ġ in UTF-8
                    } else if (c == '\n') {
                        gpt2_word += "\xc4\x8a";  // Ċ in UTF-8
                    } else if (c == '\t') {
                        gpt2_word += "\xc4\x89";  // ĉ in UTF-8
                    } else {
                        gpt2_word += c;
                    }
                }
                // Try direct vocab lookup first
                auto it = token_to_id.find(gpt2_word);
                if (it != token_to_id.end()) {
                    tokens.push_back(it->second);
                } else {
                    encode_bpe_piece(gpt2_word, tokens);
                }
            }
        }
    }

    // BPE encode a single piece (after pre-tokenization)
    void encode_bpe_piece(const std::string& piece, std::vector<int32_t>& tokens) const {
        // Initialize: each UTF-8 character (or byte) as a separate symbol
        std::vector<std::string> symbols;
        size_t i = 0;
        while (i < piece.size()) {
            size_t char_len = utf8_char_len(piece[i]);
            if (i + char_len > piece.size()) char_len = 1;
            symbols.push_back(piece.substr(i, char_len));
            i += char_len;
        }

        // Apply BPE merges iteratively
        while (symbols.size() >= 2) {
            // Find the merge with lowest rank (highest priority)
            int best_rank = INT32_MAX;
            size_t best_pos = SIZE_MAX;

            for (size_t j = 0; j + 1 < symbols.size(); ++j) {
                std::string pair = symbols[j] + " " + symbols[j + 1];
                auto it = merge_rank_.find(pair);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = j;
                }
            }

            if (best_pos == SIZE_MAX) break;  // No more merges applicable

            // Apply the merge
            symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
            symbols.erase(symbols.begin() + best_pos + 1);
        }

        // Convert symbols to token IDs
        for (const auto& sym : symbols) {
            auto it = token_to_id.find(sym);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
            } else {
                // Try byte fallback
                encode_bytes_fallback(sym, tokens);
            }
        }
    }

    // ========================================================================
    // Greedy longest-match encoding (fallback when no merges)
    // ========================================================================

    void encode_greedy(const std::string& text, std::vector<int32_t>& tokens) const {
        // Prepend space separator
        std::string processed;
        bool at_start = true;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == ' ') {
                at_start = true;
                continue;
            }
            if (at_start) {
                processed += "\xe2\x96\x81";  // ▁
                at_start = false;
            }
            processed += text[i];
        }

        size_t i = 0;
        while (i < processed.size()) {
            // Find longest matching token
            int32_t best_id = unk_id;
            size_t best_len = 0;

            // Try decreasing lengths
            size_t max_len = (std::min)(processed.size() - i, size_t(64));
            for (size_t len = max_len; len >= 1; --len) {
                std::string candidate = processed.substr(i, len);
                auto it = token_to_id.find(candidate);
                if (it != token_to_id.end()) {
                    best_id = it->second;
                    best_len = len;
                    break;
                }
            }

            if (best_len > 0) {
                tokens.push_back(best_id);
                i += best_len;
            } else {
                // Byte fallback: encode as <0xNN>
                uint8_t byte = static_cast<uint8_t>(processed[i]);
                encode_byte_fallback(byte, tokens);
                i += 1;
            }
        }
    }

    // ========================================================================
    // Byte fallback encoding
    // ========================================================================

    void encode_bytes_fallback(const std::string& s, std::vector<int32_t>& tokens) const {
        for (unsigned char c : s) {
            encode_byte_fallback(c, tokens);
        }
    }

    void encode_byte_fallback(uint8_t byte, std::vector<int32_t>& tokens) const {
        // Try <0xNN> format (SentencePiece style)
        char buf[8];
        snprintf(buf, sizeof(buf), "<0x%02X>", byte);
        auto it = token_to_id.find(std::string(buf));
        if (it != token_to_id.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(unk_id);
        }
    }

    // ========================================================================
    // Decode token string (handle special encodings)
    // ========================================================================

    std::string decode_token(const std::string& token) const {
        // Handle byte tokens: <0xNN>
        if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>') {
            char byte = static_cast<char>(std::stoi(token.substr(3, 2), nullptr, 16));
            return std::string(1, byte);
        }

        // Handle space markers
        std::string result = token;

        // SentencePiece: ▁ (U+2581) → space
        const std::string sp_marker = "\xe2\x96\x81";
        size_t pos = 0;
        while ((pos = result.find(sp_marker, pos)) != std::string::npos) {
            result.replace(pos, sp_marker.size(), " ");
            pos += 1;
        }

        // GPT-2: Ġ (U+0120, UTF-8: 0xC4 0xA0) → space
        const std::string gpt2_space = "\xc4\xa0";
        pos = 0;
        while ((pos = result.find(gpt2_space, pos)) != std::string::npos) {
            result.replace(pos, gpt2_space.size(), " ");
            pos += 1;
        }

        return result;
    }

    // ========================================================================
    // UTF-8 helpers
    // ========================================================================

    static size_t utf8_char_len(unsigned char c) {
        if ((c & 0x80) == 0) return 1;
        if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3;
        if ((c & 0xF8) == 0xF0) return 4;
        return 1;
    }
};

} // namespace io
} // namespace torch
