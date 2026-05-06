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

        // Collect special tokens (tokens with <|...|> or <...> patterns)
        for (size_t i = 0; i < vocab.size(); ++i) {
            const auto& tok = vocab[i];
            if (tok.size() >= 3 && tok[0] == '<' && tok.back() == '>') {
                // Check it's not a byte token like <0xFF>
                if (tok.size() > 4 || tok[1] != '0') {
                    special_tokens_.push_back({tok, static_cast<int32_t>(i)});
                }
            }
        }
        // Sort by length descending so longer tokens match first
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) {
                      return a.first.size() > b.first.size();
                  });

        std::cout << "[Tokenizer] Loaded: model=" << model_type
                  << ", vocab=" << vocab.size()
                  << ", merges=" << merges.size()
                  << ", bos=" << bos_id << ", eos=" << eos_id
                  << ", special=" << special_tokens_.size() << std::endl;
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

        // Split text by special tokens first, then BPE-encode non-special parts
        std::vector<std::pair<std::string, int32_t>> segments;  // text,-1 or "",token_id
        split_special_tokens(text, segments);

        for (const auto& seg : segments) {
            if (seg.second >= 0) {
                // Special token — add directly
                tokens.push_back(seg.second);
            } else if (!seg.first.empty()) {
                // Regular text — BPE encode
                if (!merges.empty()) {
                    encode_bpe(seg.first, tokens);
                } else {
                    encode_greedy(seg.first, tokens);
                }
            }
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
    std::vector<std::pair<std::string, int32_t>> special_tokens_;  // sorted by length desc

    // Split text into alternating segments: regular text (id=-1) and special tokens (id>=0)
    void split_special_tokens(const std::string& text,
                              std::vector<std::pair<std::string, int32_t>>& segments) const {
        if (special_tokens_.empty()) {
            segments.push_back({text, -1});
            return;
        }

        size_t pos = 0;
        while (pos < text.size()) {
            // Try to match a special token at current position
            bool found = false;
            for (const auto& st : special_tokens_) {
                const auto& tok_str = st.first;
                if (pos + tok_str.size() <= text.size() &&
                    text.compare(pos, tok_str.size(), tok_str) == 0) {
                    // Found special token
                    segments.push_back({"", st.second});
                    pos += tok_str.size();
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Find next special token or end of string
                size_t next_special = text.size();
                for (const auto& st : special_tokens_) {
                    size_t p = text.find(st.first, pos);
                    if (p != std::string::npos && p < next_special) {
                        next_special = p;
                    }
                }
                // Add regular text segment
                segments.push_back({text.substr(pos, next_special - pos), -1});
                pos = next_special;
            }
        }
    }

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
            // GPT-2 style pre-tokenization, реализующая регекс llama.cpp:
            //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
            //
            // BUG-13 fix (2026-05-06): на gigachat (deepseek2) старый
            // handcrafted split давал ДРУГИЕ tokens чем llama.cpp,
            // модель видела rare sequences → garbage output. Теперь
            // полная имитация регекспа через scanner.
            //
            // is_letter: ASCII letter ИЛИ UTF-8 multibyte first byte (>=0xC2).
            //   Это покрывает кириллицу (D0..D1), греческий, грузинский, китайский...
            //   Continuation bytes (0x80..0xBF) трактуются как часть текущего letter run.
            // is_digit: только ASCII 0-9.
            // is_space: ASCII space, tab, \n, \r, \v, \f.
            std::vector<std::string> words;
            const auto& s = text;
            const size_t N = s.size();
            auto is_space = [](unsigned char c) {
                return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
            };
            auto is_digit = [](unsigned char c) {
                return c >= '0' && c <= '9';
            };
            auto is_letter_first_byte = [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0xC2;
            };
            auto is_continuation = [](unsigned char c) { return c >= 0x80 && c <= 0xBF; };

            // Helpers: проверить contraction в позиции pos, вернуть длину или 0.
            auto match_contraction = [&](size_t pos) -> size_t {
                if (pos >= N || s[pos] != '\'') return 0;
                if (pos + 1 < N) {
                    char b = s[pos + 1];
                    if (b == 's' || b == 't' || b == 'm' || b == 'd') return 2;
                    if (pos + 2 < N) {
                        if (b == 'r' && s[pos + 2] == 'e') return 3;
                        if (b == 'v' && s[pos + 2] == 'e') return 3;
                        if (b == 'l' && s[pos + 2] == 'l') return 3;
                    }
                }
                return 0;
            };

            size_t i = 0;
            while (i < N) {
                // 1. Contractions: 's 't 're 've 'm 'll 'd
                if (size_t cl = match_contraction(i); cl > 0) {
                    words.push_back(s.substr(i, cl));
                    i += cl;
                    continue;
                }

                size_t start = i;
                bool had_space = false;
                if (is_space((unsigned char)s[i])) {
                    // Может быть optional leading space ИЛИ trailing whitespace run.
                    // Пробуем optional space + что-то.
                    had_space = true;
                    ++i;
                    if (i >= N || is_space((unsigned char)s[i])) {
                        // Это whitespace run без следующего непробельного, или EOF.
                        // \s+(?!\S): grab все пробелы до того, как впереди не-space.
                        // Если до конца строки только space — эмитим всё одним токеном.
                        // Если в середине — эмитим всё кроме последнего (тот пойдёт
                        // префиксом следующего слова).
                        i = start;
                        while (i < N && is_space((unsigned char)s[i])) ++i;
                        bool at_eof = (i == N);
                        size_t end = at_eof ? i : (i - 1);
                        if (end > start) {
                            words.push_back(s.substr(start, end - start));
                        }
                        if (!at_eof) {
                            i = end;  // оставшийся 1 space станет prefix следующего слова
                        }
                        continue;
                    }
                }

                unsigned char c = (unsigned char)s[i];
                if (is_letter_first_byte(c)) {
                    // Letter run (с поддержкой UTF-8 continuation)
                    while (i < N) {
                        unsigned char ch = (unsigned char)s[i];
                        if (is_letter_first_byte(ch)) { ++i; }
                        else if (is_continuation(ch)) { ++i; }
                        else break;
                    }
                    words.push_back(s.substr(start, i - start));
                } else if (is_digit(c)) {
                    while (i < N && is_digit((unsigned char)s[i])) ++i;
                    words.push_back(s.substr(start, i - start));
                } else if (!is_space(c)) {
                    // Не-letter, не-digit, не-space → punctuation/symbol run
                    while (i < N) {
                        unsigned char ch = (unsigned char)s[i];
                        if (is_space(ch) || is_letter_first_byte(ch) || is_digit(ch) ||
                            is_continuation(ch)) break;
                        ++i;
                    }
                    words.push_back(s.substr(start, i - start));
                } else {
                    // Защита от infinite loop: одиночный байт
                    words.push_back(std::string(1, s[i]));
                    ++i;
                }
                (void)had_space;
            }

            // BPE encode each word (apply GPT-2 bytes_to_unicode forward map).
            // BUG-10 fix: раньше обрабатывались только 3 spec-bytes (space/\n/\t).
            // Из-за этого cyrillic bytes (0xD0, 0xD1, ...) проходили raw и
            // вокаб не находил match — модель видела byte-fallback токены и
            // переключалась на garbage / другие языки. Теперь полная
            // 256-entry таблица: каждый byte 0-255 → unique unicode codepoint
            // в UTF-8 form, согласно GPT-2 standard.
            auto append_byte_encoded = [](std::string& dst, uint8_t b) {
                int cp;
                if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                    cp = b;  // printable bytes mapped to self
                } else if (b <= 32) {
                    cp = 0x100 + b;        // bytes 0..32 → U+0100..U+0120 (space=0x20→Ġ)
                } else if (b >= 127 && b <= 160) {
                    cp = 0x121 + (b - 127); // bytes 127..160 → U+0121..U+0142
                } else {                    // b == 173
                    cp = 0x143;
                }
                if (cp < 0x80) {
                    dst += (char)cp;
                } else if (cp < 0x800) {
                    dst += (char)(0xC0 | (cp >> 6));
                    dst += (char)(0x80 | (cp & 0x3F));
                } else {
                    dst += (char)(0xE0 | (cp >> 12));
                    dst += (char)(0x80 | ((cp >> 6) & 0x3F));
                    dst += (char)(0x80 | (cp & 0x3F));
                }
            };
            for (const auto& word : words) {
                std::string gpt2_word;
                for (unsigned char c : word) {
                    append_byte_encoded(gpt2_word, c);
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

        // GPT-2: Ċ (U+010A, UTF-8: 0xC4 0x8A) → newline
        const std::string gpt2_newline = "\xc4\x8a";
        pos = 0;
        while ((pos = result.find(gpt2_newline, pos)) != std::string::npos) {
            result.replace(pos, gpt2_newline.size(), "\n");
            pos += 1;
        }

        // GPT-2: ĉ (U+0109, UTF-8: 0xC4 0x89) → tab
        const std::string gpt2_tab = "\xc4\x89";
        pos = 0;
        while ((pos = result.find(gpt2_tab, pos)) != std::string::npos) {
            result.replace(pos, gpt2_tab.size(), "\t");
            pos += 1;
        }

        // BUG-7 fix: GPT-2 byte-to-unicode reverse mapping для cyrillic +
        // других не-ASCII bytes. qwen3, llama3, mistral используют
        // bytes_to_unicode() encoder где каждый byte 0-255 mapped в
        // "printable" unicode codepoint. На decode нужна обратная операция,
        // иначе русский (UTF-8 multi-byte) выходит как latin-1 garbage
        // ("å¤ĸ è¯Ĩ" вместо "Привет").
        std::string byte_decoded;
        byte_decoded.reserve(result.size());
        size_t bi = 0;
        while (bi < result.size()) {
            unsigned char c = (unsigned char)result[bi];
            size_t clen = utf8_char_len(c);
            if (clen == 1) {
                byte_decoded += result[bi];
                bi++;
                continue;
            }
            if (bi + clen > result.size()) {
                byte_decoded += result[bi];
                bi++;
                continue;
            }
            // Decode codepoint
            uint32_t cp = 0;
            if (clen == 2) {
                cp = ((c & 0x1F) << 6) | ((unsigned char)result[bi+1] & 0x3F);
            } else if (clen == 3) {
                cp = ((c & 0x0F) << 12) |
                     (((unsigned char)result[bi+1] & 0x3F) << 6) |
                     ((unsigned char)result[bi+2] & 0x3F);
            } else if (clen == 4) {
                cp = ((c & 0x07) << 18) |
                     (((unsigned char)result[bi+1] & 0x3F) << 12) |
                     (((unsigned char)result[bi+2] & 0x3F) << 6) |
                     ((unsigned char)result[bi+3] & 0x3F);
            }
            // Check if cp is in GPT-2 byte-level mapping range
            bool mapped = false;
            uint8_t raw = 0;
            if (cp >= 161 && cp <= 172)        { mapped = true; raw = (uint8_t)cp; }
            else if (cp >= 174 && cp <= 255)   { mapped = true; raw = (uint8_t)cp; }
            else if (cp >= 0x100 && cp <= 0x120) { mapped = true; raw = (uint8_t)(cp - 0x100); }       // bytes 0-32
            else if (cp >= 0x121 && cp <= 0x142) { mapped = true; raw = (uint8_t)(127 + (cp - 0x121)); } // bytes 127-160
            else if (cp == 0x143)                { mapped = true; raw = 173; }
            if (mapped) {
                byte_decoded += (char)raw;
            } else {
                byte_decoded.append(result, bi, clen);
            }
            bi += clen;
        }

        return byte_decoded;
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
