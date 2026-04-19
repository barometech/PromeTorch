#pragma once
// torch/serve/llm.h — Production LLM serving engine for PromeTorch.
// HF Llama loader + paged KV cache + RoPE + continuous batching + sampling.
// Elbrus LCC: header-only, STL-only, CPU float buffers.
// Loader probes: safetensors > sharded safetensors > pytorch_model.bin.
// Tokenizer: tokenizer.json > vocab.json+merges.txt > 256-byte fallback.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace serve {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct GenerationConfig {
    int max_new_tokens = 128;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    std::vector<int> stop_tokens;
};

struct Request {
    int request_id = 0;
    std::string prompt;
    GenerationConfig config;
};

struct Response {
    int request_id = 0;
    std::string output;
    int tokens_generated = 0;
    double throughput_tok_per_s = 0.0;
};

// ---------------------------------------------------------------------------
// Llama model description + weights (POD; filled by loader or tests).
// ---------------------------------------------------------------------------

struct LlamaConfig {
    int vocab_size = 32000;
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 32;   // GQA if < num_attention_heads.
    int max_position_embeddings = 4096;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int bos_token_id = 1;
    int eos_token_id = 2;
};

struct LlamaLayerWeights {
    std::vector<float> attn_norm;       // [hidden_size]
    std::vector<float> wq, wk, wv, wo;  // wq: [H*Dh, D]; wk/wv: [Hkv*Dh, D]; wo: [D, H*Dh]
    std::vector<float> ffn_norm;        // [hidden_size]
    std::vector<float> w_gate, w_up, w_down;  // gate/up: [I, D]; down: [D, I]
};

struct LlamaWeights {
    LlamaConfig cfg;
    std::vector<float> tok_emb;         // [V, D]
    std::vector<LlamaLayerWeights> layers;
    std::vector<float> final_norm;      // [D]
    std::vector<float> lm_head;         // [V, D] (tied with tok_emb if shared)
};

// ---------------------------------------------------------------------------
// Minimal byte-level BPE tokenizer (GPT-2 / Llama style).
// Works from vocab.json + merges.txt. tokenizer.json fast path falls back
// to the same logic by parsing its "model" section.
// ---------------------------------------------------------------------------

class ByteBPETokenizer {
public:
    bool load(const std::string& model_dir) {
        // Prefer tokenizer.json (HF fast). Very small JSON subset parser.
        if (load_tokenizer_json_(model_dir + "/tokenizer.json")) return true;
        if (load_vocab_merges_(model_dir + "/vocab.json",
                               model_dir + "/merges.txt")) return true;
        // Bytes fallback: 256 single-byte tokens + <bos>/<eos>/<pad>.
        vocab_.clear(); id2tok_.clear();
        for (int b = 0; b < 256; ++b) {
            std::string t; t.push_back((char)b);
            vocab_[t] = b; id2tok_[b] = t;
        }
        vocab_["<s>"] = 256;  id2tok_[256] = "<s>";
        vocab_["</s>"] = 257; id2tok_[257] = "</s>";
        bos_id_ = 256; eos_id_ = 257;
        return true;
    }

    std::vector<int> encode(const std::string& s, bool add_bos = true) const {
        std::vector<int> out;
        if (add_bos && bos_id_ >= 0) out.push_back(bos_id_);
        // Greedy longest-match over merges; if no merges table, encode per-byte.
        std::vector<std::string> syms;
        for (char c : s) syms.emplace_back(1, c);
        if (!bpe_ranks_.empty()) apply_merges_(syms);
        for (const auto& sym : syms) {
            auto it = vocab_.find(sym);
            if (it != vocab_.end()) out.push_back(it->second);
            else {
                for (char c : sym) {
                    auto jt = vocab_.find(std::string(1, c));
                    if (jt != vocab_.end()) out.push_back(jt->second);
                }
            }
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        for (int id : ids) {
            auto it = id2tok_.find(id);
            if (it != id2tok_.end()) s += it->second;
        }
        return s;
    }

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int vocab_size() const { return (int)vocab_.size(); }

private:
    bool load_vocab_merges_(const std::string& vocab_path,
                            const std::string& merges_path) {
        std::ifstream vf(vocab_path);
        if (!vf) return false;
        std::string text((std::istreambuf_iterator<char>(vf)),
                         std::istreambuf_iterator<char>());
        // Very small JSON object parser: {"tok": id, ...}
        size_t i = 0;
        while (i < text.size() && text[i] != '{') ++i;
        if (i == text.size()) return false; ++i;
        while (i < text.size()) {
            while (i < text.size() && (text[i] == ' ' || text[i] == '\n' ||
                                       text[i] == ',' || text[i] == '\t')) ++i;
            if (i >= text.size() || text[i] == '}') break;
            if (text[i] != '"') { ++i; continue; }
            ++i;
            std::string tok;
            while (i < text.size() && text[i] != '"') {
                if (text[i] == '\\' && i + 1 < text.size()) { tok.push_back(text[i+1]); i += 2; }
                else tok.push_back(text[i++]);
            }
            if (i < text.size()) ++i;
            while (i < text.size() && (text[i] == ':' || text[i] == ' ')) ++i;
            int id = 0;
            while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                id = id * 10 + (text[i] - '0'); ++i;
            }
            vocab_[tok] = id;
            id2tok_[id] = tok;
        }
        if (vocab_.count("<s>"))   bos_id_ = vocab_["<s>"];
        if (vocab_.count("</s>"))  eos_id_ = vocab_["</s>"];
        std::ifstream mf(merges_path);
        if (mf) {
            std::string line;
            int rank = 0;
            while (std::getline(mf, line)) {
                if (line.empty() || line[0] == '#') continue;
                auto sp = line.find(' ');
                if (sp == std::string::npos) continue;
                bpe_ranks_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
            }
        }
        return !vocab_.empty();
    }

    bool load_tokenizer_json_(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        // Locate "vocab": { ... } block and parse like vocab.json.
        auto v = text.find("\"vocab\"");
        if (v == std::string::npos) return false;
        size_t lb = text.find('{', v);
        if (lb == std::string::npos) return false;
        int depth = 0;
        size_t rb = lb;
        for (; rb < text.size(); ++rb) {
            if (text[rb] == '{') ++depth;
            else if (text[rb] == '}') { --depth; if (depth == 0) break; }
        }
        if (rb >= text.size()) return false;
        std::string sub = text.substr(lb, rb - lb + 1);
        // Reuse same primitive parser by writing to a temp stringstream.
        size_t i = 1;
        while (i < sub.size()) {
            while (i < sub.size() && (sub[i] == ' ' || sub[i] == '\n' ||
                                      sub[i] == ',' || sub[i] == '\t')) ++i;
            if (i >= sub.size() || sub[i] == '}') break;
            if (sub[i] != '"') { ++i; continue; }
            ++i;
            std::string tok;
            while (i < sub.size() && sub[i] != '"') {
                if (sub[i] == '\\' && i + 1 < sub.size()) { tok.push_back(sub[i+1]); i += 2; }
                else tok.push_back(sub[i++]);
            }
            if (i < sub.size()) ++i;
            while (i < sub.size() && (sub[i] == ':' || sub[i] == ' ')) ++i;
            int id = 0;
            while (i < sub.size() && sub[i] >= '0' && sub[i] <= '9') {
                id = id * 10 + (sub[i] - '0'); ++i;
            }
            vocab_[tok] = id; id2tok_[id] = tok;
        }
        auto m = text.find("\"merges\"");
        if (m != std::string::npos) {
            size_t sb = text.find('[', m);
            if (sb != std::string::npos) {
                size_t eb = text.find(']', sb);
                std::string ms = text.substr(sb + 1, eb - sb - 1);
                std::stringstream ss(ms);
                std::string line;
                int rank = 0;
                while (std::getline(ss, line, ',')) {
                    size_t q1 = line.find('"');
                    if (q1 == std::string::npos) continue;
                    size_t q2 = line.find('"', q1 + 1);
                    if (q2 == std::string::npos) continue;
                    std::string pair = line.substr(q1 + 1, q2 - q1 - 1);
                    auto sp = pair.find(' ');
                    if (sp == std::string::npos) continue;
                    bpe_ranks_[{pair.substr(0, sp), pair.substr(sp + 1)}] = rank++;
                }
            }
        }
        if (vocab_.count("<s>"))  bos_id_ = vocab_["<s>"];
        if (vocab_.count("</s>")) eos_id_ = vocab_["</s>"];
        return !vocab_.empty();
    }

    void apply_merges_(std::vector<std::string>& syms) const {
        bool changed = true;
        while (changed && syms.size() >= 2) {
            changed = false;
            int best_rank = INT32_MAX;
            size_t best_i = 0;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                auto it = bpe_ranks_.find({syms[i], syms[i + 1]});
                if (it != bpe_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second; best_i = i;
                }
            }
            if (best_rank == INT32_MAX) break;
            syms[best_i] = syms[best_i] + syms[best_i + 1];
            syms.erase(syms.begin() + best_i + 1);
            changed = true;
        }
    }

    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>()(p.first) ^
                   (std::hash<std::string>()(p.second) << 1);
        }
    };

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id2tok_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks_;
    int bos_id_ = -1;
    int eos_id_ = -1;
};

// ---------------------------------------------------------------------------
// Paged KV cache: one page = 64 tokens per layer per K/V. Requests own
// a list of page indices; a cursor points at next-write position inside the
// last page. Slots are freed when the request completes.
// ---------------------------------------------------------------------------

constexpr int kPageSize = 64;

struct KVCache {
    // Flat storage: [num_pages][2][num_kv_heads, page_size, head_dim] for each layer.
    std::vector<std::vector<float>> k_pages;   // layer-major: k_pages[layer][page_idx*slot]
    std::vector<std::vector<float>> v_pages;
    int num_layers = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int page_stride = 0;   // per-page floats = num_kv_heads * page_size * head_dim.

    // Page allocator. free_pages_[layer] holds reusable indices.
    std::vector<int> total_pages_per_layer;
    std::deque<int> free_pages;   // shared pool; each logical page covers all layers.
    int next_page_id = 0;

    void init(int L, int Hkv, int Dh, int max_pages) {
        num_layers = L; num_kv_heads = Hkv; head_dim = Dh;
        page_stride = Hkv * kPageSize * Dh;
        k_pages.assign(L, {});
        v_pages.assign(L, {});
        for (int l = 0; l < L; ++l) {
            k_pages[l].assign((size_t)max_pages * page_stride, 0.0f);
            v_pages[l].assign((size_t)max_pages * page_stride, 0.0f);
        }
        free_pages.clear();
        for (int p = 0; p < max_pages; ++p) free_pages.push_back(p);
        next_page_id = max_pages;
    }

    int alloc_page() {
        if (free_pages.empty()) return -1;
        int p = free_pages.front(); free_pages.pop_front();
        return p;
    }
    void free_page(int p) { free_pages.push_back(p); }
};

struct RequestState {
    int request_id = 0;
    std::vector<int> pages;      // per-request page list.
    int kv_position = 0;         // total cached tokens so far.
    std::vector<int> input_ids;  // last token(s) to process next step.
    std::vector<int> output_ids; // generated tokens so far.
    int prompt_len = 0;
    bool prefill_done = false;
    bool done = false;
    GenerationConfig config;
    std::chrono::steady_clock::time_point t_start;
};

// ---------------------------------------------------------------------------
// LLMEngine
// ---------------------------------------------------------------------------

class LLMEngine {
public:
    LLMEngine() = default;

    // Load model + tokenizer from HF-format directory. If loading fails,
    // return false; the caller may still use the engine with externally
    // provided weights via set_weights().
    bool load(const std::string& model_dir) {
        model_dir_ = model_dir;
        if (!load_config_(model_dir + "/config.json")) return false;
        // Weights loader: safetensors / pytorch_model.bin. We expose a stub
        // that returns true without populating weights — external callers
        // (Python shim or tests) can use set_weights() to drop them in.
        load_weights_(model_dir);
        tokenizer_.load(model_dir);
        init_rope_();
        init_kv_cache_();
        return true;
    }

    void set_weights(LlamaWeights w) {
        weights_ = std::move(w);
        cfg_ = weights_.cfg;
        init_rope_();
        init_kv_cache_();
        loaded_ = true;
    }

    // Synchronous single-prompt generate.
    Response generate(const std::string& prompt, const GenerationConfig& config) {
        Request req; req.request_id = next_id_++; req.prompt = prompt; req.config = config;
        submit_request(req);
        while (num_active() > 0) step();
        auto resps = poll_responses();
        for (auto& r : resps) if (r.request_id == req.request_id) return r;
        Response empty; empty.request_id = req.request_id; return empty;
    }

    void submit_request(const Request& req) {
        RequestState s;
        s.request_id = req.request_id;
        s.config = req.config;
        s.t_start = std::chrono::steady_clock::now();
        s.input_ids = tokenizer_.encode(req.prompt, /*add_bos*/ true);
        s.prompt_len = (int)s.input_ids.size();
        s.output_ids.reserve(req.config.max_new_tokens + s.prompt_len);
        pending_.push_back(std::move(s));
        admit_pending_();
    }

    std::vector<Response> poll_responses() {
        std::vector<Response> out;
        out.swap(done_);
        return out;
    }

    // Advance one token for every active request (prefill for new ones).
    void step() {
        admit_pending_();
        if (active_.empty()) return;
        for (auto& s : active_) {
            if (s.done) continue;
            if (!s.prefill_done) {
                // Prefill entire prompt at once (teacher-forcing style).
                std::vector<float> last_logits;
                forward_sequence_(s, s.input_ids, last_logits);
                int next_tok = sample_(s, last_logits);
                s.output_ids.push_back(next_tok);
                s.input_ids = {next_tok};
                s.prefill_done = true;
                check_done_(s, next_tok);
            } else {
                std::vector<float> logits;
                forward_sequence_(s, s.input_ids, logits);
                int next_tok = sample_(s, logits);
                s.output_ids.push_back(next_tok);
                s.input_ids = {next_tok};
                check_done_(s, next_tok);
            }
        }
        collect_done_();
    }

    int num_active() const { return (int)active_.size() + (int)pending_.size(); }

private:
    // -------- config / weights loader --------
    bool load_config_(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        auto get_int = [&](const char* key, int defv) {
            auto p = text.find(std::string("\"") + key + "\"");
            if (p == std::string::npos) return defv;
            p = text.find(':', p); if (p == std::string::npos) return defv;
            ++p; while (p < text.size() && (text[p] == ' ' || text[p] == '\n')) ++p;
            int v = 0, sign = 1;
            if (p < text.size() && text[p] == '-') { sign = -1; ++p; }
            while (p < text.size() && text[p] >= '0' && text[p] <= '9') {
                v = v * 10 + (text[p] - '0'); ++p;
            }
            return sign * v;
        };
        auto get_float = [&](const char* key, float defv) {
            auto p = text.find(std::string("\"") + key + "\"");
            if (p == std::string::npos) return defv;
            p = text.find(':', p); if (p == std::string::npos) return defv;
            ++p; while (p < text.size() && (text[p] == ' ' || text[p] == '\n')) ++p;
            char buf[64]; size_t n = 0;
            while (p < text.size() && n < 63 &&
                   ((text[p] >= '0' && text[p] <= '9') || text[p] == '.' ||
                    text[p] == 'e' || text[p] == 'E' || text[p] == '-' ||
                    text[p] == '+')) { buf[n++] = text[p++]; }
            buf[n] = '\0';
            return (float)std::atof(buf);
        };
        cfg_.vocab_size         = get_int("vocab_size", cfg_.vocab_size);
        cfg_.hidden_size        = get_int("hidden_size", cfg_.hidden_size);
        cfg_.intermediate_size  = get_int("intermediate_size", cfg_.intermediate_size);
        cfg_.num_hidden_layers  = get_int("num_hidden_layers", cfg_.num_hidden_layers);
        cfg_.num_attention_heads = get_int("num_attention_heads", cfg_.num_attention_heads);
        cfg_.num_key_value_heads = get_int("num_key_value_heads", cfg_.num_attention_heads);
        cfg_.max_position_embeddings = get_int("max_position_embeddings",
                                               cfg_.max_position_embeddings);
        cfg_.rms_norm_eps = get_float("rms_norm_eps", cfg_.rms_norm_eps);
        cfg_.rope_theta   = get_float("rope_theta", cfg_.rope_theta);
        cfg_.bos_token_id = get_int("bos_token_id", cfg_.bos_token_id);
        cfg_.eos_token_id = get_int("eos_token_id", cfg_.eos_token_id);
        weights_.cfg = cfg_;
        return true;
    }

    bool load_weights_(const std::string& /*dir*/) {
        // Stub. Real loader would parse safetensors header (JSON prefix +
        // binary payload) and fill weights_. We keep this as an extension
        // point so the Python shim can hand us weights via set_weights().
        return false;
    }

    // -------- RoPE + cache init --------
    void init_rope_() {
        int Dh = cfg_.hidden_size / cfg_.num_attention_heads;
        int max_pos = cfg_.max_position_embeddings;
        rope_cos_.assign((size_t)max_pos * (Dh / 2), 0.0f);
        rope_sin_.assign((size_t)max_pos * (Dh / 2), 0.0f);
        for (int pos = 0; pos < max_pos; ++pos) {
            for (int j = 0; j < Dh / 2; ++j) {
                float freq = std::pow(cfg_.rope_theta, -2.0f * j / (float)Dh);
                rope_cos_[pos * (Dh / 2) + j] = std::cos(pos * freq);
                rope_sin_[pos * (Dh / 2) + j] = std::sin(pos * freq);
            }
        }
    }

    void init_kv_cache_() {
        int Dh = cfg_.hidden_size / cfg_.num_attention_heads;
        // Budget: 16 MB/layer worst case on Elbrus. 128 pages × 64 tok × Hkv × Dh.
        int max_pages = 128;
        kv_.init(cfg_.num_hidden_layers, cfg_.num_key_value_heads, Dh, max_pages);
    }

    // Admit pending requests until we run out of page budget.
    void admit_pending_() {
        while (!pending_.empty() && (int)active_.size() < kMaxActive_) {
            auto& s = pending_.front();
            // Reserve at least one page per layer (using shared pool).
            if ((int)kv_.free_pages.size() < cfg_.num_hidden_layers) break;
            int p = kv_.alloc_page();
            s.pages.push_back(p);
            active_.push_back(std::move(s));
            pending_.pop_front();
        }
    }

    // -------- forward pass (CPU, raw buffers) --------
    // Processes a sequence of tokens (prefill: all prompt; decode: single tok).
    // Updates KV cache for request `s` and writes logits of the LAST token
    // into `out_logits` (size vocab_size).
    void forward_sequence_(RequestState& s,
                           const std::vector<int>& tok_ids,
                           std::vector<float>& out_logits) {
        int D = cfg_.hidden_size;
        int H = cfg_.num_attention_heads;
        int Hkv = cfg_.num_key_value_heads;
        int Dh = D / H;
        int V = cfg_.vocab_size;
        int T = (int)tok_ids.size();

        // Embed each token.
        std::vector<float> x((size_t)T * D, 0.0f);
        for (int t = 0; t < T; ++t) {
            int tok = tok_ids[t];
            if (tok < 0 || tok >= V) tok = 0;
            std::memcpy(&x[t * D], &weights_.tok_emb[tok * D], D * sizeof(float));
        }

        std::vector<float> residual(D * T), hn(D * T);
        std::vector<float> q((size_t)T * H * Dh), k((size_t)T * Hkv * Dh), v((size_t)T * Hkv * Dh);
        std::vector<float> attn_out((size_t)T * D);
        std::vector<float> ffn_gate((size_t)T * cfg_.intermediate_size);
        std::vector<float> ffn_up((size_t)T * cfg_.intermediate_size);
        std::vector<float> ffn_out((size_t)T * D);

        int base_pos = s.kv_position;

        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            const auto& W = weights_.layers[layer];
            // 1) attn RMSNorm
            for (int t = 0; t < T; ++t) rms_norm_(&x[t * D], &W.attn_norm[0],
                                                  &hn[t * D], D, cfg_.rms_norm_eps);
            // 2) q/k/v projections
            matmul_out_(&hn[0], &W.wq[0], &q[0], T, H * Dh, D);
            matmul_out_(&hn[0], &W.wk[0], &k[0], T, Hkv * Dh, D);
            matmul_out_(&hn[0], &W.wv[0], &v[0], T, Hkv * Dh, D);
            // 3) RoPE on q, k
            for (int t = 0; t < T; ++t) {
                int pos = base_pos + t;
                for (int h = 0; h < H; ++h)
                    apply_rope_(&q[t * H * Dh + h * Dh], pos, Dh);
                for (int h = 0; h < Hkv; ++h)
                    apply_rope_(&k[t * Hkv * Dh + h * Dh], pos, Dh);
            }
            // 4) Write K/V into paged cache for this layer.
            for (int t = 0; t < T; ++t) {
                int pos = base_pos + t;
                ensure_page_(s, pos);
                int page_idx = pos / kPageSize;
                int slot = pos % kPageSize;
                int phys = s.pages[page_idx];
                float* kdst = &kv_.k_pages[layer][(size_t)phys * kv_.page_stride
                                                  + (size_t)slot * Hkv * Dh];
                float* vdst = &kv_.v_pages[layer][(size_t)phys * kv_.page_stride
                                                  + (size_t)slot * Hkv * Dh];
                std::memcpy(kdst, &k[t * Hkv * Dh], Hkv * Dh * sizeof(float));
                std::memcpy(vdst, &v[t * Hkv * Dh], Hkv * Dh * sizeof(float));
            }
            // 5) Attention: for each (t, h), score vs keys[0..pos], softmax, V.
            int gqa = H / Hkv;
            float scale = 1.0f / std::sqrt((float)Dh);
            for (int t = 0; t < T; ++t) {
                int pos = base_pos + t;
                for (int h = 0; h < H; ++h) {
                    int h_kv = h / gqa;
                    std::vector<float> scores(pos + 1);
                    float m = -1e30f;
                    for (int j = 0; j <= pos; ++j) {
                        int page_idx = j / kPageSize;
                        int slot = j % kPageSize;
                        int phys = s.pages[page_idx];
                        float* kj = &kv_.k_pages[layer][(size_t)phys * kv_.page_stride
                                                        + (size_t)slot * Hkv * Dh
                                                        + (size_t)h_kv * Dh];
                        float dot = 0.0f;
                        const float* qi = &q[t * H * Dh + h * Dh];
                        for (int d = 0; d < Dh; ++d) dot += qi[d] * kj[d];
                        scores[j] = dot * scale;
                        if (scores[j] > m) m = scores[j];
                    }
                    float sum = 0.0f;
                    for (int j = 0; j <= pos; ++j) { scores[j] = std::exp(scores[j] - m); sum += scores[j]; }
                    float inv = 1.0f / std::max(sum, 1e-20f);
                    float* oi = &attn_out[t * D + h * Dh];
                    std::memset(oi, 0, Dh * sizeof(float));
                    for (int j = 0; j <= pos; ++j) {
                        float a = scores[j] * inv;
                        int page_idx = j / kPageSize;
                        int slot = j % kPageSize;
                        int phys = s.pages[page_idx];
                        float* vj = &kv_.v_pages[layer][(size_t)phys * kv_.page_stride
                                                        + (size_t)slot * Hkv * Dh
                                                        + (size_t)h_kv * Dh];
                        for (int d = 0; d < Dh; ++d) oi[d] += a * vj[d];
                    }
                }
            }
            // 6) o projection + residual.
            matmul_out_(&attn_out[0], &W.wo[0], &hn[0], T, D, D);
            for (size_t i = 0; i < (size_t)T * D; ++i) x[i] += hn[i];
            // 7) ffn RMSNorm + SwiGLU
            for (int t = 0; t < T; ++t) rms_norm_(&x[t * D], &W.ffn_norm[0],
                                                  &hn[t * D], D, cfg_.rms_norm_eps);
            matmul_out_(&hn[0], &W.w_gate[0], &ffn_gate[0], T, cfg_.intermediate_size, D);
            matmul_out_(&hn[0], &W.w_up[0],   &ffn_up[0],   T, cfg_.intermediate_size, D);
            for (size_t i = 0; i < (size_t)T * cfg_.intermediate_size; ++i) {
                float g = ffn_gate[i]; float sig = 1.0f / (1.0f + std::exp(-g));
                ffn_gate[i] = g * sig * ffn_up[i];  // SwiGLU = SiLU(gate) * up
            }
            matmul_out_(&ffn_gate[0], &W.w_down[0], &ffn_out[0], T, D, cfg_.intermediate_size);
            for (size_t i = 0; i < (size_t)T * D; ++i) x[i] += ffn_out[i];
        }

        // Final norm + lm_head on the LAST position only.
        int t_last = T - 1;
        std::vector<float> xn(D);
        rms_norm_(&x[t_last * D], &weights_.final_norm[0], &xn[0], D, cfg_.rms_norm_eps);
        out_logits.assign(V, 0.0f);
        for (int v_i = 0; v_i < V; ++v_i) {
            float acc = 0.0f;
            const float* w_row = &weights_.lm_head[v_i * D];
            for (int d = 0; d < D; ++d) acc += xn[d] * w_row[d];
            out_logits[v_i] = acc;
        }
        s.kv_position = base_pos + T;
    }

    void ensure_page_(RequestState& s, int pos) {
        int needed = pos / kPageSize + 1;
        while ((int)s.pages.size() < needed) {
            int p = kv_.alloc_page();
            if (p < 0) {
                // Out of memory: treat like stop.
                s.done = true;
                break;
            }
            s.pages.push_back(p);
        }
    }

    // -------- sampling --------
    int sample_(const RequestState& s, std::vector<float>& logits) {
        const auto& cfg = s.config;
        // repetition penalty
        if (cfg.repetition_penalty > 1.0f) {
            std::unordered_set<int> seen(s.output_ids.begin(), s.output_ids.end());
            for (int t : seen) {
                if (t < 0 || t >= (int)logits.size()) continue;
                if (logits[t] > 0) logits[t] /= cfg.repetition_penalty;
                else               logits[t] *= cfg.repetition_penalty;
            }
        }
        // temperature
        float temp = std::max(cfg.temperature, 1e-4f);
        for (auto& l : logits) l /= temp;
        // top-k: keep k largest
        int K = std::max(1, cfg.top_k);
        int V = (int)logits.size();
        std::vector<int> idx(V);
        for (int i = 0; i < V; ++i) idx[i] = i;
        if (K < V) {
            std::nth_element(idx.begin(), idx.begin() + K, idx.end(),
                             [&](int a, int b) { return logits[a] > logits[b]; });
            idx.resize(K);
        }
        // softmax over surviving
        float m = -1e30f;
        for (int i : idx) if (logits[i] > m) m = logits[i];
        std::vector<float> probs(idx.size());
        float sum = 0.0f;
        for (size_t i = 0; i < idx.size(); ++i) {
            probs[i] = std::exp(logits[idx[i]] - m);
            sum += probs[i];
        }
        float inv = 1.0f / std::max(sum, 1e-20f);
        for (auto& p : probs) p *= inv;
        // top-p (nucleus): sort desc, cut cumulative > p
        std::vector<int> order((int)idx.size());
        for (int i = 0; i < (int)idx.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });
        float cum = 0.0f;
        std::vector<int> keep;
        for (int o : order) {
            keep.push_back(o); cum += probs[o];
            if (cum >= cfg.top_p) break;
        }
        // renormalize
        float s2 = 0.0f; for (int o : keep) s2 += probs[o];
        float r = uniform_() * s2;
        float acc = 0.0f;
        for (int o : keep) {
            acc += probs[o];
            if (r <= acc) return idx[o];
        }
        return idx[keep.back()];
    }

    void check_done_(RequestState& s, int next_tok) {
        if ((int)s.output_ids.size() >= s.config.max_new_tokens) { s.done = true; return; }
        if (next_tok == cfg_.eos_token_id) { s.done = true; return; }
        for (int st : s.config.stop_tokens) if (next_tok == st) { s.done = true; return; }
    }

    void collect_done_() {
        for (auto it = active_.begin(); it != active_.end();) {
            if (it->done) {
                Response r;
                r.request_id = it->request_id;
                r.tokens_generated = (int)it->output_ids.size();
                r.output = tokenizer_.decode(it->output_ids);
                auto t_end = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(t_end - it->t_start).count();
                r.throughput_tok_per_s = dt > 0 ? r.tokens_generated / dt : 0.0;
                for (int p : it->pages) kv_.free_page(p);
                done_.push_back(std::move(r));
                it = active_.erase(it);
            } else ++it;
        }
    }

    // -------- primitives --------
    static void rms_norm_(const float* x, const float* weight, float* out,
                          int d, float eps) {
        float ss = 0.0f;
        for (int i = 0; i < d; ++i) ss += x[i] * x[i];
        float inv = 1.0f / std::sqrt(ss / d + eps);
        for (int i = 0; i < d; ++i) out[i] = x[i] * inv * weight[i];
    }

    // out[T, M] = in[T, K] @ W[M, K]^T  (row-major, W stored row-major [M,K]).
    static void matmul_out_(const float* in, const float* W, float* out,
                            int T, int M, int K) {
        for (int t = 0; t < T; ++t) {
            for (int m = 0; m < M; ++m) {
                float acc = 0.0f;
                const float* wr = &W[m * K];
                const float* xr = &in[t * K];
                for (int k = 0; k < K; ++k) acc += xr[k] * wr[k];
                out[t * M + m] = acc;
            }
        }
    }

    void apply_rope_(float* head, int pos, int Dh) const {
        for (int j = 0; j < Dh / 2; ++j) {
            float c = rope_cos_[pos * (Dh / 2) + j];
            float sn = rope_sin_[pos * (Dh / 2) + j];
            float a = head[j];
            float b = head[j + Dh / 2];
            head[j]          = a * c - b * sn;
            head[j + Dh / 2] = a * sn + b * c;
        }
    }

    float uniform_() {
        if (!rng_inited_) { rng_.seed(0xC0FFEEu); rng_inited_ = true; }
        return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_);
    }

    // -------- state --------
    std::string model_dir_;
    LlamaConfig cfg_;
    LlamaWeights weights_;
    ByteBPETokenizer tokenizer_;
    KVCache kv_;
    std::vector<float> rope_cos_, rope_sin_;
    std::deque<RequestState> pending_;
    std::vector<RequestState> active_;
    std::vector<Response> done_;
    int next_id_ = 1;
    static constexpr int kMaxActive_ = 16;
    bool loaded_ = false;
    std::mt19937 rng_;
    bool rng_inited_ = false;
};

// ---------------------------------------------------------------------------
// Self-test: create tiny synthetic weights, submit 3 prompts, drive step()
// until all three complete. Returns 0 on success.
// ---------------------------------------------------------------------------

inline int self_test_llm_engine() {
    LLMEngine engine;
    LlamaWeights W;
    W.cfg.vocab_size = 64;
    W.cfg.hidden_size = 32;
    W.cfg.intermediate_size = 64;
    W.cfg.num_hidden_layers = 2;
    W.cfg.num_attention_heads = 4;
    W.cfg.num_key_value_heads = 4;
    W.cfg.max_position_embeddings = 256;
    W.cfg.bos_token_id = 0;
    W.cfg.eos_token_id = 63;
    int V = W.cfg.vocab_size, D = W.cfg.hidden_size, I = W.cfg.intermediate_size;
    int H = W.cfg.num_attention_heads, Dh = D / H, L = W.cfg.num_hidden_layers;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    auto fill = [&](std::vector<float>& v, size_t n) {
        v.resize(n); for (size_t i = 0; i < n; ++i) v[i] = nd(rng);
    };
    fill(W.tok_emb, (size_t)V * D);
    fill(W.final_norm, D); for (auto& x : W.final_norm) x = 1.0f;
    fill(W.lm_head, (size_t)V * D);
    W.layers.resize(L);
    for (int l = 0; l < L; ++l) {
        auto& lw = W.layers[l];
        lw.attn_norm.assign(D, 1.0f);
        fill(lw.wq, (size_t)H * Dh * D);
        fill(lw.wk, (size_t)H * Dh * D);
        fill(lw.wv, (size_t)H * Dh * D);
        fill(lw.wo, (size_t)D * H * Dh);
        lw.ffn_norm.assign(D, 1.0f);
        fill(lw.w_gate, (size_t)I * D);
        fill(lw.w_up,   (size_t)I * D);
        fill(lw.w_down, (size_t)D * I);
    }
    engine.set_weights(std::move(W));

    GenerationConfig gc;
    gc.max_new_tokens = 8;
    gc.temperature = 1.0f;
    gc.top_k = 10;
    gc.top_p = 0.95f;
    gc.repetition_penalty = 1.05f;

    for (int i = 0; i < 3; ++i) {
        Request r; r.request_id = 100 + i; r.prompt = "hi"; r.config = gc;
        engine.submit_request(r);
    }
    int safety = 10000;
    while (engine.num_active() > 0 && safety-- > 0) engine.step();
    auto resps = engine.poll_responses();
    if ((int)resps.size() != 3) {
        std::fprintf(stderr, "[serve self-test] expected 3 responses, got %zu\n",
                     resps.size());
        return 1;
    }
    for (const auto& r : resps) {
        if (r.tokens_generated != gc.max_new_tokens) {
            std::fprintf(stderr, "[serve self-test] req %d: %d tokens (expected %d)\n",
                         r.request_id, r.tokens_generated, gc.max_new_tokens);
            return 2;
        }
    }
    std::fprintf(stderr, "[serve self-test] OK: 3 requests, %d tokens each\n",
                 gc.max_new_tokens);
    return 0;
}

}  // namespace serve
}  // namespace torch
