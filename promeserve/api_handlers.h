#pragma once

// ============================================================================
// PromeServe — Ollama-Compatible API Handlers
//
// Implements REST endpoints matching Ollama's API format so existing clients
// (Open WebUI, LangChain, etc.) work out of the box.
//
// Endpoints:
//   POST /api/generate  — text completion (streaming NDJSON)
//   POST /api/chat      — chat completion (streaming NDJSON)
//   GET  /api/tags      — list available models
//   POST /api/show      — model info
//   GET  /api/version   — server version
//   GET  /               — health check
// ============================================================================

#include "promeserve/http_server.h"
#include "promeserve/model_manager.h"

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <ctime>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>

namespace promeserve {

// ============================================================================
// Minimal JSON Parser (hand-written, no external deps)
// ============================================================================

namespace json {

// Skip whitespace
inline size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
        ++pos;
    return pos;
}

// Parse a JSON string value (assumes pos points to opening '"')
inline std::string parse_string(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') return "";
    ++pos; // skip opening quote
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            ++pos;
            switch (s[pos]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                case 'u': {
                    // Parse 4 hex digits for Unicode escape
                    if (pos + 4 < s.size()) {
                        std::string hex = s.substr(pos + 1, 4);
                        unsigned int codepoint = 0;
                        for (char c : hex) {
                            codepoint <<= 4;
                            if (c >= '0' && c <= '9') codepoint |= (c - '0');
                            else if (c >= 'a' && c <= 'f') codepoint |= (c - 'a' + 10);
                            else if (c >= 'A' && c <= 'F') codepoint |= (c - 'A' + 10);
                        }
                        // Simple ASCII range handling
                        if (codepoint < 0x80) {
                            result += static_cast<char>(codepoint);
                        } else if (codepoint < 0x800) {
                            result += static_cast<char>(0xC0 | (codepoint >> 6));
                            result += static_cast<char>(0x80 | (codepoint & 0x3F));
                        } else {
                            result += static_cast<char>(0xE0 | (codepoint >> 12));
                            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (codepoint & 0x3F));
                        }
                        pos += 4;
                    }
                    break;
                }
                default: result += s[pos]; break;
            }
        } else {
            result += s[pos];
        }
        ++pos;
    }
    if (pos < s.size()) ++pos; // skip closing quote
    return result;
}

// Parse a JSON number (integer or float) as string
inline std::string parse_number(const std::string& s, size_t& pos) {
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    if (pos < s.size() && s[pos] == '.') {
        ++pos;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    }
    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
        ++pos;
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    }
    return s.substr(start, pos - start);
}

// Skip a JSON value (string, number, object, array, bool, null)
inline void skip_value(const std::string& s, size_t& pos) {
    pos = skip_ws(s, pos);
    if (pos >= s.size()) return;

    if (s[pos] == '"') {
        parse_string(s, pos);
    } else if (s[pos] == '{') {
        int depth = 1;
        ++pos;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '{') ++depth;
            else if (s[pos] == '}') --depth;
            else if (s[pos] == '"') parse_string(s, pos);
            if (depth > 0) ++pos;
        }
        if (pos < s.size()) ++pos;
    } else if (s[pos] == '[') {
        int depth = 1;
        ++pos;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '[') ++depth;
            else if (s[pos] == ']') --depth;
            else if (s[pos] == '"') parse_string(s, pos);
            if (depth > 0) ++pos;
        }
        if (pos < s.size()) ++pos;
    } else {
        // number, bool, null — read until delimiter
        while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']'
               && s[pos] != ' ' && s[pos] != '\n' && s[pos] != '\r' && s[pos] != '\t')
            ++pos;
    }
}

// Represents a parsed JSON value
struct JsonValue {
    enum Type { NONE, STRING, NUMBER, BOOL, OBJECT, ARRAY, NUL };
    Type type = NONE;
    std::string str_val;
    double num_val = 0.0;
    bool bool_val = false;
    std::map<std::string, JsonValue> obj;
    std::vector<JsonValue> arr;

    bool is_string() const { return type == STRING; }
    bool is_number() const { return type == NUMBER; }
    bool is_bool() const { return type == BOOL; }
    bool is_object() const { return type == OBJECT; }
    bool is_array() const { return type == ARRAY; }
    bool is_null() const { return type == NUL; }

    std::string as_string(const std::string& def = "") const {
        return (type == STRING) ? str_val : def;
    }
    double as_number(double def = 0.0) const {
        return (type == NUMBER) ? num_val : def;
    }
    int as_int(int def = 0) const {
        return (type == NUMBER) ? static_cast<int>(num_val) : def;
    }
    bool as_bool(bool def = false) const {
        return (type == BOOL) ? bool_val : def;
    }

    bool has(const std::string& key) const {
        return type == OBJECT && obj.count(key) > 0;
    }

    const JsonValue& operator[](const std::string& key) const {
        static JsonValue empty;
        if (type != OBJECT) return empty;
        auto it = obj.find(key);
        return (it != obj.end()) ? it->second : empty;
    }
};

// Forward declaration
inline JsonValue parse_value(const std::string& s, size_t& pos);

inline JsonValue parse_object(const std::string& s, size_t& pos) {
    JsonValue val;
    val.type = JsonValue::OBJECT;
    ++pos; // skip '{'
    pos = skip_ws(s, pos);
    while (pos < s.size() && s[pos] != '}') {
        pos = skip_ws(s, pos);
        if (s[pos] != '"') break;
        std::string key = parse_string(s, pos);
        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ':') ++pos;
        pos = skip_ws(s, pos);
        val.obj[key] = parse_value(s, pos);
        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') ++pos;
    }
    if (pos < s.size()) ++pos; // skip '}'
    return val;
}

inline JsonValue parse_array(const std::string& s, size_t& pos) {
    JsonValue val;
    val.type = JsonValue::ARRAY;
    ++pos; // skip '['
    pos = skip_ws(s, pos);
    while (pos < s.size() && s[pos] != ']') {
        val.arr.push_back(parse_value(s, pos));
        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') ++pos;
        pos = skip_ws(s, pos);
    }
    if (pos < s.size()) ++pos; // skip ']'
    return val;
}

inline JsonValue parse_value(const std::string& s, size_t& pos) {
    pos = skip_ws(s, pos);
    if (pos >= s.size()) return JsonValue{};

    JsonValue val;
    if (s[pos] == '"') {
        val.type = JsonValue::STRING;
        val.str_val = parse_string(s, pos);
    } else if (s[pos] == '{') {
        return parse_object(s, pos);
    } else if (s[pos] == '[') {
        return parse_array(s, pos);
    } else if (s[pos] == 't' && s.substr(pos, 4) == "true") {
        val.type = JsonValue::BOOL;
        val.bool_val = true;
        pos += 4;
    } else if (s[pos] == 'f' && s.substr(pos, 5) == "false") {
        val.type = JsonValue::BOOL;
        val.bool_val = false;
        pos += 5;
    } else if (s[pos] == 'n' && s.substr(pos, 4) == "null") {
        val.type = JsonValue::NUL;
        pos += 4;
    } else {
        // Number
        val.type = JsonValue::NUMBER;
        std::string num_str = parse_number(s, pos);
        val.num_val = std::stod(num_str);
    }
    return val;
}

inline JsonValue parse(const std::string& s) {
    size_t pos = 0;
    return parse_value(s, pos);
}

}  // namespace json

// ============================================================================
// JSON Serialization helpers
// ============================================================================

inline std::string json_escape(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // Control character — encode as \u00XX
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

inline std::string iso8601_now() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    struct tm tm_buf;
#ifdef _WIN32
    gmtime_s(&tm_buf, &time_t_now);
#else
    gmtime_r(&time_t_now, &tm_buf);
#endif

    char buf[64];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
             tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
             static_cast<int>(ms.count()));
    return std::string(buf);
}

inline std::string duration_ns_str(double seconds) {
    // Ollama reports durations in nanoseconds as integers
    int64_t ns = static_cast<int64_t>(seconds * 1e9);
    return std::to_string(ns);
}

// ============================================================================
// API Handlers
// ============================================================================

class ApiHandlers {
public:
    ApiHandlers(ModelManager& models) : models_(models), server_timeout_ms_(60000) {}

    // Set per-request generation timeout (milliseconds). 0 = no timeout.
    void set_timeout_ms(int ms) { server_timeout_ms_ = ms; }
    int timeout_ms() const { return server_timeout_ms_; }

    // Register all routes on the given server
    void register_routes(HttpServer& server) {
        // Pick up timeout from server config so the CLI only needs to set it once.
        server_timeout_ms_ = server.server_timeout_ms();

        // Health check
        server.route("GET", "/", [this](const HttpRequest& req) -> HttpResponse {
            return handle_health(req);
        });

        // Version
        server.route("GET", "/api/version", [this](const HttpRequest& req) -> HttpResponse {
            return handle_version(req);
        });

        // List models
        server.route("GET", "/api/tags", [this](const HttpRequest& req) -> HttpResponse {
            return handle_tags(req);
        });

        // Show model info
        server.route("POST", "/api/show", [this](const HttpRequest& req) -> HttpResponse {
            return handle_show(req);
        });

        // Generate (streaming)
        server.route_stream("POST", "/api/generate",
            [this](const HttpRequest& req, StreamWriter& writer) -> HttpResponse {
                return handle_generate(req, writer);
            });

        // Chat (streaming)
        server.route_stream("POST", "/api/chat",
            [this](const HttpRequest& req, StreamWriter& writer) -> HttpResponse {
                return handle_chat(req, writer);
            });

        // Embeddings stub — 501 Not Implemented (better than 404)
        server.route("POST", "/api/embeddings", [this](const HttpRequest& req) -> HttpResponse {
            return handle_embeddings(req);
        });
        server.route("GET", "/api/embeddings", [this](const HttpRequest& req) -> HttpResponse {
            return handle_embeddings(req);
        });
    }

private:
    // ========================================================================
    // GET / — serve Web UI (index.html) or health check
    // ========================================================================
    HttpResponse handle_health(const HttpRequest& req) {
        HttpResponse resp;
        resp.status = 200;

        // Try to serve index.html from web/ directory
        std::vector<std::string> paths = {
            "promeserve/web/index.html",
            "../promeserve/web/index.html",
            "web/index.html"
        };
        for (const auto& path : paths) {
            std::ifstream f(path);
            if (f.good()) {
                std::string html((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
                resp.body = html;
                resp.headers["Content-Type"] = "text/html; charset=utf-8";
                return resp;
            }
        }

        // Fallback: plain text health check
        resp.body = "PromeServe is running";
        resp.headers["Content-Type"] = "text/plain";
        return resp;
    }

    // ========================================================================
    // GET /api/version
    // ========================================================================
    HttpResponse handle_version(const HttpRequest&) {
        HttpResponse resp;
        resp.set_json("{\"version\":\"0.1.0-promeserve\"}");
        return resp;
    }

    // ========================================================================
    // GET /api/tags — list available models
    // ========================================================================
    HttpResponse handle_tags(const HttpRequest&) {
        auto model_list = models_.list_models();

        std::ostringstream oss;
        oss << "{\"models\":[";
        bool first = true;
        for (auto& m : model_list) {
            if (!first) oss << ",";
            first = false;
            oss << "{\"name\":\"" << json_escape(m.name) << "\""
                << ",\"model\":\"" << json_escape(m.name) << "\""
                << ",\"modified_at\":\"" << iso8601_now() << "\""
                << ",\"size\":" << m.size_bytes
                << ",\"digest\":\"" << json_escape(m.digest) << "\""
                << ",\"details\":{\"family\":\"" << json_escape(m.family) << "\""
                << ",\"format\":\"gguf\""
                << ",\"parameter_size\":\"" << json_escape(m.tag) << "\""
                << "}}";
        }
        oss << "]}";

        HttpResponse resp;
        resp.set_json(oss.str());
        return resp;
    }

    // ========================================================================
    // POST /api/show — model information
    // ========================================================================
    HttpResponse handle_show(const HttpRequest& req) {
        auto body = json::parse(req.body);
        std::string model_name = body["name"].as_string(body["model"].as_string());

        if (model_name.empty()) {
            HttpResponse resp;
            resp.status = 400;
            resp.set_json("{\"error\":\"model name required\"}");
            return resp;
        }

        if (!models_.has_model(model_name)) {
            HttpResponse resp;
            resp.status = 404;
            resp.set_json("{\"error\":\"model not found: " + json_escape(model_name) + "\"}");
            return resp;
        }

        auto info = models_.get_model_info(model_name);

        std::ostringstream oss;
        oss << "{\"modelfile\":\"FROM " << json_escape(model_name) << "\""
            << ",\"parameters\":\"\""
            << ",\"template\":\"\""
            << ",\"details\":{"
            << "\"family\":\"" << json_escape(info.family) << "\""
            << ",\"format\":\"gguf\""
            << ",\"parameter_size\":\"" << json_escape(info.tag) << "\""
            << "}"
            << ",\"model_info\":{"
            << "\"general.architecture\":\"" << json_escape(info.architecture) << "\""
            << ",\"general.file_size\":" << info.size_bytes
            << "}}";

        HttpResponse resp;
        resp.set_json(oss.str());
        return resp;
    }

    // ========================================================================
    // POST /api/embeddings — not implemented (stub returns 501)
    // ========================================================================
    HttpResponse handle_embeddings(const HttpRequest&) {
        HttpResponse resp;
        resp.status = 501;
        resp.set_json("{\"error\":\"embeddings not implemented\"}");
        return resp;
    }

    // ========================================================================
    // POST /api/generate — text generation (streaming NDJSON)
    // ========================================================================
    HttpResponse handle_generate(const HttpRequest& req, StreamWriter& writer) {
#ifdef PT_DEBUG_HTTP
        std::cerr << "[Generate] Request body (" << req.body.size() << " bytes): "
                  << req.body.substr(0, 200) << std::endl;
#endif
        auto body = json::parse(req.body);
        std::string model_name = body["model"].as_string();
        std::string prompt = body["prompt"].as_string();
        bool stream = body.has("stream") ? body["stream"].as_bool(true) : true;
#ifdef PT_DEBUG_HTTP
        std::cerr << "[Generate] Parsed: model=\"" << model_name
                  << "\" prompt=\"" << prompt.substr(0, 80)
                  << "\" stream=" << stream << std::endl;
#endif

        // Generation parameters (Ollama-compatible)
        int max_tokens = 128;
        float temperature = 0.7f;
        int top_k = 40;
        float top_p = 0.9f;
        float repeat_penalty = 1.05f;

        if (body.has("options")) {
            auto& opts = body["options"];
            if (opts.has("num_predict")) max_tokens = opts["num_predict"].as_int(128);
            if (opts.has("temperature")) temperature = static_cast<float>(opts["temperature"].as_number(0.7));
            if (opts.has("top_k")) top_k = opts["top_k"].as_int(40);
            if (opts.has("top_p")) top_p = static_cast<float>(opts["top_p"].as_number(0.9));
            if (opts.has("repeat_penalty")) repeat_penalty = static_cast<float>(opts["repeat_penalty"].as_number(1.05));
        }

        // NOTE: Server already sent 200 + streaming headers before calling us.
        // All responses must go through writer as NDJSON chunks.
        HttpResponse resp;
        resp.status = 200;
        resp.set_ndjson_streaming();

        if (model_name.empty()) {
            writer.write("{\"error\":\"model name required\",\"done\":true}\n");
            writer.finish();
            return resp;
        }

        // Load model if needed
        if (!ensure_model_loaded(model_name)) {
            std::cerr << "[Generate] ERROR: failed to load model: " << model_name << std::endl;
            writer.write("{\"error\":\"failed to load model: " + json_escape(model_name) + "\",\"done\":true}\n");
            writer.finish();
            return resp;
        }

        auto t_total_start = std::chrono::high_resolution_clock::now();

        // We need a lock since the model is shared
        std::lock_guard<std::mutex> lock(generate_mutex_);

        auto* model = models_.get_loaded_model();
        if (!model) {
            std::cerr << "[Generate] ERROR: model pointer is null after load" << std::endl;
            writer.write("{\"error\":\"model not loaded\",\"done\":true}\n");
            writer.finish();
            return resp;
        }

#ifdef PT_DEBUG_HTTP
        std::cerr << "[Generate] Model ready: " << model_name
                  << " use_cuda=" << model->use_cuda_
                  << " prompt=\"" << prompt.substr(0, 100) << "\"" << std::endl;
#endif

        // Auto-apply chat template if prompt doesn't already have one
        bool needs_template = (prompt.find("<|im_start|>") == std::string::npos &&
                               prompt.find("<s>") == std::string::npos &&
                               prompt.find("[INST]") == std::string::npos);

        try {
            generate_streaming(model, model_name, prompt, needs_template,
                              max_tokens, temperature, top_k, top_p, repeat_penalty,
                              stream, writer, t_total_start);
        } catch (const std::exception& e) {
            std::cerr << "[Generate] ERROR: " << e.what() << std::endl;
            if (!writer.is_closed()) {
                writer.write("{\"error\":\"internal error: " + json_escape(e.what()) + "\",\"done\":true}\n");
                writer.finish();
            }
        } catch (...) {
            std::cerr << "[Generate] ERROR: unknown exception" << std::endl;
            if (!writer.is_closed()) {
                writer.write("{\"error\":\"internal error: unknown exception\",\"done\":true}\n");
                writer.finish();
            }
        }

        return resp;
    }

    // ========================================================================
    // POST /api/chat — chat completion (streaming NDJSON)
    // ========================================================================
    HttpResponse handle_chat(const HttpRequest& req, StreamWriter& writer) {
#ifdef PT_DEBUG_HTTP
        std::cerr << "[Chat] Request body (" << req.body.size() << " bytes): "
                  << req.body.substr(0, 200) << std::endl;
#endif
        auto body = json::parse(req.body);
        std::string model_name = body["model"].as_string();
        bool stream = body.has("stream") ? body["stream"].as_bool(true) : true;
#ifdef PT_DEBUG_HTTP
        std::cerr << "[Chat] Parsed: model=\"" << model_name
                  << "\" stream=" << stream << std::endl;
#endif

        // Parse messages array
        std::string formatted_prompt;
        if (body.has("messages") && body["messages"].is_array()) {
            formatted_prompt = format_chat_messages(body["messages"], model_name);
        }

        // Generation parameters
        int max_tokens = 128;
        float temperature = 0.7f;
        int top_k = 40;
        float top_p = 0.9f;
        float repeat_penalty = 1.05f;

        if (body.has("options")) {
            auto& opts = body["options"];
            if (opts.has("num_predict")) max_tokens = opts["num_predict"].as_int(128);
            if (opts.has("temperature")) temperature = static_cast<float>(opts["temperature"].as_number(0.7));
            if (opts.has("top_k")) top_k = opts["top_k"].as_int(40);
            if (opts.has("top_p")) top_p = static_cast<float>(opts["top_p"].as_number(0.9));
            if (opts.has("repeat_penalty")) repeat_penalty = static_cast<float>(opts["repeat_penalty"].as_number(1.05));
        }

        HttpResponse resp;
        resp.status = 200;
        resp.set_ndjson_streaming();

        if (model_name.empty()) {
            writer.write("{\"error\":\"model name required\",\"done\":true}\n");
            writer.finish();
            return resp;
        }

        if (formatted_prompt.empty()) {
            writer.write("{\"error\":\"messages array required\",\"done\":true}\n");
            writer.finish();
            return resp;
        }

        // Load model if needed
        if (!ensure_model_loaded(model_name)) {
            std::cerr << "[Chat] ERROR: failed to load model: " << model_name << std::endl;
            writer.write("{\"error\":\"failed to load model: " + json_escape(model_name) + "\",\"done\":true}\n");
            writer.finish();
            return resp;
        }

        auto t_total_start = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(generate_mutex_);

        auto* model = models_.get_loaded_model();
        if (!model) {
            std::cerr << "[Chat] ERROR: model pointer is null after load" << std::endl;
            writer.write("{\"error\":\"model not loaded\",\"done\":true}\n");
            writer.finish();
            return resp;
        }

#ifdef PT_DEBUG_HTTP
        std::cerr << "[Chat] Model ready: " << model_name
                  << " use_cuda=" << model->use_cuda_
                  << " prompt_len=" << formatted_prompt.size() << std::endl;
#endif

        // For chat, the prompt is already formatted with chat template
        try {
            generate_streaming_chat(model, model_name, formatted_prompt,
                                   max_tokens, temperature, top_k, top_p, repeat_penalty,
                                   stream, writer, t_total_start);
        } catch (const std::exception& e) {
            std::cerr << "[Chat] ERROR: " << e.what() << std::endl;
            if (!writer.is_closed()) {
                writer.write("{\"error\":\"internal error: " + json_escape(e.what()) + "\",\"done\":true}\n");
                writer.finish();
            }
        } catch (...) {
            std::cerr << "[Chat] ERROR: unknown exception" << std::endl;
            if (!writer.is_closed()) {
                writer.write("{\"error\":\"internal error: unknown exception\",\"done\":true}\n");
                writer.finish();
            }
        }

        return resp;
    }

    // ========================================================================
    // Streaming generation core
    // ========================================================================

    void generate_streaming(torch::io::GGUFModel* model, const std::string& model_name,
                           const std::string& prompt, bool apply_template,
                           int max_tokens, float temperature, int top_k, float top_p,
                           float repeat_penalty, bool stream,
                           StreamWriter& writer,
                           std::chrono::high_resolution_clock::time_point t_total_start) {

        std::string created_at = iso8601_now();

#ifdef PT_DEBUG_HTTP
        std::cerr << "[Generate] START model=" << model_name
                  << " use_cuda=" << model->use_cuda_
                  << " prompt_len=" << prompt.size()
                  << " max_tokens=" << max_tokens
                  << " temp=" << temperature
                  << " stream=" << stream << std::endl;
#endif

        // Reset KV cache
        model->kv_cache.reset();
        int64_t kv_dim = model->config.num_kv_heads * model->config.head_dim;
        // Fixed KV cache size to avoid reallocation (which invalidates CUDA Graph)
        int64_t max_total_seq = 4096;
        if (max_total_seq > model->config.context_length)
            max_total_seq = model->config.context_length;

        if (!model->kv_cache.allocated || model->kv_cache.max_seq < max_total_seq) {
            // Invalidate CUDA Graph — KV cache pointers will change
            model->invalidate_graph();
            try {
                model->kv_cache.allocate(model->config.num_layers, max_total_seq, kv_dim,
                                         model->use_cuda_);
            } catch (const std::exception& e) {
                std::cerr << "[Generate] ERROR: KV cache allocation failed: " << e.what() << std::endl;
                std::string err = "{\"model\":\"" + json_escape(model_name) + "\""
                    + ",\"created_at\":\"" + created_at + "\""
                    + ",\"response\":\"\",\"done\":true"
                    + ",\"done_reason\":\"error: KV cache alloc failed: " + json_escape(e.what()) + "\"}\n";
                writer.write(err);
                writer.finish();
                return;
            }
        }

        // Encode prompt
        std::string actual_prompt = apply_template ? model->apply_chat_template(prompt) : prompt;
        auto input_tokens = model->tokenizer.encode(actual_prompt, true);
        int prompt_tokens = static_cast<int>(input_tokens.size());

        if (input_tokens.empty()) {
            std::string err = "{\"model\":\"" + json_escape(model_name) + "\""
                + ",\"created_at\":\"" + created_at + "\""
                + ",\"response\":\"\",\"done\":true"
                + ",\"done_reason\":\"error: empty prompt after tokenization\"}\n";
            writer.write(err);
            writer.finish();
            return;
        }

        // Prefill
        at::Tensor logits;
        auto t_prompt_start = std::chrono::high_resolution_clock::now();
        try {
            std::vector<int64_t> tokens_i64(input_tokens.begin(), input_tokens.end());
            logits = model->forward(tokens_i64, true);
        } catch (const std::exception& e) {
            std::cerr << "[Generate] ERROR in prefill forward(): " << e.what() << std::endl;
            std::string err = "{\"model\":\"" + json_escape(model_name) + "\""
                + ",\"created_at\":\"" + created_at + "\""
                + ",\"response\":\"\",\"done\":true"
                + ",\"done_reason\":\"error: prefill failed: " + json_escape(e.what()) + "\"}\n";
            writer.write(err);
            writer.finish();
            return;
        }
        auto t_prompt_end = std::chrono::high_resolution_clock::now();

        double prompt_eval_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();

        // Decode tokens one by one
        auto t_eval_start = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> generated;
        std::string full_response;

        const int timeout_ms_local = server_timeout_ms_;
        bool timed_out = false;
        for (int step = 0; step < max_tokens; ++step) {
            // Request timeout check — cheap, once per decode step
            if (timeout_ms_local > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(now - t_total_start).count();
                if (elapsed_ms > timeout_ms_local) {
                    writer.write("{\"error\":\"timeout\",\"done\":true}\n");
                    timed_out = true;
                    break;
                }
            }

            try {
                int64_t last_pos = logits.size(0) - 1;
                at::Tensor last_logits = model->get_row(logits, last_pos);

                // Apply repetition penalty on CPU
                if (repeat_penalty > 1.0f && !generated.empty()) {
                    float* logit_data = last_logits.mutable_data_ptr<float>();
                    for (int32_t prev : generated) {
                        if (prev >= 0 && prev < static_cast<int32_t>(model->tokenizer.vocab.size())) {
                            if (logit_data[prev] > 0) logit_data[prev] /= repeat_penalty;
                            else logit_data[prev] *= repeat_penalty;
                        }
                    }
                }

                int32_t next_token = model->sample_token(last_logits, temperature, top_k, top_p);

                // Check stop conditions
                if (next_token == model->tokenizer.eos_id || model->is_stop_token(next_token)) {
                    break;
                }

                generated.push_back(next_token);
                std::string token_str = model->tokenizer.decode_token(next_token);
                full_response += token_str;

                // Stream this token
                if (stream) {
                    std::ostringstream chunk;
                    chunk << "{\"model\":\"" << json_escape(model_name) << "\""
                          << ",\"created_at\":\"" << created_at << "\""
                          << ",\"response\":\"" << json_escape(token_str) << "\""
                          << ",\"done\":false}\n";
                    if (!writer.write(chunk.str())) {
                        return;  // Client gone, stop generating
                    }
                }

                // Next forward pass
#ifdef PT_USE_CUDA
                if (model->use_cuda_) {
                    logits = model->forward_decode(static_cast<int64_t>(next_token));
                } else
#endif
                if (model->use_quant_gemv_) {
                    // Zero-allocation optimized CPU decode path (13.5 tok/s)
                    logits = model->forward_decode_cpu(static_cast<int64_t>(next_token));
                } else {
                    std::vector<int64_t> next_input = {static_cast<int64_t>(next_token)};
                    logits = model->forward(next_input, true);
                }
            } catch (const std::exception& e) {
                std::cerr << "[Generate] ERROR at decode step " << step << ": " << e.what() << std::endl;
                // Send error in stream and break
                std::string err_chunk = "{\"model\":\"" + json_escape(model_name) + "\""
                    + ",\"created_at\":\"" + created_at + "\""
                    + ",\"response\":\"\",\"done\":true"
                    + ",\"done_reason\":\"error: decode failed at step " + std::to_string(step)
                    + ": " + json_escape(e.what()) + "\"}\n";
                writer.write(err_chunk);
                writer.finish();
                return;
            }
        }
        if (timed_out) {
            writer.finish();
            return;
        }

        auto t_eval_end = std::chrono::high_resolution_clock::now();
        double eval_ms = std::chrono::duration<double, std::milli>(t_eval_end - t_eval_start).count();
        auto t_total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

        int eval_count = static_cast<int>(generated.size());

        // Send final response with stats
        std::ostringstream final_chunk;
        final_chunk << "{\"model\":\"" << json_escape(model_name) << "\""
                    << ",\"created_at\":\"" << created_at << "\"";

        if (stream) {
            final_chunk << ",\"response\":\"\"";
        } else {
            final_chunk << ",\"response\":\"" << json_escape(full_response) << "\"";
        }

        final_chunk << ",\"done\":true"
                    << ",\"total_duration\":" << duration_ns_str(total_ms / 1000.0)
                    << ",\"load_duration\":0"
                    << ",\"prompt_eval_count\":" << prompt_tokens
                    << ",\"prompt_eval_duration\":" << duration_ns_str(prompt_eval_ms / 1000.0)
                    << ",\"eval_count\":" << eval_count
                    << ",\"eval_duration\":" << duration_ns_str(eval_ms / 1000.0)
                    << "}\n";
        writer.write(final_chunk.str());
        writer.finish();

        double tok_per_sec = (eval_ms > 0) ? (eval_count / (eval_ms / 1000.0)) : 0;
        std::cout << "[Generate] " << eval_count << " tokens, "
                  << tok_per_sec << " tok/s" << std::endl;
    }

    // ========================================================================
    // Streaming chat generation (messages already formatted)
    // ========================================================================

    void generate_streaming_chat(torch::io::GGUFModel* model, const std::string& model_name,
                                const std::string& formatted_prompt,
                                int max_tokens, float temperature, int top_k, float top_p,
                                float repeat_penalty, bool stream,
                                StreamWriter& writer,
                                std::chrono::high_resolution_clock::time_point t_total_start) {

        std::string created_at = iso8601_now();

#ifdef PT_DEBUG_HTTP
        std::cerr << "[Chat] START model=" << model_name
                  << " use_cuda=" << model->use_cuda_
                  << " prompt_len=" << formatted_prompt.size()
                  << " max_tokens=" << max_tokens
                  << " temp=" << temperature
                  << " stream=" << stream << std::endl;
#endif

        // Reset KV cache
        model->kv_cache.reset();
        int64_t kv_dim = model->config.num_kv_heads * model->config.head_dim;
        // Fixed KV cache size to avoid reallocation (which invalidates CUDA Graph)
        int64_t max_total_seq = 4096;
        if (max_total_seq > model->config.context_length)
            max_total_seq = model->config.context_length;

        if (!model->kv_cache.allocated || model->kv_cache.max_seq < max_total_seq) {
            model->invalidate_graph();
            try {
                model->kv_cache.allocate(model->config.num_layers, max_total_seq, kv_dim,
                                         model->use_cuda_);
            } catch (const std::exception& e) {
                std::cerr << "[Chat] ERROR: KV cache allocation failed: " << e.what() << std::endl;
                std::string err = "{\"model\":\"" + json_escape(model_name) + "\""
                    + ",\"created_at\":\"" + created_at + "\""
                    + ",\"message\":{\"role\":\"assistant\",\"content\":\"\"}"
                    + ",\"done\":true,\"done_reason\":\"error: KV cache alloc failed: " + json_escape(e.what()) + "\"}\n";
                writer.write(err);
                writer.finish();
                return;
            }
        }

        // Encode (already formatted)
        auto input_tokens = model->tokenizer.encode(formatted_prompt, true);
        int prompt_tokens = static_cast<int>(input_tokens.size());

        if (input_tokens.empty()) {
            std::string err = "{\"model\":\"" + json_escape(model_name) + "\""
                + ",\"created_at\":\"" + created_at + "\""
                + ",\"message\":{\"role\":\"assistant\",\"content\":\"\"}"
                + ",\"done\":true,\"done_reason\":\"error: empty prompt\"}\n";
            writer.write(err);
            writer.finish();
            return;
        }

        // Prefill
        at::Tensor logits;
        auto t_prompt_start = std::chrono::high_resolution_clock::now();
        try {
            std::vector<int64_t> tokens_i64(input_tokens.begin(), input_tokens.end());
            logits = model->forward(tokens_i64, true);
        } catch (const std::exception& e) {
            std::cerr << "[Chat] ERROR in prefill forward(): " << e.what() << std::endl;
            std::string err = "{\"model\":\"" + json_escape(model_name) + "\""
                + ",\"created_at\":\"" + created_at + "\""
                + ",\"message\":{\"role\":\"assistant\",\"content\":\"\"}"
                + ",\"done\":true,\"done_reason\":\"error: prefill failed: "
                + json_escape(e.what()) + "\"}\n";
            writer.write(err);
            writer.finish();
            return;
        }
        auto t_prompt_end = std::chrono::high_resolution_clock::now();
        double prompt_eval_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();

        // Decode
        auto t_eval_start = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> generated;
        std::string full_response;

        const int timeout_ms_local = server_timeout_ms_;
        bool timed_out = false;
        for (int step = 0; step < max_tokens; ++step) {
            // Request timeout check
            if (timeout_ms_local > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(now - t_total_start).count();
                if (elapsed_ms > timeout_ms_local) {
                    writer.write("{\"error\":\"timeout\",\"done\":true}\n");
                    timed_out = true;
                    break;
                }
            }

            try {
                int64_t last_pos = logits.size(0) - 1;
                at::Tensor last_logits = model->get_row(logits, last_pos);

                if (repeat_penalty > 1.0f && !generated.empty()) {
                    float* logit_data = last_logits.mutable_data_ptr<float>();
                    for (int32_t prev : generated) {
                        if (prev >= 0 && prev < static_cast<int32_t>(model->tokenizer.vocab.size())) {
                            if (logit_data[prev] > 0) logit_data[prev] /= repeat_penalty;
                            else logit_data[prev] *= repeat_penalty;
                        }
                    }
                }

                int32_t next_token = model->sample_token(last_logits, temperature, top_k, top_p);

                if (next_token == model->tokenizer.eos_id || model->is_stop_token(next_token)) {
                    break;
                }

                generated.push_back(next_token);
                std::string token_str = model->tokenizer.decode_token(next_token);
                full_response += token_str;

                // Chat streaming format: {"message":{"role":"assistant","content":"token"}}
                if (stream) {
                    std::ostringstream chunk;
                    chunk << "{\"model\":\"" << json_escape(model_name) << "\""
                          << ",\"created_at\":\"" << created_at << "\""
                          << ",\"message\":{\"role\":\"assistant\",\"content\":\"" << json_escape(token_str) << "\"}"
                          << ",\"done\":false}\n";
                    if (!writer.write(chunk.str())) {
                        return;
                    }
                }

#ifdef PT_USE_CUDA
                if (model->use_cuda_) {
                    logits = model->forward_decode(static_cast<int64_t>(next_token));
                } else
#endif
                if (model->use_quant_gemv_) {
                    logits = model->forward_decode_cpu(static_cast<int64_t>(next_token));
                } else {
                    std::vector<int64_t> next_input = {static_cast<int64_t>(next_token)};
                    logits = model->forward(next_input, true);
                }
            } catch (const std::exception& e) {
                std::cerr << "[Chat] ERROR at decode step " << step << ": " << e.what() << std::endl;
                std::string err_chunk = "{\"model\":\"" + json_escape(model_name) + "\""
                    + ",\"created_at\":\"" + created_at + "\""
                    + ",\"message\":{\"role\":\"assistant\",\"content\":\"\"}"
                    + ",\"done\":true,\"done_reason\":\"error: decode failed at step "
                    + std::to_string(step) + ": " + json_escape(e.what()) + "\"}\n";
                writer.write(err_chunk);
                writer.finish();
                return;
            }
        }
        if (timed_out) {
            writer.finish();
            return;
        }

        auto t_eval_end = std::chrono::high_resolution_clock::now();
        double eval_ms = std::chrono::duration<double, std::milli>(t_eval_end - t_eval_start).count();
        auto t_total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
        int eval_count = static_cast<int>(generated.size());

        // Final message
        std::ostringstream final_chunk;
        final_chunk << "{\"model\":\"" << json_escape(model_name) << "\""
                    << ",\"created_at\":\"" << created_at << "\"";

        if (stream) {
            final_chunk << ",\"message\":{\"role\":\"assistant\",\"content\":\"\"}";
        } else {
            final_chunk << ",\"message\":{\"role\":\"assistant\",\"content\":\""
                        << json_escape(full_response) << "\"}";
        }

        final_chunk << ",\"done\":true"
                    << ",\"total_duration\":" << duration_ns_str(total_ms / 1000.0)
                    << ",\"load_duration\":0"
                    << ",\"prompt_eval_count\":" << prompt_tokens
                    << ",\"prompt_eval_duration\":" << duration_ns_str(prompt_eval_ms / 1000.0)
                    << ",\"eval_count\":" << eval_count
                    << ",\"eval_duration\":" << duration_ns_str(eval_ms / 1000.0)
                    << "}\n";
        writer.write(final_chunk.str());
        writer.finish();

        double tok_per_sec = (eval_ms > 0) ? (eval_count / (eval_ms / 1000.0)) : 0;
        std::cout << "[Chat] " << eval_count << " tokens, "
                  << tok_per_sec << " tok/s" << std::endl;
    }

    // ========================================================================
    // Format chat messages into a prompt string
    // ========================================================================

    std::string format_chat_messages(const json::JsonValue& messages, const std::string& model_name) {
        // Determine architecture from loaded model or model name
        std::string arch;
        auto* model = models_.get_loaded_model();
        if (model) {
            arch = model->config.architecture;
        } else {
            // Guess from model name
            if (model_name.find("qwen") != std::string::npos) arch = "qwen3";
            else if (model_name.find("gemma") != std::string::npos) arch = "gemma3";
            else if (model_name.find("llama") != std::string::npos) arch = "llama";
        }

        std::string result;

        for (auto& msg : messages.arr) {
            std::string role = msg["role"].as_string();
            std::string content = msg["content"].as_string();

            if (arch == "qwen3") {
                result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
            } else if (arch == "gemma3" || arch == "gemma2") {
                if (role == "user") {
                    result += "<start_of_turn>user\n" + content + "<end_of_turn>\n";
                } else if (role == "assistant") {
                    result += "<start_of_turn>model\n" + content + "<end_of_turn>\n";
                } else if (role == "system") {
                    result += "<start_of_turn>user\n[System: " + content + "]\n<end_of_turn>\n";
                }
            } else if (arch == "llama") {
                if (role == "system") {
                    result += "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                              + content + "<|eot_id|>";
                } else if (role == "user") {
                    result += "<|start_header_id|>user<|end_header_id|>\n\n"
                              + content + "<|eot_id|>";
                } else if (role == "assistant") {
                    result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                              + content + "<|eot_id|>";
                }
            } else {
                // Generic fallback
                result += role + ": " + content + "\n";
            }
        }

        // Add assistant prompt suffix
        if (arch == "qwen3") {
            result += "<|im_start|>assistant\n";
        } else if (arch == "gemma3" || arch == "gemma2") {
            result += "<start_of_turn>model\n";
        } else if (arch == "llama") {
            result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        } else {
            result += "assistant: ";
        }

        return result;
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    bool ensure_model_loaded(const std::string& model_name) {
        if (models_.loaded_model_name() == model_name && models_.get_loaded_model() != nullptr) {
            return true;
        }
        return models_.load_model(model_name);
    }

    ModelManager& models_;
    std::mutex generate_mutex_;  // Serialize generation (one at a time)
    int server_timeout_ms_;      // Per-request generation timeout (ms); 0 = none
};

}  // namespace promeserve
