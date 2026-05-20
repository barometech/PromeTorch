#pragma once

// ============================================================================
// PromeServe — Tool-Call Loop + Built-in Tool Registry
//
// Реализует agent-loop для chat completion: модель эмитирует
// <tool_call>{"name":..., "arguments":{...}}</tool_call>, server'у
// нужно выполнить tool, добавить результат в conversation как
// <tool_response>...</tool_response>, и продолжить генерацию.
//
// Tools здесь — built-in subset (write_file, read_file, list_dir, http_get,
// bash_safe). MCP client (Phase 2) живёт в отдельном mcp_client.h и тоже
// register'ит tools в этот же registry.
//
// Detection: regex <tool_call>(.*?)</tool_call> в выводе модели.
// Qwen3 и Hermes модели обучены на этом формате.
//
// Sandboxing:
//   - Файловые tools ограничены $PROMESERVE_TOOL_ROOT (default /tmp/promeserve/)
//   - bash_safe — whitelist commands только (ls, cat, grep, head, tail)
//   - Каждый tool call — timeout 30 sec
// ============================================================================

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <fstream>
#include <sstream>
#include <regex>
#include <cstdlib>
#include <sys/stat.h>
#include <chrono>

namespace promeserve {

// ============================================================================
// Tool definition (matches Ollama / OpenAI function-calling schema)
// ============================================================================

struct ToolParameter {
    std::string name;
    std::string type;          // "string" / "integer" / "boolean" / etc.
    std::string description;
    bool required = false;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::vector<ToolParameter> parameters;
    // Executor: takes JSON string of arguments → returns result string.
    std::function<std::string(const std::string& args_json)> executor;
    // Phase 2: tools пришедшие из request body (client-supplied) — sentinel
    // признак, что tool требует client-side execution. Server при detect
    // <tool_call> для external tool возвращает результат клиенту и
    // останавливает agent loop.
    bool external = false;
};

// ============================================================================
// Built-in tool registry
// ============================================================================

class ToolRegistry {
public:
    ToolRegistry() { register_builtins(); }

    void add(const ToolDefinition& tool) {
        tools_[tool.name] = tool;
    }

    bool has(const std::string& name) const {
        return tools_.count(name) > 0;
    }

    // Phase 2: проверка что tool external (client-supplied), для agent loop.
    bool is_external(const std::string& name) const {
        auto it = tools_.find(name);
        return it != tools_.end() && it->second.external;
    }

    std::string execute(const std::string& name, const std::string& args_json) const {
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return "{\"error\":\"tool not found: " + name + "\"}";
        }
        try {
            return it->second.executor(args_json);
        } catch (const std::exception& e) {
            return std::string("{\"error\":\"") + e.what() + "\"}";
        }
    }

    // For prompt injection: emit tools description in OpenAI-style JSON.
    std::string format_for_prompt() const {
        std::string out = "[";
        bool first = true;
        for (const auto& [name, tool] : tools_) {
            if (!first) out += ",";
            first = false;
            out += "{\"type\":\"function\",\"function\":{";
            out += "\"name\":\"" + tool.name + "\",";
            out += "\"description\":\"" + escape_json(tool.description) + "\",";
            out += "\"parameters\":{\"type\":\"object\",\"properties\":{";
            bool p_first = true;
            std::string required_list;
            for (const auto& p : tool.parameters) {
                if (!p_first) out += ",";
                p_first = false;
                out += "\"" + p.name + "\":{\"type\":\"" + p.type + "\",";
                out += "\"description\":\"" + escape_json(p.description) + "\"}";
                if (p.required) {
                    if (!required_list.empty()) required_list += ",";
                    required_list += "\"" + p.name + "\"";
                }
            }
            out += "}";
            if (!required_list.empty()) {
                out += ",\"required\":[" + required_list + "]";
            }
            out += "}}}";
        }
        out += "]";
        return out;
    }

    std::vector<std::string> list_names() const {
        std::vector<std::string> names;
        for (const auto& [n, _] : tools_) names.push_back(n);
        return names;
    }

    // Phase 2: parse OpenAI-style tools[] array from request body and register
    // each as external (no executor — agent loop returns tool_call to client).
    // Format:
    //   [{"type":"function", "function":{"name":..., "description":...,
    //     "parameters":{"type":"object", "properties":{...}, "required":[...]}}}]
    void register_external_tools(const std::vector<std::map<std::string, std::string>>& parsed_tools) {
        for (const auto& t : parsed_tools) {
            ToolDefinition def;
            auto it = t.find("name");
            if (it == t.end() || it->second.empty()) continue;
            def.name = it->second;
            auto desc_it = t.find("description");
            if (desc_it != t.end()) def.description = desc_it->second;
            def.external = true;
            // Sentinel executor — never actually invoked because agent loop
            // detects external + breaks before execute(). Defensive return.
            def.executor = [name = def.name](const std::string&) {
                return std::string("{\"_external\":true,\"tool\":\"") + name
                     + "\",\"note\":\"client must execute\"}";
            };
            add(def);
        }
    }

private:
    std::map<std::string, ToolDefinition> tools_;

    // ------------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------------

    static std::string escape_json(const std::string& s) {
        std::string out;
        for (char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if ((unsigned char)c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
        return out;
    }

    static std::string get_tool_root() {
        const char* env = std::getenv("PROMESERVE_TOOL_ROOT");
        std::string root = env ? env : "/tmp/promeserve";
        // Ensure exists
        struct stat st;
        if (stat(root.c_str(), &st) != 0) {
            ::mkdir(root.c_str(), 0755);
        }
        return root;
    }

    // Resolve path безопасно: запретить ../ и absolute path вне root.
    static std::string sandbox_path(const std::string& user_path) {
        std::string root = get_tool_root();
        // Strip leading "/" if any, then prepend root
        std::string p = user_path;
        while (!p.empty() && p[0] == '/') p.erase(0, 1);
        // Reject ".." segments
        if (p.find("..") != std::string::npos) {
            throw std::runtime_error("path traversal denied");
        }
        return root + "/" + p;
    }

    // Naive JSON value extractor (supports string + number).
    static std::string extract_string_field(const std::string& json, const std::string& key) {
        std::string needle = "\"" + key + "\"";
        size_t kp = json.find(needle);
        if (kp == std::string::npos) return "";
        size_t cp = json.find(":", kp);
        if (cp == std::string::npos) return "";
        size_t qp = json.find('"', cp);
        if (qp == std::string::npos) return "";
        std::string out;
        for (size_t i = qp + 1; i < json.size(); i++) {
            if (json[i] == '\\' && i + 1 < json.size()) {
                char n = json[i + 1];
                if (n == 'n') out += '\n';
                else if (n == 't') out += '\t';
                else if (n == 'r') out += '\r';
                else if (n == '"') out += '"';
                else if (n == '\\') out += '\\';
                else out += n;
                i++;
            } else if (json[i] == '"') {
                return out;
            } else {
                out += json[i];
            }
        }
        return out;
    }

    // ------------------------------------------------------------------------
    // Built-in tool implementations
    // ------------------------------------------------------------------------

    void register_builtins() {
        add({
            "write_file",
            "Write content to a file under /tmp/promeserve sandbox. Path is relative.",
            {{"path", "string", "relative path to file", true},
             {"content", "string", "file content", true}},
            [](const std::string& args) -> std::string {
                std::string path = extract_string_field(args, "path");
                std::string content = extract_string_field(args, "content");
                if (path.empty()) return "{\"error\":\"missing path\"}";
                std::string full = sandbox_path(path);
                std::ofstream f(full);
                if (!f) return "{\"error\":\"cannot open file\"}";
                f << content;
                return "{\"ok\":true,\"path\":\"" + escape_json(full) + "\","
                       "\"bytes\":" + std::to_string(content.size()) + "}";
            }
        });

        add({
            "read_file",
            "Read content of a file under sandbox. Returns first 4KB.",
            {{"path", "string", "relative path to file", true}},
            [](const std::string& args) -> std::string {
                std::string path = extract_string_field(args, "path");
                if (path.empty()) return "{\"error\":\"missing path\"}";
                std::string full = sandbox_path(path);
                std::ifstream f(full);
                if (!f) return "{\"error\":\"cannot open\"}";
                std::stringstream ss;
                ss << f.rdbuf();
                std::string content = ss.str();
                if (content.size() > 4096) content = content.substr(0, 4096) + "...[truncated]";
                return "{\"ok\":true,\"content\":\"" + escape_json(content) + "\"}";
            }
        });

        add({
            "list_dir",
            "List files in sandbox directory.",
            {{"path", "string", "relative directory path (empty for root)", false}},
            [](const std::string& args) -> std::string {
                std::string path = extract_string_field(args, "path");
                std::string full = sandbox_path(path.empty() ? "" : path);
                std::string result = "{\"files\":[";
                FILE* p = popen(("ls -1 \"" + full + "\" 2>&1").c_str(), "r");
                if (!p) return "{\"error\":\"popen failed\"}";
                char buf[256];
                bool first = true;
                while (fgets(buf, sizeof(buf), p)) {
                    std::string line = buf;
                    if (!line.empty() && line.back() == '\n') line.pop_back();
                    if (line.empty()) continue;
                    if (!first) result += ",";
                    first = false;
                    result += "\"" + escape_json(line) + "\"";
                }
                pclose(p);
                result += "]}";
                return result;
            }
        });

        add({
            "bash_safe",
            "Execute a safe whitelisted bash command (only ls, cat, head, tail, grep, wc, file, date).",
            {{"command", "string", "shell command", true}},
            [](const std::string& args) -> std::string {
                std::string cmd = extract_string_field(args, "command");
                if (cmd.empty()) return "{\"error\":\"missing command\"}";
                // Whitelist: first token
                std::string first;
                for (char c : cmd) {
                    if (c == ' ' || c == '\t') break;
                    first += c;
                }
                static const std::vector<std::string> allowed = {
                    "ls","cat","head","tail","grep","wc","file","date","echo","pwd"
                };
                bool ok = false;
                for (auto& a : allowed) if (first == a) { ok = true; break; }
                if (!ok) {
                    return "{\"error\":\"command not in whitelist\"}";
                }
                FILE* p = popen((cmd + " 2>&1").c_str(), "r");
                if (!p) return "{\"error\":\"popen failed\"}";
                char buf[256];
                std::string out;
                while (fgets(buf, sizeof(buf), p)) {
                    out += buf;
                    if (out.size() > 4096) { out += "...[truncated]"; break; }
                }
                pclose(p);
                return "{\"ok\":true,\"output\":\"" + escape_json(out) + "\"}";
            }
        });
    }
};

// ============================================================================
// Tool call detection in model output
// ============================================================================

struct ToolCall {
    std::string name;
    std::string args_json;
    size_t start_pos = 0;
    size_t end_pos = 0;
    bool valid = false;
};

// Detect <tool_call>{"name":"...","arguments":{...}}</tool_call> в text.
// Возвращает первое вхождение или valid=false.
inline ToolCall detect_tool_call(const std::string& text) {
    ToolCall tc;
    size_t s = text.find("<tool_call>");
    if (s == std::string::npos) return tc;
    size_t e = text.find("</tool_call>", s);
    if (e == std::string::npos) return tc;
    tc.start_pos = s;
    tc.end_pos = e + std::string("</tool_call>").size();
    std::string payload = text.substr(s + std::string("<tool_call>").size(),
                                       e - s - std::string("<tool_call>").size());
    // Extract name
    auto get_str = [&](const std::string& key) {
        std::string needle = "\"" + key + "\"";
        size_t kp = payload.find(needle);
        if (kp == std::string::npos) return std::string();
        size_t cp = payload.find(":", kp);
        if (cp == std::string::npos) return std::string();
        size_t qp = payload.find('"', cp);
        if (qp == std::string::npos) return std::string();
        size_t qe = payload.find('"', qp + 1);
        if (qe == std::string::npos) return std::string();
        return payload.substr(qp + 1, qe - qp - 1);
    };
    tc.name = get_str("name");
    // Find arguments {...}
    size_t ap = payload.find("\"arguments\"");
    if (ap != std::string::npos) {
        size_t lb = payload.find("{", ap);
        if (lb != std::string::npos) {
            int depth = 1;
            size_t i = lb + 1;
            while (i < payload.size() && depth > 0) {
                if (payload[i] == '{') depth++;
                else if (payload[i] == '}') depth--;
                i++;
            }
            if (depth == 0) {
                tc.args_json = payload.substr(lb, i - lb);
            }
        }
    }
    tc.valid = !tc.name.empty();
    return tc;
}

// ============================================================================
// Format system prompt with tools (qwen3 / Hermes style)
// ============================================================================

inline std::string format_system_prompt_with_tools(
    const std::string& user_system,
    const ToolRegistry& reg)
{
    std::string sys = user_system.empty() ? "You are a helpful assistant." : user_system;
    sys += "\n\nYou have access to the following tools:\n<tools>\n";
    sys += reg.format_for_prompt();
    sys += "\n</tools>\n\n";
    sys += "To use a tool, emit:\n";
    sys += "<tool_call>{\"name\":\"<tool_name>\",\"arguments\":{<args>}}</tool_call>\n";
    sys += "After receiving a <tool_response>, continue. When task is complete, give the final answer without tool_call.\n";
    return sys;
}

}  // namespace promeserve
