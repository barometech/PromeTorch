#pragma once

// ============================================================================
// PromeServe — Tool-Call Loop + Built-in Tool Registry (Этап 1)
//
// Поддерживается Qwen3 / Hermes-style <tool_call>{json}</tool_call> формат.
// Парсер уже совместим с Qwen3 и Phi-3 tunes (см. AGENT_STACK_R2.md §2.7).
//
// Built-in tools (sandbox = $PROMESERVE_TOOL_ROOT, default /tmp/promeserve):
//   write_file, read_file, list_dir, bash_safe        — базовые FS + shell
//   fetch_url, http_get                               — HTTP GET (libcurl/curl-exec)
//   web_search                                        — DuckDuckGo html (no API key)
//   execute_python                                    — `python3 -c` в sandbox
//   git                                               — read-only git log/diff/status/show
//   sqlite                                            — sqlite3 query (read-only)
//
// Расширения Этапа 1:
//   - detect_all_tool_calls(text) → vector<ToolCall>  для parallel calls.
//   - StreamingToolCallDetector                       для streaming stop.
//   - parallel_execute(...)                            через std::thread пул.
//   - JSON-Schema валидация args (input_schema).
//   - Audit log в ~/.promeserve/audit.jsonl.
//   - Max-iter guard через PROMESERVE_MAX_TOOL_ITER.
//   - MCP-backed tools регистрируются через add_external/add_mcp.
//
// Внешних C++ deps нет (только stdlib + опционально libcurl).
// Портабельно: MSVC, LCC (Elbrus), gcc/clang. C++17.
// ============================================================================

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <sstream>
#include <regex>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <cctype>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <memory>
#include <stdexcept>
#include <cstdio>

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <direct.h>
#  define PROMESERVE_MKDIR(p) ::_mkdir(p)
#else
#  include <unistd.h>
#  define PROMESERVE_MKDIR(p) ::mkdir((p), 0755)
#endif

namespace promeserve {

// ============================================================================
// Forward decl — MCP client integration. MCPManager определён в mcp_client.h.
// Чтобы не тянуть тяжёлый include и не создавать циклическую зависимость,
// объявляем абстрактный интерфейс forward-routing'а здесь.
// ============================================================================

struct MCPToolRoute {
    std::string server_name;
    std::string remote_tool_name;
};

// Колбэк, который ToolRegistry дёргает для MCP-tools.
using MCPDispatcher =
    std::function<std::string(const MCPToolRoute& route, const std::string& args_json)>;

// ============================================================================
// Tool definition (compatible with OpenAI/Ollama function-calling schema)
// ============================================================================

struct ToolParameter {
    std::string name;
    std::string type;          // "string" / "integer" / "boolean" / "number" / "object"
    std::string description;
    bool required = false;
};

enum class ToolKind {
    BUILTIN,    // executor живёт в этом процессе
    EXTERNAL,   // client-supplied; agent loop отдаёт tool_call наружу
    MCP         // прокси в MCP server через MCPManager
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::vector<ToolParameter> parameters;
    std::function<std::string(const std::string& args_json)> executor;
    ToolKind kind = ToolKind::BUILTIN;
    MCPToolRoute mcp_route;            // valid когда kind == MCP
    // JSON-Schema input_schema (если задан — выполняется validate).
    // Формат — упрощённый JSON-Schema subset (см. validate_args_schema).
    std::string input_schema_json;
    // Phase 2 совместимость: external==true ⇔ kind == EXTERNAL.
    bool external = false;
};

// ============================================================================
// Простая naive-парсилка JSON (без зависимостей).
// Полная парсилка живёт в api_handlers.h::json; здесь — мини-версия только
// для extract'а строк/чисел из args.
// ============================================================================
namespace tc_json {

inline std::string extract_string_field(const std::string& json, const std::string& key) {
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
            else if (n == '/') out += '/';
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

inline double extract_number_field(const std::string& json, const std::string& key, double def = 0.0) {
    std::string needle = "\"" + key + "\"";
    size_t kp = json.find(needle);
    if (kp == std::string::npos) return def;
    size_t cp = json.find(":", kp);
    if (cp == std::string::npos) return def;
    size_t i = cp + 1;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) i++;
    size_t s = i;
    while (i < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[i])) ||
            json[i] == '-' || json[i] == '+' ||
            json[i] == '.' || json[i] == 'e' || json[i] == 'E')) i++;
    if (s == i) return def;
    try { return std::stod(json.substr(s, i - s)); } catch (...) { return def; }
}

inline bool has_field(const std::string& json, const std::string& key) {
    return json.find("\"" + key + "\"") != std::string::npos;
}

inline std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
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

}  // namespace tc_json

// ============================================================================
// Audit log — JSONL append-only в ~/.promeserve/audit.jsonl
//
// Каждая запись: {"ts":"...","tool":"...","args":"...","result":"...",
//                 "ok":bool,"duration_ms":N}
// Singleton, thread-safe (mutex). Никогда не fail'ит request — если
// open() сломался, тихо ignore.
// ============================================================================

class AuditLog {
public:
    static AuditLog& instance() {
        static AuditLog inst;
        return inst;
    }

    void log(const std::string& tool_name,
             const std::string& args_json,
             const std::string& result_json,
             bool ok,
             double duration_ms) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!ensure_open_()) return;

        // ISO8601 timestamp
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     now.time_since_epoch()).count() % 1000;
        struct tm tm_buf;
#ifdef _WIN32
        gmtime_s(&tm_buf, &t);
#else
        gmtime_r(&t, &tm_buf);
#endif
        char ts[64];
        std::snprintf(ts, sizeof(ts), "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
                      tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                      tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                      static_cast<long>(ms));

        // Усечь длинные поля
        std::string args_t = args_json.size() > 2048
            ? args_json.substr(0, 2048) + "...[truncated]" : args_json;
        std::string res_t = result_json.size() > 4096
            ? result_json.substr(0, 4096) + "...[truncated]" : result_json;

        std::ostringstream line;
        line << "{\"ts\":\"" << ts << "\""
             << ",\"tool\":\"" << tc_json::escape_json(tool_name) << "\""
             << ",\"args\":\"" << tc_json::escape_json(args_t) << "\""
             << ",\"result\":\"" << tc_json::escape_json(res_t) << "\""
             << ",\"ok\":" << (ok ? "true" : "false")
             << ",\"duration_ms\":" << static_cast<int64_t>(duration_ms)
             << "}\n";
        ofs_ << line.str();
        ofs_.flush();
    }

    // Прочесть последние N строк (для UI / /api/mcp/audit endpoint).
    std::vector<std::string> tail(size_t n) {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<std::string> out;
        std::ifstream f(path_);
        if (!f) return out;
        std::string line;
        std::vector<std::string> all;
        while (std::getline(f, line)) all.push_back(line);
        size_t start = all.size() > n ? all.size() - n : 0;
        for (size_t i = start; i < all.size(); i++) out.push_back(all[i]);
        return out;
    }

private:
    AuditLog() = default;
    bool ensure_open_() {
        if (ofs_.is_open()) return true;
        path_ = audit_path_();
        if (path_.empty()) return false;
        ofs_.open(path_, std::ios::app);
        return ofs_.is_open();
    }

    static std::string audit_path_() {
        const char* home = std::getenv(
#ifdef _WIN32
            "USERPROFILE"
#else
            "HOME"
#endif
        );
        if (!home) return "";
        std::string dir = std::string(home) + "/.promeserve";
        PROMESERVE_MKDIR(dir.c_str());
        return dir + "/audit.jsonl";
    }

    std::mutex mu_;
    std::ofstream ofs_;
    std::string path_;
};

// ============================================================================
// JSON-Schema мини-validator (subset: type, properties.required, типы string,
// integer, number, boolean, object, array). Возвращает "" если ok, иначе
// human-readable error.
// ============================================================================

inline std::string validate_args_schema(const std::string& schema_json,
                                        const std::string& args_json) {
    if (schema_json.empty()) return "";

    // Проверим required: ["a","b",...]
    size_t rp = schema_json.find("\"required\"");
    if (rp != std::string::npos) {
        size_t lb = schema_json.find('[', rp);
        size_t rb = schema_json.find(']', lb);
        if (lb != std::string::npos && rb != std::string::npos) {
            std::string body = schema_json.substr(lb + 1, rb - lb - 1);
            size_t i = 0;
            while (i < body.size()) {
                while (i < body.size() && body[i] != '"') i++;
                if (i >= body.size()) break;
                size_t j = ++i;
                while (j < body.size() && body[j] != '"') j++;
                if (j >= body.size()) break;
                std::string field = body.substr(i, j - i);
                i = j + 1;
                if (!field.empty() && !tc_json::has_field(args_json, field)) {
                    return "missing required field: " + field;
                }
            }
        }
    }

    // Проверка типов (мягкая, только наличие). Полноценный type-check
    // оставим на future: модель чаще всего ошибается в required, а не в
    // type, а строгая проверка ломает совместимость со множеством tunes.
    return "";
}

// ============================================================================
// Tool registry
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

    const ToolDefinition* get(const std::string& name) const {
        auto it = tools_.find(name);
        return it == tools_.end() ? nullptr : &it->second;
    }

    bool is_external(const std::string& name) const {
        auto it = tools_.find(name);
        return it != tools_.end() &&
               (it->second.kind == ToolKind::EXTERNAL || it->second.external);
    }

    bool is_mcp(const std::string& name) const {
        auto it = tools_.find(name);
        return it != tools_.end() && it->second.kind == ToolKind::MCP;
    }

    void set_mcp_dispatcher(MCPDispatcher d) { mcp_dispatch_ = std::move(d); }

    // Зарегистрировать MCP-tool. name обычно вида `mcp__<server>__<tool>`.
    void add_mcp_tool(const std::string& name,
                      const std::string& description,
                      const std::string& input_schema_json,
                      const std::string& server,
                      const std::string& remote_name) {
        ToolDefinition def;
        def.name = name;
        def.description = description;
        def.kind = ToolKind::MCP;
        def.mcp_route = {server, remote_name};
        def.input_schema_json = input_schema_json;
        def.executor = [this, server, remote_name](const std::string& args_json) {
            if (!mcp_dispatch_) {
                return std::string("{\"error\":\"no MCP dispatcher registered\"}");
            }
            return mcp_dispatch_({server, remote_name}, args_json);
        };
        add(def);
    }

    // Удалить tools соответствующего MCP-сервера (для disconnect/reconnect).
    void remove_mcp_tools(const std::string& server) {
        for (auto it = tools_.begin(); it != tools_.end(); ) {
            if (it->second.kind == ToolKind::MCP &&
                it->second.mcp_route.server_name == server) {
                it = tools_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Главная точка входа в исполнение tool. Validate schema + audit log +
    // measure duration + catch exceptions. Возвращает JSON-строку.
    std::string execute(const std::string& name, const std::string& args_json) {
        auto t0 = std::chrono::steady_clock::now();
        std::string result;
        bool ok = false;
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            result = "{\"error\":\"tool not found: " + name + "\"}";
        } else {
            // Schema validation
            std::string err = validate_args_schema(it->second.input_schema_json, args_json);
            if (!err.empty()) {
                result = "{\"error\":\"schema validation: " + tc_json::escape_json(err) + "\"}";
            } else {
                try {
                    result = it->second.executor(args_json);
                    ok = (result.find("\"error\"") == std::string::npos);
                } catch (const std::exception& e) {
                    result = std::string("{\"error\":\"") + tc_json::escape_json(e.what()) + "\"}";
                } catch (...) {
                    result = "{\"error\":\"unknown exception\"}";
                }
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        AuditLog::instance().log(name, args_json, result, ok, ms);
        return result;
    }

    // Описание для prompt'а (OpenAI-style array).
    std::string format_for_prompt() const {
        std::string out = "[";
        bool first = true;
        for (const auto& [name, tool] : tools_) {
            if (!first) out += ",";
            first = false;
            out += "{\"type\":\"function\",\"function\":{";
            out += "\"name\":\"" + tool.name + "\",";
            out += "\"description\":\"" + tc_json::escape_json(tool.description) + "\",";
            out += "\"parameters\":{\"type\":\"object\",\"properties\":{";
            bool p_first = true;
            std::string required_list;
            for (const auto& p : tool.parameters) {
                if (!p_first) out += ",";
                p_first = false;
                out += "\"" + p.name + "\":{\"type\":\"" + p.type + "\",";
                out += "\"description\":\"" + tc_json::escape_json(p.description) + "\"}";
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

    std::vector<ToolDefinition> list_all() const {
        std::vector<ToolDefinition> out;
        for (const auto& [_, t] : tools_) out.push_back(t);
        return out;
    }

    // Phase 2 compat — client-supplied external tools.
    void register_external_tools(
        const std::vector<std::map<std::string, std::string>>& parsed_tools) {
        for (const auto& t : parsed_tools) {
            ToolDefinition def;
            auto it = t.find("name");
            if (it == t.end() || it->second.empty()) continue;
            def.name = it->second;
            auto desc_it = t.find("description");
            if (desc_it != t.end()) def.description = desc_it->second;
            def.kind = ToolKind::EXTERNAL;
            def.external = true;
            def.executor = [name = def.name](const std::string&) {
                return std::string("{\"_external\":true,\"tool\":\"") + name
                     + "\",\"note\":\"client must execute\"}";
            };
            add(def);
        }
    }

    // ========================================================================
    // Sandbox helpers (доступны и наружу — MCP server endpoints их использует).
    // ========================================================================
    static std::string get_tool_root() {
        const char* env = std::getenv("PROMESERVE_TOOL_ROOT");
#ifdef _WIN32
        std::string root = env ? env : (std::string(std::getenv("TEMP") ? std::getenv("TEMP") : "C:/tmp") + "/promeserve");
#else
        std::string root = env ? env : "/tmp/promeserve";
#endif
        struct ::stat st;
        if (::stat(root.c_str(), &st) != 0) {
            PROMESERVE_MKDIR(root.c_str());
        }
        return root;
    }

    // Hardened sandbox: leading-slash strip + ".." reject + weakly_canonical
    // prefix-check. Последнее закрывает symlink-escape (файл внутри root,
    // указывающий наружу) и любые edge-cases нормализации — итоговый
    // реальный путь ОБЯЗАН лежать под root (audit 2026-06-02 _security #1,
    // RCE chain: write_file ../../.promeserve/mcp.json → MCP exec).
    static std::string sandbox_path(const std::string& user_path) {
        namespace fs = std::filesystem;
        std::string root = get_tool_root();
        std::string p = user_path;
        while (!p.empty() && (p[0] == '/' || p[0] == '\\')) p.erase(0, 1);
        if (p.find("..") != std::string::npos) {
            throw std::runtime_error("path traversal denied (.. component)");
        }
        fs::path full = fs::path(root) / p;
        // weakly_canonical резолвит существующие компоненты (включая
        // симлинки), не требуя существования хвоста (для write_file).
        std::error_code ec;
        fs::path canon_root = fs::weakly_canonical(fs::path(root), ec);
        fs::path canon_full = fs::weakly_canonical(full, ec);
        if (ec) {
            // Не удалось нормализовать — безопаснее отказать.
            throw std::runtime_error("path resolution failed");
        }
        // Проверяем что canon_full под canon_root (по компонентам, чтобы
        // /root-evil не прошёл префикс /root).
        auto rb = canon_root.begin(), re = canon_root.end();
        auto fb = canon_full.begin();
        for (; rb != re; ++rb, ++fb) {
            if (fb == canon_full.end() || *fb != *rb) {
                throw std::runtime_error("path escapes sandbox root");
            }
        }
        return canon_full.string();
    }

private:
    std::map<std::string, ToolDefinition> tools_;
    MCPDispatcher mcp_dispatch_;

    // ------------------------------------------------------------------------
    // Helper: безопасное чтение env флага.
    // ------------------------------------------------------------------------
    static bool env_flag(const char* name) {
        const char* v = std::getenv(name);
        if (!v) return false;
        if (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y') return true;
        return false;
    }

    // ------------------------------------------------------------------------
    // Built-in tool implementations
    // ------------------------------------------------------------------------
    void register_builtins() {
        // --- write_file ---------------------------------------------------------
        ToolDefinition wf;
        wf.name = "write_file";
        wf.description = "Write content to a file under sandbox. Path is relative to PROMESERVE_TOOL_ROOT.";
        wf.parameters = {
            {"path", "string", "relative path to file", true},
            {"content", "string", "file content", true}};
        wf.input_schema_json = R"({"type":"object","required":["path","content"]})";
        wf.executor = [](const std::string& args) -> std::string {
            std::string path = tc_json::extract_string_field(args, "path");
            std::string content = tc_json::extract_string_field(args, "content");
            if (path.empty()) return "{\"error\":\"missing path\"}";
            std::string full = ToolRegistry::sandbox_path(path);
            std::ofstream f(full, std::ios::binary);
            if (!f) return "{\"error\":\"cannot open file\"}";
            f << content;
            return "{\"ok\":true,\"path\":\"" + tc_json::escape_json(full) + "\","
                   "\"bytes\":" + std::to_string(content.size()) + "}";
        };
        add(wf);

        // --- read_file ----------------------------------------------------------
        ToolDefinition rf;
        rf.name = "read_file";
        rf.description = "Read content of a file under sandbox. Returns first 4KB.";
        rf.parameters = {{"path", "string", "relative path to file", true}};
        rf.input_schema_json = R"({"type":"object","required":["path"]})";
        rf.executor = [](const std::string& args) -> std::string {
            std::string path = tc_json::extract_string_field(args, "path");
            if (path.empty()) return "{\"error\":\"missing path\"}";
            std::string full = ToolRegistry::sandbox_path(path);
            std::ifstream f(full, std::ios::binary);
            if (!f) return "{\"error\":\"cannot open\"}";
            std::stringstream ss;
            ss << f.rdbuf();
            std::string content = ss.str();
            if (content.size() > 4096) content = content.substr(0, 4096) + "...[truncated]";
            return "{\"ok\":true,\"content\":\"" + tc_json::escape_json(content) + "\"}";
        };
        add(rf);

        // --- list_dir -----------------------------------------------------------
        ToolDefinition ld;
        ld.name = "list_dir";
        ld.description = "List files in sandbox directory.";
        ld.parameters = {{"path", "string", "relative directory path (empty for root)", false}};
        ld.input_schema_json = R"({"type":"object"})";
        ld.executor = [](const std::string& args) -> std::string {
            std::string path = tc_json::extract_string_field(args, "path");
            std::string full = ToolRegistry::sandbox_path(path.empty() ? "" : path);
            std::string result = "{\"files\":[";
#ifdef _WIN32
            std::string pattern = full + "/*";
            WIN32_FIND_DATAA fd;
            HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
            bool first = true;
            if (h != INVALID_HANDLE_VALUE) {
                do {
                    std::string n = fd.cFileName;
                    if (n == "." || n == "..") continue;
                    if (!first) result += ",";
                    first = false;
                    result += "\"" + tc_json::escape_json(n) + "\"";
                } while (FindNextFileA(h, &fd));
                FindClose(h);
            }
#else
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
                result += "\"" + tc_json::escape_json(line) + "\"";
            }
            pclose(p);
#endif
            result += "]}";
            return result;
        };
        add(ld);

        // --- bash_safe ----------------------------------------------------------
        ToolDefinition bs;
        bs.name = "bash_safe";
        bs.description = "Execute a safe whitelisted bash command (only ls, cat, head, tail, grep, wc, file, date, echo, pwd).";
        bs.parameters = {{"command", "string", "shell command", true}};
        bs.input_schema_json = R"({"type":"object","required":["command"]})";
        bs.executor = [](const std::string& args) -> std::string {
            std::string cmd = tc_json::extract_string_field(args, "command");
            if (cmd.empty()) return "{\"error\":\"missing command\"}";
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
            if (!ok) return "{\"error\":\"command not in whitelist\"}";
            FILE* p =
#ifdef _WIN32
                _popen((cmd + " 2>&1").c_str(), "r");
#else
                popen((cmd + " 2>&1").c_str(), "r");
#endif
            if (!p) return "{\"error\":\"popen failed\"}";
            char buf[256];
            std::string out;
            while (fgets(buf, sizeof(buf), p)) {
                out += buf;
                if (out.size() > 4096) { out += "...[truncated]"; break; }
            }
#ifdef _WIN32
            _pclose(p);
#else
            pclose(p);
#endif
            return "{\"ok\":true,\"output\":\"" + tc_json::escape_json(out) + "\"}";
        };
        add(bs);

        // --- fetch_url (== http_get) -------------------------------------------
        // Делегирует в curl/Invoke-WebRequest. Без отдельного libcurl link,
        // чтобы остаться без deps. Timeout 30 сек.
        auto fetch_impl = [](const std::string& args) -> std::string {
            std::string url = tc_json::extract_string_field(args, "url");
            if (url.empty()) return "{\"error\":\"missing url\"}";
            // Защита: только http(s)
            if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
                return "{\"error\":\"only http(s) urls allowed\"}";
            }
            // Quote-safe — заменяем " на пусто
            std::string safe;
            for (char c : url) {
                if (c == '"' || c == '`' || c == '\'' || c == '$' || c == ';' || c == '|' || c == '&') continue;
                safe += c;
            }
#ifdef _WIN32
            std::string cmd = "curl.exe -sSL --max-time 30 \"" + safe + "\" 2>nul";
            FILE* p = _popen(cmd.c_str(), "r");
#else
            std::string cmd = "curl -sSL --max-time 30 \"" + safe + "\" 2>/dev/null";
            FILE* p = popen(cmd.c_str(), "r");
#endif
            if (!p) return "{\"error\":\"curl not available\"}";
            std::string out;
            char buf[1024];
            while (fgets(buf, sizeof(buf), p)) {
                out += buf;
                if (out.size() > 64 * 1024) { out += "...[truncated]"; break; }
            }
#ifdef _WIN32
            _pclose(p);
#else
            pclose(p);
#endif
            return "{\"ok\":true,\"body\":\"" + tc_json::escape_json(out) + "\"}";
        };

        ToolDefinition fu;
        fu.name = "fetch_url";
        fu.description = "HTTP GET request. Returns body (truncated to 64KB). Timeout 30s.";
        fu.parameters = {{"url", "string", "http(s) URL", true}};
        fu.input_schema_json = R"({"type":"object","required":["url"]})";
        fu.executor = fetch_impl;
        add(fu);

        ToolDefinition hg;
        hg.name = "http_get";
        hg.description = "Alias for fetch_url.";
        hg.parameters = fu.parameters;
        hg.input_schema_json = fu.input_schema_json;
        hg.executor = fetch_impl;
        add(hg);

        // --- web_search (DuckDuckGo HTML, no API key required) -----------------
        if (env_flag("PROMESERVE_ENABLE_WEB_SEARCH")) {
            ToolDefinition ws;
            ws.name = "web_search";
            ws.description = "Search the web via DuckDuckGo HTML. Returns top results as JSON array.";
            ws.parameters = {
                {"query", "string", "search query", true},
                {"top_n", "integer", "max results (default 5)", false}};
            ws.input_schema_json = R"({"type":"object","required":["query"]})";
            ws.executor = [](const std::string& args) -> std::string {
                std::string q = tc_json::extract_string_field(args, "query");
                if (q.empty()) return "{\"error\":\"missing query\"}";
                int top_n = static_cast<int>(tc_json::extract_number_field(args, "top_n", 5));
                if (top_n < 1) top_n = 1;
                if (top_n > 20) top_n = 20;

                // URL-encode query
                std::string enc;
                for (char c : q) {
                    if (std::isalnum(static_cast<unsigned char>(c)) ||
                        c == '-' || c == '_' || c == '.' || c == '~') {
                        enc += c;
                    } else {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "%%%02X",
                                      static_cast<unsigned char>(c));
                        enc += buf;
                    }
                }
                std::string url = "https://duckduckgo.com/html/?q=" + enc;
#ifdef _WIN32
                std::string cmd = "curl.exe -sSL --max-time 30 -A \"Mozilla/5.0\" \"" + url + "\" 2>nul";
                FILE* p = _popen(cmd.c_str(), "r");
#else
                std::string cmd = "curl -sSL --max-time 30 -A 'Mozilla/5.0' \"" + url + "\" 2>/dev/null";
                FILE* p = popen(cmd.c_str(), "r");
#endif
                if (!p) return "{\"error\":\"curl not available\"}";
                std::string html;
                char buf[2048];
                while (fgets(buf, sizeof(buf), p)) {
                    html += buf;
                    if (html.size() > 512 * 1024) break;
                }
#ifdef _WIN32
                _pclose(p);
#else
                pclose(p);
#endif
                // Парсим result_link__a: <a class="result__a" href="...">title</a>
                std::vector<std::pair<std::string,std::string>> hits;
                size_t pos = 0;
                while (hits.size() < (size_t)top_n) {
                    size_t a = html.find("result__a", pos);
                    if (a == std::string::npos) break;
                    size_t href = html.find("href=\"", a);
                    if (href == std::string::npos) break;
                    href += 6;
                    size_t hend = html.find("\"", href);
                    if (hend == std::string::npos) break;
                    std::string link = html.substr(href, hend - href);
                    size_t gt = html.find('>', hend);
                    if (gt == std::string::npos) break;
                    size_t tend = html.find("</a>", gt);
                    if (tend == std::string::npos) break;
                    std::string title = html.substr(gt + 1, tend - gt - 1);
                    // strip tags
                    std::string clean;
                    bool in_tag = false;
                    for (char c : title) {
                        if (c == '<') in_tag = true;
                        else if (c == '>') in_tag = false;
                        else if (!in_tag) clean += c;
                    }
                    hits.emplace_back(link, clean);
                    pos = tend + 4;
                }
                std::string out = "{\"results\":[";
                for (size_t i = 0; i < hits.size(); i++) {
                    if (i) out += ",";
                    out += "{\"url\":\"" + tc_json::escape_json(hits[i].first) + "\""
                         + ",\"title\":\"" + tc_json::escape_json(hits[i].second) + "\"}";
                }
                out += "]}";
                return out;
            };
            add(ws);
        }

        // --- execute_python -----------------------------------------------------
        // Только если PROMESERVE_ENABLE_PYTHON=1. python3 -c в sandbox.
        if (env_flag("PROMESERVE_ENABLE_PYTHON")) {
            ToolDefinition ep;
            ep.name = "execute_python";
            ep.description = "Execute a Python snippet (python3 -c). Timeout 15s, stdout truncated to 8KB.";
            ep.parameters = {{"code", "string", "python source", true}};
            ep.input_schema_json = R"({"type":"object","required":["code"]})";
            ep.executor = [](const std::string& args) -> std::string {
                std::string code = tc_json::extract_string_field(args, "code");
                if (code.empty()) return "{\"error\":\"missing code\"}";
                // Запретим вызовы вне sandbox: cd в TOOL_ROOT.
                std::string root = ToolRegistry::get_tool_root();
                // Sanitize: запишем код во временный файл, чтобы избежать
                // экранирования shell.
                std::string code_path = root + "/__exec.py";
                {
                    std::ofstream f(code_path, std::ios::binary);
                    if (!f) return "{\"error\":\"cannot write code file\"}";
                    f << code;
                }
                std::string cmd;
#ifdef _WIN32
                cmd = "python -X utf8 \"" + code_path + "\" 2>&1";
                FILE* p = _popen(cmd.c_str(), "r");
#else
                cmd = "cd \"" + root + "\" && timeout 15 python3 \"" + code_path + "\" 2>&1";
                FILE* p = popen(cmd.c_str(), "r");
#endif
                if (!p) return "{\"error\":\"popen failed\"}";
                std::string out;
                char buf[1024];
                while (fgets(buf, sizeof(buf), p)) {
                    out += buf;
                    if (out.size() > 8 * 1024) { out += "...[truncated]"; break; }
                }
#ifdef _WIN32
                _pclose(p);
#else
                pclose(p);
#endif
                return "{\"ok\":true,\"stdout\":\"" + tc_json::escape_json(out) + "\"}";
            };
            add(ep);
        }

        // --- git (read-only) ----------------------------------------------------
        ToolDefinition gt;
        gt.name = "git";
        gt.description = "Read-only git command (log, diff, status, show, branch, blame). Runs in PROMESERVE_TOOL_ROOT.";
        gt.parameters = {{"command", "string", "git args, e.g. 'log -5 --oneline'", true}};
        gt.input_schema_json = R"({"type":"object","required":["command"]})";
        gt.executor = [](const std::string& args) -> std::string {
            std::string cmd = tc_json::extract_string_field(args, "command");
            if (cmd.empty()) return "{\"error\":\"missing command\"}";
            std::string first;
            for (char c : cmd) {
                if (c == ' ' || c == '\t') break;
                first += c;
            }
            static const std::vector<std::string> allowed = {
                "log","diff","status","show","branch","blame","ls-files","rev-parse","describe"
            };
            bool ok = false;
            for (auto& a : allowed) if (first == a) { ok = true; break; }
            if (!ok) return "{\"error\":\"git subcommand not in read-only whitelist\"}";
            std::string root = ToolRegistry::get_tool_root();
            // Sanitize — режем shell-metacharacters
            for (char c : cmd) {
                if (c == ';' || c == '|' || c == '&' || c == '`' || c == '$' || c == '>') {
                    return "{\"error\":\"shell metacharacters denied\"}";
                }
            }
            std::string full;
#ifdef _WIN32
            full = "cd /d \"" + root + "\" && git " + cmd + " 2>&1";
            FILE* p = _popen(full.c_str(), "r");
#else
            full = "cd \"" + root + "\" && git " + cmd + " 2>&1";
            FILE* p = popen(full.c_str(), "r");
#endif
            if (!p) return "{\"error\":\"popen failed\"}";
            std::string out;
            char buf[1024];
            while (fgets(buf, sizeof(buf), p)) {
                out += buf;
                if (out.size() > 16 * 1024) { out += "...[truncated]"; break; }
            }
#ifdef _WIN32
            _pclose(p);
#else
            pclose(p);
#endif
            return "{\"ok\":true,\"output\":\"" + tc_json::escape_json(out) + "\"}";
        };
        add(gt);

        // --- sqlite (read-only SELECT) ------------------------------------------
        ToolDefinition sq;
        sq.name = "sqlite";
        sq.description = "Run a read-only SQL query (SELECT only) against an sqlite db under sandbox.";
        sq.parameters = {
            {"db_path", "string", "relative path to sqlite db", true},
            {"query",   "string", "SELECT statement", true}};
        sq.input_schema_json = R"({"type":"object","required":["db_path","query"]})";
        sq.executor = [](const std::string& args) -> std::string {
            std::string db = tc_json::extract_string_field(args, "db_path");
            std::string q  = tc_json::extract_string_field(args, "query");
            if (db.empty() || q.empty()) return "{\"error\":\"missing db_path or query\"}";
            // Защита: только SELECT/PRAGMA/WITH
            std::string up;
            for (char c : q) up += static_cast<char>(std::toupper((unsigned char)c));
            // trim leading whitespace
            size_t qs = 0;
            while (qs < up.size() && std::isspace(static_cast<unsigned char>(up[qs]))) qs++;
            if (up.compare(qs, 6, "SELECT") != 0 &&
                up.compare(qs, 6, "PRAGMA") != 0 &&
                up.compare(qs, 4, "WITH") != 0) {
                return "{\"error\":\"only SELECT/PRAGMA/WITH allowed\"}";
            }
            // No semicolons (multi-stmt protection)
            if (q.find(';') != std::string::npos &&
                q.find(';') != q.size() - 1) {
                return "{\"error\":\"multiple statements denied\"}";
            }
            std::string full_db = ToolRegistry::sandbox_path(db);
            // sqlite3 CLI binary
            std::string sanitized_q;
            for (char c : q) {
                if (c == '"' || c == '`' || c == '\\') continue;
                sanitized_q += c;
            }
            std::string cmd;
#ifdef _WIN32
            cmd = "sqlite3.exe -readonly -json \"" + full_db + "\" \"" + sanitized_q + "\" 2>&1";
            FILE* p = _popen(cmd.c_str(), "r");
#else
            cmd = "sqlite3 -readonly -json \"" + full_db + "\" \"" + sanitized_q + "\" 2>&1";
            FILE* p = popen(cmd.c_str(), "r");
#endif
            if (!p) return "{\"error\":\"sqlite3 not available\"}";
            std::string out;
            char buf[1024];
            while (fgets(buf, sizeof(buf), p)) {
                out += buf;
                if (out.size() > 32 * 1024) { out += "...[truncated]"; break; }
            }
#ifdef _WIN32
            _pclose(p);
#else
            pclose(p);
#endif
            return "{\"ok\":true,\"rows\":\"" + tc_json::escape_json(out) + "\"}";
        };
        add(sq);
    }
};

// ============================================================================
// Tool call detection — single & all
// ============================================================================

struct ToolCall {
    std::string name;
    std::string args_json;
    size_t start_pos = 0;
    size_t end_pos = 0;
    bool valid = false;
};

inline ToolCall detect_tool_call_at(const std::string& text, size_t from) {
    ToolCall tc;
    size_t s = text.find("<tool_call>", from);
    if (s == std::string::npos) return tc;
    size_t e = text.find("</tool_call>", s);
    if (e == std::string::npos) return tc;
    tc.start_pos = s;
    tc.end_pos = e + std::string("</tool_call>").size();
    std::string payload = text.substr(s + std::string("<tool_call>").size(),
                                       e - s - std::string("<tool_call>").size());
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
            if (depth == 0) tc.args_json = payload.substr(lb, i - lb);
        }
    }
    tc.valid = !tc.name.empty();
    return tc;
}

inline ToolCall detect_tool_call(const std::string& text) {
    return detect_tool_call_at(text, 0);
}

// Parallel parse — собрать все <tool_call> блоки.
inline std::vector<ToolCall> detect_all_tool_calls(const std::string& text) {
    std::vector<ToolCall> out;
    size_t pos = 0;
    while (true) {
        ToolCall tc = detect_tool_call_at(text, pos);
        if (!tc.valid) break;
        out.push_back(tc);
        pos = tc.end_pos;
    }
    return out;
}

// ============================================================================
// Streaming-stop detector. Используется в decode-loop:
//     StreamingToolCallDetector det;
//     for each token:
//        det.feed(token_str);
//        if (det.should_stop()) break;
// На EOS-вызывающую границу `</tool_call>` декодер останавливается, агенту
// возвращается всё, что накопилось до этого момента — он сам найдёт
// блок через detect_all_tool_calls(buffer).
// ============================================================================

class StreamingToolCallDetector {
public:
    void feed(const std::string& chunk) {
        buf_.append(chunk);
        if (!stop_) {
            size_t p = buf_.find("</tool_call>");
            if (p != std::string::npos) {
                stop_ = true;
                stop_pos_ = p + std::string("</tool_call>").size();
            }
        }
    }
    bool should_stop() const { return stop_; }
    size_t stop_pos() const { return stop_pos_; }
    const std::string& buffer() const { return buf_; }
    void reset() { buf_.clear(); stop_ = false; stop_pos_ = 0; }

private:
    std::string buf_;
    bool stop_ = false;
    size_t stop_pos_ = 0;
};

// ============================================================================
// Parallel execution of multiple tool calls.
// Использует std::thread + future (не c10::ThreadPool — он broadcast-dispatch,
// не подходит для разнородных task'ов с разным временем выполнения).
// ============================================================================

struct ToolCallResult {
    std::string name;
    std::string args_json;
    std::string result_json;
    double duration_ms = 0.0;
    bool external = false;   // tool — client-supplied, нужно вернуть наружу
};

inline std::vector<ToolCallResult> parallel_execute(
    ToolRegistry& registry,
    const std::vector<ToolCall>& calls,
    size_t max_parallel = 4) {

    std::vector<ToolCallResult> results(calls.size());
    if (calls.empty()) return results;

    // Сериализуем external-tools без exec — agent loop сам вернёт их.
    std::vector<size_t> to_run;
    for (size_t i = 0; i < calls.size(); i++) {
        results[i].name = calls[i].name;
        results[i].args_json = calls[i].args_json;
        if (registry.is_external(calls[i].name)) {
            results[i].external = true;
            results[i].result_json = "{\"_external\":true}";
        } else {
            to_run.push_back(i);
        }
    }

    if (to_run.size() <= 1 || max_parallel <= 1) {
        // Serial path
        for (size_t idx : to_run) {
            auto t0 = std::chrono::steady_clock::now();
            results[idx].result_json = registry.execute(calls[idx].name, calls[idx].args_json);
            results[idx].duration_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
        }
        return results;
    }

    // Parallel — std::async с ограничением concurrency.
    std::vector<std::future<void>> futures;
    std::atomic<size_t> in_flight{0};

    for (size_t idx : to_run) {
        // Простой semaphore через busy-wait (max_parallel мал — обычно 4).
        while (in_flight.load() >= max_parallel) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        in_flight.fetch_add(1);
        futures.push_back(std::async(std::launch::async,
            [&registry, &calls, &results, idx, &in_flight]() {
                auto t0 = std::chrono::steady_clock::now();
                try {
                    results[idx].result_json =
                        registry.execute(calls[idx].name, calls[idx].args_json);
                } catch (const std::exception& e) {
                    results[idx].result_json =
                        std::string("{\"error\":\"") + e.what() + "\"}";
                }
                results[idx].duration_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                in_flight.fetch_sub(1);
            }));
    }
    for (auto& f : futures) {
        if (f.valid()) f.wait();
    }
    return results;
}

// ============================================================================
// Max-iter guard helper.
// ============================================================================
inline int max_tool_iter() {
    const char* v = std::getenv("PROMESERVE_MAX_TOOL_ITER");
    if (!v) return 10;
    try {
        int n = std::stoi(v);
        return (n > 0 && n < 1000) ? n : 10;
    } catch (...) { return 10; }
}

// ============================================================================
// Format system prompt with tools (qwen3 / Hermes style).
// ============================================================================

inline std::string format_system_prompt_with_tools(
    const std::string& user_system,
    const ToolRegistry& reg)
{
    std::string sys = user_system.empty() ? "You are a helpful assistant." : user_system;
    sys += "\n\nYou have access to the following tools:\n<tools>\n";
    sys += reg.format_for_prompt();
    sys += "\n</tools>\n\n";
    sys += "To use one or more tools, emit one or several blocks (parallel calls OK):\n";
    sys += "<tool_call>{\"name\":\"<tool_name>\",\"arguments\":{<args>}}</tool_call>\n";
    sys += "After receiving <tool_response>, continue. When task is complete, give the final answer without tool_call.\n";
    return sys;
}

}  // namespace promeserve
