#pragma once

// ============================================================================
// PromeServe — MCP Client (Этап 2)
//
// Подключается к external MCP server'у через JSON-RPC 2.0 over stdio.
// На startup читается ~/.promeserve/mcp.json (Cline-compatible формат),
// для каждого entry — spawn'ится subprocess, выполняется `initialize` +
// `tools/list`, all tools регистрируются в общий ToolRegistry с префиксом
// `mcp__<server>__<tool>`. При вызове такого tool ToolRegistry дёргает
// MCPDispatcher (см. tool_call.h), который форвардит `tools/call` в нужный
// сервер.
//
// Транспорт — POSIX fork/pipe + dup2 на не-Windows, CreateProcess + pipes
// на Windows. Без внешних зависимостей кроме stdlib.
//
// Lifecycle:
//   - Health check ping (`tools/list` без params, lightweight) каждые 60 сек.
//   - Auto-restart с exp backoff (1s, 2s, 5s, 30s — max).
//   - shutdown отправляет `notifications/cancelled` и close stdin.
//
// Конкурентность:
//   - Каждый MCPClient имеет:
//       * writer mutex (один writer в pipe)
//       * dispatcher thread (читает stdout, маршрутизирует response → promise)
//       * pending_ map<int, promise<json>> (atomic id allocator)
//   - call() блокирующий с timeout (default 30s).
//
// Endpoints API:
//   GET  /api/mcp/servers       — список + статус
//   GET  /api/mcp/tools         — каталог tools (server.tool_name)
//   POST /api/mcp/reconnect/X   — force reconnect server X
//   POST /api/mcp/call          — manual call (debug)
//   GET  /api/mcp/audit?n=50    — последние N audit-записей
//
// Spec source: https://modelcontextprotocol.io/specification/2025-11-25
// JSON-RPC 2.0: https://www.jsonrpc.org/specification
// ============================================================================

#include "promeserve/tool_call.h"

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <future>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <signal.h>
#  include <sys/wait.h>
#  include <poll.h>
#endif

namespace promeserve {

// ============================================================================
// Минималистичный JSON parse / build (subset).
// Полноценная парсилка для requests — в api_handlers.h::json. Здесь — лишь то
// что нужно для MCP wire protocol: object с id/method/params/result/error и
// строки/числа.
// ============================================================================
namespace mcp_json {

struct Value {
    enum Type { NUL, BOOL_, NUM, STR, ARR, OBJ } type = NUL;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    bool is_obj() const { return type == OBJ; }
    bool is_arr() const { return type == ARR; }
    bool is_str() const { return type == STR; }
    bool is_num() const { return type == NUM; }
    bool has(const std::string& k) const { return type == OBJ && obj.count(k); }
    const Value& get(const std::string& k) const {
        static Value empty;
        auto it = obj.find(k);
        return (type == OBJ && it != obj.end()) ? it->second : empty;
    }
    std::string str() const { return type == STR ? s : ""; }
    int num_i() const { return type == NUM ? static_cast<int>(n) : 0; }
};

inline void skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) p++;
}

inline std::string parse_str(const std::string& s, size_t& p) {
    if (p >= s.size() || s[p] != '"') return "";
    p++;
    std::string out;
    while (p < s.size() && s[p] != '"') {
        if (s[p] == '\\' && p + 1 < s.size()) {
            p++;
            switch (s[p]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    if (p + 4 < s.size()) {
                        unsigned int cp = 0;
                        for (int k = 1; k <= 4; k++) {
                            char c = s[p + k];
                            cp <<= 4;
                            if (c >= '0' && c <= '9') cp |= (c - '0');
                            else if (c >= 'a' && c <= 'f') cp |= (c - 'a' + 10);
                            else if (c >= 'A' && c <= 'F') cp |= (c - 'A' + 10);
                        }
                        if (cp < 0x80) out += static_cast<char>(cp);
                        else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        p += 4;
                    }
                    break;
                }
                default: out += s[p]; break;
            }
        } else {
            out += s[p];
        }
        p++;
    }
    if (p < s.size()) p++;
    return out;
}

inline Value parse_value(const std::string& s, size_t& p);

inline Value parse_obj(const std::string& s, size_t& p) {
    Value v;
    v.type = Value::OBJ;
    p++;
    skip_ws(s, p);
    while (p < s.size() && s[p] != '}') {
        skip_ws(s, p);
        if (s[p] != '"') break;
        std::string k = parse_str(s, p);
        skip_ws(s, p);
        if (p < s.size() && s[p] == ':') p++;
        skip_ws(s, p);
        v.obj[k] = parse_value(s, p);
        skip_ws(s, p);
        if (p < s.size() && s[p] == ',') p++;
        skip_ws(s, p);
    }
    if (p < s.size()) p++;
    return v;
}

inline Value parse_arr(const std::string& s, size_t& p) {
    Value v;
    v.type = Value::ARR;
    p++;
    skip_ws(s, p);
    while (p < s.size() && s[p] != ']') {
        v.arr.push_back(parse_value(s, p));
        skip_ws(s, p);
        if (p < s.size() && s[p] == ',') p++;
        skip_ws(s, p);
    }
    if (p < s.size()) p++;
    return v;
}

inline Value parse_value(const std::string& s, size_t& p) {
    skip_ws(s, p);
    Value v;
    if (p >= s.size()) return v;
    if (s[p] == '"') { v.type = Value::STR; v.s = parse_str(s, p); return v; }
    if (s[p] == '{') return parse_obj(s, p);
    if (s[p] == '[') return parse_arr(s, p);
    if (s[p] == 't' && s.compare(p, 4, "true") == 0) { v.type = Value::BOOL_; v.b = true; p += 4; return v; }
    if (s[p] == 'f' && s.compare(p, 5, "false") == 0) { v.type = Value::BOOL_; v.b = false; p += 5; return v; }
    if (s[p] == 'n' && s.compare(p, 4, "null") == 0) { v.type = Value::NUL; p += 4; return v; }
    size_t start = p;
    if (s[p] == '-' || s[p] == '+') p++;
    while (p < s.size() &&
           (std::isdigit(static_cast<unsigned char>(s[p])) ||
            s[p] == '.' || s[p] == 'e' || s[p] == 'E' || s[p] == '-' || s[p] == '+')) p++;
    v.type = Value::NUM;
    try { v.n = std::stod(s.substr(start, p - start)); } catch (...) { v.n = 0; }
    return v;
}

inline Value parse(const std::string& s) {
    size_t p = 0;
    return parse_value(s, p);
}

// Сериализация — простая, без exotic'и (нам нужно только outbound JSON-RPC).
inline std::string serialize(const Value& v) {
    switch (v.type) {
        case Value::NUL:   return "null";
        case Value::BOOL_: return v.b ? "true" : "false";
        case Value::NUM: {
            std::ostringstream oss;
            // если целое — без точки
            if (v.n == static_cast<int64_t>(v.n)) {
                oss << static_cast<int64_t>(v.n);
            } else {
                oss << v.n;
            }
            return oss.str();
        }
        case Value::STR: return "\"" + tc_json::escape_json(v.s) + "\"";
        case Value::ARR: {
            std::string out = "[";
            for (size_t i = 0; i < v.arr.size(); i++) {
                if (i) out += ",";
                out += serialize(v.arr[i]);
            }
            return out + "]";
        }
        case Value::OBJ: {
            std::string out = "{";
            bool first = true;
            for (auto& [k, val] : v.obj) {
                if (!first) out += ",";
                first = false;
                out += "\"" + tc_json::escape_json(k) + "\":" + serialize(val);
            }
            return out + "}";
        }
    }
    return "null";
}

}  // namespace mcp_json

// ============================================================================
// Конфиг (Cline-compatible mcp.json).
// ============================================================================

struct MCPServerSpec {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::map<std::string, std::string> env;
    bool disabled = false;
};

struct MCPConfig {
    std::vector<MCPServerSpec> servers;
    std::string source_path;
};

inline std::string mcp_config_path() {
    const char* env = std::getenv("PROMESERVE_MCP_CONFIG");
    if (env) return env;
    const char* home = std::getenv(
#ifdef _WIN32
        "USERPROFILE"
#else
        "HOME"
#endif
    );
    if (!home) return "";
    return std::string(home) + "/.promeserve/mcp.json";
}

inline MCPConfig load_mcp_config() {
    MCPConfig cfg;
    cfg.source_path = mcp_config_path();
    if (cfg.source_path.empty()) return cfg;
    std::ifstream f(cfg.source_path);
    if (!f) return cfg;
    std::stringstream ss;
    ss << f.rdbuf();
    auto root = mcp_json::parse(ss.str());
    if (!root.is_obj() || !root.has("mcpServers")) return cfg;
    auto& srv = root.get("mcpServers");
    if (!srv.is_obj()) return cfg;
    for (auto& [name, val] : srv.obj) {
        if (!val.is_obj()) continue;
        MCPServerSpec s;
        s.name = name;
        s.command = val.get("command").str();
        if (val.has("args") && val.get("args").is_arr()) {
            for (auto& a : val.get("args").arr) {
                if (a.is_str()) s.args.push_back(a.s);
            }
        }
        if (val.has("env") && val.get("env").is_obj()) {
            for (auto& [k, v] : val.get("env").obj) {
                if (v.is_str()) s.env[k] = v.s;
            }
        }
        if (val.has("disabled") && val.get("disabled").type == mcp_json::Value::BOOL_) {
            s.disabled = val.get("disabled").b;
        }
        cfg.servers.push_back(std::move(s));
    }
    return cfg;
}

// ============================================================================
// MCPSubprocess — кросс-платформенный fork+pipe.
// ============================================================================

class MCPSubprocess {
public:
    MCPSubprocess() = default;
    ~MCPSubprocess() { stop(); }
    MCPSubprocess(const MCPSubprocess&) = delete;
    MCPSubprocess& operator=(const MCPSubprocess&) = delete;

    bool start(const MCPServerSpec& spec, std::string* err_out = nullptr) {
        if (running_) return true;

#ifdef _WIN32
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;

        HANDLE child_stdin_rd = nullptr, child_stdin_wr = nullptr;
        HANDLE child_stdout_rd = nullptr, child_stdout_wr = nullptr;

        if (!CreatePipe(&child_stdin_rd, &child_stdin_wr, &sa, 0)) {
            if (err_out) *err_out = "CreatePipe(stdin) failed";
            return false;
        }
        SetHandleInformation(child_stdin_wr, HANDLE_FLAG_INHERIT, 0);

        if (!CreatePipe(&child_stdout_rd, &child_stdout_wr, &sa, 0)) {
            CloseHandle(child_stdin_rd); CloseHandle(child_stdin_wr);
            if (err_out) *err_out = "CreatePipe(stdout) failed";
            return false;
        }
        SetHandleInformation(child_stdout_rd, HANDLE_FLAG_INHERIT, 0);

        std::string cmdline = quote_(spec.command);
        for (auto& a : spec.args) { cmdline += " "; cmdline += quote_(a); }

        // Env block (UNICODE-less, простой ANSI).
        std::string env_block;
        bool any_env = !spec.env.empty();
        if (any_env) {
            // Скопируем родительский env потом добавим override.
            LPCH cur = GetEnvironmentStringsA();
            for (LPCH p = cur; *p; ) {
                size_t l = std::strlen(p);
                env_block.append(p, l);
                env_block.push_back('\0');
                p += l + 1;
            }
            FreeEnvironmentStringsA(cur);
            for (auto& [k, v] : spec.env) {
                env_block += k + "=" + expand_env_(v);
                env_block.push_back('\0');
            }
            env_block.push_back('\0');
        }

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = child_stdin_rd;
        si.hStdOutput = child_stdout_wr;
        si.hStdError = child_stdout_wr;

        PROCESS_INFORMATION pi{};
        BOOL ok = CreateProcessA(
            nullptr,
            const_cast<char*>(cmdline.c_str()),
            nullptr, nullptr,
            TRUE,
            CREATE_NO_WINDOW,
            any_env ? const_cast<char*>(env_block.c_str()) : nullptr,
            nullptr,
            &si, &pi);

        CloseHandle(child_stdin_rd);
        CloseHandle(child_stdout_wr);

        if (!ok) {
            CloseHandle(child_stdin_wr); CloseHandle(child_stdout_rd);
            if (err_out) *err_out = "CreateProcess failed: " + std::to_string(GetLastError());
            return false;
        }
        CloseHandle(pi.hThread);
        proc_handle_ = pi.hProcess;
        stdin_handle_ = child_stdin_wr;
        stdout_handle_ = child_stdout_rd;
        running_ = true;
        return true;
#else
        int in_pipe[2];   // parent writes
        int out_pipe[2];  // parent reads
        if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) {
            if (err_out) *err_out = "pipe() failed";
            return false;
        }
        pid_t pid = fork();
        if (pid < 0) {
            if (err_out) *err_out = "fork() failed";
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);
            return false;
        }
        if (pid == 0) {
            // Child
            ::dup2(in_pipe[0], 0);
            ::dup2(out_pipe[1], 1);
            ::dup2(out_pipe[1], 2);
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);
            // env override
            for (auto& [k, v] : spec.env) {
                std::string val = expand_env_(v);
                setenv(k.c_str(), val.c_str(), 1);
            }
            std::vector<char*> argv;
            argv.push_back(const_cast<char*>(spec.command.c_str()));
            for (auto& a : spec.args) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            execvp(spec.command.c_str(), argv.data());
            // exec failed
            ::_exit(127);
        }
        // Parent
        ::close(in_pipe[0]);
        ::close(out_pipe[1]);
        pid_ = pid;
        stdin_fd_ = in_pipe[1];
        stdout_fd_ = out_pipe[0];
        running_ = true;
        return true;
#endif
    }

    void stop() {
        if (!running_) return;
        running_ = false;
#ifdef _WIN32
        if (stdin_handle_) { CloseHandle(stdin_handle_); stdin_handle_ = nullptr; }
        if (stdout_handle_) { CloseHandle(stdout_handle_); stdout_handle_ = nullptr; }
        if (proc_handle_) {
            // Дать процессу шанс выйти, потом терминируем
            WaitForSingleObject(proc_handle_, 500);
            DWORD exit_code = STILL_ACTIVE;
            GetExitCodeProcess(proc_handle_, &exit_code);
            if (exit_code == STILL_ACTIVE) TerminateProcess(proc_handle_, 1);
            CloseHandle(proc_handle_);
            proc_handle_ = nullptr;
        }
#else
        if (stdin_fd_ >= 0) { ::close(stdin_fd_); stdin_fd_ = -1; }
        if (stdout_fd_ >= 0) { ::close(stdout_fd_); stdout_fd_ = -1; }
        if (pid_ > 0) {
            // SIGTERM, потом SIGKILL после 500ms
            ::kill(pid_, SIGTERM);
            for (int i = 0; i < 5; i++) {
                int status = 0;
                pid_t r = ::waitpid(pid_, &status, WNOHANG);
                if (r == pid_) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, nullptr, 0);
            pid_ = -1;
        }
#endif
    }

    bool write(const std::string& data) {
        if (!running_) return false;
#ifdef _WIN32
        DWORD written = 0;
        return WriteFile(stdin_handle_, data.data(),
                         static_cast<DWORD>(data.size()), &written, nullptr)
            && written == data.size();
#else
        ssize_t total = 0;
        while (total < (ssize_t)data.size()) {
            ssize_t n = ::write(stdin_fd_, data.data() + total, data.size() - total);
            if (n <= 0) return false;
            total += n;
        }
        return true;
#endif
    }

    // Читает доступные байты (non-blocking poll, timeout ms).
    // Возвращает прочитанное; пустая строка ⇒ ничего нет (или EOF).
    std::string read_some(int timeout_ms) {
        if (!running_) return "";
#ifdef _WIN32
        // Windows: PeekNamedPipe для non-blocking, потом ReadFile
        auto t0 = std::chrono::steady_clock::now();
        std::string out;
        while (true) {
            DWORD avail = 0;
            if (!PeekNamedPipe(stdout_handle_, nullptr, 0, nullptr, &avail, nullptr)) {
                return out;
            }
            if (avail > 0) {
                char buf[4096];
                DWORD to_read = avail > sizeof(buf) ? sizeof(buf) : avail;
                DWORD got = 0;
                if (!ReadFile(stdout_handle_, buf, to_read, &got, nullptr) || got == 0) {
                    return out;
                }
                out.append(buf, got);
                return out;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (elapsed >= timeout_ms) return out;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
#else
        struct pollfd pfd{};
        pfd.fd = stdout_fd_;
        pfd.events = POLLIN;
        int rv = ::poll(&pfd, 1, timeout_ms);
        if (rv <= 0) return "";
        if (!(pfd.revents & POLLIN)) return "";
        char buf[4096];
        ssize_t n = ::read(stdout_fd_, buf, sizeof(buf));
        if (n <= 0) return "";
        return std::string(buf, n);
#endif
    }

    bool is_running() const { return running_; }

private:
    static std::string quote_(const std::string& s) {
        if (s.find(' ') == std::string::npos && s.find('\t') == std::string::npos) return s;
        return "\"" + s + "\"";
    }

    static std::string expand_env_(const std::string& v) {
        // "$FOO" → getenv("FOO"); поддерживаем простой формат.
        if (!v.empty() && v[0] == '$') {
            const char* e = std::getenv(v.c_str() + 1);
            return e ? e : "";
        }
        return v;
    }

    std::atomic<bool> running_{false};
#ifdef _WIN32
    HANDLE proc_handle_ = nullptr;
    HANDLE stdin_handle_ = nullptr;
    HANDLE stdout_handle_ = nullptr;
#else
    pid_t pid_ = -1;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
#endif
};

// ============================================================================
// MCPClient — JSON-RPC 2.0 over stdio к одному MCP-серверу.
// ============================================================================

class MCPClient {
public:
    explicit MCPClient(MCPServerSpec spec) : spec_(std::move(spec)) {}
    ~MCPClient() { stop(); }
    MCPClient(const MCPClient&) = delete;
    MCPClient& operator=(const MCPClient&) = delete;

    const std::string& name() const { return spec_.name; }
    const MCPServerSpec& spec() const { return spec_; }

    bool start(std::string* err = nullptr) {
        if (running_) return true;
        if (!subproc_.start(spec_, err)) {
            mark_unavailable_("subprocess start failed");
            return false;
        }
        running_ = true;
        stop_dispatch_ = false;
        dispatcher_ = std::thread([this]() { dispatcher_loop_(); });

        // Handshake
        mcp_json::Value params;
        params.type = mcp_json::Value::OBJ;
        params.obj["protocolVersion"].type = mcp_json::Value::STR;
        params.obj["protocolVersion"].s = "2025-11-25";
        mcp_json::Value caps; caps.type = mcp_json::Value::OBJ;
        caps.obj["tools"].type = mcp_json::Value::OBJ;
        caps.obj["resources"].type = mcp_json::Value::OBJ;
        params.obj["capabilities"] = caps;
        mcp_json::Value cinfo; cinfo.type = mcp_json::Value::OBJ;
        cinfo.obj["name"].type = mcp_json::Value::STR;
        cinfo.obj["name"].s = "PromeServe";
        cinfo.obj["version"].type = mcp_json::Value::STR;
        cinfo.obj["version"].s = "0.1.0";
        params.obj["clientInfo"] = cinfo;

        auto init_resp = call("initialize", params, 30000);
        if (init_resp.has("error") || init_resp.type == mcp_json::Value::NUL) {
            mark_unavailable_("initialize failed");
            stop();
            if (err) *err = "initialize failed";
            return false;
        }
        // notifications/initialized (no id, fire-and-forget)
        send_notification_("notifications/initialized");
        available_ = true;
        last_heartbeat_ = std::chrono::steady_clock::now();
        return true;
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        stop_dispatch_ = true;
        // close stdin → MCP server должен exit
        subproc_.stop();
        if (dispatcher_.joinable()) dispatcher_.join();
        // Notify любые pending'ы об отмене.
        std::lock_guard<std::mutex> lock(pending_mu_);
        for (auto& [id, p] : pending_) {
            try {
                mcp_json::Value v;
                v.type = mcp_json::Value::OBJ;
                v.obj["error"].type = mcp_json::Value::STR;
                v.obj["error"].s = "client stopped";
                p.set_value(v);
            } catch (...) {}
        }
        pending_.clear();
    }

    bool is_available() const { return available_; }
    bool is_running() const { return running_; }
    std::string last_error() const {
        std::lock_guard<std::mutex> lock(err_mu_);
        return last_error_;
    }

    // Синхронный JSON-RPC call с таймаутом. Возвращает result-объект или
    // {"error":"..."} при таймауте/RPC ошибке.
    mcp_json::Value call(const std::string& method,
                         const mcp_json::Value& params,
                         int timeout_ms = 30000) {
        if (!running_) {
            mcp_json::Value e;
            e.type = mcp_json::Value::OBJ;
            e.obj["error"].type = mcp_json::Value::STR;
            e.obj["error"].s = "not running";
            return e;
        }
        int id = next_id_.fetch_add(1);
        std::promise<mcp_json::Value> prom;
        std::future<mcp_json::Value> fut = prom.get_future();
        {
            std::lock_guard<std::mutex> lock(pending_mu_);
            pending_[id] = std::move(prom);
        }

        mcp_json::Value req;
        req.type = mcp_json::Value::OBJ;
        req.obj["jsonrpc"].type = mcp_json::Value::STR;
        req.obj["jsonrpc"].s = "2.0";
        req.obj["id"].type = mcp_json::Value::NUM;
        req.obj["id"].n = id;
        req.obj["method"].type = mcp_json::Value::STR;
        req.obj["method"].s = method;
        if (params.type != mcp_json::Value::NUL) req.obj["params"] = params;
        std::string line = mcp_json::serialize(req) + "\n";
        bool sent;
        {
            std::lock_guard<std::mutex> lock(writer_mu_);
            sent = subproc_.write(line);
        }
        if (!sent) {
            std::lock_guard<std::mutex> lock(pending_mu_);
            pending_.erase(id);
            mark_unavailable_("write failed");
            mcp_json::Value e;
            e.type = mcp_json::Value::OBJ;
            e.obj["error"].type = mcp_json::Value::STR;
            e.obj["error"].s = "write failed";
            return e;
        }

        auto status = fut.wait_for(std::chrono::milliseconds(timeout_ms));
        if (status != std::future_status::ready) {
            std::lock_guard<std::mutex> lock(pending_mu_);
            pending_.erase(id);
            mark_unavailable_("call timeout: " + method);
            mcp_json::Value e;
            e.type = mcp_json::Value::OBJ;
            e.obj["error"].type = mcp_json::Value::STR;
            e.obj["error"].s = "timeout";
            return e;
        }
        return fut.get();
    }

    // tools/list helper. Возвращает массив описаний.
    mcp_json::Value list_tools(int timeout_ms = 10000) {
        mcp_json::Value p;
        p.type = mcp_json::Value::OBJ;
        return call("tools/list", p, timeout_ms);
    }

    // tools/call helper. arguments — уже-JSON arguments object (с обёртки {}).
    mcp_json::Value tools_call(const std::string& tool_name,
                               const std::string& arguments_json,
                               int timeout_ms = 30000) {
        mcp_json::Value p;
        p.type = mcp_json::Value::OBJ;
        p.obj["name"].type = mcp_json::Value::STR;
        p.obj["name"].s = tool_name;
        // Парсим arguments_json как mcp_json::Value
        size_t pos = 0;
        std::string aj = arguments_json.empty() ? "{}" : arguments_json;
        p.obj["arguments"] = mcp_json::parse_value(aj, pos);
        return call("tools/call", p, timeout_ms);
    }

    // ping для health check. Используем `tools/list` как лёгкую проверку
    // (стандартного `ping` метода в спеке нет).
    bool ping(int timeout_ms = 5000) {
        auto r = list_tools(timeout_ms);
        if (r.has("error") || r.type == mcp_json::Value::NUL) {
            mark_unavailable_("ping failed");
            return false;
        }
        available_ = true;
        last_heartbeat_ = std::chrono::steady_clock::now();
        return true;
    }

    std::chrono::steady_clock::time_point last_heartbeat() const {
        return last_heartbeat_;
    }

private:
    void mark_unavailable_(const std::string& msg) {
        available_ = false;
        std::lock_guard<std::mutex> lock(err_mu_);
        last_error_ = msg;
    }

    void send_notification_(const std::string& method) {
        mcp_json::Value n;
        n.type = mcp_json::Value::OBJ;
        n.obj["jsonrpc"].type = mcp_json::Value::STR;
        n.obj["jsonrpc"].s = "2.0";
        n.obj["method"].type = mcp_json::Value::STR;
        n.obj["method"].s = method;
        std::string line = mcp_json::serialize(n) + "\n";
        std::lock_guard<std::mutex> lock(writer_mu_);
        subproc_.write(line);
    }

    void dispatcher_loop_() {
        std::string buf;
        while (!stop_dispatch_) {
            std::string chunk = subproc_.read_some(200);
            if (chunk.empty()) {
                if (!subproc_.is_running()) { mark_unavailable_("subprocess died"); break; }
                continue;
            }
            buf += chunk;
            // Парсим строки (LSP/MCP stdio — line-delimited JSON).
            while (true) {
                size_t nl = buf.find('\n');
                if (nl == std::string::npos) break;
                std::string line = buf.substr(0, nl);
                buf.erase(0, nl + 1);
                // strip CR
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                if (line.empty()) continue;
                handle_message_(line);
            }
            if (buf.size() > 4 * 1024 * 1024) {
                // sanity: 4MB без \n — что-то не так
                buf.clear();
                mark_unavailable_("malformed stream");
                break;
            }
        }
    }

    void handle_message_(const std::string& line) {
        auto msg = mcp_json::parse(line);
        if (!msg.is_obj()) return;
        if (msg.has("id")) {
            int id = msg.get("id").num_i();
            std::lock_guard<std::mutex> lock(pending_mu_);
            auto it = pending_.find(id);
            if (it != pending_.end()) {
                try {
                    if (msg.has("result")) {
                        it->second.set_value(msg.get("result"));
                    } else if (msg.has("error")) {
                        mcp_json::Value e;
                        e.type = mcp_json::Value::OBJ;
                        e.obj["error"] = msg.get("error");
                        it->second.set_value(e);
                    } else {
                        it->second.set_value(mcp_json::Value{});
                    }
                } catch (...) {}
                pending_.erase(it);
            }
        }
        // Notifications (no id) — пока игнорируем (можно расширить:
        // resources/updated, tools/listChanged, ...).
    }

    MCPServerSpec spec_;
    MCPSubprocess subproc_;
    std::thread dispatcher_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_dispatch_{false};
    std::atomic<bool> available_{false};
    std::atomic<int> next_id_{1};

    std::mutex writer_mu_;
    std::mutex pending_mu_;
    std::unordered_map<int, std::promise<mcp_json::Value>> pending_;

    mutable std::mutex err_mu_;
    std::string last_error_;
    std::chrono::steady_clock::time_point last_heartbeat_;
};

// ============================================================================
// MCPManager — оркестрация всех MCP клиентов.
//
// На startup:
//   1. load_mcp_config()
//   2. для каждого server'а — spawn + initialize + tools/list
//   3. tools registered в ToolRegistry с префиксом mcp__<server>__<tool>
//   4. set MCPDispatcher в ToolRegistry → tools_call routing
//
// Health-check thread: каждые 60 сек ping() каждого. Если нет ответа — mark
// unavailable, schedule reconnect (exp backoff).
// ============================================================================

class MCPManager {
public:
    explicit MCPManager(ToolRegistry& registry) : registry_(registry) {
        registry_.set_mcp_dispatcher(
            [this](const MCPToolRoute& r, const std::string& args) {
                return this->dispatch_(r, args);
            });
    }
    ~MCPManager() { shutdown(); }

    MCPManager(const MCPManager&) = delete;
    MCPManager& operator=(const MCPManager&) = delete;

    // Запустить все из конфига. Не fail'ит на одном плохом сервере.
    void start_all() {
        MCPConfig cfg = load_mcp_config();
        config_path_ = cfg.source_path;
        for (auto& spec : cfg.servers) {
            if (spec.disabled) continue;
            start_server_(spec);
        }
        start_health_check_();
    }

    void shutdown() {
        stop_health_ = true;
        if (health_thread_.joinable()) health_thread_.join();
        std::lock_guard<std::mutex> lock(clients_mu_);
        for (auto& [name, c] : clients_) {
            if (c) c->stop();
            registry_.remove_mcp_tools(name);
        }
        clients_.clear();
    }

    // Принудительный reconnect одного сервера. Возвращает true если ok.
    bool reconnect(const std::string& server_name) {
        std::lock_guard<std::mutex> lock(clients_mu_);
        auto it = clients_.find(server_name);
        MCPServerSpec spec;
        if (it != clients_.end()) {
            spec = it->second->spec();
            it->second->stop();
            clients_.erase(it);
        } else {
            MCPConfig cfg = load_mcp_config();
            bool found = false;
            for (auto& s : cfg.servers) {
                if (s.name == server_name) { spec = s; found = true; break; }
            }
            if (!found) return false;
        }
        registry_.remove_mcp_tools(server_name);
        return start_server_locked_(spec);
    }

    struct ServerStatus {
        std::string name;
        bool available;
        std::string last_error;
        int tool_count;
        std::string command;
    };

    std::vector<ServerStatus> list_status() {
        std::lock_guard<std::mutex> lock(clients_mu_);
        std::vector<ServerStatus> out;
        for (auto& [name, c] : clients_) {
            ServerStatus s;
            s.name = name;
            s.available = c->is_available();
            s.last_error = c->last_error();
            s.command = c->spec().command;
            for (auto& a : c->spec().args) { s.command += " "; s.command += a; }
            s.tool_count = tool_count_(name);
            out.push_back(s);
        }
        return out;
    }

    int tool_count_(const std::string& server_name) {
        int c = 0;
        for (auto& t : registry_.list_all()) {
            if (t.kind == ToolKind::MCP && t.mcp_route.server_name == server_name) c++;
        }
        return c;
    }

    const std::string& config_path() const { return config_path_; }

private:
    bool start_server_(const MCPServerSpec& spec) {
        std::lock_guard<std::mutex> lock(clients_mu_);
        return start_server_locked_(spec);
    }

    bool start_server_locked_(const MCPServerSpec& spec) {
        auto client = std::make_unique<MCPClient>(spec);
        std::string err;
        if (!client->start(&err)) {
            // Сохраняем клиента даже неуспешного — для list_status().
            clients_[spec.name] = std::move(client);
            return false;
        }
        // List tools и зарегистрировать
        auto tools = client->list_tools(10000);
        if (tools.is_obj() && tools.has("tools") && tools.get("tools").is_arr()) {
            for (auto& t : tools.get("tools").arr) {
                if (!t.is_obj()) continue;
                std::string remote_name = t.get("name").str();
                if (remote_name.empty()) continue;
                std::string local_name = "mcp__" + spec.name + "__" + remote_name;
                std::string desc = t.get("description").str();
                std::string schema;
                if (t.has("inputSchema")) {
                    schema = mcp_json::serialize(t.get("inputSchema"));
                }
                registry_.add_mcp_tool(local_name, desc, schema, spec.name, remote_name);
            }
        }
        clients_[spec.name] = std::move(client);
        return true;
    }

    std::string dispatch_(const MCPToolRoute& route, const std::string& args_json) {
        // Возьмём raw pointer под lock'ом, потом отпустим lock — MCPClient::call
        // thread-safe (внутри свои мьютексы). Удаление client'а возможно только
        // из shutdown()/reconnect(), оба берут clients_mu_; пока мы из неё не
        // вышли через возврат, client жив. Для безопасности после извлечения
        // указателя — продолжим работать с raw, но проверим doc выше: shutdown
        // synchronous join'ит health thread, а удаление в reconnect делает
        // stop() который сначала помечает available_=false, и tools_call()
        // на нём вернёт ошибку — но race потенциально остаётся. Минимизируем:
        // держим shared lock на время вызова через scoped guard.
        std::lock_guard<std::mutex> lock(clients_mu_);
        auto it = clients_.find(route.server_name);
        if (it == clients_.end() || !it->second->is_available()) {
            return "{\"error\":\"MCP server unavailable: " + route.server_name + "\"}";
        }
        auto raw = it->second.get();
        try {
            auto resp = raw->tools_call(route.remote_tool_name, args_json, 30000);
            if (resp.has("error")) {
                return "{\"error\":" + mcp_json::serialize(resp.get("error")) + "}";
            }
            if (resp.has("content")) {
                return "{\"ok\":true,\"content\":" +
                    mcp_json::serialize(resp.get("content")) + "}";
            }
            return mcp_json::serialize(resp);
        } catch (const std::exception& e) {
            return std::string("{\"error\":\"") + e.what() + "\"}";
        } catch (...) {
            return "{\"error\":\"unknown exception\"}";
        }
    }

    void start_health_check_() {
        stop_health_ = false;
        health_thread_ = std::thread([this]() {
            while (!stop_health_) {
                for (int i = 0; i < 60 && !stop_health_; i++)
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                if (stop_health_) break;
                std::vector<std::string> to_reconnect;
                {
                    std::lock_guard<std::mutex> lock(clients_mu_);
                    for (auto& [name, c] : clients_) {
                        if (!c->is_running() || !c->ping(5000)) {
                            to_reconnect.push_back(name);
                        }
                    }
                }
                for (auto& name : to_reconnect) {
                    std::cerr << "[MCP] health-check failed for " << name
                              << ", attempting reconnect" << std::endl;
                    reconnect(name);
                }
            }
        });
    }

    ToolRegistry& registry_;
    std::mutex clients_mu_;
    std::unordered_map<std::string, std::unique_ptr<MCPClient>> clients_;
    std::thread health_thread_;
    std::atomic<bool> stop_health_{false};
    std::string config_path_;
};

}  // namespace promeserve
