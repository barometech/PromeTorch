#pragma once

// ============================================================================
// PromeServe — Minimal HTTP/1.1 Server
//
// Raw socket implementation, no external dependencies.
// Winsock2 on Windows, POSIX sockets on Linux.
// Thread-per-connection model suitable for LLM serving.
// ============================================================================

#include <string>
#include <map>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdint>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
#define INVALID_SOCK INVALID_SOCKET
#define CLOSE_SOCKET closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
using socket_t = int;
#define INVALID_SOCK (-1)
#define CLOSE_SOCKET ::close
#endif

namespace promeserve {

// ============================================================================
// HTTP Request / Response structures
// ============================================================================

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
    std::string query_string;

    std::string get_header(const std::string& name, const std::string& default_val = "") const {
        // Case-insensitive header lookup
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        for (auto& kv : headers) {
            std::string lower_key = kv.first;
            std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
            if (lower_key == lower_name) return kv.second;
        }
        return default_val;
    }

    int64_t content_length() const {
        std::string val = get_header("content-length", "0");
        return std::stoll(val);
    }
};

struct HttpResponse {
    int status = 200;
    std::string body;
    std::map<std::string, std::string> headers;
    bool streaming = false;

    void set_json(const std::string& json_body) {
        body = json_body;
        headers["Content-Type"] = "application/json";
    }

    void set_ndjson_streaming() {
        streaming = true;
        headers["Content-Type"] = "application/x-ndjson";
        headers["Transfer-Encoding"] = "chunked";
    }

    std::string status_text() const {
        switch (status) {
            case 200: return "OK";
            case 201: return "Created";
            case 204: return "No Content";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 500: return "Internal Server Error";
            default:  return "Unknown";
        }
    }

    std::string serialize() const {
        std::ostringstream oss;
        oss << "HTTP/1.1 " << status << " " << status_text() << "\r\n";
        for (auto& kv : headers) {
            oss << kv.first << ": " << kv.second << "\r\n";
        }
        if (!streaming && headers.find("Content-Length") == headers.end()) {
            oss << "Content-Length: " << body.size() << "\r\n";
        }
        oss << "\r\n";
        if (!streaming) {
            oss << body;
        }
        return oss.str();
    }
};

// ============================================================================
// Streaming writer — sends chunked transfer encoding to client
// ============================================================================

class StreamWriter {
public:
    explicit StreamWriter(socket_t sock) : sock_(sock), closed_(false) {}

    // Send a single NDJSON line (one JSON object + newline)
    bool write(const std::string& data) {
        if (closed_) return false;
        // Chunked transfer encoding: size in hex, CRLF, data, CRLF
        std::ostringstream chunk;
        chunk << std::hex << data.size() << "\r\n" << data << "\r\n";
        std::string payload = chunk.str();
        return send_raw(payload);
    }

    // Send the terminating zero-length chunk
    bool finish() {
        if (closed_) return false;
        closed_ = true;
        return send_raw("0\r\n\r\n");
    }

    bool is_closed() const { return closed_; }

private:
    bool send_raw(const std::string& data) {
        const char* ptr = data.c_str();
        int remaining = static_cast<int>(data.size());
        while (remaining > 0) {
            int sent = ::send(sock_, ptr, remaining, 0);
            if (sent <= 0) {
                closed_ = true;
                return false;
            }
            ptr += sent;
            remaining -= sent;
        }
        return true;
    }

    socket_t sock_;
    bool closed_;
};

// ============================================================================
// Handler types
// ============================================================================

// Standard handler: receives request, returns response
using Handler = std::function<HttpResponse(const HttpRequest&)>;

// Streaming handler: receives request + stream writer, returns response headers
// The handler writes chunks via StreamWriter, then the server finishes the stream
using StreamHandler = std::function<HttpResponse(const HttpRequest&, StreamWriter&)>;

struct Route {
    std::string method;
    std::string path;
    Handler handler;
    StreamHandler stream_handler;
    bool is_streaming = false;
};

// ============================================================================
// HTTP Server
// ============================================================================

class HttpServer {
public:
    HttpServer() : listen_sock_(INVALID_SOCK), running_(false), port_(0) {}

    ~HttpServer() {
        stop();
    }

    // Register a standard (non-streaming) route
    void route(const std::string& method, const std::string& path, Handler handler) {
        Route r;
        r.method = method;
        r.path = path;
        r.handler = handler;
        r.is_streaming = false;
        routes_.push_back(std::move(r));
    }

    // Register a streaming route (for generate/chat endpoints)
    void route_stream(const std::string& method, const std::string& path, StreamHandler handler) {
        Route r;
        r.method = method;
        r.path = path;
        r.stream_handler = handler;
        r.is_streaming = true;
        routes_.push_back(std::move(r));
    }

    // Start listening on the given port (blocking)
    void start(int port) {
        port_ = port;

#ifdef _WIN32
        WSADATA wsa_data;
        int wsa_result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
        if (wsa_result != 0) {
            std::cerr << "[PromeServe] WSAStartup failed: " << wsa_result << std::endl;
            return;
        }
#else
        // Ignore SIGPIPE so we don't crash on broken connections
        signal(SIGPIPE, SIG_IGN);
#endif

        listen_sock_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listen_sock_ == INVALID_SOCK) {
            std::cerr << "[PromeServe] Failed to create socket" << std::endl;
            return;
        }

        // Allow port reuse
        int opt = 1;
#ifdef _WIN32
        setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
        setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port));

        if (::bind(listen_sock_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
            std::cerr << "[PromeServe] Failed to bind to port " << port << std::endl;
            CLOSE_SOCKET(listen_sock_);
            listen_sock_ = INVALID_SOCK;
            return;
        }

        if (::listen(listen_sock_, 16) != 0) {
            std::cerr << "[PromeServe] Failed to listen" << std::endl;
            CLOSE_SOCKET(listen_sock_);
            listen_sock_ = INVALID_SOCK;
            return;
        }

        running_ = true;
        std::cout << "[PromeServe] Listening on http://0.0.0.0:" << port << std::endl;

        accept_loop();
    }

    void stop() {
        running_ = false;
        if (listen_sock_ != INVALID_SOCK) {
            CLOSE_SOCKET(listen_sock_);
            listen_sock_ = INVALID_SOCK;
        }

        // Wait for worker threads to finish
        std::lock_guard<std::mutex> lock(threads_mutex_);
        for (auto& t : threads_) {
            if (t.joinable()) {
                t.detach();  // Don't block on long-running connections
            }
        }
        threads_.clear();

#ifdef _WIN32
        WSACleanup();
#endif
    }

    bool is_running() const { return running_; }
    int port() const { return port_; }

private:
    void accept_loop() {
        while (running_) {
            struct sockaddr_in client_addr;
#ifdef _WIN32
            int addr_len = sizeof(client_addr);
#else
            socklen_t addr_len = sizeof(client_addr);
#endif
            socket_t client = ::accept(listen_sock_,
                                       reinterpret_cast<struct sockaddr*>(&client_addr),
                                       &addr_len);
            if (client == INVALID_SOCK) {
                if (running_) {
                    // Transient error, continue
                    continue;
                }
                break;  // Server shutting down
            }

            // Spawn a thread for this connection
            std::lock_guard<std::mutex> lock(threads_mutex_);
            threads_.emplace_back([this, client]() {
                handle_connection(client);
            });
            threads_.back().detach();
        }
    }

    void handle_connection(socket_t client) {
        // Receive full request
        HttpRequest request;
        if (!parse_request(client, request)) {
            CLOSE_SOCKET(client);
            return;
        }

        // Match route
        const Route* matched = nullptr;
        for (auto& r : routes_) {
            if (r.method == request.method && r.path == request.path) {
                matched = &r;
                break;
            }
        }

        // Handle OPTIONS (CORS preflight)
        if (request.method == "OPTIONS") {
            HttpResponse resp;
            resp.status = 204;
            add_cors_headers(resp);
            resp.headers["Content-Length"] = "0";
            std::string raw = resp.serialize();
            ::send(client, raw.c_str(), static_cast<int>(raw.size()), 0);
            CLOSE_SOCKET(client);
            return;
        }

        if (!matched) {
            // HEAD on existing GET routes
            if (request.method == "HEAD") {
                for (auto& r : routes_) {
                    if (r.method == "GET" && r.path == request.path) {
                        matched = &r;
                        break;
                    }
                }
            }

            if (!matched) {
                HttpResponse resp;
                resp.status = 404;
                resp.set_json("{\"error\":\"not found\"}");
                add_cors_headers(resp);
                std::string raw = resp.serialize();
                ::send(client, raw.c_str(), static_cast<int>(raw.size()), 0);
                CLOSE_SOCKET(client);
                return;
            }
        }

        if (matched->is_streaming) {
            // Streaming response
            StreamWriter writer(client);
            HttpResponse resp = matched->stream_handler(request, writer);
            add_cors_headers(resp);
            // Send headers first
            std::string header_raw = resp.serialize();
            ::send(client, header_raw.c_str(), static_cast<int>(header_raw.size()), 0);
            // Handler has already written chunks via writer
            // (in practice, the handler calls writer.write() and writer.finish())
            // If handler didn't finish, do it now
            if (!writer.is_closed()) {
                writer.finish();
            }
        } else {
            HttpResponse resp;
            if (request.method == "HEAD" && matched->handler) {
                // For HEAD, run handler but don't send body
                resp = matched->handler(request);
                resp.body.clear();
            } else {
                resp = matched->handler(request);
            }
            add_cors_headers(resp);
            std::string raw = resp.serialize();
            ::send(client, raw.c_str(), static_cast<int>(raw.size()), 0);
        }

        CLOSE_SOCKET(client);
    }

    // ========================================================================
    // HTTP Request Parser
    // ========================================================================

    bool parse_request(socket_t client, HttpRequest& req) {
        // Read data in chunks until we have the full headers
        std::string buffer;
        char chunk[4096];
        bool headers_complete = false;
        size_t header_end = std::string::npos;

        // Set a receive timeout (30 seconds)
#ifdef _WIN32
        DWORD timeout = 30000;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
        struct timeval tv;
        tv.tv_sec = 30;
        tv.tv_usec = 0;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

        while (!headers_complete) {
            int n = ::recv(client, chunk, sizeof(chunk), 0);
            if (n <= 0) return false;
            buffer.append(chunk, n);
            header_end = buffer.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                headers_complete = true;
            }
            if (buffer.size() > 1024 * 1024) {
                // Protect against absurdly large headers
                return false;
            }
        }

        // Parse request line
        size_t first_line_end = buffer.find("\r\n");
        if (first_line_end == std::string::npos) return false;
        std::string request_line = buffer.substr(0, first_line_end);

        // METHOD /path HTTP/1.1
        size_t sp1 = request_line.find(' ');
        size_t sp2 = request_line.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) return false;

        req.method = request_line.substr(0, sp1);
        std::string full_path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

        // Split path and query string
        size_t qmark = full_path.find('?');
        if (qmark != std::string::npos) {
            req.path = full_path.substr(0, qmark);
            req.query_string = full_path.substr(qmark + 1);
        } else {
            req.path = full_path;
        }

        // Parse headers
        size_t pos = first_line_end + 2;
        while (pos < header_end) {
            size_t line_end = buffer.find("\r\n", pos);
            if (line_end == std::string::npos || line_end >= header_end) break;
            std::string line = buffer.substr(pos, line_end - pos);
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string key = line.substr(0, colon);
                std::string value = line.substr(colon + 1);
                // Trim leading whitespace from value
                size_t val_start = value.find_first_not_of(" \t");
                if (val_start != std::string::npos) {
                    value = value.substr(val_start);
                }
                req.headers[key] = value;
            }
            pos = line_end + 2;
        }

        // Read body if Content-Length present
        int64_t content_len = req.content_length();
        if (content_len > 0) {
            size_t body_start = header_end + 4;
            size_t already_read = buffer.size() - body_start;
            req.body = buffer.substr(body_start);

            // Read remaining body data
            while (static_cast<int64_t>(req.body.size()) < content_len) {
                int n = ::recv(client, chunk, sizeof(chunk), 0);
                if (n <= 0) break;
                req.body.append(chunk, n);
            }

            // Trim body to exact content length
            if (static_cast<int64_t>(req.body.size()) > content_len) {
                req.body.resize(static_cast<size_t>(content_len));
            }
        }

        return true;
    }

    void add_cors_headers(HttpResponse& resp) {
        resp.headers["Access-Control-Allow-Origin"] = "*";
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE";
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
        resp.headers["Access-Control-Max-Age"] = "86400";
    }

    // Members
    socket_t listen_sock_;
    std::atomic<bool> running_;
    int port_;
    std::vector<Route> routes_;
    std::vector<std::thread> threads_;
    std::mutex threads_mutex_;
};

}  // namespace promeserve
