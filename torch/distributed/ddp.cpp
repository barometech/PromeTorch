// ============================================================================
// ddp.cpp — POSIX TCP implementation of DDP collectives (CPU/Elbrus-friendly)
// ============================================================================
#include "torch/distributed/ddp.h"

#include "aten/src/ATen/core/Tensor.h"
#include "torch/csrc/autograd/autograd_meta.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
  // Stubs so the file at least parses on Windows. Real DDP runs on POSIX.
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "Ws2_32.lib")
  using ssize_t = long long;
  static int close_sock(int fd) { return ::closesocket((SOCKET)fd); }
  static int set_reuseaddr(int fd) {
      char yes = 1;
      return ::setsockopt((SOCKET)fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  }
#else
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/types.h>
  #include <unistd.h>
  static int close_sock(int fd) { return ::close(fd); }
  static int set_reuseaddr(int fd) {
      int yes = 1;
      return ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  }
#endif

namespace torch {
namespace distributed {

// ============================================================================
// Singleton state
// ============================================================================
namespace {

struct PGState {
    std::mutex             mu;
    bool                   initialized = false;
    int                    rank        = 0;
    int                    world_size  = 1;
    int                    timeout_sec = 300;
    // On rank 0: peers_[r] = socket to rank r (peers_[0] unused).
    // On rank r>0: peers_[0] = socket to rank 0; other slots unused.
    std::vector<int>       peers;
    std::vector<char>      io_buf;        // scratch for send/recv
};

PGState& pg() {
    static PGState s;
    return s;
}

// ----------------------------------------------------------------------------
// Endian helpers (8-byte size prefix is big-endian on the wire)
// ----------------------------------------------------------------------------
void pack_u64_be(uint64_t v, unsigned char out[8]) {
    for (int i = 0; i < 8; ++i) out[7 - i] = (unsigned char)((v >> (i * 8)) & 0xFF);
}
uint64_t unpack_u64_be(const unsigned char in[8]) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | (uint64_t)in[i];
    return v;
}

// ----------------------------------------------------------------------------
// Robust send/recv (loop until N bytes or error)
// ----------------------------------------------------------------------------
bool send_all(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    size_t left = n;
    while (left > 0) {
#if defined(_WIN32)
        int chunk = (int)(left > (1u << 20) ? (1u << 20) : left);
        int got = ::send((SOCKET)fd, p, chunk, 0);
#else
        ssize_t got = ::send(fd, p, left, MSG_NOSIGNAL);
#endif
        if (got <= 0) return false;
        p    += got;
        left -= (size_t)got;
    }
    return true;
}

bool recv_all(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    size_t left = n;
    while (left > 0) {
#if defined(_WIN32)
        int chunk = (int)(left > (1u << 20) ? (1u << 20) : left);
        int got = ::recv((SOCKET)fd, p, chunk, 0);
#else
        ssize_t got = ::recv(fd, p, left, 0);
#endif
        if (got <= 0) return false;
        p    += got;
        left -= (size_t)got;
    }
    return true;
}

bool send_msg(int fd, const void* buf, size_t n) {
    unsigned char hdr[8];
    pack_u64_be((uint64_t)n, hdr);
    if (!send_all(fd, hdr, 8)) return false;
    if (n == 0) return true;
    return send_all(fd, buf, n);
}

bool recv_msg(int fd, void* buf, size_t expect_n) {
    unsigned char hdr[8];
    if (!recv_all(fd, hdr, 8)) return false;
    uint64_t got = unpack_u64_be(hdr);
    if (got != expect_n) return false;
    if (expect_n == 0) return true;
    return recv_all(fd, buf, expect_n);
}

void set_socket_timeout(int fd, int seconds) {
#if defined(_WIN32)
    DWORD ms = (DWORD)seconds * 1000;
    ::setsockopt((SOCKET)fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof(ms));
    ::setsockopt((SOCKET)fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof(ms));
#else
    struct timeval tv;
    tv.tv_sec  = seconds;
    tv.tv_usec = 0;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

void enable_tcp_nodelay(int fd) {
#if defined(TCP_NODELAY)
    int yes = 1;
#if defined(_WIN32)
    ::setsockopt((SOCKET)fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&yes, sizeof(yes));
#else
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
#endif
#endif
}

#if defined(_WIN32)
struct WinsockInit {
    WinsockInit() { WSADATA d; ::WSAStartup(MAKEWORD(2, 2), &d); }
    ~WinsockInit(){ ::WSACleanup(); }
};
WinsockInit g_wsa_init;
#endif

}  // anonymous namespace

// ============================================================================
// init_process_group / destroy_process_group
// ============================================================================
void init_process_group(const DDPConfig& cfg) {
    auto& s = pg();
    std::lock_guard<std::mutex> lk(s.mu);

    if (s.initialized) {
        throw std::runtime_error("init_process_group: already initialized");
    }
    if (cfg.world_size < 1 || cfg.rank < 0 || cfg.rank >= cfg.world_size) {
        throw std::runtime_error("init_process_group: bad rank/world_size");
    }

    s.rank        = cfg.rank;
    s.world_size  = cfg.world_size;
    s.timeout_sec = cfg.timeout_sec;
    s.peers.assign(cfg.world_size, -1);

    if (cfg.world_size == 1) {
        s.initialized = true;
        return;
    }

    if (cfg.rank == 0) {
        // ---- HUB: listen, accept N-1 workers ----
        int listen_fd = (int)::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) throw std::runtime_error("DDP: socket() failed");
        set_reuseaddr(listen_fd);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons((uint16_t)cfg.master_port);
        if (::inet_pton(AF_INET, cfg.master_addr.c_str(), &addr.sin_addr) <= 0) {
            // Fall back to INADDR_ANY for robustness on Elbrus.
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
        }
        if (::bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
            close_sock(listen_fd);
            throw std::runtime_error("DDP: bind() failed on " + cfg.master_addr +
                                     ":" + std::to_string(cfg.master_port));
        }
        if (::listen(listen_fd, cfg.world_size) < 0) {
            close_sock(listen_fd);
            throw std::runtime_error("DDP: listen() failed");
        }

        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(cfg.timeout_sec);

        for (int accepted = 0; accepted < cfg.world_size - 1; ++accepted) {
            sockaddr_in peer{};
            socklen_t   plen = sizeof(peer);
            int fd = (int)::accept(listen_fd, (sockaddr*)&peer, &plen);
            if (fd < 0) {
                close_sock(listen_fd);
                throw std::runtime_error("DDP: accept() failed");
            }
            // Worker first sends its rank as 4 bytes big-endian.
            unsigned char rb[4];
            if (!recv_all(fd, rb, 4)) {
                close_sock(fd); close_sock(listen_fd);
                throw std::runtime_error("DDP: failed to read worker rank");
            }
            int worker_rank = (rb[0]<<24) | (rb[1]<<16) | (rb[2]<<8) | rb[3];
            if (worker_rank <= 0 || worker_rank >= cfg.world_size) {
                close_sock(fd); close_sock(listen_fd);
                throw std::runtime_error("DDP: bogus worker rank " +
                                         std::to_string(worker_rank));
            }
            set_socket_timeout(fd, cfg.timeout_sec);
            enable_tcp_nodelay(fd);
            s.peers[worker_rank] = fd;
            if (std::chrono::steady_clock::now() > deadline) {
                close_sock(listen_fd);
                throw std::runtime_error("DDP: rank-0 accept timeout");
            }
        }
        close_sock(listen_fd);

        // Send "READY" ack to every worker.
        unsigned char ack = 1;
        for (int r = 1; r < cfg.world_size; ++r) {
            if (!send_all(s.peers[r], &ack, 1)) {
                throw std::runtime_error("DDP: failed to send ack to rank " +
                                         std::to_string(r));
            }
        }
    } else {
        // ---- WORKER: connect to rank 0 ----
        int fd = (int)::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("DDP: socket() failed");

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons((uint16_t)cfg.master_port);
        if (::inet_pton(AF_INET, cfg.master_addr.c_str(), &addr.sin_addr) <= 0) {
            close_sock(fd);
            throw std::runtime_error("DDP: bad master_addr " + cfg.master_addr);
        }

        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(cfg.timeout_sec);
        bool connected = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) == 0) {
                connected = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!connected) {
            close_sock(fd);
            throw std::runtime_error("DDP: connect timeout to " + cfg.master_addr +
                                     ":" + std::to_string(cfg.master_port));
        }
        set_socket_timeout(fd, cfg.timeout_sec);
        enable_tcp_nodelay(fd);

        // Send our rank.
        unsigned char rb[4];
        rb[0] = (unsigned char)((cfg.rank >> 24) & 0xFF);
        rb[1] = (unsigned char)((cfg.rank >> 16) & 0xFF);
        rb[2] = (unsigned char)((cfg.rank >>  8) & 0xFF);
        rb[3] = (unsigned char)( cfg.rank        & 0xFF);
        if (!send_all(fd, rb, 4)) {
            close_sock(fd);
            throw std::runtime_error("DDP: failed to send rank");
        }
        // Wait for ack from rank 0.
        unsigned char ack = 0;
        if (!recv_all(fd, &ack, 1) || ack != 1) {
            close_sock(fd);
            throw std::runtime_error("DDP: no ack from rank 0");
        }
        s.peers[0] = fd;
    }

    s.initialized = true;
}

void destroy_process_group() {
    auto& s = pg();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.initialized) return;
    for (int fd : s.peers) {
        if (fd >= 0) close_sock(fd);
    }
    s.peers.clear();
    s.io_buf.clear();
    s.initialized = false;
    s.rank = 0;
    s.world_size = 1;
}

bool is_initialized() { return pg().initialized; }
int  get_rank()       { return pg().rank; }
int  get_world_size() { return pg().world_size; }

// ============================================================================
// Collectives
// ============================================================================
void all_reduce(at::Tensor& tensor) {
    auto& s = pg();
    if (!s.initialized) {
        throw std::runtime_error("all_reduce: process group not initialized");
    }
    if (s.world_size == 1) return;
    if (!tensor.defined() || tensor.numel() == 0) return;

    int64_t numel = tensor.numel();
    size_t  nbytes = (size_t)numel * sizeof(float);
    float*  data   = tensor.mutable_data_ptr<float>();

    if (s.rank == 0) {
        // 1) recv from each worker, 2) sum into local, 3) broadcast result.
        if (s.io_buf.size() < nbytes) s.io_buf.resize(nbytes);
        float* tmp = reinterpret_cast<float*>(s.io_buf.data());

        for (int r = 1; r < s.world_size; ++r) {
            if (!recv_msg(s.peers[r], tmp, nbytes)) {
                throw std::runtime_error("all_reduce: recv from rank " +
                                         std::to_string(r) + " failed");
            }
            for (int64_t i = 0; i < numel; ++i) data[i] += tmp[i];
        }
        for (int r = 1; r < s.world_size; ++r) {
            if (!send_msg(s.peers[r], data, nbytes)) {
                throw std::runtime_error("all_reduce: send to rank " +
                                         std::to_string(r) + " failed");
            }
        }
    } else {
        if (!send_msg(s.peers[0], data, nbytes)) {
            throw std::runtime_error("all_reduce: send to rank 0 failed");
        }
        if (!recv_msg(s.peers[0], data, nbytes)) {
            throw std::runtime_error("all_reduce: recv from rank 0 failed");
        }
    }
}

void broadcast(at::Tensor& tensor, int src_rank) {
    auto& s = pg();
    if (!s.initialized) {
        throw std::runtime_error("broadcast: process group not initialized");
    }
    if (s.world_size == 1) return;
    if (!tensor.defined() || tensor.numel() == 0) return;
    if (src_rank < 0 || src_rank >= s.world_size) {
        throw std::runtime_error("broadcast: bad src_rank");
    }

    int64_t numel = tensor.numel();
    size_t  nbytes = (size_t)numel * sizeof(float);
    float*  data   = tensor.mutable_data_ptr<float>();

    if (src_rank == 0) {
        if (s.rank == 0) {
            for (int r = 1; r < s.world_size; ++r) {
                if (!send_msg(s.peers[r], data, nbytes))
                    throw std::runtime_error("broadcast: send to " + std::to_string(r));
            }
        } else {
            if (!recv_msg(s.peers[0], data, nbytes))
                throw std::runtime_error("broadcast: recv from rank 0");
        }
    } else {
        // Star topology: route through rank 0.
        if (s.rank == src_rank) {
            if (!send_msg(s.peers[0], data, nbytes))
                throw std::runtime_error("broadcast: src->0 failed");
        } else if (s.rank == 0) {
            if (s.io_buf.size() < nbytes) s.io_buf.resize(nbytes);
            if (!recv_msg(s.peers[src_rank], s.io_buf.data(), nbytes))
                throw std::runtime_error("broadcast: 0<-src failed");
            std::memcpy(data, s.io_buf.data(), nbytes);
            for (int r = 1; r < s.world_size; ++r) {
                if (r == src_rank) continue;
                if (!send_msg(s.peers[r], data, nbytes))
                    throw std::runtime_error("broadcast: 0->r failed");
            }
        } else {
            if (!recv_msg(s.peers[0], data, nbytes))
                throw std::runtime_error("broadcast: r<-0 failed");
        }
    }
}

void barrier() {
    auto& s = pg();
    if (!s.initialized) {
        throw std::runtime_error("barrier: process group not initialized");
    }
    if (s.world_size == 1) return;

    // Single-byte ping/pong: workers send 0xAA, rank 0 collects all then replies.
    unsigned char tok = 0xAA;
    if (s.rank == 0) {
        unsigned char rx;
        for (int r = 1; r < s.world_size; ++r) {
            if (!recv_all(s.peers[r], &rx, 1))
                throw std::runtime_error("barrier: recv from " + std::to_string(r));
        }
        for (int r = 1; r < s.world_size; ++r) {
            if (!send_all(s.peers[r], &tok, 1))
                throw std::runtime_error("barrier: send to " + std::to_string(r));
        }
    } else {
        if (!send_all(s.peers[0], &tok, 1))
            throw std::runtime_error("barrier: worker send");
        unsigned char rx;
        if (!recv_all(s.peers[0], &rx, 1))
            throw std::runtime_error("barrier: worker recv");
    }
}

// ============================================================================
// DistributedDataParallel wrapper
// ============================================================================
DistributedDataParallel::DistributedDataParallel(std::shared_ptr<nn::Module> module,
                                                 const DDPConfig& cfg,
                                                 bool broadcast_init_params)
    : nn::Module("DistributedDataParallel"),
      module_(std::move(module)),
      cfg_(cfg)
{
    if (!module_) throw std::runtime_error("DDP: module is null");
    register_module("module", module_);

    if (!is_initialized()) {
        throw std::runtime_error(
            "DDP: call torch::distributed::init_process_group(cfg) before "
            "constructing DistributedDataParallel");
    }

    if (broadcast_init_params && cfg_.world_size > 1) {
        // Identical starting weights: broadcast all parameters from rank 0.
        auto params = module_->parameters(/*recurse=*/true);
        for (auto* p : params) {
            if (!p) continue;
            at::Tensor& t = p->data();
            if (!t.defined() || t.numel() == 0) continue;
            broadcast(t, /*src_rank=*/0);
        }
        auto bufs = module_->buffers(/*recurse=*/true);
        for (auto* b : bufs) {
            if (!b) continue;
            at::Tensor& t = b->data();
            if (!t.defined() || t.numel() == 0) continue;
            broadcast(t, /*src_rank=*/0);
        }
    }
}

at::Tensor DistributedDataParallel::forward(const at::Tensor& input) {
    return module_->forward(input);
}
at::Tensor DistributedDataParallel::forward(const at::Tensor& a, const at::Tensor& b) {
    return module_->forward(a, b);
}
at::Tensor DistributedDataParallel::forward(const std::vector<at::Tensor>& inputs) {
    return module_->forward(inputs);
}

void DistributedDataParallel::allreduce_grads() {
    if (cfg_.world_size <= 1) return;
    // no_sync(): user is doing gradient accumulation across micro-batches.
    // Skip the AllReduce; .grad accumulates locally for the next forward.
    if (!require_grad_sync_) return;

    const float inv = 1.0f / (float)cfg_.world_size;
    auto params = module_->parameters(/*recurse=*/true);
    for (auto* p : params) {
        if (!p) continue;
        at::Tensor& w = p->data();
        if (!w.defined() || !w.requires_grad()) continue;
        auto* meta = w.autograd_meta();
        if (!meta || !meta->grad_) continue;

        at::Tensor g(meta->grad_);
        if (!g.defined() || g.numel() == 0) continue;

        all_reduce(g);

        float*  gd = g.mutable_data_ptr<float>();
        int64_t n  = g.numel();
        for (int64_t i = 0; i < n; ++i) gd[i] *= inv;
    }
}

}  // namespace distributed
}  // namespace torch
