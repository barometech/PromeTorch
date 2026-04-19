#pragma once
// ============================================================================
// launcher.h — Multi-process launcher for distributed training
// ============================================================================
// POSIX fork()-based launcher (PyTorch torch.distributed.launch convention).
// Each child gets MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE env vars and
// runs `worker_fn(rank, world_size)`. Parent waits for all children, returns
// the maximum exit code observed.
//
// Build: header-only. POSIX path requires <unistd.h>, <sys/wait.h>.
// On Windows the launch() function returns -1 (not supported); parse_dist_args
// is portable.
//
// Self-test (compiled into examples/distributed/test_launcher.cpp):
//   launches 4 procs that each print "hello from rank X of 4" and exit 0.
// ============================================================================

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/types.h>
  #include <sys/wait.h>
  #include <unistd.h>
  #define PT_LAUNCHER_POSIX 1
#else
  #define PT_LAUNCHER_POSIX 0
#endif

namespace torch {
namespace distributed {

// ----------------------------------------------------------------------------
// Distributed CLI args
// ----------------------------------------------------------------------------
struct DistArgs {
    int         rank        = 0;
    int         world_size  = 1;
    std::string master_addr = "127.0.0.1";
    int         master_port = 29500;
};

// Parse `--rank N --world_size N --master_addr X --master_port N` from argv.
// Unknown flags are ignored (caller may parse the rest). Also accepts the
// PyTorch-style alternates `--nprocs` and `--nnodes` for world_size.
inline DistArgs parse_dist_args(int argc, char** argv) {
    DistArgs out;
    auto next_str = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("parse_dist_args: missing value for ") + argv[i]);
        }
        return argv[++i];
    };
    auto next_int = [&](int& i) -> int {
        const char* s = next_str(i);
        return std::atoi(s);
    };

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--rank") == 0)              out.rank        = next_int(i);
        else if (std::strcmp(a, "--world_size") == 0
              || std::strcmp(a, "--world-size") == 0
              || std::strcmp(a, "--nprocs") == 0
              || std::strcmp(a, "--nnodes") == 0)       out.world_size  = next_int(i);
        else if (std::strcmp(a, "--master_addr") == 0
              || std::strcmp(a, "--master-addr") == 0)  out.master_addr = next_str(i);
        else if (std::strcmp(a, "--master_port") == 0
              || std::strcmp(a, "--master-port") == 0)  out.master_port = next_int(i);
    }
    return out;
}

// ----------------------------------------------------------------------------
// Internal: setenv wrapper (POSIX has setenv; we have a portable fallback).
// ----------------------------------------------------------------------------
namespace detail {
inline void set_env(const char* name, const std::string& value) {
#if PT_LAUNCHER_POSIX
    setenv(name, value.c_str(), /*overwrite=*/1);
#else
    std::string assign = std::string(name) + "=" + value;
    // Windows _putenv copies the string.
    _putenv(assign.c_str());
#endif
}
}  // namespace detail

// ----------------------------------------------------------------------------
// launch — fork `world_size` children, each runs worker_fn(rank, world_size).
// Parent waits for all children to exit and returns max(exit_code) seen.
// Children that crash via signal contribute exit code 128+signo.
// ----------------------------------------------------------------------------
inline int launch(int world_size,
                  std::function<int(int rank, int world_size)> worker_fn,
                  const std::string& master_addr = "127.0.0.1",
                  int master_port = 29500) {
    if (world_size <= 0) {
        throw std::invalid_argument("launch: world_size must be > 0");
    }
    if (!worker_fn) {
        throw std::invalid_argument("launch: worker_fn must be non-null");
    }

#if !PT_LAUNCHER_POSIX
    (void)worker_fn; (void)master_addr; (void)master_port;
    std::fprintf(stderr,
                 "torch::distributed::launch: fork() not supported on this platform\n");
    return -1;
#else
    // Set shared env vars (children inherit).
    detail::set_env("MASTER_ADDR", master_addr);
    detail::set_env("MASTER_PORT", std::to_string(master_port));
    detail::set_env("WORLD_SIZE", std::to_string(world_size));

    std::vector<pid_t> pids;
    pids.reserve(static_cast<size_t>(world_size));

    for (int rank = 0; rank < world_size; ++rank) {
        pid_t pid = fork();
        if (pid < 0) {
            // Fork failed: kill any children we already spawned, then throw.
            int saved_errno = errno;
            for (pid_t p : pids) kill(p, SIGTERM);
            for (pid_t p : pids) {
                int st = 0; waitpid(p, &st, 0);
            }
            throw std::runtime_error(std::string("launch: fork() failed: ")
                                     + std::strerror(saved_errno));
        }
        if (pid == 0) {
            // ---- CHILD ----
            detail::set_env("RANK", std::to_string(rank));
            int rc = 1;
            try {
                rc = worker_fn(rank, world_size);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[rank %d] uncaught exception: %s\n", rank, e.what());
                rc = 1;
            } catch (...) {
                std::fprintf(stderr, "[rank %d] uncaught non-std exception\n", rank);
                rc = 1;
            }
            // _exit avoids running parent atexit handlers / global dtors twice.
            _exit(rc);
        }
        // ---- PARENT ----
        pids.push_back(pid);
    }

    // Wait for all children. Track the maximum exit code (worst failure).
    int max_rc = 0;
    for (pid_t p : pids) {
        int status = 0;
        pid_t r = waitpid(p, &status, 0);
        if (r < 0) {
            std::fprintf(stderr, "launch: waitpid(%d) failed: %s\n",
                         (int)p, std::strerror(errno));
            if (max_rc < 1) max_rc = 1;
            continue;
        }
        int code = 0;
        if (WIFEXITED(status))        code = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) code = 128 + WTERMSIG(status);
        else                          code = 1;
        if (code > max_rc) max_rc = code;
    }
    return max_rc;
#endif
}

}  // namespace distributed
}  // namespace torch
