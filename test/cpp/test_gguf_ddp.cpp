// ============================================================================
// test_gguf_ddp.cpp — multi-process tensor-parallel GGUF inference self-test
// ============================================================================
// Fork-based POSIX test. Spawns `world_size` processes; each loads the SAME
// GGUF model, rank 0 additionally runs a 1-process reference decode, and
// ranks 0..N-1 run tensor-parallel decode. Rank 0 compares logits (cos-sim)
// and prints PASS/FAIL.
//
// Usage:
//   ./gguf_ddp_tests <model_path_or_ollama_name> [world_size=2]
//
// The test reads an env var GGUF_DDP_MODEL if argv not set, otherwise skips.
// On Windows, prints "skipped" (no fork).
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
int main() {
    std::printf("test_gguf_ddp: SKIPPED on Windows (no fork)\n");
    return 0;
}
#else

#include "torch/io/gguf_model.h"
#include "torch/distributed/ddp.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cmath>
#include <vector>

namespace td = torch::distributed;
namespace tio = torch::io;

// Simple cosine similarity between two float arrays.
static double cosine_sim(const float* a, const float* b, int64_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Max absolute difference
static float max_abs_diff(const float* a, const float* b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static bool is_path(const std::string& s) {
    return s.find('/')  != std::string::npos ||
           s.find('\\') != std::string::npos ||
           s.find(".gguf") != std::string::npos;
}

static void load_model(tio::GGUFModel& model, const std::string& name) {
    if (is_path(name)) {
        model.load(name);
    } else {
        model.load_ollama(name);
    }
}

// Run a single rank's TP decode and write logits to `out`.
// token_id is the single input token (no prefill, just a single decode step).
static int run_rank_tp(int rank, int world, int port, const std::string& model_name,
                       int32_t token_id, std::vector<float>& out_logits) {
    td::DDPConfig cfg;
    cfg.rank        = rank;
    cfg.world_size  = world;
    cfg.master_addr = "127.0.0.1";
    cfg.master_port = port;
    cfg.timeout_sec = 300;

    try {
        td::init_process_group(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] init_process_group failed: %s\n", rank, e.what());
        return 10;
    }

    try {
        tio::GGUFModel model;
        load_model(model, model_name);

        if (!model.init_tensor_parallel(rank, world)) {
            std::fprintf(stderr, "[rank %d] init_tensor_parallel failed\n", rank);
            td::destroy_process_group();
            return 11;
        }
        // Allocate KV cache
        model.tp_allocate_kv_cache(256);

        // Run a single decode step.
        at::Tensor logits = model.forward_decode_cpu_tp(static_cast<int64_t>(token_id));
        int64_t V = logits.numel();
        out_logits.assign(V, 0.0f);
        std::memcpy(out_logits.data(), logits.data_ptr<float>(), V * sizeof(float));

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] TP decode failed: %s\n", rank, e.what());
        td::destroy_process_group();
        return 12;
    }

    td::destroy_process_group();
    return 0;
}

// Run single-process reference decode (no DDP).
static int run_single_process_reference(const std::string& model_name,
                                        int32_t token_id,
                                        std::vector<float>& out_logits) {
    try {
        tio::GGUFModel model;
        load_model(model, model_name);

        // Allocate KV cache (forward_decode_cpu() reads kv_cache.key_cache[i]
        // without allocating itself — generate() normally handles this).
        int64_t kv_dim = model.config.num_kv_heads * model.config.head_dim;
        model.kv_cache.allocate(model.config.num_layers, 256, kv_dim, false);

        at::Tensor logits = model.forward_decode_cpu(static_cast<int64_t>(token_id));
        int64_t V = logits.numel();
        out_logits.assign(V, 0.0f);
        std::memcpy(out_logits.data(), logits.data_ptr<float>(), V * sizeof(float));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[reference] failed: %s\n", e.what());
        return 20;
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string model_name;
    int world_size = 2;
    if (argc >= 2) {
        model_name = argv[1];
    } else if (const char* env = std::getenv("GGUF_DDP_MODEL")) {
        model_name = env;
    } else {
        std::printf("test_gguf_ddp: SKIPPED (pass model path or set GGUF_DDP_MODEL)\n");
        return 0;
    }
    if (argc >= 3) {
        world_size = std::atoi(argv[2]);
    }
    int port = 29600 + (int)(getpid() % 200);

    // Pick a deterministic token id (a BOS-like id). 1 is safe for most models.
    int32_t token_id = 1;

    // 1) Reference: single-process decode.
    std::vector<float> ref_logits;
    int rc_ref = run_single_process_reference(model_name, token_id, ref_logits);
    if (rc_ref != 0) {
        std::fprintf(stderr, "test_gguf_ddp: reference decode failed rc=%d\n", rc_ref);
        return 1;
    }
    std::printf("[reference] logits[0..4]=%f %f %f %f %f  (V=%zu)\n",
                ref_logits[0], ref_logits[1], ref_logits[2], ref_logits[3], ref_logits[4],
                ref_logits.size());

    // 2) Spawn world_size-1 child ranks, run rank 0 in parent.
    std::vector<pid_t> children;
    for (int r = 1; r < world_size; ++r) {
        pid_t pid = fork();
        if (pid < 0) { std::perror("fork"); return 2; }
        if (pid == 0) {
            // Child: run TP rank r and exit.
            std::vector<float> dummy;
            int rc = run_rank_tp(r, world_size, port, model_name, token_id, dummy);
            std::exit(rc);
        } else {
            children.push_back(pid);
        }
    }

    // Rank 0 runs in the parent.
    std::vector<float> tp_logits;
    int rc0 = run_rank_tp(0, world_size, port, model_name, token_id, tp_logits);

    // Wait for children
    int rc_children = 0;
    for (pid_t pid : children) {
        int status = 0;
        waitpid(pid, &status, 0);
        int rc = WIFEXITED(status) ? WEXITSTATUS(status) : 99;
        if (rc != 0) {
            std::fprintf(stderr, "child pid %d exited rc=%d\n", (int)pid, rc);
            rc_children = rc;
        }
    }

    if (rc0 != 0 || rc_children != 0) {
        std::printf("test_gguf_ddp: FAIL (rank0=%d, children=%d)\n", rc0, rc_children);
        return 1;
    }

    // 3) Compare rank-0 TP logits vs single-process logits.
    if (tp_logits.size() != ref_logits.size()) {
        std::printf("test_gguf_ddp: FAIL (size mismatch %zu vs %zu)\n",
                    tp_logits.size(), ref_logits.size());
        return 1;
    }
    double cos = cosine_sim(tp_logits.data(), ref_logits.data(), (int64_t)tp_logits.size());
    float mad = max_abs_diff(tp_logits.data(), ref_logits.data(), (int64_t)tp_logits.size());

    std::printf("[rank 0 TP] logits[0..4]=%f %f %f %f %f\n",
                tp_logits[0], tp_logits[1], tp_logits[2], tp_logits[3], tp_logits[4]);
    std::printf("test_gguf_ddp: cos_sim=%.6f max_abs_diff=%g\n", cos, (double)mad);

    // Greedy argmax should match.
    int64_t V = (int64_t)tp_logits.size();
    int32_t ref_arg = 0, tp_arg = 0;
    float ref_best = ref_logits[0], tp_best = tp_logits[0];
    for (int64_t i = 1; i < V; ++i) {
        if (ref_logits[i] > ref_best) { ref_best = ref_logits[i]; ref_arg = (int32_t)i; }
        if (tp_logits[i]  > tp_best)  { tp_best  = tp_logits[i];  tp_arg  = (int32_t)i; }
    }
    std::printf("test_gguf_ddp: argmax ref=%d tp=%d\n", ref_arg, tp_arg);

    bool ok = (cos >= 0.999) && (ref_arg == tp_arg);
    std::printf("test_gguf_ddp: %s (cos_sim >= 0.999 and argmax match required)\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif  // !_WIN32
