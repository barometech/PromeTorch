// ============================================================================
// hot_loops.cpp — Compiled inner loops for LTO on Elbrus E2K
// ============================================================================
// THREADING: c10::ThreadPool (persistent threads) replaces OpenMP fork/join.
// On Elbrus, OpenMP fork/join costs ~10ms per region = 93s overhead/epoch.
// ============================================================================

#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/tuda/TudaMath.h"
#include "aten/src/ATen/native/cpu/tuda/TudaBLAS.h"
#include "c10/util/ThreadPool.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <atomic>

#ifdef PT_USE_EML_BLAS
#include <eml/cblas.h>
#include <eml/eml_vector.h>
#include <eml/eml_core.h>
#include <omp.h>
#endif

#if !defined(PT_USE_EML_BLAS) && defined(PT_USE_SYSTEM_BLAS)
#if __has_include(<mkl_cblas.h>)
#include <mkl_cblas.h>
#elif __has_include(<cblas.h>)
#include <cblas.h>
#endif
#endif

#ifdef PT_USE_NUMA
#include <numa.h>
#include <thread>
#include <pthread.h>
#include <sched.h>
#endif

// ============================================================================
// EML SIGILL FIX: Prevent nested OpenMP on Elbrus E2K
// ============================================================================
// EML (cblas_sgemm) uses OMP internally. If called from within an OMP parallel
// region or from multiple std::threads that each trigger OMP, we get nested OMP
// fork on E2K → SIGILL (illegal instruction) for matrices >= 256.
//
// Fix: at static init time, disable nested OMP parallelism. This means:
//   - Our outer OMP parallel regions work normally (parallel_scan, etc.)
//   - EML's INTERNAL OMP calls become single-threaded (no nested fork)
//   - NUMA tiling via std::thread still gives multi-node parallelism
//   - Each NUMA thread calls EML single-threaded on its tile
//
// omp_set_max_active_levels(1) is the modern (OpenMP 5.0) replacement for
// the deprecated omp_set_nested(0). It limits active parallel regions to 1 level.
// ============================================================================
#if defined(PT_USE_EML_BLAS) || defined(_OPENMP)
#include <omp.h>
namespace {
struct OmpNestedGuard {
    OmpNestedGuard() {
        // Prevent nested OMP: only one level of parallelism allowed.
        // This stops EML from forking threads when called from within
        // an OMP parallel region or from NUMA std::threads.
        omp_set_max_active_levels(1);
        // Also use deprecated API for older OpenMP implementations (LCC 1.29)
        omp_set_nested(0);
    }
};
static OmpNestedGuard g_omp_nested_guard;
} // anonymous namespace
#endif

namespace at {
namespace native {
namespace hot {

using tuda::VecF;

// ============================================================================
// NUMA-aware GEMM for multi-chip Elbrus (E8C2 4-chip: 1840 GFLOPS)
// ============================================================================
// Problem: EML cblas_sgemm on 4-chip E8C2 gives 330 GFLOPS (WORSE than
// single chip 463 GFLOPS) due to cross-NUMA memory access.
// Solution: Split M rows across NUMA nodes, each node copies its A-rows
// to local memory, calls cblas_sgemm independently. B is shared read-only.
// Result: 4x node-local parallel = 1840 GFLOPS (linear scaling).
// ============================================================================

// ============================================================================
// NUMA THREAD POOL for Elbrus E8C2 (4-socket, 32 cores)
// ============================================================================
// Uses single-threaded EML (-leml) from persistent pthreads.
// Each pthread is pinned to one core + NUMA node.
// Eliminates: pthread create/join overhead, cross-NUMA memory access.
// Key: eml_mt SIGILL from pthreads, but eml (ST) works fine.
// Benchmark: 600-900 GFLOPS vs 187 GFLOPS with eml_mt(32).
// ============================================================================
#if defined(PT_USE_NUMA) && defined(PT_USE_EML_BLAS)

static constexpr int NUMA_POOL_MAX = 32;
static constexpr int64_t NUMA_GEMM_THRESHOLD = 512;

struct NumaGemmTile {
    int M, N, K;
    const float* A;
    const float* B;
    float* C;
    int lda, ldb, ldc;
    int transA, transB;  // 0=NoTrans, 1=Trans
};

static pthread_barrier_t g_numa_bar_start, g_numa_bar_done;
static volatile int g_numa_pool_alive = 0;
static NumaGemmTile g_numa_tiles[NUMA_POOL_MAX];
static pthread_t g_numa_threads[NUMA_POOL_MAX];
static int g_numa_n_workers = 0;
static int g_numa_n_nodes = 0;
static int g_numa_cpn = 0;

// B matrix replicated on each NUMA node (persistent cache)
static float* g_B_cache[4] = {nullptr, nullptr, nullptr, nullptr};
static int64_t g_B_cache_size[4] = {0, 0, 0, 0};

static void* numa_gemm_worker(void* arg) {
    int id = (int)(long)arg;
    int node = id / g_numa_cpn;
    if (node >= g_numa_n_nodes) node = g_numa_n_nodes - 1;

    numa_run_on_node(node);
    // Also pin to specific core for best cache locality
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(id, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);

    while (1) {
        pthread_barrier_wait(&g_numa_bar_start);
        if (!g_numa_pool_alive) break;

        NumaGemmTile* t = &g_numa_tiles[id];
        if (t->M > 0) {
            CBLAS_TRANSPOSE ta = (t->transA == 1) ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE tb = (t->transB == 1) ? CblasTrans : CblasNoTrans;
            cblas_sgemm(CblasRowMajor, ta, tb,
                        t->M, t->N, t->K, 1.0f,
                        t->A, t->lda, t->B, t->ldb,
                        0.0f, t->C, t->ldc);
        }
        pthread_barrier_wait(&g_numa_bar_done);
    }
    return NULL;
}

static void numa_pool_init() {
    if (g_numa_pool_alive) return;

    g_numa_n_nodes = numa_max_node() + 1;
    if (g_numa_n_nodes <= 0) g_numa_n_nodes = 1;

    // Use all available cores (detect from /proc/cpuinfo or sysconf)
    g_numa_n_workers = g_numa_n_nodes * 8;  // E8C2: 8 cores per chip
    if (g_numa_n_workers > NUMA_POOL_MAX) g_numa_n_workers = NUMA_POOL_MAX;
    g_numa_cpn = g_numa_n_workers / g_numa_n_nodes;

    g_numa_pool_alive = 1;
    pthread_barrier_init(&g_numa_bar_start, NULL, g_numa_n_workers + 1);
    pthread_barrier_init(&g_numa_bar_done, NULL, g_numa_n_workers + 1);

    for (int i = 0; i < g_numa_n_workers; i++) {
        memset(&g_numa_tiles[i], 0, sizeof(NumaGemmTile));
        pthread_create(&g_numa_threads[i], NULL, numa_gemm_worker, (void*)(long)i);
    }
}

// Ensure B is replicated on each NUMA node
static const float* numa_get_B(int node, const float* B_src, int64_t B_elems) {
    int64_t B_bytes = B_elems * sizeof(float);
    if (!g_B_cache[node] || g_B_cache_size[node] < B_bytes) {
        if (g_B_cache[node]) numa_free(g_B_cache[node], g_B_cache_size[node]);
        g_B_cache[node] = (float*)numa_alloc_onnode(B_bytes, node);
        g_B_cache_size[node] = B_bytes;
    }
    memcpy(g_B_cache[node], B_src, B_bytes);
    return g_B_cache[node];
}

// Dispatch tiled GEMM: C[M,N] = A[M,K] @ B[K,N] (or transposed variants)
static void numa_tiled_sgemm(int64_t M, int64_t K, int64_t N,
                              const float* A, int64_t lda,
                              const float* B, int64_t ldb,
                              float* C, int64_t ldc,
                              int transA, int transB) {
    numa_pool_init();

    int nw = g_numa_n_workers;
    // For transA=0: split M rows across workers
    // For transA=1: A is [K,M] but we compute C[M,N], still split output M rows
    int64_t out_rows = M;
    int64_t rp = (out_rows + nw - 1) / nw;

    // Replicate B on each NUMA node
    int64_t B_elems = (transB == 0) ? K * N : N * K;
    for (int n = 0; n < g_numa_n_nodes; n++)
        numa_get_B(n, B, B_elems);

    for (int i = 0; i < nw; i++) {
        int node = i / g_numa_cpn;
        if (node >= g_numa_n_nodes) node = g_numa_n_nodes - 1;
        int64_t ms = i * rp;
        int64_t rows = (ms + rp <= out_rows) ? rp : out_rows - ms;
        if (ms >= out_rows) rows = 0;

        g_numa_tiles[i].M = (int)rows;
        g_numa_tiles[i].N = (int)N;
        g_numa_tiles[i].K = (int)K;
        g_numa_tiles[i].transA = transA;
        g_numa_tiles[i].transB = transB;

        if (transA == 0) {
            g_numa_tiles[i].A = A + ms * lda;
            g_numa_tiles[i].lda = (int)lda;
        } else {
            // A is [K, M], transposed to [M, K]. Offset by columns.
            g_numa_tiles[i].A = A + ms;  // column offset in A[K,M]
            g_numa_tiles[i].lda = (int)lda;
        }
        g_numa_tiles[i].B = g_B_cache[node];
        g_numa_tiles[i].ldb = (int)ldb;
        g_numa_tiles[i].C = C + ms * ldc;
        g_numa_tiles[i].ldc = (int)ldc;
    }

    pthread_barrier_wait(&g_numa_bar_start);
    pthread_barrier_wait(&g_numa_bar_done);
}

#elif defined(PT_USE_NUMA) && (defined(PT_USE_SYSTEM_BLAS))
static constexpr int64_t NUMA_GEMM_THRESHOLD = 256;

// NUMA-aware NN: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
void sgemm_numa(int64_t M, int64_t K, int64_t N, float alpha,
                const float* A, int64_t lda, const float* B, int64_t ldb,
                float beta, float* C, int64_t ldc) {
    int num_nodes = numa_max_node() + 1;
    if (num_nodes <= 1 || M < 64) {
        // Single node or tiny M: fall back to plain EML
        tuda::blas::sgemm(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // Split M rows across NUMA nodes
    int64_t rows_per_node = (M + num_nodes - 1) / num_nodes;
    std::vector<std::thread> threads(num_nodes);

    for (int n = 0; n < num_nodes; n++) {
        threads[n] = std::thread([&, n]() {
            // Pin this thread to NUMA node n
            numa_run_on_node(n);

            int64_t m_start = n * rows_per_node;
            int64_t m_end = std::min(m_start + rows_per_node, M);
            if (m_start >= M) return;
            int64_t m_tile = m_end - m_start;

            // Allocate A tile on local NUMA node memory
            // This is the key: A rows are copied to node-local RAM,
            // avoiding cross-chip memory access during GEMM.
            size_t a_bytes = (size_t)m_tile * K * sizeof(float);
            float* A_local = (float*)numa_alloc_onnode(a_bytes, n);
            if (!A_local) {
                // Fallback: use original A pointer (slower, cross-NUMA)
                A_local = nullptr;
            }

            const float* A_src = A + m_start * lda;
            const float* A_use = A_local ? A_local : A_src;

            if (A_local) {
                // Copy A rows to node-local memory
                // Use lda-aware copy (rows may have padding)
                if (lda == K) {
                    // Contiguous: single memcpy
                    std::memcpy(A_local, A_src, a_bytes);
                } else {
                    // Strided: copy row by row
                    for (int64_t i = 0; i < m_tile; i++) {
                        std::memcpy(A_local + i * K, A_src + i * lda, K * sizeof(float));
                    }
                }
            }

            // Each node calls EML sgemm on its tile of rows
            // B is shared (read-only, acceptable cross-NUMA since it's in cache)
            // C is written directly (each node writes disjoint rows)
            // SIGILL FIX: Force EML single-threaded in this worker thread.
            // Without this, EML's internal OMP fork from multiple std::threads
            // causes SIGILL on E2K for matrices >= 256.
#ifdef _OPENMP
            omp_set_num_threads(1);
#endif
            tuda::blas::sgemm(
                        (int)m_tile, (int)N, (int)K,
                        alpha,
                        A_use, A_local ? (int)K : (int)lda,
                        B, (int)ldb,
                        beta,
                        C + m_start * ldc, (int)ldc);

            if (A_local) {
                numa_free(A_local, a_bytes);
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

// NUMA-aware NT: C[M,N] = alpha * A[M,K] @ B^T[N,K] + beta * C[M,N]
void sgemm_nt_numa(int64_t M, int64_t K, int64_t N, float alpha,
                   const float* A, int64_t lda, const float* B, int64_t ldb,
                   float beta, float* C, int64_t ldc) {
    int num_nodes = numa_max_node() + 1;
    if (num_nodes <= 1 || M < 64) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)M, (int)N, (int)K,
                    alpha, A, (int)lda, B, (int)ldb,
                    beta, C, (int)ldc);
        return;
    }

    int64_t rows_per_node = (M + num_nodes - 1) / num_nodes;
    std::vector<std::thread> threads(num_nodes);

    for (int n = 0; n < num_nodes; n++) {
        threads[n] = std::thread([&, n]() {
            numa_run_on_node(n);

            int64_t m_start = n * rows_per_node;
            int64_t m_end = std::min(m_start + rows_per_node, M);
            if (m_start >= M) return;
            int64_t m_tile = m_end - m_start;

            size_t a_bytes = (size_t)m_tile * K * sizeof(float);
            float* A_local = (float*)numa_alloc_onnode(a_bytes, n);

            const float* A_src = A + m_start * lda;
            const float* A_use = A_local ? A_local : A_src;

            if (A_local) {
                if (lda == K) {
                    std::memcpy(A_local, A_src, a_bytes);
                } else {
                    for (int64_t i = 0; i < m_tile; i++)
                        std::memcpy(A_local + i * K, A_src + i * lda, K * sizeof(float));
                }
            }

            // B^T: B is [N,K], shared read-only across all nodes
            // SIGILL FIX: Force EML single-threaded in NUMA worker thread
#ifdef _OPENMP
            omp_set_num_threads(1);
#endif
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)m_tile, (int)N, (int)K,
                        alpha,
                        A_use, A_local ? (int)K : (int)lda,
                        B, (int)ldb,
                        beta,
                        C + m_start * ldc, (int)ldc);

            if (A_local) numa_free(A_local, a_bytes);
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

#endif // PT_USE_NUMA

// BLAS wrappers — NUMA-aware dispatch for Elbrus E8C2
// PT_NO_NUMA_POOL=1: bypass pthread pool, call EML directly from main thread.
//   Use with libeml_mt + numactl --cpunodebind=N for 245 GFLOPS/node.
// Without env var: uses 32-pthread pool + ST EML (180 GFLOPS total).
#if defined(PT_USE_NUMA)
static bool g_numa_pool_disabled = (getenv("PT_NO_NUMA_POOL") != nullptr);
#endif

void sgemm(int64_t M, int64_t K, int64_t N, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) {
#if defined(PT_USE_EML_BLAS)
  #if defined(PT_USE_NUMA)
    if (!g_numa_pool_disabled && M >= NUMA_GEMM_THRESHOLD) {
        numa_tiled_sgemm(M, K, N, A, lda, B, ldb, C, ldc, 0, 0);
        return;
    }
  #endif
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
#elif defined(PT_USE_SYSTEM_BLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
#else
    tuda::blas::sgemm(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}
void sgemm_nt(int64_t M, int64_t K, int64_t N, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) {
#if defined(PT_USE_EML_BLAS)
  #if defined(PT_USE_NUMA)
    if (!g_numa_pool_disabled && M >= NUMA_GEMM_THRESHOLD) {
        numa_tiled_sgemm(M, K, N, A, lda, B, ldb, C, ldc, 0, 1);
        return;
    }
  #endif
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
#elif defined(PT_USE_SYSTEM_BLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)M, (int)N, (int)K,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
#else
    tuda::blas::sgemm_nt(M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}
void sgemv(int64_t M, int64_t N, float alpha, const float* A, int64_t lda, const float* x, float beta, float* y) {
#if defined(PT_USE_EML_BLAS)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)M, (int)N,
                alpha, A, (int)lda, x, 1, beta, y, 1);
#elif defined(PT_USE_SYSTEM_BLAS)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)M, (int)N,
                alpha, A, (int)lda, x, 1, beta, y, 1);
#else
    tuda::blas::sgemv(M, N, alpha, A, lda, x, beta, y);
#endif
}
float sdot(int64_t n, const float* a, const float* b) {
#if defined(PT_USE_EML_BLAS)
    return cblas_sdot((int)n, a, 1, b, 1);
#elif defined(PT_USE_SYSTEM_BLAS)
    return cblas_sdot((int)n, a, 1, b, 1);
#else
    return tuda::blas::sdot(n, a, b);
#endif
}

// ============================================================================
// Element-wise binary loops
// ============================================================================

void add_loop(const float* a, const float* b, float* out, int64_t n, float alpha) {
#ifdef PT_USE_EML_BLAS
    if (alpha == 1.0f) {
        eml_Vector_Add_32F(a, b, out, (int)n);
        return;
    } else {
        // out = a + alpha*b: copy a to out, then saxpy
        if (out != a) std::memcpy(out, a, n * sizeof(float));
        cblas_saxpy((int)n, alpha, b, 1, out, 1);
        return;
    }
#endif
    constexpr int W = VecF::width;
    if (alpha == 1.0f) {
        c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
            int64_t i = start;
            for (; i < end && (i % W != 0); ++i) out[i] = a[i] + b[i];
            int64_t se = end - ((end - i) % (4*W));
            for (; i < se; i += 4*W) {
                (VecF::load(a+i) + VecF::load(b+i)).store(out+i);
                (VecF::load(a+i+W) + VecF::load(b+i+W)).store(out+i+W);
                (VecF::load(a+i+2*W) + VecF::load(b+i+2*W)).store(out+i+2*W);
                (VecF::load(a+i+3*W) + VecF::load(b+i+3*W)).store(out+i+3*W);
            }
            for (; i + W <= end; i += W) (VecF::load(a+i) + VecF::load(b+i)).store(out+i);
            for (; i < end; ++i) out[i] = a[i] + b[i];
        });
    } else {
        VecF va = VecF::broadcast(alpha);
        c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
            int64_t i = start;
            for (; i < end && (i % W != 0); ++i) out[i] = a[i] + alpha * b[i];
            int64_t se = end - ((end - i) % (4*W));
            for (; i < se; i += 4*W) {
                VecF::fmadd(va, VecF::load(b+i), VecF::load(a+i)).store(out+i);
                VecF::fmadd(va, VecF::load(b+i+W), VecF::load(a+i+W)).store(out+i+W);
                VecF::fmadd(va, VecF::load(b+i+2*W), VecF::load(a+i+2*W)).store(out+i+2*W);
                VecF::fmadd(va, VecF::load(b+i+3*W), VecF::load(a+i+3*W)).store(out+i+3*W);
            }
            for (; i + W <= end; i += W) VecF::fmadd(va, VecF::load(b+i), VecF::load(a+i)).store(out+i);
            for (; i < end; ++i) out[i] = a[i] + alpha * b[i];
        });
    }
}

void sub_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) out[i] = a[i] - b[i];
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            (VecF::load(a+i) - VecF::load(b+i)).store(out+i);
            (VecF::load(a+i+W) - VecF::load(b+i+W)).store(out+i+W);
            (VecF::load(a+i+2*W) - VecF::load(b+i+2*W)).store(out+i+2*W);
            (VecF::load(a+i+3*W) - VecF::load(b+i+3*W)).store(out+i+3*W);
        }
        for (; i + W <= end; i += W) (VecF::load(a+i) - VecF::load(b+i)).store(out+i);
        for (; i < end; ++i) out[i] = a[i] - b[i];
    });
}

void mul_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) out[i] = a[i] * b[i];
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            (VecF::load(a+i) * VecF::load(b+i)).store(out+i);
            (VecF::load(a+i+W) * VecF::load(b+i+W)).store(out+i+W);
            (VecF::load(a+i+2*W) * VecF::load(b+i+2*W)).store(out+i+2*W);
            (VecF::load(a+i+3*W) * VecF::load(b+i+3*W)).store(out+i+3*W);
        }
        for (; i + W <= end; i += W) (VecF::load(a+i) * VecF::load(b+i)).store(out+i);
        for (; i < end; ++i) out[i] = a[i] * b[i];
    });
}

void div_loop(const float* a, const float* b, float* out, int64_t n) {
    constexpr int W = VecF::width;
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) out[i] = a[i] / b[i];
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            (VecF::load(a+i) / VecF::load(b+i)).store(out+i);
            (VecF::load(a+i+W) / VecF::load(b+i+W)).store(out+i+W);
            (VecF::load(a+i+2*W) / VecF::load(b+i+2*W)).store(out+i+2*W);
            (VecF::load(a+i+3*W) / VecF::load(b+i+3*W)).store(out+i+3*W);
        }
        for (; i + W <= end; i += W) (VecF::load(a+i) / VecF::load(b+i)).store(out+i);
        for (; i < end; ++i) out[i] = a[i] / b[i];
    });
}

void add_broadcast_loop(const float* a, const float* b, float* out, int64_t outer, int64_t inner, float alpha) {
    constexpr int W = VecF::width;
    if (alpha == 1.0f) {
        c10::parallel_for_1d(outer, [&](int64_t start, int64_t end) {
            for (int64_t o = start; o < end; ++o) {
                const float* ra = a + o * inner; float* ro = out + o * inner;
                int64_t j = 0;
                for (; j + W <= inner; j += W) (VecF::load(ra+j) + VecF::load(b+j)).store(ro+j);
                for (; j < inner; ++j) ro[j] = ra[j] + b[j];
            }
        }, 16);
    } else {
        VecF va = VecF::broadcast(alpha);
        c10::parallel_for_1d(outer, [&](int64_t start, int64_t end) {
            for (int64_t o = start; o < end; ++o) {
                const float* ra = a + o * inner; float* ro = out + o * inner;
                int64_t j = 0;
                for (; j + W <= inner; j += W) VecF::fmadd(va, VecF::load(b+j), VecF::load(ra+j)).store(ro+j);
                for (; j < inner; ++j) ro[j] = ra[j] + alpha * b[j];
            }
        }, 16);
    }
}

// ============================================================================
// Unary loops
// ============================================================================

#define DEFINE_HOT_UNARY(name, vec_expr, scalar_expr) \
void name##_loop(const float* in, float* out, int64_t n) { \
    constexpr int W = VecF::width; \
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) { \
        int64_t i = start; \
        for (; i < end && (i % W != 0); ++i) { float x = in[i]; out[i] = (scalar_expr); } \
        int64_t se = end - ((end - i) % (4*W)); \
        for (; i < se; i += 4*W) { \
            { VecF v = VecF::load(in+i);     (vec_expr).store(out+i); } \
            { VecF v = VecF::load(in+i+W);   (vec_expr).store(out+i+W); } \
            { VecF v = VecF::load(in+i+2*W); (vec_expr).store(out+i+2*W); } \
            { VecF v = VecF::load(in+i+3*W); (vec_expr).store(out+i+3*W); } \
        } \
        for (; i + W <= end; i += W) { VecF v = VecF::load(in+i); (vec_expr).store(out+i); } \
        for (; i < end; ++i) { float x = in[i]; out[i] = (scalar_expr); } \
    }); \
}

DEFINE_HOT_UNARY(neg,        tuda::neg_vec(v),        -x)
DEFINE_HOT_UNARY(abs,        tuda::abs_vec(v),        std::abs(x))
DEFINE_HOT_UNARY(sqrt,       tuda::sqrt_vec(v),       std::sqrt(x))
DEFINE_HOT_UNARY(rsqrt,      tuda::rsqrt_vec(v),      1.0f / std::sqrt(x))
DEFINE_HOT_UNARY(square,     tuda::square_vec(v),     x * x)
DEFINE_HOT_UNARY(exp,        tuda::exp_vec(v),        std::exp(x))
DEFINE_HOT_UNARY(log,        tuda::log_vec(v),        std::log(x))
DEFINE_HOT_UNARY(log2,       tuda::log2_vec(v),       std::log2(x))
DEFINE_HOT_UNARY(log10,      tuda::log10_vec(v),      std::log10(x))
DEFINE_HOT_UNARY(sin,        tuda::sin_vec(v),        std::sin(x))
DEFINE_HOT_UNARY(cos,        tuda::cos_vec(v),        std::cos(x))
DEFINE_HOT_UNARY(tan,        tuda::tan_vec(v),        std::tan(x))
DEFINE_HOT_UNARY(tanh,       tuda::tanh_vec(v),       std::tanh(x))
DEFINE_HOT_UNARY(sigmoid,    tuda::sigmoid_vec(v),    1.0f / (1.0f + std::exp(-x)))
DEFINE_HOT_UNARY(relu,       tuda::relu_vec(v),       (x > 0.0f ? x : 0.0f))
DEFINE_HOT_UNARY(reciprocal, tuda::reciprocal_vec(v), 1.0f / x)
DEFINE_HOT_UNARY(ceil,       tuda::ceil_vec(v),       std::ceil(x))
DEFINE_HOT_UNARY(floor,      tuda::floor_vec(v),      std::floor(x))
DEFINE_HOT_UNARY(round,      tuda::round_vec(v),      std::round(x))
DEFINE_HOT_UNARY(sign,       tuda::sign_vec(v),       ((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)))

#undef DEFINE_HOT_UNARY

// ============================================================================
// Reduction loops
// ============================================================================

float sum_loop(const float* data, int64_t n) {
    if (n >= 65536) {
        auto& pool = c10::get_thread_pool();
        int nt = pool.num_threads();
        std::vector<float> partial(nt, 0.0f);
        std::atomic<int> cid{0};
        pool.parallel_for(n, [&](int64_t s, int64_t e) {
            float ls = tuda::vec_sum(data + s, e - s);
            int id = cid.fetch_add(1, std::memory_order_relaxed);
            partial[id] = ls;
        });
        float t = 0; for (int i = 0; i < nt; ++i) t += partial[i];
        return t;
    }
    return tuda::vec_sum(data, n);
}

void sum_dim_loop(const float* in, float* out, int64_t outer_size, int64_t reduce_size, int64_t inner_size) {
    constexpr int W = VecF::width;
    if (inner_size == 1) {
        c10::parallel_for_1d(outer_size, [&](int64_t start, int64_t end) {
            for (int64_t o = start; o < end; ++o)
                out[o] = tuda::vec_sum(in + o * reduce_size, reduce_size);
        }, 16);
    } else {
        c10::parallel_for_1d(outer_size, [&](int64_t start, int64_t end) {
            for (int64_t o = start; o < end; ++o) {
                float* orow = out + o * inner_size;
                std::memset(orow, 0, inner_size * sizeof(float));
                for (int64_t r = 0; r < reduce_size; ++r) {
                    const float* irow = in + (o * reduce_size + r) * inner_size;
                    int64_t j = 0;
                    for (; j + 4*W <= inner_size; j += 4*W) {
                        (VecF::load(orow+j) + VecF::load(irow+j)).store(orow+j);
                        (VecF::load(orow+j+W) + VecF::load(irow+j+W)).store(orow+j+W);
                        (VecF::load(orow+j+2*W) + VecF::load(irow+j+2*W)).store(orow+j+2*W);
                        (VecF::load(orow+j+3*W) + VecF::load(irow+j+3*W)).store(orow+j+3*W);
                    }
                    for (; j + W <= inner_size; j += W) (VecF::load(orow+j) + VecF::load(irow+j)).store(orow+j);
                    for (; j < inner_size; ++j) orow[j] += irow[j];
                }
            }
        }, 16);
    }
}

// ============================================================================
// In-place scalar loops (optimizer hot paths)
// ============================================================================

void mul_scalar_inplace(float* data, float scalar, int64_t n) {
    constexpr int W = VecF::width;
    VecF vs = VecF::broadcast(scalar);
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) data[i] *= scalar;
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            (VecF::load(data+i) * vs).store(data+i);
            (VecF::load(data+i+W) * vs).store(data+i+W);
            (VecF::load(data+i+2*W) * vs).store(data+i+2*W);
            (VecF::load(data+i+3*W) * vs).store(data+i+3*W);
        }
        for (; i + W <= end; i += W) (VecF::load(data+i) * vs).store(data+i);
        for (; i < end; ++i) data[i] *= scalar;
    });
}

void axpy_inplace(float* data, float scalar, const float* other, int64_t n) {
    constexpr int W = VecF::width;
    VecF vs = VecF::broadcast(scalar);
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) data[i] += scalar * other[i];
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            VecF::fmadd(vs, VecF::load(other+i), VecF::load(data+i)).store(data+i);
            VecF::fmadd(vs, VecF::load(other+i+W), VecF::load(data+i+W)).store(data+i+W);
            VecF::fmadd(vs, VecF::load(other+i+2*W), VecF::load(data+i+2*W)).store(data+i+2*W);
            VecF::fmadd(vs, VecF::load(other+i+3*W), VecF::load(data+i+3*W)).store(data+i+3*W);
        }
        for (; i + W <= end; i += W) VecF::fmadd(vs, VecF::load(other+i), VecF::load(data+i)).store(data+i);
        for (; i < end; ++i) data[i] += scalar * other[i];
    });
}

void adam_step_loop(float* param, const float* grad, float* exp_avg, float* exp_avg_sq,
                    int64_t n, float lr, float beta1, float beta2, float eps, float weight_decay,
                    float bias_correction1, float bias_correction2, bool amsgrad, float* max_exp_avg_sq) {
    constexpr int W = VecF::width;
    VecF vb1 = VecF::broadcast(beta1), vb2 = VecF::broadcast(beta2);
    VecF v1b1 = VecF::broadcast(1.0f - beta1), v1b2 = VecF::broadcast(1.0f - beta2);
    VecF ve = VecF::broadcast(eps);
    float ss = lr / bias_correction1;
    VecF vss = VecF::broadcast(ss);
    VecF vbc2 = VecF::broadcast(1.0f / std::sqrt(bias_correction2));
    VecF vwd = VecF::broadcast(weight_decay);

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) {
            float g = grad[i], p = param[i];
            if (weight_decay != 0.0f) g += weight_decay * p;
            exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
            exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
            float dv;
            if (amsgrad && max_exp_avg_sq) { max_exp_avg_sq[i] = std::max(max_exp_avg_sq[i], exp_avg_sq[i]); dv = std::sqrt(max_exp_avg_sq[i] / bias_correction2) + eps; }
            else dv = std::sqrt(exp_avg_sq[i] / bias_correction2) + eps;
            param[i] = p - ss * exp_avg[i] / dv;
        }
        int64_t se = end - ((end - i) % W);
        for (; i < se; i += W) {
            VecF g = VecF::load(grad+i), p = VecF::load(param+i);
            if (weight_decay != 0.0f) g = g + vwd * p;
            VecF m = VecF::fmadd(vb1, VecF::load(exp_avg+i), v1b1 * g); m.store(exp_avg+i);
            VecF v = VecF::fmadd(vb2, VecF::load(exp_avg_sq+i), v1b2 * g * g); v.store(exp_avg_sq+i);
            VecF dn;
            if (amsgrad && max_exp_avg_sq) {
                float *mp = max_exp_avg_sq+i, *vp = exp_avg_sq+i;
                for (int j = 0; j < W; ++j) if (vp[j] > mp[j]) mp[j] = vp[j];
                dn = tuda::sqrt_vec(VecF::load(max_exp_avg_sq+i) * vbc2 * vbc2) + ve;
            } else dn = tuda::sqrt_vec(v * vbc2 * vbc2) + ve;
            (p - vss * m / dn).store(param+i);
        }
        for (; i < end; ++i) {
            float g = grad[i], p = param[i];
            if (weight_decay != 0.0f) g += weight_decay * p;
            exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
            exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
            float dv;
            if (amsgrad && max_exp_avg_sq) { max_exp_avg_sq[i] = std::max(max_exp_avg_sq[i], exp_avg_sq[i]); dv = std::sqrt(max_exp_avg_sq[i] / bias_correction2) + eps; }
            else dv = std::sqrt(exp_avg_sq[i] / bias_correction2) + eps;
            param[i] = p - ss * exp_avg[i] / dv;
        }
    });
}

void sgd_step_loop(float* param, const float* grad, float* momentum_buf,
                   int64_t n, float lr, float momentum, float dampening,
                   float weight_decay, bool nesterov) {
    constexpr int W = VecF::width;
    VecF vlr = VecF::broadcast(lr), vwd = VecF::broadcast(weight_decay);
    VecF vmom = VecF::broadcast(momentum), vdamp = VecF::broadcast(1.0f - dampening);

    if (momentum == 0.0f) {
        c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
            int64_t i = start;
            for (; i < end && (i % W != 0); ++i) { float g = grad[i]; if (weight_decay != 0.0f) g += weight_decay * param[i]; param[i] -= lr * g; }
            int64_t se = end - ((end - i) % W);
            for (; i < se; i += W) { VecF g = VecF::load(grad+i), p = VecF::load(param+i); if (weight_decay != 0.0f) g = g + vwd * p; (p - vlr * g).store(param+i); }
            for (; i < end; ++i) { float g = grad[i]; if (weight_decay != 0.0f) g += weight_decay * param[i]; param[i] -= lr * g; }
        });
    } else {
        c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
            int64_t i = start;
            for (; i < end && (i % W != 0); ++i) {
                float g = grad[i]; if (weight_decay != 0.0f) g += weight_decay * param[i];
                momentum_buf[i] = momentum * momentum_buf[i] + (1.0f - dampening) * g;
                param[i] -= nesterov ? lr * (g + momentum * momentum_buf[i]) : lr * momentum_buf[i];
            }
            int64_t se = end - ((end - i) % W);
            for (; i < se; i += W) {
                VecF g = VecF::load(grad+i), p = VecF::load(param+i);
                if (weight_decay != 0.0f) g = g + vwd * p;
                VecF buf = VecF::fmadd(vmom, VecF::load(momentum_buf+i), vdamp * g); buf.store(momentum_buf+i);
                if (nesterov) p = p - vlr * (g + vmom * buf); else p = p - vlr * buf;
                p.store(param+i);
            }
            for (; i < end; ++i) {
                float g = grad[i]; if (weight_decay != 0.0f) g += weight_decay * param[i];
                momentum_buf[i] = momentum * momentum_buf[i] + (1.0f - dampening) * g;
                param[i] -= nesterov ? lr * (g + momentum * momentum_buf[i]) : lr * momentum_buf[i];
            }
        });
    }
}

// Missing implementations added below

// sgemm_tn: C[K,N] = alpha * A^T[K,M] @ B[M,N] + beta * C
// A is [M, K] in row-major, A^T is [K, M]
void sgemm_tn(int64_t M, int64_t K, int64_t N, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) {
#if defined(PT_USE_EML_BLAS)
  #if defined(PT_USE_NUMA)
    if (!g_numa_pool_disabled && K >= NUMA_GEMM_THRESHOLD) {
        numa_tiled_sgemm(K, M, N, A, lda, B, ldb, C, ldc, 1, 0);
        return;
    }
  #endif
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)K, (int)N, (int)M,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
#elif defined(PT_USE_SYSTEM_BLAS)
    // Direct system BLAS with CblasTrans — no transpose buffer needed
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)K, (int)N, (int)M,
                alpha, A, (int)lda, B, (int)ldb,
                beta, C, (int)ldc);
#else
    // Fallback: use thread-local buffer to avoid heap alloc per call
    // For MNIST sizes (M=64, K=512): 64*512*4 = 128KB — fits in stack/TLS
    static thread_local std::vector<float> At_buf;
    int64_t sz = K * M;
    if ((int64_t)At_buf.size() < sz) At_buf.resize(sz);
    float* At = At_buf.data();
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k)
            At[k * M + m] = A[m * lda + k];
    tuda::blas::sgemm(K, M, N, alpha, At, M, B, ldb, beta, C, ldc);
#endif
}

void relu_mask_mul(const float* grad, const float* mask, float* out, int64_t n) {
    // ReLU backward: out = grad * (mask > 0)
    // Use sign(mask) trick: relu(mask) > 0 means mask > 0.
    // Compute: out = grad * min(1, relu(mask) * LARGE) — but that's convoluted.
    // Platform-specific SIMD blendv is fastest:
#if defined(TUDA_AVX2)
    __m256 vzero = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 m0 = _mm256_loadu_ps(mask+i),    g0 = _mm256_loadu_ps(grad+i);
        __m256 m1 = _mm256_loadu_ps(mask+i+8),  g1 = _mm256_loadu_ps(grad+i+8);
        __m256 m2 = _mm256_loadu_ps(mask+i+16), g2 = _mm256_loadu_ps(grad+i+16);
        __m256 m3 = _mm256_loadu_ps(mask+i+24), g3 = _mm256_loadu_ps(grad+i+24);
        _mm256_storeu_ps(out+i,    _mm256_and_ps(g0, _mm256_cmp_ps(m0, vzero, _CMP_GT_OS)));
        _mm256_storeu_ps(out+i+8,  _mm256_and_ps(g1, _mm256_cmp_ps(m1, vzero, _CMP_GT_OS)));
        _mm256_storeu_ps(out+i+16, _mm256_and_ps(g2, _mm256_cmp_ps(m2, vzero, _CMP_GT_OS)));
        _mm256_storeu_ps(out+i+24, _mm256_and_ps(g3, _mm256_cmp_ps(m3, vzero, _CMP_GT_OS)));
    }
    for (; i + 8 <= n; i += 8) {
        __m256 m = _mm256_loadu_ps(mask+i), g = _mm256_loadu_ps(grad+i);
        _mm256_storeu_ps(out+i, _mm256_and_ps(g, _mm256_cmp_ps(m, vzero, _CMP_GT_OS)));
    }
    for (; i < n; ++i) out[i] = mask[i] > 0.0f ? grad[i] : 0.0f;
#else
    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i)
            out[i] = mask[i] > 0.0f ? grad[i] : 0.0f;
    });
#endif
}

void col_sum(const float* data, float* out, int64_t rows, int64_t cols) {
    // out[j] = sum over rows of data[i*cols + j]
    constexpr int W = VecF::width;
    // Zero output
    int64_t j = 0;
    for (; j + W <= cols; j += W) VecF::zero().store(out + j);
    for (; j < cols; ++j) out[j] = 0;
    // Accumulate rows
    for (int64_t i = 0; i < rows; ++i) {
        const float* row = data + i * cols;
        j = 0;
        for (; j + 4*W <= cols; j += 4*W) {
            (VecF::load(out+j)     + VecF::load(row+j)).store(out+j);
            (VecF::load(out+j+W)   + VecF::load(row+j+W)).store(out+j+W);
            (VecF::load(out+j+2*W) + VecF::load(row+j+2*W)).store(out+j+2*W);
            (VecF::load(out+j+3*W) + VecF::load(row+j+3*W)).store(out+j+3*W);
        }
        for (; j + W <= cols; j += W)
            (VecF::load(out+j) + VecF::load(row+j)).store(out+j);
        for (; j < cols; ++j) out[j] += row[j];
    }
}

void add_inplace(float* dst, const float* src, int64_t n) {
    constexpr int W = VecF::width;
    int64_t i = 0;
    for (; i + 4*W <= n; i += 4*W) {
        (VecF::load(dst+i)     + VecF::load(src+i)).store(dst+i);
        (VecF::load(dst+i+W)   + VecF::load(src+i+W)).store(dst+i+W);
        (VecF::load(dst+i+2*W) + VecF::load(src+i+2*W)).store(dst+i+2*W);
        (VecF::load(dst+i+3*W) + VecF::load(src+i+3*W)).store(dst+i+3*W);
    }
    for (; i + W <= n; i += W)
        (VecF::load(dst+i) + VecF::load(src+i)).store(dst+i);
    for (; i < n; ++i) dst[i] += src[i];
}

void fused_adam_multi(AdamParamPack* params, int num_params,
                     float lr, float beta1, float beta2, float eps,
                     float weight_decay, int step, bool amsgrad, float** max_exp_avg_sq) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    for (int p = 0; p < num_params; p++) {
        if (!amsgrad) {
            // Fast path: use fused_adam_avx2 (single-pass, no amsgrad)
            fused_adam_avx2(params[p].param, params[p].grad,
                           params[p].exp_avg, params[p].exp_avg_sq,
                           params[p].numel, lr, beta1, beta2, eps,
                           weight_decay, bc1, bc2);
        } else {
            // AMSGrad path: fallback to adam_step_loop (handles max_exp_avg_sq)
            adam_step_loop(params[p].param, params[p].grad,
                           params[p].exp_avg, params[p].exp_avg_sq,
                           params[p].numel, lr, beta1, beta2, eps, weight_decay,
                           bc1, bc2,
                           true, max_exp_avg_sq ? max_exp_avg_sq[p] : nullptr);
        }
    }
}

void fused_sgd_multi(SGDParamPack* params, int num_params,
                    float lr, float momentum, float dampening,
                    float weight_decay, bool nesterov) {
    for (int p = 0; p < num_params; p++) {
        if (momentum != 0.0f && !nesterov && weight_decay == 0.0f
            && params[p].momentum_buf != nullptr) {
            // Fast path: fused SGD+momentum (single-pass buf+param update)
            fused_sgd_momentum_avx2(params[p].param, params[p].grad,
                                    params[p].momentum_buf, params[p].numel,
                                    lr, momentum, dampening);
        } else {
            // General path: handles nesterov, weight_decay, no-momentum
            sgd_step_loop(params[p].param, params[p].grad, params[p].momentum_buf,
                          params[p].numel, lr, momentum, dampening,
                          weight_decay, nesterov);
        }
    }
}

void cross_entropy_fused(const float* logits, const int64_t* targets,
                         float* loss_out, float* grad_out,
                         int64_t batch, int64_t classes) {
    // Clamp range: prevents Inf-Inf=NaN when logits contain Inf/-Inf.
    // 88.0f is close to log(FLT_MAX); beyond that expf() overflows.
    constexpr float CLAMP_MAX = 88.0f;
    constexpr float CLAMP_MIN = -88.0f;

    float total_loss = 0.0f;
    for (int64_t b = 0; b < batch; b++) {
        const float* row = logits + b * classes;
        float* grow = grad_out + b * classes;
        int64_t tgt = targets[b];

        // Find max (with clamp to prevent Inf)
        float mx = std::max(CLAMP_MIN, std::min(row[0], CLAMP_MAX));
        for (int64_t c = 1; c < classes; c++) {
            float v = std::max(CLAMP_MIN, std::min(row[c], CLAMP_MAX));
            if (v > mx) mx = v;
        }

        // Compute exp(logit - max) and sum_exp
        float sum_exp = 0;
        for (int64_t c = 0; c < classes; c++) {
            float v = std::max(CLAMP_MIN, std::min(row[c], CLAMP_MAX));
            grow[c] = expf(v - mx);
            sum_exp += grow[c];
        }

        // Softmax and gradient: (softmax - one_hot) / batch
        float inv = 1.0f / sum_exp;
        float inv_batch = 1.0f / (float)batch;
        for (int64_t c = 0; c < classes; c++) {
            float sm = grow[c] * inv;
            grow[c] = (sm - (c == tgt ? 1.0f : 0.0f)) * inv_batch;
        }

        // Loss: -log(softmax[tgt]) = -(logit[tgt] - max - log(sum_exp))
        float logit_tgt = std::max(CLAMP_MIN, std::min(row[tgt], CLAMP_MAX));
        total_loss += mx - logit_tgt + logf(sum_exp);
    }
    *loss_out = total_loss / (float)batch;
}


void bias_relu_fused(float* out, const float* bias, int64_t M, int64_t N) {
    constexpr int W = VecF::width;
#if defined(TUDA_AVX2)
    __m256 vzero = _mm256_setzero_ps();
    for (int64_t i = 0; i < M; ++i) {
        float* row = out + i * N;
        int64_t j = 0;
        for (; j + 4*W <= N; j += 4*W) {
            __m256 r0 = _mm256_add_ps(_mm256_loadu_ps(row+j),      _mm256_loadu_ps(bias+j));
            __m256 r1 = _mm256_add_ps(_mm256_loadu_ps(row+j+W),    _mm256_loadu_ps(bias+j+W));
            __m256 r2 = _mm256_add_ps(_mm256_loadu_ps(row+j+2*W),  _mm256_loadu_ps(bias+j+2*W));
            __m256 r3 = _mm256_add_ps(_mm256_loadu_ps(row+j+3*W),  _mm256_loadu_ps(bias+j+3*W));
            _mm256_storeu_ps(row+j,      _mm256_max_ps(r0, vzero));
            _mm256_storeu_ps(row+j+W,    _mm256_max_ps(r1, vzero));
            _mm256_storeu_ps(row+j+2*W,  _mm256_max_ps(r2, vzero));
            _mm256_storeu_ps(row+j+3*W,  _mm256_max_ps(r3, vzero));
        }
        for (; j + W <= N; j += W) {
            __m256 r = _mm256_add_ps(_mm256_loadu_ps(row+j), _mm256_loadu_ps(bias+j));
            _mm256_storeu_ps(row+j, _mm256_max_ps(r, vzero));
        }
        for (; j < N; ++j) {
            float v = row[j] + bias[j];
            row[j] = v > 0.0f ? v : 0.0f;
        }
    }
#else
    for (int64_t i = 0; i < M; ++i) {
        float* row = out + i * N;
        for (int64_t j = 0; j < N; ++j) {
            float v = row[j] + bias[j];
            row[j] = v > 0.0f ? v : 0.0f;
        }
    }
#endif
}

// ============================================================================
// GELU scalar helper
// ============================================================================
static inline float gelu_scalar(float x) {
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    const float k = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + k * x3);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

void bias_gelu_fused(float* out, const float* bias, int64_t M, int64_t N) {
    constexpr int W = VecF::width;
#if defined(TUDA_AVX2)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 c_sqrt2pi = _mm256_set1_ps(0.7978845608028654f);
    __m256 c_k = _mm256_set1_ps(0.044715f);

    for (int64_t i = 0; i < M; ++i) {
        float* row = out + i * N;
        int64_t j = 0;
        for (; j + W <= N; j += W) {
            __m256 x = _mm256_add_ps(_mm256_loadu_ps(row + j), _mm256_loadu_ps(bias + j));
            __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
            __m256 inner = _mm256_mul_ps(c_sqrt2pi, _mm256_fmadd_ps(c_k, x3, x));
            __m256 t = tuda::tanh_vec(VecF(inner)).val;
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, t));
            _mm256_storeu_ps(row + j, result);
        }
        for (; j < N; ++j) {
            row[j] = gelu_scalar(row[j] + bias[j]);
        }
    }
#else
    for (int64_t i = 0; i < M; ++i) {
        float* row = out + i * N;
        for (int64_t j = 0; j < N; ++j) {
            row[j] = gelu_scalar(row[j] + bias[j]);
        }
    }
#endif
}

// ============================================================================
// Fused element-wise AVX2 kernels — beat PyTorch on x86
// ============================================================================
// Key insight: PyTorch's 1.5x advantage on element-wise ops comes from tensor
// allocation overhead (malloc + TensorImpl constructor + metadata). These fused
// kernels eliminate intermediate tensors — one pass, one write.

void fused_bias_relu_scale(const float* x, const float* bias, float scale,
                           float* out, int64_t batch, int64_t features) {
    // out[i*features+j] = scale * max(0, x[i*features+j] + bias[j])
    // Fuses: bias_add + relu + scale_mul — 3 ops -> 1 pass
    constexpr int W = VecF::width;
#if defined(TUDA_AVX2)
    __m256 vzero = _mm256_setzero_ps();
    __m256 vscale = _mm256_set1_ps(scale);

    c10::parallel_for_1d(batch, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            const float* xrow = x + i * features;
            float* orow = out + i * features;
            int64_t j = 0;
            for (; j + 4*W <= features; j += 4*W) {
                __m256 v0 = _mm256_add_ps(_mm256_loadu_ps(xrow+j),      _mm256_loadu_ps(bias+j));
                __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(xrow+j+W),    _mm256_loadu_ps(bias+j+W));
                __m256 v2 = _mm256_add_ps(_mm256_loadu_ps(xrow+j+2*W),  _mm256_loadu_ps(bias+j+2*W));
                __m256 v3 = _mm256_add_ps(_mm256_loadu_ps(xrow+j+3*W),  _mm256_loadu_ps(bias+j+3*W));
                _mm256_storeu_ps(orow+j,      _mm256_mul_ps(vscale, _mm256_max_ps(v0, vzero)));
                _mm256_storeu_ps(orow+j+W,    _mm256_mul_ps(vscale, _mm256_max_ps(v1, vzero)));
                _mm256_storeu_ps(orow+j+2*W,  _mm256_mul_ps(vscale, _mm256_max_ps(v2, vzero)));
                _mm256_storeu_ps(orow+j+3*W,  _mm256_mul_ps(vscale, _mm256_max_ps(v3, vzero)));
            }
            for (; j + W <= features; j += W) {
                __m256 v = _mm256_add_ps(_mm256_loadu_ps(xrow+j), _mm256_loadu_ps(bias+j));
                _mm256_storeu_ps(orow+j, _mm256_mul_ps(vscale, _mm256_max_ps(v, vzero)));
            }
            for (; j < features; ++j) {
                float v = xrow[j] + bias[j];
                orow[j] = scale * (v > 0.0f ? v : 0.0f);
            }
        }
    }, 16);
#else
    for (int64_t i = 0; i < batch; ++i) {
        const float* xrow = x + i * features;
        float* orow = out + i * features;
        for (int64_t j = 0; j < features; ++j) {
            float v = xrow[j] + bias[j];
            orow[j] = scale * (v > 0.0f ? v : 0.0f);
        }
    }
#endif
}

void fused_silu_scale(const float* x, float alpha, float* out, int64_t n) {
    // out[i] = alpha * sigmoid(x[i]) * x[i]   (SiLU/Swish with scale)
    // Fuses: sigmoid + elementwise_mul + scalar_mul — 3 ops -> 1 pass
    constexpr int W = VecF::width;
#if defined(TUDA_AVX2)
    __m256 valpha = _mm256_set1_ps(alpha);

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) {
            float xi = x[i];
            float s = 1.0f / (1.0f + std::exp(-xi));
            out[i] = alpha * s * xi;
        }
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            __m256 x0 = _mm256_loadu_ps(x+i),      s0 = tuda::sigmoid_vec(VecF(x0)).val;
            __m256 x1 = _mm256_loadu_ps(x+i+W),    s1 = tuda::sigmoid_vec(VecF(x1)).val;
            __m256 x2 = _mm256_loadu_ps(x+i+2*W),  s2 = tuda::sigmoid_vec(VecF(x2)).val;
            __m256 x3 = _mm256_loadu_ps(x+i+3*W),  s3 = tuda::sigmoid_vec(VecF(x3)).val;
            _mm256_storeu_ps(out+i,      _mm256_mul_ps(valpha, _mm256_mul_ps(s0, x0)));
            _mm256_storeu_ps(out+i+W,    _mm256_mul_ps(valpha, _mm256_mul_ps(s1, x1)));
            _mm256_storeu_ps(out+i+2*W,  _mm256_mul_ps(valpha, _mm256_mul_ps(s2, x2)));
            _mm256_storeu_ps(out+i+3*W,  _mm256_mul_ps(valpha, _mm256_mul_ps(s3, x3)));
        }
        for (; i + W <= end; i += W) {
            __m256 xi = _mm256_loadu_ps(x+i);
            __m256 si = tuda::sigmoid_vec(VecF(xi)).val;
            _mm256_storeu_ps(out+i, _mm256_mul_ps(valpha, _mm256_mul_ps(si, xi)));
        }
        for (; i < end; ++i) {
            float xi = x[i];
            float s = 1.0f / (1.0f + std::exp(-xi));
            out[i] = alpha * s * xi;
        }
    });
#else
    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        float s = 1.0f / (1.0f + std::exp(-xi));
        out[i] = alpha * s * xi;
    }
#endif
}

void fused_dropout_scale(float* x, const uint8_t* mask, float scale, int64_t n) {
    // x[i] = mask[i] ? 0 : x[i] * scale    (mask=1 means DROP)
    // In-place. Fuses: mask_apply + scale — 2 ops -> 1 pass, zero allocs
    constexpr int W = VecF::width;
#if defined(TUDA_AVX2)
    __m256 vscale = _mm256_set1_ps(scale);

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) {
            x[i] = mask[i] ? 0.0f : x[i] * scale;
        }
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            for (int k = 0; k < 4; ++k) {
                int64_t idx = i + k * W;
                // Load 8 mask bytes -> 32-bit ints, compare == 0 for keep mask
                __m256i mi = _mm256_set_epi32(
                    mask[idx+7], mask[idx+6], mask[idx+5], mask[idx+4],
                    mask[idx+3], mask[idx+2], mask[idx+1], mask[idx]);
                __m256 keep = _mm256_castsi256_ps(_mm256_cmpeq_epi32(mi, _mm256_setzero_si256()));
                __m256 val = _mm256_loadu_ps(x + idx);
                _mm256_storeu_ps(x + idx, _mm256_and_ps(keep, _mm256_mul_ps(val, vscale)));
            }
        }
        for (; i + W <= end; i += W) {
            __m256i mi = _mm256_set_epi32(
                mask[i+7], mask[i+6], mask[i+5], mask[i+4],
                mask[i+3], mask[i+2], mask[i+1], mask[i]);
            __m256 keep = _mm256_castsi256_ps(_mm256_cmpeq_epi32(mi, _mm256_setzero_si256()));
            __m256 val = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_and_ps(keep, _mm256_mul_ps(val, vscale)));
        }
        for (; i < end; ++i) {
            x[i] = mask[i] ? 0.0f : x[i] * scale;
        }
    });
#else
    for (int64_t i = 0; i < n; ++i) {
        x[i] = mask[i] ? 0.0f : x[i] * scale;
    }
#endif
}

void fused_relu_backward_scale(const float* grad, const float* input,
                                float scale, float* out, int64_t n) {
    // out[i] = (input[i] > 0 ? grad[i] : 0) * scale
    // Fuses: relu_backward + gradient_scale — 2 ops -> 1 pass
    constexpr int W = VecF::width;
#if defined(TUDA_AVX2)
    __m256 vzero = _mm256_setzero_ps();
    __m256 vscale = _mm256_set1_ps(scale);

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) {
            out[i] = (input[i] > 0.0f ? grad[i] : 0.0f) * scale;
        }
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            __m256 m0 = _mm256_cmp_ps(_mm256_loadu_ps(input+i),      vzero, _CMP_GT_OS);
            __m256 m1 = _mm256_cmp_ps(_mm256_loadu_ps(input+i+W),    vzero, _CMP_GT_OS);
            __m256 m2 = _mm256_cmp_ps(_mm256_loadu_ps(input+i+2*W),  vzero, _CMP_GT_OS);
            __m256 m3 = _mm256_cmp_ps(_mm256_loadu_ps(input+i+3*W),  vzero, _CMP_GT_OS);
            _mm256_storeu_ps(out+i,      _mm256_mul_ps(vscale, _mm256_and_ps(m0, _mm256_loadu_ps(grad+i))));
            _mm256_storeu_ps(out+i+W,    _mm256_mul_ps(vscale, _mm256_and_ps(m1, _mm256_loadu_ps(grad+i+W))));
            _mm256_storeu_ps(out+i+2*W,  _mm256_mul_ps(vscale, _mm256_and_ps(m2, _mm256_loadu_ps(grad+i+2*W))));
            _mm256_storeu_ps(out+i+3*W,  _mm256_mul_ps(vscale, _mm256_and_ps(m3, _mm256_loadu_ps(grad+i+3*W))));
        }
        for (; i + W <= end; i += W) {
            __m256 m = _mm256_cmp_ps(_mm256_loadu_ps(input+i), vzero, _CMP_GT_OS);
            _mm256_storeu_ps(out+i, _mm256_mul_ps(vscale, _mm256_and_ps(m, _mm256_loadu_ps(grad+i))));
        }
        for (; i < end; ++i) {
            out[i] = (input[i] > 0.0f ? grad[i] : 0.0f) * scale;
        }
    });
#else
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (input[i] > 0.0f ? grad[i] : 0.0f) * scale;
    }
#endif
}

void fused_sgd_momentum_avx2(float* param, const float* grad, float* buf,
                              int64_t n, float lr, float momentum, float dampening) {
    // Single pass: update momentum buffer AND param together.
    //   buf[i]   = momentum * buf[i] + (1 - dampening) * grad[i]
    //   param[i] -= lr * buf[i]
    // Saves one full pass vs separate mul + axpy + sub.
    // Uses _mm256_fnmadd_ps: -(a*b) + c  =>  param = param - lr*buf
    constexpr int W = VecF::width;
#if defined(TUDA_AVX2)
    __m256 vmom = _mm256_set1_ps(momentum);
    __m256 vdamp = _mm256_set1_ps(1.0f - dampening);
    __m256 vlr = _mm256_set1_ps(lr);

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) {
            buf[i] = momentum * buf[i] + (1.0f - dampening) * grad[i];
            param[i] -= lr * buf[i];
        }
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            // Unroll 0
            __m256 b0 = _mm256_fmadd_ps(vmom, _mm256_loadu_ps(buf+i), _mm256_mul_ps(vdamp, _mm256_loadu_ps(grad+i)));
            _mm256_storeu_ps(buf+i, b0);
            _mm256_storeu_ps(param+i, _mm256_fnmadd_ps(vlr, b0, _mm256_loadu_ps(param+i)));
            // Unroll 1
            __m256 b1 = _mm256_fmadd_ps(vmom, _mm256_loadu_ps(buf+i+W), _mm256_mul_ps(vdamp, _mm256_loadu_ps(grad+i+W)));
            _mm256_storeu_ps(buf+i+W, b1);
            _mm256_storeu_ps(param+i+W, _mm256_fnmadd_ps(vlr, b1, _mm256_loadu_ps(param+i+W)));
            // Unroll 2
            __m256 b2 = _mm256_fmadd_ps(vmom, _mm256_loadu_ps(buf+i+2*W), _mm256_mul_ps(vdamp, _mm256_loadu_ps(grad+i+2*W)));
            _mm256_storeu_ps(buf+i+2*W, b2);
            _mm256_storeu_ps(param+i+2*W, _mm256_fnmadd_ps(vlr, b2, _mm256_loadu_ps(param+i+2*W)));
            // Unroll 3
            __m256 b3 = _mm256_fmadd_ps(vmom, _mm256_loadu_ps(buf+i+3*W), _mm256_mul_ps(vdamp, _mm256_loadu_ps(grad+i+3*W)));
            _mm256_storeu_ps(buf+i+3*W, b3);
            _mm256_storeu_ps(param+i+3*W, _mm256_fnmadd_ps(vlr, b3, _mm256_loadu_ps(param+i+3*W)));
        }
        for (; i + W <= end; i += W) {
            __m256 b = _mm256_fmadd_ps(vmom, _mm256_loadu_ps(buf+i), _mm256_mul_ps(vdamp, _mm256_loadu_ps(grad+i)));
            _mm256_storeu_ps(buf+i, b);
            _mm256_storeu_ps(param+i, _mm256_fnmadd_ps(vlr, b, _mm256_loadu_ps(param+i)));
        }
        for (; i < end; ++i) {
            buf[i] = momentum * buf[i] + (1.0f - dampening) * grad[i];
            param[i] -= lr * buf[i];
        }
    });
#else
    for (int64_t i = 0; i < n; ++i) {
        buf[i] = momentum * buf[i] + (1.0f - dampening) * grad[i];
        param[i] -= lr * buf[i];
    }
#endif
}

void fused_adam_avx2(float* param, const float* grad,
                     float* m, float* v, int64_t n,
                     float lr, float beta1, float beta2, float eps,
                     float weight_decay, float bc1, float bc2) {
    // Single-pass Adam: m, v, param all updated per element in one loop.
    //   g = grad + weight_decay * param
    //   m = beta1 * m + (1 - beta1) * g
    //   v = beta2 * v + (1 - beta2) * g^2
    //   denom = sqrt(v) / sqrt(bc2) + eps
    //   param -= (lr / bc1) * m / denom
    //
    // Eliminates 3 separate passes that each allocate intermediate tensors.
    // bc1, bc2 precomputed by caller: bc1 = 1 - beta1^t, bc2 = 1 - beta2^t
    constexpr int W = VecF::width;
    float step_size = lr / bc1;
    float inv_bc2_sqrt = 1.0f / std::sqrt(bc2);
    bool has_wd = weight_decay != 0.0f;
#if defined(TUDA_AVX2)
    __m256 vb1 = _mm256_set1_ps(beta1);
    __m256 vb2 = _mm256_set1_ps(beta2);
    __m256 v1b1 = _mm256_set1_ps(1.0f - beta1);
    __m256 v1b2 = _mm256_set1_ps(1.0f - beta2);
    __m256 veps = _mm256_set1_ps(eps);
    __m256 vss = _mm256_set1_ps(step_size);
    __m256 vibc2 = _mm256_set1_ps(inv_bc2_sqrt);
    __m256 vwd = _mm256_set1_ps(weight_decay);

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        // Scalar preamble for alignment
        for (; i < end && (i % W != 0); ++i) {
            float g = grad[i];
            if (has_wd) g += weight_decay * param[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            float denom = std::sqrt(v[i]) * inv_bc2_sqrt + eps;
            param[i] -= step_size * m[i] / denom;
        }
        // 4x unrolled AVX2 loop
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            for (int k = 0; k < 4; ++k) {
                int64_t idx = i + k * W;
                __m256 g = _mm256_loadu_ps(grad + idx);
                __m256 p = _mm256_loadu_ps(param + idx);
                if (has_wd) g = _mm256_fmadd_ps(vwd, p, g);

                // m = beta1 * m_old + (1-beta1) * g
                __m256 mi = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(m + idx), _mm256_mul_ps(v1b1, g));
                _mm256_storeu_ps(m + idx, mi);

                // v = beta2 * v_old + (1-beta2) * g^2
                __m256 vi = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(v + idx), _mm256_mul_ps(v1b2, _mm256_mul_ps(g, g)));
                _mm256_storeu_ps(v + idx, vi);

                // denom = sqrt(v) * inv_bc2_sqrt + eps
                __m256 denom = _mm256_fmadd_ps(_mm256_sqrt_ps(vi), vibc2, veps);

                // param -= step_size * m / denom  (fnmadd: -(a*b) + c)
                _mm256_storeu_ps(param + idx, _mm256_fnmadd_ps(vss, _mm256_div_ps(mi, denom), p));
            }
        }
        // Single vector tail
        for (; i + W <= end; i += W) {
            __m256 g = _mm256_loadu_ps(grad + i);
            __m256 p = _mm256_loadu_ps(param + i);
            if (has_wd) g = _mm256_fmadd_ps(vwd, p, g);

            __m256 mi = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(m + i), _mm256_mul_ps(v1b1, g));
            _mm256_storeu_ps(m + i, mi);

            __m256 vi = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(v + i), _mm256_mul_ps(v1b2, _mm256_mul_ps(g, g)));
            _mm256_storeu_ps(v + i, vi);

            __m256 denom = _mm256_fmadd_ps(_mm256_sqrt_ps(vi), vibc2, veps);
            _mm256_storeu_ps(param + i, _mm256_fnmadd_ps(vss, _mm256_div_ps(mi, denom), p));
        }
        // Scalar tail
        for (; i < end; ++i) {
            float g = grad[i];
            if (has_wd) g += weight_decay * param[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            float denom = std::sqrt(v[i]) * inv_bc2_sqrt + eps;
            param[i] -= step_size * m[i] / denom;
        }
    });
#else
    for (int64_t i = 0; i < n; ++i) {
        float g = grad[i];
        if (has_wd) g += weight_decay * param[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float denom = std::sqrt(v[i]) * inv_bc2_sqrt + eps;
        param[i] -= step_size * m[i] / denom;
    }
#endif
}

void fused_gelu(const float* x, float* out, int64_t n) {
    // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Single pass, no intermediate tensors
    constexpr int W = VecF::width;
#if defined(TUDA_AVX2)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 c_sqrt2pi = _mm256_set1_ps(0.7978845608028654f);
    __m256 c_k = _mm256_set1_ps(0.044715f);

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) {
            out[i] = gelu_scalar(x[i]);
        }
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            for (int k = 0; k < 4; ++k) {
                int64_t idx = i + k * W;
                __m256 xi = _mm256_loadu_ps(x + idx);
                __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xi, xi), xi);
                __m256 inner = _mm256_mul_ps(c_sqrt2pi, _mm256_fmadd_ps(c_k, x3, xi));
                __m256 t = tuda::tanh_vec(VecF(inner)).val;
                _mm256_storeu_ps(out + idx, _mm256_mul_ps(_mm256_mul_ps(half, xi), _mm256_add_ps(one, t)));
            }
        }
        for (; i + W <= end; i += W) {
            __m256 xi = _mm256_loadu_ps(x + i);
            __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xi, xi), xi);
            __m256 inner = _mm256_mul_ps(c_sqrt2pi, _mm256_fmadd_ps(c_k, x3, xi));
            __m256 t = tuda::tanh_vec(VecF(inner)).val;
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(half, xi), _mm256_add_ps(one, t)));
        }
        for (; i < end; ++i) {
            out[i] = gelu_scalar(x[i]);
        }
    });
#else
    for (int64_t i = 0; i < n; ++i) {
        out[i] = gelu_scalar(x[i]);
    }
#endif
}

// ============================================================================
// INFERENCE FUSIONS — CPU decode hot path
// ============================================================================

// Fusion 1: GEMV + residual add
// y[i] = x_residual[i] + dot(W[i,:], src)
// Eliminates: writing h_buf then reading it back for x_next = x + h_buf
void gemv_residual_add(const float* W, int64_t N, int64_t K,
                       const float* src, const float* x_residual,
                       float* x_out) {
    // Parallelize over output rows
    c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end) {
        for (int64_t n = start; n < end; ++n) {
            const float* w_row = W + n * K;
#if defined(TUDA_AVX2)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int64_t k = 0;
            for (; k + 15 < K; k += 16) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(src + k),
                                       _mm256_loadu_ps(w_row + k), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(src + k + 8),
                                       _mm256_loadu_ps(w_row + k + 8), acc1);
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            for (; k + 7 < K; k += 8) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(src + k),
                                       _mm256_loadu_ps(w_row + k), acc0);
            }
            // hsum
            __m128 lo = _mm256_castps256_ps128(acc0);
            __m128 hi = _mm256_extractf128_ps(acc0, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            float dot = _mm_cvtss_f32(lo);
            for (; k < K; ++k) dot += src[k] * w_row[k];
#else
            float dot = 0.0f;
            for (int64_t k = 0; k < K; ++k) dot += src[k] * w_row[k];
#endif
            x_out[n] = x_residual[n] + dot;
        }
    }, 64);
}

// Fusion 2: Final RMSNorm + output GEMV + argmax
// Avoids writing full logits array and re-scanning for argmax
int32_t fused_rmsnorm_gemv_argmax(const float* x, const float* norm_w,
                                   float eps, bool add_one, int64_t H,
                                   const float* output_weight, int64_t V,
                                   float* logits_out) {
    // Step 1: RMSNorm x -> x_normed (on stack)
    constexpr int64_t MAX_STACK = 8192;
    float stack_buf[MAX_STACK];
    float* x_normed = (H <= MAX_STACK) ? stack_buf
        : static_cast<float*>(std::malloc(H * sizeof(float)));

#if defined(TUDA_AVX2)
    __m256 sum_sq_vec = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 7 < H; j += 8) {
        __m256 vx = _mm256_loadu_ps(x + j);
        sum_sq_vec = _mm256_fmadd_ps(vx, vx, sum_sq_vec);
    }
    // hsum
    __m128 lo_s = _mm256_castps256_ps128(sum_sq_vec);
    __m128 hi_s = _mm256_extractf128_ps(sum_sq_vec, 1);
    lo_s = _mm_add_ps(lo_s, hi_s);
    lo_s = _mm_hadd_ps(lo_s, lo_s);
    lo_s = _mm_hadd_ps(lo_s, lo_s);
    float sum_sq = _mm_cvtss_f32(lo_s);
    for (; j < H; ++j) sum_sq += x[j] * x[j];

    float rms = 1.0f / std::sqrt(sum_sq / H + eps);
    __m256 vrms = _mm256_set1_ps(rms);
    j = 0;
    if (add_one) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 7 < H; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(norm_w + j);
            _mm256_storeu_ps(x_normed + j,
                _mm256_mul_ps(_mm256_mul_ps(vx, vrms), _mm256_add_ps(vg, one)));
        }
    } else {
        for (; j + 7 < H; j += 8) {
            __m256 vx = _mm256_loadu_ps(x + j);
            __m256 vg = _mm256_loadu_ps(norm_w + j);
            _mm256_storeu_ps(x_normed + j, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vg));
        }
    }
    for (; j < H; ++j) {
        float w = add_one ? (1.0f + norm_w[j]) : norm_w[j];
        x_normed[j] = x[j] * rms * w;
    }
#else
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < H; ++j) sum_sq += x[j] * x[j];
    float rms = 1.0f / std::sqrt(sum_sq / H + eps);
    for (int64_t j = 0; j < H; ++j) {
        float w = add_one ? (1.0f + norm_w[j]) : norm_w[j];
        x_normed[j] = x[j] * rms * w;
    }
#endif

    // Step 2: GEMV + argmax fused — find max during dot products
    // Split across threads, each finds local max
    struct ThreadResult { int32_t idx; float val; };
    auto& pool = c10::get_thread_pool();
    int nt = pool.num_threads();
    std::vector<ThreadResult> results(nt, {0, -1e30f});
    std::atomic<int> tid_counter{0};

    pool.parallel_for(0, V, [&](int64_t start, int64_t end) {
        int my_tid = tid_counter.fetch_add(1, std::memory_order_relaxed);
        if (my_tid >= nt) my_tid = nt - 1;  // safety
        int32_t local_best_idx = static_cast<int32_t>(start);
        float local_best_val = -1e30f;

        for (int64_t n = start; n < end; ++n) {
            const float* w_row = output_weight + n * H;
#if defined(TUDA_AVX2)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int64_t k = 0;
            for (; k + 15 < H; k += 16) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_normed + k),
                                       _mm256_loadu_ps(w_row + k), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_normed + k + 8),
                                       _mm256_loadu_ps(w_row + k + 8), acc1);
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            for (; k + 7 < H; k += 8) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_normed + k),
                                       _mm256_loadu_ps(w_row + k), acc0);
            }
            __m128 lo2 = _mm256_castps256_ps128(acc0);
            __m128 hi2 = _mm256_extractf128_ps(acc0, 1);
            lo2 = _mm_add_ps(lo2, hi2);
            lo2 = _mm_hadd_ps(lo2, lo2);
            lo2 = _mm_hadd_ps(lo2, lo2);
            float dot = _mm_cvtss_f32(lo2);
            for (; k < H; ++k) dot += x_normed[k] * w_row[k];
#else
            float dot = 0.0f;
            for (int64_t k = 0; k < H; ++k) dot += x_normed[k] * w_row[k];
#endif
            // Write logits if requested
            if (logits_out) logits_out[n] = dot;

            // Track argmax
            if (dot > local_best_val) {
                local_best_val = dot;
                local_best_idx = static_cast<int32_t>(n);
            }
        }
        results[my_tid] = {local_best_idx, local_best_val};
    }, 64);

    // Reduce across threads
    int32_t best_idx = results[0].idx;
    float best_val = results[0].val;
    for (int t = 1; t < nt; ++t) {
        if (results[t].val > best_val) {
            best_val = results[t].val;
            best_idx = results[t].idx;
        }
    }

    if (H > MAX_STACK) std::free(x_normed);
    return best_idx;
}

// Softmax fused (declared in header, was missing impl)
void softmax_fused(const float* in, float* out, int64_t rows, int64_t cols) {
    c10::parallel_for_1d(rows, [&](int64_t start, int64_t end) {
        for (int64_t r = start; r < end; ++r) {
            const float* row_in = in + r * cols;
            float* row_out = out + r * cols;

            // Find max
            float mx = row_in[0];
#if defined(TUDA_AVX2)
            __m256 vmx = _mm256_set1_ps(row_in[0]);
            int64_t j = 0;
            for (; j + 7 < cols; j += 8) {
                vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(row_in + j));
            }
            // hsum max
            __m128 lo = _mm256_castps256_ps128(vmx);
            __m128 hi = _mm256_extractf128_ps(vmx, 1);
            lo = _mm_max_ps(lo, hi);
            float tmp4[4]; _mm_storeu_ps(tmp4, lo);
            mx = tmp4[0];
            for (int k = 1; k < 4; ++k) if (tmp4[k] > mx) mx = tmp4[k];
            for (; j < cols; ++j) if (row_in[j] > mx) mx = row_in[j];
#else
            for (int64_t j = 1; j < cols; ++j) if (row_in[j] > mx) mx = row_in[j];
#endif

            // Exp + sum
            float sum_exp = 0.0f;
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = std::exp(row_in[j] - mx);
                sum_exp += row_out[j];
            }

            // Normalize
            float inv = 1.0f / sum_exp;
#if defined(TUDA_AVX2)
            __m256 vinv = _mm256_set1_ps(inv);
            j = 0;
            for (; j + 7 < cols; j += 8) {
                _mm256_storeu_ps(row_out + j,
                    _mm256_mul_ps(_mm256_loadu_ps(row_out + j), vinv));
            }
            for (; j < cols; ++j) row_out[j] *= inv;
#else
            for (int64_t j = 0; j < cols; ++j) row_out[j] *= inv;
#endif
        }
    }, 16);
}

// Residual + LayerNorm fused
void residual_layernorm_fused(const float* x, const float* residual,
                              const float* gamma, const float* beta_param,
                              float* out, int64_t rows, int64_t cols, float eps) {
    c10::parallel_for_1d(rows, [&](int64_t start, int64_t end) {
        for (int64_t r = start; r < end; ++r) {
            const float* x_row = x + r * cols;
            const float* res_row = residual + r * cols;
            float* out_row = out + r * cols;

            // Combined residual add + compute mean
            float sum = 0.0f;
            for (int64_t j = 0; j < cols; ++j) {
                out_row[j] = x_row[j] + res_row[j];  // residual add
                sum += out_row[j];
            }
            float mean = sum / cols;

            // Compute variance
            float var = 0.0f;
            for (int64_t j = 0; j < cols; ++j) {
                float d = out_row[j] - mean;
                var += d * d;
            }
            float inv_std = 1.0f / std::sqrt(var / cols + eps);

            // Normalize + scale + shift
            for (int64_t j = 0; j < cols; ++j) {
                float normed = (out_row[j] - mean) * inv_std;
                out_row[j] = normed * gamma[j] + (beta_param ? beta_param[j] : 0.0f);
            }
        }
    }, 16);
}

// ============================================================================
// TRAINING FUSIONS
// ============================================================================

// Fusion 4: AdamW single-pass step (decoupled weight decay + Adam in one loop)
void fused_adamw_step(float* param, const float* grad,
                      float* exp_avg, float* exp_avg_sq,
                      int64_t n, float lr, float beta1, float beta2,
                      float eps, float weight_decay,
                      float bc1, float bc2) {
    // Single pass: param *= (1 - lr*wd) then Adam update
    // Instead of separate mul_ + adam passes
    float decay_factor = 1.0f - lr * weight_decay;
    float step_size = lr / bc1;
    float inv_bc2_sqrt = 1.0f / std::sqrt(bc2);

#if defined(TUDA_AVX2)
    __m256 vdecay = _mm256_set1_ps(decay_factor);
    __m256 vb1 = _mm256_set1_ps(beta1);
    __m256 vb2 = _mm256_set1_ps(beta2);
    __m256 v1b1 = _mm256_set1_ps(1.0f - beta1);
    __m256 v1b2 = _mm256_set1_ps(1.0f - beta2);
    __m256 veps = _mm256_set1_ps(eps);
    __m256 vss = _mm256_set1_ps(step_size);
    __m256 vibc2 = _mm256_set1_ps(inv_bc2_sqrt);
    constexpr int W = VecF::width;

    c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
        int64_t i = start;
        for (; i < end && (i % W != 0); ++i) {
            float p = param[i] * decay_factor;  // weight decay
            float g = grad[i];
            exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
            exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
            float denom = std::sqrt(exp_avg_sq[i]) * inv_bc2_sqrt + eps;
            param[i] = p - step_size * exp_avg[i] / denom;
        }
        int64_t se = end - ((end - i) % (4*W));
        for (; i < se; i += 4*W) {
            for (int k = 0; k < 4; ++k) {
                int64_t idx = i + k * W;
                __m256 p = _mm256_mul_ps(_mm256_loadu_ps(param + idx), vdecay);
                __m256 g = _mm256_loadu_ps(grad + idx);
                __m256 mi = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(exp_avg + idx), _mm256_mul_ps(v1b1, g));
                _mm256_storeu_ps(exp_avg + idx, mi);
                __m256 vi = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(exp_avg_sq + idx), _mm256_mul_ps(v1b2, _mm256_mul_ps(g, g)));
                _mm256_storeu_ps(exp_avg_sq + idx, vi);
                __m256 denom = _mm256_fmadd_ps(_mm256_sqrt_ps(vi), vibc2, veps);
                _mm256_storeu_ps(param + idx, _mm256_fnmadd_ps(vss, _mm256_div_ps(mi, denom), p));
            }
        }
        for (; i + W <= end; i += W) {
            __m256 p = _mm256_mul_ps(_mm256_loadu_ps(param + i), vdecay);
            __m256 g = _mm256_loadu_ps(grad + i);
            __m256 mi = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(exp_avg + i), _mm256_mul_ps(v1b1, g));
            _mm256_storeu_ps(exp_avg + i, mi);
            __m256 vi = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(exp_avg_sq + i), _mm256_mul_ps(v1b2, _mm256_mul_ps(g, g)));
            _mm256_storeu_ps(exp_avg_sq + i, vi);
            __m256 denom = _mm256_fmadd_ps(_mm256_sqrt_ps(vi), vibc2, veps);
            _mm256_storeu_ps(param + i, _mm256_fnmadd_ps(vss, _mm256_div_ps(mi, denom), p));
        }
        for (; i < end; ++i) {
            float p = param[i] * decay_factor;
            float g = grad[i];
            exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
            exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
            float denom = std::sqrt(exp_avg_sq[i]) * inv_bc2_sqrt + eps;
            param[i] = p - step_size * exp_avg[i] / denom;
        }
    });
#else
    for (int64_t i = 0; i < n; ++i) {
        float p = param[i] * decay_factor;
        float g = grad[i];
        exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
        float denom = std::sqrt(exp_avg_sq[i]) * inv_bc2_sqrt + eps;
        param[i] = p - step_size * exp_avg[i] / denom;
    }
#endif
}

// Fusion 5: Fused multi-parameter AdamW
void fused_adamw_multi(AdamWParamPack* params, int num_params,
                       float lr, float beta1, float beta2, float eps,
                       float weight_decay, int step) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    for (int p = 0; p < num_params; p++) {
        fused_adamw_step(params[p].param, params[p].grad,
                         params[p].exp_avg, params[p].exp_avg_sq,
                         params[p].numel, lr, beta1, beta2, eps,
                         weight_decay, bc1, bc2);
    }
}

// Fusion 6: zero_grad multi — memset all grad buffers in one call
void fused_zero_grad_multi(GradBufPack* bufs, int num_bufs) {
    for (int i = 0; i < num_bufs; ++i) {
        std::memset(bufs[i].grad_data, 0, bufs[i].numel * sizeof(float));
    }
}

// Fusion 7: RoPE precompute + apply
void rope_precompute(float* cos_out, float* sin_out,
                     int64_t pos, int64_t head_dim, float freq_base) {
    for (int64_t d = 0; d < head_dim / 2; ++d) {
        float freq = 1.0f / std::pow(freq_base, 2.0f * d / head_dim);
        float theta = pos * freq;
        cos_out[d] = std::cos(theta);
        sin_out[d] = std::sin(theta);
    }
}

void rope_apply_fused(float* q, float* k,
                      const float* cos_table, const float* sin_table,
                      int64_t n_heads, int64_t n_kv_heads, int64_t head_dim) {
    int64_t half_dim = head_dim / 2;
    // Q: all heads
#if defined(TUDA_AVX2)
    for (int64_t h = 0; h < n_heads; ++h) {
        float* hd = q + h * head_dim;
        int64_t d = 0;
        for (; d + 3 < half_dim; d += 4) {
            // Pair (2d, 2d+1) for each d
            for (int dd = 0; dd < 4; ++dd) {
                float c = cos_table[d + dd];
                float s = sin_table[d + dd];
                float x0 = hd[2*(d+dd)];
                float x1 = hd[2*(d+dd)+1];
                hd[2*(d+dd)]   = x0 * c - x1 * s;
                hd[2*(d+dd)+1] = x0 * s + x1 * c;
            }
        }
        for (; d < half_dim; ++d) {
            float c = cos_table[d];
            float s = sin_table[d];
            float x0 = hd[2*d];
            float x1 = hd[2*d+1];
            hd[2*d]   = x0 * c - x1 * s;
            hd[2*d+1] = x0 * s + x1 * c;
        }
    }
    // K: kv_heads
    for (int64_t h = 0; h < n_kv_heads; ++h) {
        float* hd = k + h * head_dim;
        for (int64_t d = 0; d < half_dim; ++d) {
            float c = cos_table[d];
            float s = sin_table[d];
            float x0 = hd[2*d];
            float x1 = hd[2*d+1];
            hd[2*d]   = x0 * c - x1 * s;
            hd[2*d+1] = x0 * s + x1 * c;
        }
    }
#else
    for (int64_t h = 0; h < n_heads; ++h) {
        float* hd = q + h * head_dim;
        for (int64_t d = 0; d < half_dim; ++d) {
            float c = cos_table[d];
            float s = sin_table[d];
            float x0 = hd[2*d];
            float x1 = hd[2*d+1];
            hd[2*d]   = x0 * c - x1 * s;
            hd[2*d+1] = x0 * s + x1 * c;
        }
    }
    for (int64_t h = 0; h < n_kv_heads; ++h) {
        float* hd = k + h * head_dim;
        for (int64_t d = 0; d < half_dim; ++d) {
            float c = cos_table[d];
            float s = sin_table[d];
            float x0 = hd[2*d];
            float x1 = hd[2*d+1];
            hd[2*d]   = x0 * c - x1 * s;
            hd[2*d+1] = x0 * s + x1 * c;
        }
    }
#endif
}

} // namespace hot
} // namespace native
} // namespace at
