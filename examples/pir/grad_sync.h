// ============================================================================
// grad_sync.h — Shared memory gradient synchronization for 4-NUMA Elbrus
// ============================================================================
// Data-parallel training: 4 processes × 1 NUMA-node each.
// Each process computes forward+backward independently, then sync_gradients()
// averages gradients across all 4 processes via POSIX shared memory.
// Result: 1 model trained with effective batch = 4 × per-process batch.
// ============================================================================
#pragma once

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct GradSync {
    static constexpr int MAX_PROCS = 4;

    struct SharedControl {
        pthread_barrier_t bar_grads_ready;  // all grads written
        pthread_barrier_t bar_reduce_done;  // reduce complete
        pthread_barrier_t bar_weights_done; // weights synced
        int initialized;
    };

    int rank_ = -1;
    int nprocs_ = 0;
    int64_t total_params_ = 0;
    int64_t chunk_size_ = 0;

    SharedControl* ctrl_ = nullptr;
    float* shm_grad_ = nullptr;  // [nprocs × total_params] shared gradient buffer
    float* shm_weights_ = nullptr;  // [total_params] shared weight buffer (rank 0 writes)

    size_t grad_bytes_ = 0;
    size_t weight_bytes_ = 0;
    int fd_ctrl_ = -1;
    int fd_grad_ = -1;
    int fd_weights_ = -1;

    bool init(int rank, int nprocs, int64_t total_params) {
        rank_ = rank;
        nprocs_ = nprocs;
        total_params_ = total_params;
        chunk_size_ = total_params / nprocs;

        // 1. Control block (barriers)
        fd_ctrl_ = shm_open("/pt_grad_ctrl", O_CREAT | O_RDWR, 0666);
        if (fd_ctrl_ < 0) { perror("shm_open ctrl"); return false; }
        ftruncate(fd_ctrl_, sizeof(SharedControl));
        ctrl_ = (SharedControl*)mmap(nullptr, sizeof(SharedControl),
                                      PROT_READ | PROT_WRITE, MAP_SHARED, fd_ctrl_, 0);

        if (rank == 0) {
            memset(ctrl_, 0, sizeof(SharedControl));
            pthread_barrierattr_t attr;
            pthread_barrierattr_init(&attr);
            pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
            pthread_barrier_init(&ctrl_->bar_grads_ready, &attr, nprocs);
            pthread_barrier_init(&ctrl_->bar_reduce_done, &attr, nprocs);
            pthread_barrier_init(&ctrl_->bar_weights_done, &attr, nprocs);
            ctrl_->initialized = 1;
            pthread_barrierattr_destroy(&attr);
        }

        // Wait for rank 0 to init (simple spin)
        while (!ctrl_->initialized) usleep(1000);

        // 2. Gradient buffer: each process writes its grads to its slice
        grad_bytes_ = (size_t)nprocs * total_params * sizeof(float);
        fd_grad_ = shm_open("/pt_grad_data", O_CREAT | O_RDWR, 0666);
        if (fd_grad_ < 0) { perror("shm_open grad"); return false; }
        ftruncate(fd_grad_, grad_bytes_);
        shm_grad_ = (float*)mmap(nullptr, grad_bytes_,
                                  PROT_READ | PROT_WRITE, MAP_SHARED, fd_grad_, 0);

        // 3. Weight buffer: rank 0 writes, others read
        weight_bytes_ = (size_t)total_params * sizeof(float);
        fd_weights_ = shm_open("/pt_weights", O_CREAT | O_RDWR, 0666);
        if (fd_weights_ < 0) { perror("shm_open weights"); return false; }
        ftruncate(fd_weights_, weight_bytes_);
        shm_weights_ = (float*)mmap(nullptr, weight_bytes_,
                                     PROT_READ | PROT_WRITE, MAP_SHARED, fd_weights_, 0);

        return true;
    }

    // Sync gradients: simple all-average (each process reads all, computes full mean)
    // No complex reduce-scatter/allgather — simpler = correct.
    // Cost: 189M × 4 reads = ~3GB → ~100ms on DDR4 (0.3% of 34s step)
    void sync(float* flat_grads, float* flat_params) {
        // Step 1: Write my gradients to shared memory (my row)
        memcpy(shm_grad_ + (size_t)rank_ * total_params_,
               flat_grads, total_params_ * sizeof(float));

        // Step 2: Wait for all processes to write
        pthread_barrier_wait(&ctrl_->bar_grads_ready);

        // Step 3: Every process computes the FULL average (no allgather needed)
        float inv_n = 1.0f / nprocs_;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < total_params_; i++) {
            float sum = 0.0f;
            for (int p = 0; p < nprocs_; p++) {
                sum += shm_grad_[(size_t)p * total_params_ + i];
            }
            flat_grads[i] = sum * inv_n;
        }

        // Step 4: Wait for all to finish reading (prevents next iter overwriting)
        pthread_barrier_wait(&ctrl_->bar_reduce_done);
    }

    // After Adam: sync weights so all processes have identical params
    void sync_weights(float* flat_params) {
        if (rank_ == 0) {
            memcpy(shm_weights_, flat_params, total_params_ * sizeof(float));
        }
        pthread_barrier_wait(&ctrl_->bar_weights_done);
        if (rank_ != 0) {
            memcpy(flat_params, shm_weights_, total_params_ * sizeof(float));
        }
        // FIX Bug2: second barrier ensures all reads complete before next phase
        pthread_barrier_wait(&ctrl_->bar_weights_done);
    }

    void cleanup() {
        if (shm_grad_) munmap(shm_grad_, grad_bytes_);
        if (shm_weights_) munmap(shm_weights_, weight_bytes_);
        if (ctrl_) {
            if (rank_ == 0) {
                pthread_barrier_destroy(&ctrl_->bar_grads_ready);
                pthread_barrier_destroy(&ctrl_->bar_reduce_done);
                pthread_barrier_destroy(&ctrl_->bar_weights_done);
                shm_unlink("/pt_grad_ctrl");
                shm_unlink("/pt_grad_data");
                shm_unlink("/pt_weights");
            }
            munmap(ctrl_, sizeof(SharedControl));
        }
    }
};

#else
// Non-Linux stub
struct GradSync {
    bool init(int, int, int64_t) { return false; }
    void sync(float*, float*) {}
    void sync_weights(float*) {}
    void cleanup() {}
    int rank_ = 0;
};
#endif
