// ============================================================================
// grad_sync.h — File-based weight sync for Local SGD on Elbrus
// ============================================================================
// No POSIX shared memory, no pthread barriers, no atomics.
// Each process writes weights to /tmp/pir_w_$rank.bin.
// Rank 0 averages all files, writes result.
// Others poll for result file, read it.
// Works on any NUMA configuration without deadlocks.
// ============================================================================
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <unistd.h>

struct GradSync {
    int rank_ = -1;
    int nprocs_ = 0;
    int64_t total_params_ = 0;
    int sync_gen_ = 0;  // generation counter for unique filenames

    bool init(int rank, int nprocs, int64_t total_params) {
        rank_ = rank;
        nprocs_ = nprocs;
        total_params_ = total_params;
        // Clean old files
        if (rank == 0) {
            for (int r = 0; r < nprocs; r++) {
                char f[128];
                snprintf(f, sizeof(f), "/tmp/pir_w_%d.bin", r);
                unlink(f);
            }
            unlink("/tmp/pir_w_avg.bin");
            unlink("/tmp/pir_w_avg.ready");
        }
        // Barrier: wait for rank 0 cleanup
        char init_file[128];
        if (rank == 0) {
            snprintf(init_file, sizeof(init_file), "/tmp/pir_sync_init");
            FILE* f = fopen(init_file, "w");
            if (f) { fprintf(f, "ok"); fclose(f); }
        }
        snprintf(init_file, sizeof(init_file), "/tmp/pir_sync_init");
        while (access(init_file, F_OK) != 0) usleep(1000);
        usleep(10000);  // let rank 0 finish cleanup
        return true;
    }

    // Compute simple hash of float array (XOR of bit-cast ints, every 1000th element)
    // Used to verify all ranks have same weights after sync.
    static uint64_t weights_hash(const float* w, int64_t n) {
        uint64_t h = 0;
        for (int64_t i = 0; i < n; i += 1000) {
            uint32_t bits;
            std::memcpy(&bits, w + i, 4);
            h ^= ((uint64_t)bits << 32) | bits;
            h = (h * 1099511628211ULL) ^ i;
        }
        return h;
    }

    // sync() is used for weight averaging in Local SGD
    void sync(float* flat_weights, float* /*unused*/) {
        sync_gen_++;
        uint64_t h_before = weights_hash(flat_weights, total_params_);

        // 1. Write my weights to file
        char my_file[128];
        snprintf(my_file, sizeof(my_file), "/tmp/pir_w_%d.bin", rank_);
        FILE* f = fopen(my_file, "wb");
        if (f) {
            size_t wrote = fwrite(flat_weights, sizeof(float), total_params_, f);
            fclose(f);
            if ((int64_t)wrote != total_params_) {
                fprintf(stderr, "GradSync rank %d: SHORT WRITE %zu/%lld at gen %d\n",
                        rank_, wrote, (long long)total_params_, sync_gen_);
            }
        } else {
            fprintf(stderr, "GradSync rank %d: fopen FAILED for %s\n", rank_, my_file);
        }

        // Signal: I wrote my weights
        char ready_file[128];
        snprintf(ready_file, sizeof(ready_file), "/tmp/pir_w_%d.ready", rank_);
        f = fopen(ready_file, "w");
        if (f) { fprintf(f, "%d", sync_gen_); fclose(f); }

        // 2. Rank 0: wait for all, average, write result
        if (rank_ == 0) {
            // Wait for all ranks to write (timeout 120s)
            for (int r = 0; r < nprocs_; r++) {
                char rf[128];
                snprintf(rf, sizeof(rf), "/tmp/pir_w_%d.ready", r);
                int wait_us = 0;
                while (access(rf, F_OK) != 0) {
                    usleep(100);
                    wait_us += 100;
                    if (wait_us > 120 * 1000000) {
                        fprintf(stderr, "GradSync TIMEOUT waiting for rank %d at gen %d — aborting\n", r, sync_gen_);
                        std::_Exit(2);
                    }
                }
            }

            // Average all weights
            std::vector<float> avg(total_params_, 0.0f);
            std::vector<float> tmp(total_params_);
            float inv_n = 1.0f / nprocs_;

            for (int r = 0; r < nprocs_; r++) {
                char wf[128];
                snprintf(wf, sizeof(wf), "/tmp/pir_w_%d.bin", r);
                FILE* wfp = fopen(wf, "rb");
                if (wfp) {
                    size_t got = fread(tmp.data(), sizeof(float), total_params_, wfp);
                    fclose(wfp);
                    if ((int64_t)got != total_params_) {
                        fprintf(stderr, "GradSync rank 0: SHORT READ from rank %d: %zu/%lld at gen %d\n",
                                r, got, (long long)total_params_, sync_gen_);
                    }
                    uint64_t h_r = weights_hash(tmp.data(), total_params_);
                    fprintf(stderr, "[GradSync avg gen=%d] rank %d hash=%016llx\n",
                            sync_gen_, r, (unsigned long long)h_r);
                    for (int64_t i = 0; i < total_params_; i++)
                        avg[i] += tmp[i] * inv_n;
                } else {
                    fprintf(stderr, "GradSync rank 0: fopen FAILED for %s\n", wf);
                }
            }

            // Write averaged weights
            f = fopen("/tmp/pir_w_avg.bin", "wb");
            if (f) {
                fwrite(avg.data(), sizeof(float), total_params_, f);
                fclose(f);
            }

            // Signal done
            f = fopen("/tmp/pir_w_avg.ready", "w");
            if (f) { fprintf(f, "%d", sync_gen_); fclose(f); }

            // Copy to my own buffer
            memcpy(flat_weights, avg.data(), total_params_ * sizeof(float));

            // Cleanup ready files for next sync
            usleep(50000);  // let others read
            for (int r = 0; r < nprocs_; r++) {
                char rf[128];
                snprintf(rf, sizeof(rf), "/tmp/pir_w_%d.ready", r);
                unlink(rf);
            }
            unlink("/tmp/pir_w_avg.ready");
        } else {
            // 3. Other ranks: wait for average, read it (timeout 120s)
            int wait_us = 0;
            while (access("/tmp/pir_w_avg.ready", F_OK) != 0) {
                usleep(100);
                wait_us += 100;
                if (wait_us > 120 * 1000000) {
                    fprintf(stderr, "GradSync TIMEOUT rank %d waiting avg at gen %d — aborting\n", rank_, sync_gen_);
                    std::_Exit(2);
                }
            }
            f = fopen("/tmp/pir_w_avg.bin", "rb");
            if (f) {
                size_t got = fread(flat_weights, sizeof(float), total_params_, f);
                fclose(f);
                if ((int64_t)got != total_params_) {
                    fprintf(stderr, "GradSync rank %d: SHORT READ %zu/%lld at gen %d\n",
                            rank_, got, (long long)total_params_, sync_gen_);
                }
            }
        }
        // Hash AFTER sync — all ranks should print same value
        uint64_t h_after = weights_hash(flat_weights, total_params_);
        fprintf(stderr, "[GradSync rank=%d gen=%d] before=%016llx after=%016llx\n",
                rank_, sync_gen_, (unsigned long long)h_before, (unsigned long long)h_after);
    }

    void sync_weights(float* flat_params) {
        sync(flat_params, nullptr);
    }

    void cleanup() {
        if (rank_ == 0) {
            for (int r = 0; r < nprocs_; r++) {
                char f[128];
                snprintf(f, sizeof(f), "/tmp/pir_w_%d.bin", r);
                unlink(f);
                snprintf(f, sizeof(f), "/tmp/pir_w_%d.ready", r);
                unlink(f);
            }
            unlink("/tmp/pir_w_avg.bin");
            unlink("/tmp/pir_w_avg.ready");
            unlink("/tmp/pir_sync_init");
        }
    }
};
