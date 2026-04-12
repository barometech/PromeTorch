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

    // sync() is used for weight averaging in Local SGD
    void sync(float* flat_weights, float* /*unused*/) {
        sync_gen_++;

        // 1. Write my weights to file
        char my_file[128];
        snprintf(my_file, sizeof(my_file), "/tmp/pir_w_%d.bin", rank_);
        FILE* f = fopen(my_file, "wb");
        if (f) {
            fwrite(flat_weights, sizeof(float), total_params_, f);
            fclose(f);
        }

        // Signal: I wrote my weights
        char ready_file[128];
        snprintf(ready_file, sizeof(ready_file), "/tmp/pir_w_%d.ready", rank_);
        f = fopen(ready_file, "w");
        if (f) { fprintf(f, "%d", sync_gen_); fclose(f); }

        // 2. Rank 0: wait for all, average, write result
        if (rank_ == 0) {
            // Wait for all ranks to write
            for (int r = 0; r < nprocs_; r++) {
                char rf[128];
                snprintf(rf, sizeof(rf), "/tmp/pir_w_%d.ready", r);
                while (access(rf, F_OK) != 0) usleep(100);
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
                    fread(tmp.data(), sizeof(float), total_params_, wfp);
                    fclose(wfp);
                    for (int64_t i = 0; i < total_params_; i++)
                        avg[i] += tmp[i] * inv_n;
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
            // 3. Other ranks: wait for average, read it
            while (access("/tmp/pir_w_avg.ready", F_OK) != 0) usleep(100);
            f = fopen("/tmp/pir_w_avg.bin", "rb");
            if (f) {
                fread(flat_weights, sizeof(float), total_params_, f);
                fclose(f);
            }
        }
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
