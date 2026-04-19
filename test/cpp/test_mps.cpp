// ============================================================================
// test_mps.cpp — MPS backend self-test (macOS only).
// ============================================================================
// On any non-Apple host this test compiles to a trivial main() that prints
// "SKIP" and exits 0. On macOS with PT_USE_MPS=ON it:
//   1. Allocates two tensors on MPS (contents backed by id<MTLBuffer>).
//   2. Runs launch_add_mps on the MTLCommandQueue.
//   3. Synchronises, copies the result back to CPU, compares element-wise.
//
// This file is wired up in CMake only when PT_USE_MPS=ON.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#if defined(__APPLE__) && defined(PT_USE_MPS)

#include "aten/src/ATen/mps/MPSDispatch.h"
#include "aten/src/ATen/mps/MPSDevice.h"
#include "aten/src/ATen/core/TensorFactory.h"

int main() {
    using namespace at;

    if (!at::mps::MPSDevice::get().is_available()) {
        std::printf("MPS: no MTLDevice available — SKIP\n");
        return 0;
    }

    // 1. Create two CPU tensors with known data.
    const int N = 16;
    std::vector<float> ha(N), hb(N), hexp(N);
    for (int i = 0; i < N; ++i) {
        ha[i]   = float(i);
        hb[i]   = float(2 * i + 1);
        hexp[i] = ha[i] + hb[i];
    }
    auto a = empty({N}, TensorOptions().dtype(c10::ScalarType::Float));
    auto b = empty({N}, TensorOptions().dtype(c10::ScalarType::Float));
    std::memcpy(a.mutable_data_ptr<float>(), ha.data(), N * sizeof(float));
    std::memcpy(b.mutable_data_ptr<float>(), hb.data(), N * sizeof(float));

    // 2. Push to MPS, run add, synchronize, pull back.
    auto a_mps = to_mps(a);
    auto b_mps = to_mps(b);
    auto c_mps = mps_ops::add(a_mps, b_mps);
    auto c_cpu = to_cpu_from_mps(c_mps);

    // 3. Compare.
    const float* got = c_cpu.data_ptr<float>();
    for (int i = 0; i < N; ++i) {
        if (std::fabs(got[i] - hexp[i]) > 1e-5f) {
            std::printf("MPS add mismatch at %d: got %f, want %f\n",
                        i, got[i], hexp[i]);
            return 1;
        }
    }
    std::printf("MPS self-test OK (%d elems)\n", N);
    return 0;
}

#else  // non-Apple / PT_USE_MPS=OFF

int main() {
    std::printf("MPS: non-Apple or PT_USE_MPS=OFF — SKIP\n");
    return 0;
}

#endif
