#pragma once

// ============================================================================
// Futex.h — низкоуровневые wait/wake primitives для intra-process sync
// ============================================================================
// Linux:  syscall(SYS_futex, FUTEX_WAIT/FUTEX_WAKE)
// Windows: WaitOnAddress / WakeByAddress* (Windows 8+, synchronization.lib)
// Other:  fallback на std::this_thread::yield (busy poll)
//
// Намеренно НЕ используется FUTEX_PRIVATE_FLAG: на Эльбрусе E2K (ядро 6.1)
// он работал нестабильно с ddp.cpp SHM (cross-process). Для intra-process
// тоже остаёмся на public futex для единообразия.
// ============================================================================

#include <atomic>
#include <cstdint>

#if defined(__linux__) && !defined(_WIN32)
#  include <linux/futex.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#  include <climits>
#elif defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <synchapi.h>
#  if defined(_MSC_VER)
#    pragma comment(lib, "synchronization.lib")
#  endif
#else
#  include <thread>
#endif

namespace c10 {

// Sleep до тех пор, пока *addr != expected (или wake event).
// Безопасно к race: если addr уже != expected, возвращается мгновенно.
inline void futex_wait(std::atomic<uint32_t>* addr, uint32_t expected) {
#if defined(__linux__) && !defined(_WIN32) && defined(SYS_futex)
    ::syscall(SYS_futex,
              reinterpret_cast<uint32_t*>(addr),
              FUTEX_WAIT,
              expected,
              nullptr, nullptr, 0);
#elif defined(_WIN32)
    uint32_t cmp = expected;
    WaitOnAddress(reinterpret_cast<volatile void*>(addr),
                  &cmp, sizeof(cmp), INFINITE);
#else
    while (addr->load(std::memory_order_acquire) == expected) {
        std::this_thread::yield();
    }
#endif
}

// Sleep до timeout_ms или wake. Returns true если wake'нулся раньше timeout.
inline bool futex_wait_timed(std::atomic<uint32_t>* addr,
                              uint32_t expected,
                              uint32_t timeout_ms) {
#if defined(__linux__) && !defined(_WIN32) && defined(SYS_futex)
    struct timespec ts;
    ts.tv_sec  = timeout_ms / 1000;
    ts.tv_nsec = (timeout_ms % 1000) * 1000000L;
    long rc = ::syscall(SYS_futex,
                        reinterpret_cast<uint32_t*>(addr),
                        FUTEX_WAIT,
                        expected,
                        &ts, nullptr, 0);
    return rc == 0;  // 0 = woken, -1 = timeout/error
#elif defined(_WIN32)
    uint32_t cmp = expected;
    return WaitOnAddress(reinterpret_cast<volatile void*>(addr),
                         &cmp, sizeof(cmp), timeout_ms) != 0;
#else
    (void)timeout_ms;
    futex_wait(addr, expected);
    return true;
#endif
}

inline void futex_wake_all(std::atomic<uint32_t>* addr) {
#if defined(__linux__) && !defined(_WIN32) && defined(SYS_futex)
    ::syscall(SYS_futex,
              reinterpret_cast<uint32_t*>(addr),
              FUTEX_WAKE,
              INT_MAX, nullptr, nullptr, 0);
#elif defined(_WIN32)
    WakeByAddressAll(reinterpret_cast<void*>(addr));
#else
    (void)addr;
#endif
}

inline void futex_wake_one(std::atomic<uint32_t>* addr) {
#if defined(__linux__) && !defined(_WIN32) && defined(SYS_futex)
    ::syscall(SYS_futex,
              reinterpret_cast<uint32_t*>(addr),
              FUTEX_WAKE,
              1, nullptr, nullptr, 0);
#elif defined(_WIN32)
    WakeByAddressSingle(reinterpret_cast<void*>(addr));
#else
    (void)addr;
#endif
}

}  // namespace c10
