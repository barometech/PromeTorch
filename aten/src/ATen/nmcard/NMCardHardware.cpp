// ============================================================================
// NMCardHardware.cpp - Real NM Card Mini Hardware Backend
// ============================================================================
// Singleton in .cpp (DLL safety — same pattern as NMCardEmulator.cpp)
// Dynamically loads nm_card_load.dll via LoadLibraryA/GetProcAddress

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "aten/src/ATen/nmcard/NMCardHardware.h"
#include <chrono>
#include <thread>
#include <cstring>

#ifdef _WIN32
static const char* DEFAULT_DLL_PATHS[] = {
    "C:\\Program Files\\Module\\NM_Card\\libload\\bin\\nm_card_load.dll",
    "nm_card_load.dll",
    nullptr
};
#else
static const char* DEFAULT_DLL_PATHS[] = {
    "libnm_card_load.so",
    nullptr
};
#endif

namespace at {
namespace nmcard {

// ============================================================================
// Singleton
// ============================================================================

static NMCardHardware g_nmcard_hw;

ATEN_NMCARD_API NMCardHardware& NMCardHardware::get() {
    return g_nmcard_hw;
}

// Private helper to free the DLL
static void free_dll_handle(HMODULE_t& dll) {
    if (dll) {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(dll));
#else
        dlclose(dll);
#endif
        dll = nullptr;
    }
}

// ============================================================================
// Destructor
// ============================================================================

NMCardHardware::~NMCardHardware() {
    if (initialized_) {
        // Don't throw from destructor — just try to clean up
        try { shutdown(); } catch (...) {}
    }
}

// ============================================================================
// DLL Loading
// ============================================================================

bool NMCardHardware::load_dll() {
    const char* env_path = std::getenv("NMCARD_DLL_PATH");

#ifdef _WIN32
    auto try_load = [&](const char* path) -> bool {
        HMODULE h = LoadLibraryA(path);
        if (h) { dll_ = h; return true; }
        return false;
    };
#else
    auto try_load = [&](const char* path) -> bool {
        void* h = dlopen(path, RTLD_LAZY);
        if (h) { dll_ = h; return true; }
        return false;
    };
#endif

    if (env_path && try_load(env_path)) return true;
    for (int i = 0; DEFAULT_DLL_PATHS[i]; ++i) {
        if (try_load(DEFAULT_DLL_PATHS[i])) return true;
    }
    return false;
}

bool NMCardHardware::resolve_functions() {
    auto resolve = [&](const char* name) -> void* {
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(dll_), name));
#else
        return dlsym(dll_, name);
#endif
    };

    #define RESOLVE(name) \
        fn_##name##_ = reinterpret_cast<PL_##name##_fn>(resolve("PL_" #name)); \
        if (!fn_##name##_) { \
            std::cerr << "NMCard: cannot resolve PL_" #name << std::endl; \
            return false; \
        }

    RESOLVE(GetBoardCount)
    RESOLVE(GetBoardDesc)
    RESOLVE(ResetBoard)
    RESOLVE(LoadInitCode)
    RESOLVE(GetAccess)
    RESOLVE(LoadProgramFile)
    RESOLVE(WriteMemBlock)
    RESOLVE(ReadMemBlock)
    RESOLVE(CloseAccess)
    RESOLVE(CloseBoardDesc)

    #undef RESOLVE

    // GetVersion is optional
    fn_GetVersion_ = reinterpret_cast<PL_GetVersion_fn>(resolve("PL_GetVersion"));

    return true;
}

// ============================================================================
// Initialization
// ============================================================================

bool NMCardHardware::init(const std::string& dispatcher_path) {
    if (initialized_) return true;

    // Step 1: Load DLL
    if (!load_dll()) {
        std::cerr << "NMCard: nm_card_load.dll not found" << std::endl;
        return false;
    }

    // Step 2: Resolve function pointers
    if (!resolve_functions()) {
        std::cerr << "NMCard: failed to resolve DLL functions" << std::endl;
        free_dll_handle(dll_);
        return false;
    }

    // Step 3: Print version if available
    if (fn_GetVersion_) {
        unsigned int major = 0, minor = 0;
        fn_GetVersion_(&major, &minor);
        std::cout << "NMCard DLL version: " << major << "." << minor << std::endl;
    }

    // Step 4: Enumerate boards
    unsigned int board_count = 0;
    int ret = fn_GetBoardCount_(&board_count);
    if (ret != PL_OK || board_count == 0) {
        std::cerr << "NMCard: no boards found (count=" << board_count << ")" << std::endl;
        free_dll_handle(dll_);
        return false;
    }
    std::cout << "NMCard: found " << board_count << " board(s)" << std::endl;

    // Step 5: Open board 0
    ret = fn_GetBoardDesc_(0, &board_);
    if (ret != PL_OK) {
        std::cerr << "NMCard: PL_GetBoardDesc failed (ret=" << ret << ")" << std::endl;
        return false;
    }

    // Step 6: Reset board
    ret = fn_ResetBoard_(board_);
    if (ret != PL_OK) {
        std::cerr << "NMCard: PL_ResetBoard failed (ret=" << ret << ")" << std::endl;
        return false;
    }

    // Step 7: Load init code (DDR controller setup)
    ret = fn_LoadInitCode_(board_);
    if (ret != PL_OK) {
        std::cerr << "NMCard: PL_LoadInitCode failed (ret=" << ret << ")" << std::endl;
        return false;
    }

    // Step 8: Get access to core 0 (cluster 0, nm 0)
    // For now, single-core mode only
    PL_CoreNo core = {0, 0};
    ret = fn_GetAccess_(board_, &core, &access_[0]);
    if (ret != PL_OK) {
        std::cerr << "NMCard: PL_GetAccess failed for core 0 (ret=" << ret << ")" << std::endl;
        return false;
    }
    num_cores_ = 1;

    // Step 9: Load dispatcher program
    std::string abs_path = dispatcher_path;
    if (abs_path.empty()) {
        // Try default locations
        const char* defaults[] = {
            "dispatcher.abs",
            "nmc_programs/dispatcher.abs",
            nullptr
        };
        for (int i = 0; defaults[i]; ++i) {
            abs_path = defaults[i];
            // Just try it — PL_LoadProgramFile will tell us if it fails
            break;
        }
    }

    ret = fn_LoadProgramFile_(access_[0], abs_path.c_str());
    if (ret != PL_OK) {
        std::cerr << "NMCard: PL_LoadProgramFile failed for '" << abs_path
                  << "' (ret=" << ret << ")" << std::endl;
        fn_CloseAccess_(access_[0]);
        access_[0] = nullptr;
        return false;
    }

    // Step 10: Wait briefly for dispatcher to start polling
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify: read command word — should be OP_NOP (0) if dispatcher is running
    uint32_t cmd_word = 0xFF;
    fn_ReadMemBlock_(access_[0], &cmd_word, cmd_block_addr(0), 1);
    if (cmd_word != 0) {  // OP_NOP
        std::cerr << "NMCard: dispatcher not responding (cmd_word=" << cmd_word << ")" << std::endl;
        fn_CloseAccess_(access_[0]);
        access_[0] = nullptr;
        return false;
    }

    initialized_ = true;
    ddr_.reset();
    std::cout << "NMCard: hardware initialized (1 core, dispatcher loaded)" << std::endl;
    return true;
}

// ============================================================================
// Shutdown
// ============================================================================

void NMCardHardware::shutdown() {
    if (!initialized_) return;

    // Send OP_EXIT to each active core
    for (int c = 0; c < num_cores_; ++c) {
        if (access_[c]) {
            uint32_t exit_op = 255;  // OP_EXIT
            uint32_t zero = 0;
            fn_WriteMemBlock_(access_[c], &zero, cmd_block_addr(c) + STATUS_OFFSET, 1);
            fn_WriteMemBlock_(access_[c], &exit_op, cmd_block_addr(c), 1);

            // Wait for acknowledgment
            uint32_t status = 0;
            for (int i = 0; i < 100 && status != 1; ++i) {
                fn_ReadMemBlock_(access_[c], &status, cmd_block_addr(c) + STATUS_OFFSET, 1);
                if (status != 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }

            fn_CloseAccess_(access_[c]);
            access_[c] = nullptr;
        }
    }

    if (board_) {
        fn_CloseBoardDesc_(board_);
        board_ = nullptr;
    }

    free_dll_handle(dll_);

    initialized_ = false;
    num_cores_ = 0;
    ddr_.reset();
    std::cout << "NMCard: hardware shutdown complete" << std::endl;
}

// ============================================================================
// DDR Memory Transfer
// ============================================================================

void NMCardHardware::write_to_ddr(const void* host_src, uint32_t ddr_addr, uint32_t word_count) {
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 4 bytes");
    int ret = fn_WriteMemBlock_(access_[0],
                                 const_cast<uint32_t*>(static_cast<const uint32_t*>(host_src)),
                                 ddr_addr, word_count);
    if (ret != PL_OK) {
        throw std::runtime_error("NMCard: PL_WriteMemBlock failed (ret=" + std::to_string(ret) + ")");
    }
}

void NMCardHardware::read_from_ddr(void* host_dst, uint32_t ddr_addr, uint32_t word_count) {
    int ret = fn_ReadMemBlock_(access_[0],
                                static_cast<uint32_t*>(host_dst),
                                ddr_addr, word_count);
    if (ret != PL_OK) {
        throw std::runtime_error("NMCard: PL_ReadMemBlock failed (ret=" + std::to_string(ret) + ")");
    }
}

// ============================================================================
// Upload / Download Helpers
// ============================================================================

uint32_t NMCardHardware::upload(const float* host_ptr, int64_t count) {
    uint32_t words = static_cast<uint32_t>(count);
    uint32_t ddr_addr = ddr_.alloc(count * sizeof(float));
    write_to_ddr(host_ptr, ddr_addr, words);
    return ddr_addr;
}

void NMCardHardware::download(float* host_ptr, uint32_t ddr_addr, int64_t count) {
    read_from_ddr(host_ptr, ddr_addr, static_cast<uint32_t>(count));
}

// ============================================================================
// Dispatch Protocol
// ============================================================================

void NMCardHardware::dispatch_op(uint32_t opcode, const uint32_t* args, int nargs, int core) {
    uint32_t base = cmd_block_addr(core);

    // 1. Wait for card ready (mem[0] == OP_NOP = 0)
    uint32_t cmd_word = 0xFF;
    int spins = 0;
    while (cmd_word != 0) {
        fn_ReadMemBlock_(access_[core], &cmd_word, base, 1);
        if (++spins > 1000000) {
            throw std::runtime_error("NMCard: dispatch timeout waiting for NOP on core " +
                                     std::to_string(core));
        }
    }

    // 2. Write args to mem[1..nargs]
    if (nargs > 0) {
        fn_WriteMemBlock_(access_[core], const_cast<uint32_t*>(args),
                          base + 1, static_cast<uint32_t>(nargs));
    }

    // 3. Set STATUS = 0 (busy)
    uint32_t zero = 0;
    fn_WriteMemBlock_(access_[core], &zero, base + STATUS_OFFSET, 1);

    // 4. Write opcode to mem[0] — TRIGGERS execution
    fn_WriteMemBlock_(access_[core], &opcode, base, 1);
}

void NMCardHardware::wait_done(int core, float timeout_sec) {
    uint32_t base = cmd_block_addr(core);
    uint32_t status = 0;

    auto start = std::chrono::steady_clock::now();

    while (status == 0) {
        fn_ReadMemBlock_(access_[core], &status, base + STATUS_OFFSET, 1);

        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start).count();
        if (elapsed > timeout_sec) {
            throw std::runtime_error("NMCard: operation timeout on core " +
                                     std::to_string(core) + " after " +
                                     std::to_string(timeout_sec) + "s");
        }
    }

    if (status == 2) {
        throw std::runtime_error("NMCard: hardware error (unknown opcode?) on core " +
                                 std::to_string(core));
    }
}

// ============================================================================
// High-level Operations
// ============================================================================
// Pattern: reset DDR allocator per-op → upload inputs → dispatch → wait → download
// Note: DDR allocator is reset before each op to avoid fragmentation.
// This is safe because we're synchronous (single op at a time).

void NMCardHardware::matmul(const float* A, const float* B, float* C,
                             int64_t M, int64_t K, int64_t N) {
    ddr_.reset();
    uint32_t addr_A = upload(A, M * K);
    uint32_t addr_B = upload(B, K * N);
    uint32_t addr_C = ddr_.alloc(M * N * sizeof(float));

    uint32_t args[6] = {
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(K),
        static_cast<uint32_t>(N),
        addr_A, addr_B, addr_C
    };
    dispatch_op(1, args, 6);  // OP_MATMUL
    wait_done();
    download(C, addr_C, M * N);
}

void NMCardHardware::rmsnorm(const float* input, float* output, const float* gamma,
                              int64_t batch, int64_t hidden) {
    ddr_.reset();
    uint32_t addr_in    = upload(input, batch * hidden);
    uint32_t addr_gamma = upload(gamma, hidden);
    uint32_t addr_out   = ddr_.alloc(batch * hidden * sizeof(float));

    uint32_t args[5] = {
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(hidden),
        addr_in, addr_out, addr_gamma
    };
    dispatch_op(2, args, 5);  // OP_RMSNORM
    wait_done();
    download(output, addr_out, batch * hidden);
}

void NMCardHardware::softmax(const float* input, float* output,
                              int64_t batch, int64_t dim) {
    ddr_.reset();
    uint32_t addr_in  = upload(input, batch * dim);
    uint32_t addr_out = ddr_.alloc(batch * dim * sizeof(float));

    uint32_t args[4] = {
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(dim),
        addr_in, addr_out
    };
    dispatch_op(3, args, 4);  // OP_SOFTMAX
    wait_done();
    download(output, addr_out, batch * dim);
}

void NMCardHardware::silu(const float* input, float* output, int64_t count) {
    ddr_.reset();
    uint32_t addr_in  = upload(input, count);
    uint32_t addr_out = ddr_.alloc(count * sizeof(float));

    uint32_t args[3] = {
        static_cast<uint32_t>(count),
        addr_in, addr_out
    };
    dispatch_op(4, args, 3);  // OP_SILU
    wait_done();
    download(output, addr_out, count);
}

void NMCardHardware::rope(const float* input, float* output, const float* freqs,
                           int64_t seq_len, int64_t head_dim, int64_t pos_offset) {
    ddr_.reset();
    uint32_t addr_in    = upload(input, seq_len * head_dim);
    uint32_t addr_freqs = upload(freqs, head_dim / 2);
    uint32_t addr_out   = ddr_.alloc(seq_len * head_dim * sizeof(float));

    uint32_t args[6] = {
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(pos_offset),
        addr_in, addr_out, addr_freqs
    };
    dispatch_op(5, args, 6);  // OP_ROPE
    wait_done();
    download(output, addr_out, seq_len * head_dim);
}

void NMCardHardware::elem_add(const float* a, const float* b, float* out, int64_t count) {
    ddr_.reset();
    uint32_t addr_a   = upload(a, count);
    uint32_t addr_b   = upload(b, count);
    uint32_t addr_out = ddr_.alloc(count * sizeof(float));

    uint32_t args[4] = {
        static_cast<uint32_t>(count),
        addr_a, addr_b, addr_out
    };
    dispatch_op(10, args, 4);  // OP_ELEM_ADD
    wait_done();
    download(out, addr_out, count);
}

void NMCardHardware::elem_mul(const float* a, const float* b, float* out, int64_t count) {
    ddr_.reset();
    uint32_t addr_a   = upload(a, count);
    uint32_t addr_b   = upload(b, count);
    uint32_t addr_out = ddr_.alloc(count * sizeof(float));

    uint32_t args[4] = {
        static_cast<uint32_t>(count),
        addr_a, addr_b, addr_out
    };
    dispatch_op(11, args, 4);  // OP_ELEM_MUL
    wait_done();
    download(out, addr_out, count);
}

void NMCardHardware::gate_mul(const float* a, const float* b, float* out, int64_t count) {
    ddr_.reset();
    uint32_t addr_a   = upload(a, count);
    uint32_t addr_b   = upload(b, count);
    uint32_t addr_out = ddr_.alloc(count * sizeof(float));

    uint32_t args[4] = {
        static_cast<uint32_t>(count),
        addr_a, addr_b, addr_out
    };
    dispatch_op(13, args, 4);  // OP_GATE_MUL
    wait_done();
    download(out, addr_out, count);
}

} // namespace nmcard
} // namespace at
