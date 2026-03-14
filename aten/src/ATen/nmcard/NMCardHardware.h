#pragma once
// ============================================================================
// NMCardHardware.h - Real NM Card Mini Hardware Backend
// ============================================================================
// Dynamically loads nm_card_load.dll and communicates with the NMC4 cores
// via DDR memory-mapped command/response protocol.
//
// Usage:
//   auto& hw = NMCardHardware::get();
//   if (hw.init("path/to/dispatcher.abs")) { ... }
//
// If the DLL is not found or no board is present, is_available() returns false
// and all operations fall through to the emulator.

#include "c10/macros/Macros.h"
#include <cstdint>
#include <string>
#include <map>
#include <mutex>
#include <stdexcept>
#include <iostream>

// Forward-declare platform handle types (avoid including windows.h in header)
#ifdef _WIN32
struct HINSTANCE__;
using HMODULE_t = HINSTANCE__*;
#else
using HMODULE_t = void*;
#endif

#ifndef ATEN_NMCARD_API
#if defined(PT_PLATFORM_WINDOWS) || defined(_MSC_VER)
    #if defined(ATEN_NMCARD_EXPORTS)
        #define ATEN_NMCARD_API __declspec(dllexport)
    #else
        #define ATEN_NMCARD_API __declspec(dllimport)
    #endif
#else
    #define ATEN_NMCARD_API __attribute__((visibility("default")))
#endif
#endif

namespace at {
namespace nmcard {

// ============================================================================
// DDR Memory Constants (from nm_card_mini_as_TRAINER/nmruntime/ops.py)
// ============================================================================

constexpr uint32_t DDR_BASE        = 0x00340000;
constexpr uint32_t CMD_BLOCK_SIZE  = 32;        // 32 words per core
constexpr uint32_t STATUS_OFFSET   = 30;        // word offset in cmd block
constexpr uint32_t WATCHDOG_OFFSET = 31;
constexpr uint32_t DATA_START      = DDR_BASE + 512;  // after 16 cmd blocks
constexpr uint32_t WEIGHT_START    = DDR_BASE + 65536;
constexpr uint32_t DDR_END         = 0x1FF00000;      // ~500MB DDR limit

// Opcodes: reuse NMCardOp enum from NMCardEmulator.h
// NMCardOp::MATMUL=1, RMSNORM=2, SOFTMAX=3, SILU=4, ROPE=5,
// ELEM_ADD=10, ELEM_MUL=11, GATE_MUL=13, EXIT=255

// PL_* return codes
constexpr int PL_OK        = 0;
constexpr int PL_ERROR     = 1;
constexpr int PL_TIMEOUT   = 2;

// ============================================================================
// PL_CoreNo structure (matches nm_card_load.dll)
// ============================================================================

struct PL_CoreNo {
    int nm_id;
    int cluster_id;
};

// ============================================================================
// Function pointer typedefs for nm_card_load.dll
// ============================================================================

using PL_GetBoardCount_fn  = int(*)(unsigned int*);
using PL_GetBoardDesc_fn   = int(*)(unsigned int, void**);
using PL_ResetBoard_fn     = int(*)(void*);
using PL_LoadInitCode_fn   = int(*)(void*);
using PL_GetAccess_fn      = int(*)(void*, PL_CoreNo*, void**);
using PL_LoadProgramFile_fn = int(*)(void*, const char*);
using PL_WriteMemBlock_fn  = int(*)(void*, uint32_t*, uint32_t, uint32_t);
using PL_ReadMemBlock_fn   = int(*)(void*, uint32_t*, uint32_t, uint32_t);
using PL_CloseAccess_fn    = int(*)(void*);
using PL_CloseBoardDesc_fn = int(*)(void*);
using PL_GetVersion_fn     = int(*)(unsigned int*, unsigned int*);

// ============================================================================
// DDR Bump Allocator
// ============================================================================
// Manages the DDR address space on the NM Card. Tracks host ptr → DDR addr
// mappings so that launch_* functions can find where data lives on-card.

struct DDRBlock {
    uint32_t addr;   // DDR word address
    uint32_t words;  // size in 32-bit words
};

class DDRAllocator {
public:
    DDRAllocator() : next_(DATA_START) {}

    // Allocate DDR space, returns word address
    uint32_t alloc(size_t nbytes) {
        uint32_t words = static_cast<uint32_t>((nbytes + 3) / 4);
        // Align to 16 words (64 bytes, cache line)
        words = (words + 15) & ~15u;

        if (next_ + words > DDR_END) {
            throw std::runtime_error("NMCard DDR: out of memory");
        }

        uint32_t addr = next_;
        next_ += words;
        return addr;
    }

    // Register a host pointer → DDR address mapping
    void register_mapping(const void* host_ptr, uint32_t ddr_addr, uint32_t words) {
        std::lock_guard<std::mutex> lock(mu_);
        host_to_ddr_[host_ptr] = {ddr_addr, words};
    }

    // Look up DDR address for a host pointer, returns 0 if not found
    uint32_t lookup(const void* host_ptr) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = host_to_ddr_.find(host_ptr);
        if (it != host_to_ddr_.end()) {
            return it->second.addr;
        }
        return 0;
    }

    // Remove mapping
    void unmap(const void* host_ptr) {
        std::lock_guard<std::mutex> lock(mu_);
        host_to_ddr_.erase(host_ptr);
    }

    // Reset allocator (e.g., between operations or epochs)
    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        next_ = DATA_START;
        host_to_ddr_.clear();
    }

    uint32_t bytes_used() const { return (next_ - DATA_START) * 4; }

private:
    uint32_t next_;
    mutable std::mutex mu_;
    std::map<const void*, DDRBlock> host_to_ddr_;
};

// ============================================================================
// NMCardHardware — Real Hardware Backend
// ============================================================================

class ATEN_NMCARD_API NMCardHardware {
public:
    // Singleton — implemented in NMCardHardware.cpp
    static NMCardHardware& get();

    // Initialize: load DLL, detect board, reset, load dispatcher
    // dispatcher_path: path to dispatcher.abs (empty = search default locations)
    // Returns true if hardware is ready
    bool init(const std::string& dispatcher_path = "");

    // Is hardware available and initialized?
    bool is_available() const { return initialized_; }

    // Graceful shutdown: send OP_EXIT, close handles, unload DLL
    void shutdown();

    // ========================================================================
    // DDR Memory Transfer
    // ========================================================================

    void write_to_ddr(const void* host_src, uint32_t ddr_addr, uint32_t word_count);
    void read_from_ddr(void* host_dst, uint32_t ddr_addr, uint32_t word_count);

    // ========================================================================
    // Low-level Dispatch Protocol
    // ========================================================================

    // Send opcode + args to a core, wait for completion
    void dispatch_op(uint32_t opcode, const uint32_t* args, int nargs, int core = 0);
    void wait_done(int core = 0, float timeout_sec = 10.0f);

    // ========================================================================
    // High-level Operations
    // ========================================================================
    // Each method: upload inputs → alloc DDR output → dispatch → wait → download

    void matmul(const float* A, const float* B, float* C,
                int64_t M, int64_t K, int64_t N);

    void rmsnorm(const float* input, float* output, const float* gamma,
                 int64_t batch, int64_t hidden);

    void softmax(const float* input, float* output,
                 int64_t batch, int64_t dim);

    void silu(const float* input, float* output, int64_t count);

    void rope(const float* input, float* output, const float* freqs,
              int64_t seq_len, int64_t head_dim, int64_t pos_offset);

    void elem_add(const float* a, const float* b, float* out, int64_t count);
    void elem_mul(const float* a, const float* b, float* out, int64_t count);
    void gate_mul(const float* a, const float* b, float* out, int64_t count);

    // ========================================================================
    // Info
    // ========================================================================

    int num_cores() const { return num_cores_; }
    DDRAllocator& ddr() { return ddr_; }
    const DDRAllocator& ddr() const { return ddr_; }

    NMCardHardware() = default;
    ~NMCardHardware();

private:
    // DLL handle
    HMODULE_t dll_ = nullptr;

    // Board handles
    void* board_ = nullptr;
    void* access_[16] = {};
    int num_cores_ = 0;
    bool initialized_ = false;

    // DDR allocator
    DDRAllocator ddr_;

    // Function pointers
    PL_GetBoardCount_fn  fn_GetBoardCount_  = nullptr;
    PL_GetBoardDesc_fn   fn_GetBoardDesc_   = nullptr;
    PL_ResetBoard_fn     fn_ResetBoard_     = nullptr;
    PL_LoadInitCode_fn   fn_LoadInitCode_   = nullptr;
    PL_GetAccess_fn      fn_GetAccess_      = nullptr;
    PL_LoadProgramFile_fn fn_LoadProgramFile_ = nullptr;
    PL_WriteMemBlock_fn  fn_WriteMemBlock_  = nullptr;
    PL_ReadMemBlock_fn   fn_ReadMemBlock_   = nullptr;
    PL_CloseAccess_fn    fn_CloseAccess_    = nullptr;
    PL_CloseBoardDesc_fn fn_CloseBoardDesc_ = nullptr;
    PL_GetVersion_fn     fn_GetVersion_     = nullptr;

    // Helpers
    bool load_dll();
    bool resolve_functions();
    uint32_t cmd_block_addr(int core) const {
        return DDR_BASE + static_cast<uint32_t>(core) * CMD_BLOCK_SIZE;
    }

    // Upload a host buffer to DDR, return DDR address
    uint32_t upload(const float* host_ptr, int64_t count);
    // Download from DDR to host buffer
    void download(float* host_ptr, uint32_t ddr_addr, int64_t count);
};

} // namespace nmcard
} // namespace at
