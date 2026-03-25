#pragma once
// ============================================================================
// NMQuadHardware.h — Host-side NM QUAD board control
// ============================================================================
// Wraps nm_quad_load.dll API for 4x NM6408 chip access.
// Handles: init, reset, program loading, DDR read/write, sync.
// Remote execution via nmrb proxy is transparent (same DLL API).

#include <string>
#include <array>
#include <mutex>
#include <cstdint>
#include <stdexcept>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef _WIN32
// Linux: linked directly against libnm_quad_load.so — declare at global scope
extern "C" {
    struct PL_Board;
    struct PL_Access;
    typedef struct { int nm_id; int cluster_id; } PL_CoreNo_t;
    typedef uint32_t PL_Word;
    typedef uint32_t PL_Addr;

    int PL_GetBoardCount(unsigned int*);
    int PL_GetBoardDesc(unsigned int, PL_Board**);
    int PL_CloseBoardDesc(PL_Board*);
    int PL_ResetBoard(PL_Board*);
    int PL_LoadInitCode(PL_Board*);
    int PL_GetAccess(PL_Board*, PL_CoreNo_t*, PL_Access**);
    int PL_CloseAccess(PL_Access*);
    int PL_LoadProgramFile(PL_Access*, const char*);
    int PL_ReadMemBlock(PL_Access*, PL_Word*, PL_Addr, uint32_t);
    int PL_WriteMemBlock(PL_Access*, const PL_Word*, PL_Addr, uint32_t);
    int PL_Sync(PL_Access*, int, int*);
    int PL_SyncArray(PL_Access*, int, PL_Addr, PL_Word, int*, PL_Addr*, PL_Word*);
    int PL_SetTimeout(uint32_t);
    int PL_GetStatus(PL_Access*, PL_Word*);
    int PL_GetResult(PL_Access*, PL_Word*);
}
#endif

namespace at {
namespace nmquad {

constexpr int NUM_CHIPS = 4;

// DDR memory layout per chip (from nm6408brd.lds)
constexpr uint32_t DDR_START = 0x00340000;
constexpr uint32_t DDR_END   = 0x1FCC0000;
constexpr uint32_t DDR_SIZE  = DDR_END - DDR_START;  // ~509 MB

// Command block region
constexpr uint32_t CMD_BLOCK_BASE  = DDR_START;
constexpr uint32_t CMD_BLOCK_SIZE  = 128;  // 32 words per core, 4 cores
constexpr uint32_t DATA_AREA_BASE  = DDR_START + 0x200;  // After cmd blocks

// NMMB (internal memory)
constexpr uint32_t NMMB_START = 0x00000800;
constexpr uint32_t NMMB_END   = 0x0007F800;
constexpr uint32_t NMMB_SIZE  = NMMB_END - NMMB_START;  // 510 KB

// PL_ types — use global declarations from above on Linux, local on Windows
#ifdef _WIN32
struct PL_Board;
struct PL_Access;
struct PL_CoreNo { int nm_id; int cluster_id; };
using PL_Word = uint32_t;
using PL_Addr = uint32_t;
#else
using PL_CoreNo = PL_CoreNo_t;
#endif

// Return codes
enum PLRetValue {
    PL_OK = 0, PL_ERROR = 1, PL_TIMEOUT = 2,
    PL_FILE = 3, PL_BADADDRESS = 4, PL_AGAIN = 5
};

// ============================================================================
// Per-chip state
// ============================================================================
struct ChipState {
    PL_Access* access = nullptr;
    bool initialized = false;
    bool dispatcher_loaded = false;
    uint32_t data_ptr = DATA_AREA_BASE;  // Bump allocator for DDR

    // Allocate DDR space on this chip
    uint32_t alloc_ddr(size_t words) {
        uint32_t addr = data_ptr;
        data_ptr += static_cast<uint32_t>(words);
        if (data_ptr > DDR_END) {
            throw std::runtime_error("NM QUAD: DDR out of memory on chip");
        }
        return addr;
    }

    void reset_ddr() {
        data_ptr = DATA_AREA_BASE;
    }
};

// ============================================================================
// NMQuadHardware — singleton managing all 4 chips
// ============================================================================
class NMQuadHardware {
public:
    static NMQuadHardware& get() {
        static NMQuadHardware instance;
        return instance;
    }

    // Initialize board and get access to chips
    bool init(int num_chips = NUM_CHIPS) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return true;

#ifdef _WIN32
        // Load DLL dynamically
        dll_ = LoadLibraryA("nm_quad_load.dll");
        if (!dll_) {
            last_error_ = "Failed to load nm_quad_load.dll";
            return false;
        }

        // Resolve functions
        #define RESOLVE(name) \
            fn_##name = (decltype(fn_##name))GetProcAddress(dll_, #name); \
            if (!fn_##name) { last_error_ = "Missing: " #name; return false; }

        RESOLVE(PL_GetBoardCount)
        RESOLVE(PL_GetBoardDesc)
        RESOLVE(PL_CloseBoardDesc)
        RESOLVE(PL_ResetBoard)
        RESOLVE(PL_LoadInitCode)
        RESOLVE(PL_GetAccess)
        RESOLVE(PL_CloseAccess)
        RESOLVE(PL_LoadProgramFile)
        RESOLVE(PL_ReadMemBlock)
        RESOLVE(PL_WriteMemBlock)
        RESOLVE(PL_Sync)
        RESOLVE(PL_SyncArray)
        RESOLVE(PL_SetTimeout)
        RESOLVE(PL_GetStatus)
        RESOLVE(PL_GetResult)
        #undef RESOLVE
#else
        // Linux: functions linked directly via -lnm_quad_load
        fn_PL_GetBoardCount = ::PL_GetBoardCount;
        fn_PL_GetBoardDesc = ::PL_GetBoardDesc;
        fn_PL_CloseBoardDesc = ::PL_CloseBoardDesc;
        fn_PL_ResetBoard = ::PL_ResetBoard;
        fn_PL_LoadInitCode = ::PL_LoadInitCode;
        fn_PL_GetAccess = ::PL_GetAccess;
        fn_PL_CloseAccess = ::PL_CloseAccess;
        fn_PL_LoadProgramFile = ::PL_LoadProgramFile;
        fn_PL_ReadMemBlock = ::PL_ReadMemBlock;
        fn_PL_WriteMemBlock = ::PL_WriteMemBlock;
        fn_PL_Sync = ::PL_Sync;
        fn_PL_SyncArray = ::PL_SyncArray;
        fn_PL_SetTimeout = ::PL_SetTimeout;
        fn_PL_GetStatus = ::PL_GetStatus;
        fn_PL_GetResult = ::PL_GetResult;
#endif

        // Get board
        unsigned int board_count = 0;
        if (fn_PL_GetBoardCount(&board_count) != PL_OK || board_count == 0) {
            last_error_ = "No NM QUAD boards detected";
            return false;
        }

        if (fn_PL_GetBoardDesc(0, &board_) != PL_OK) {
            last_error_ = "Failed to get board descriptor";
            return false;
        }

        // Reset and init
        fn_PL_ResetBoard(board_);
        fn_PL_LoadInitCode(board_);

        // Set timeout
        fn_PL_SetTimeout(10000);  // 10 seconds

        // Get access to each chip
        num_chips_ = std::min(num_chips, NUM_CHIPS);
        for (int i = 0; i < num_chips_; ++i) {
            PL_CoreNo core = {0, i};  // nm_id=0, cluster_id=chip
            if (fn_PL_GetAccess(board_, &core, &chips_[i].access) != PL_OK) {
                last_error_ = "Failed to get access to chip " + std::to_string(i);
                num_chips_ = i;
                break;
            }
            chips_[i].initialized = true;
        }

        initialized_ = true;
        return true;
    }

    // Load dispatcher program onto a chip
    bool load_dispatcher(int chip_id, const std::string& abs_path) {
        if (!check_chip(chip_id)) return false;

        if (fn_PL_LoadProgramFile(chips_[chip_id].access, abs_path.c_str()) != PL_OK) {
            last_error_ = "Failed to load program on chip " + std::to_string(chip_id);
            return false;
        }
        chips_[chip_id].dispatcher_loaded = true;
        return true;
    }

    // Write data to chip DDR
    bool write_ddr(int chip_id, const void* data, uint32_t addr, size_t words) {
        if (!check_chip(chip_id)) return false;
        return fn_PL_WriteMemBlock(chips_[chip_id].access,
                                    (PL_Word*)data, addr, (uint32_t)words) == PL_OK;
    }

    // Read data from chip DDR
    bool read_ddr(int chip_id, void* data, uint32_t addr, size_t words) {
        if (!check_chip(chip_id)) return false;
        return fn_PL_ReadMemBlock(chips_[chip_id].access,
                                   (PL_Word*)data, addr, (uint32_t)words) == PL_OK;
    }

    // Sync with chip (barrier)
    int sync(int chip_id, int value) {
        if (!check_chip(chip_id)) return -1;
        int ret = 0;
        fn_PL_Sync(chips_[chip_id].access, value, &ret);
        return ret;
    }

    // SyncArray — exchange data address/length
    int sync_array(int chip_id, int value, uint32_t out_addr, uint32_t out_len,
                   uint32_t* in_addr = nullptr, uint32_t* in_len = nullptr) {
        if (!check_chip(chip_id)) return -1;
        int ret = 0;
        fn_PL_SyncArray(chips_[chip_id].access, value, out_addr, out_len,
                        &ret, in_addr, in_len);
        return ret;
    }

    // Get chip status
    uint32_t get_status(int chip_id) {
        if (!check_chip(chip_id)) return 0xFFFFFFFF;
        uint32_t status = 0;
        fn_PL_GetStatus(chips_[chip_id].access, &status);
        return status;
    }

    // Get result from chip
    uint32_t get_result(int chip_id) {
        if (!check_chip(chip_id)) return 0;
        uint32_t result = 0;
        fn_PL_GetResult(chips_[chip_id].access, &result);
        return result;
    }

    // Allocate DDR on a chip
    uint32_t alloc_ddr(int chip_id, size_t words) {
        if (!check_chip(chip_id)) return 0;
        return chips_[chip_id].alloc_ddr(words);
    }

    // Reset DDR allocator
    void reset_ddr(int chip_id) {
        if (check_chip(chip_id)) chips_[chip_id].reset_ddr();
    }

    // Accessors
    int num_chips() const { return num_chips_; }
    bool is_initialized() const { return initialized_; }
    const std::string& last_error() const { return last_error_; }
    ChipState& chip(int id) { return chips_[id]; }

    // Shutdown
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return;

        for (int i = 0; i < num_chips_; ++i) {
            if (chips_[i].access) {
                fn_PL_CloseAccess(chips_[i].access);
                chips_[i].access = nullptr;
            }
            chips_[i].initialized = false;
            chips_[i].dispatcher_loaded = false;
        }

        if (board_) {
            fn_PL_CloseBoardDesc(board_);
            board_ = nullptr;
        }

#ifdef _WIN32
        if (dll_) {
            // Don't FreeLibrary — same pattern as CUDA (avoid shutdown crashes)
            dll_ = nullptr;
        }
#endif
        initialized_ = false;
    }

private:
    NMQuadHardware() = default;
    ~NMQuadHardware() { /* Don't shutdown — avoid order-of-destruction issues */ }

    bool check_chip(int id) const {
        return initialized_ && id >= 0 && id < num_chips_ && chips_[id].initialized;
    }

    bool initialized_ = false;
    int num_chips_ = 0;
    PL_Board* board_ = nullptr;
    std::array<ChipState, NUM_CHIPS> chips_;
    std::string last_error_;
    std::mutex mutex_;

#ifdef _WIN32
    HMODULE dll_ = nullptr;
#endif

    // Function pointers (resolved from DLL on Windows, direct link on Linux)
    int (*fn_PL_GetBoardCount)(unsigned int*) = nullptr;
    int (*fn_PL_GetBoardDesc)(unsigned int, PL_Board**) = nullptr;
    int (*fn_PL_CloseBoardDesc)(PL_Board*) = nullptr;
    int (*fn_PL_ResetBoard)(PL_Board*) = nullptr;
    int (*fn_PL_LoadInitCode)(PL_Board*) = nullptr;
    int (*fn_PL_GetAccess)(PL_Board*, PL_CoreNo*, PL_Access**) = nullptr;
    int (*fn_PL_CloseAccess)(PL_Access*) = nullptr;
    int (*fn_PL_LoadProgramFile)(PL_Access*, const char*) = nullptr;
    int (*fn_PL_ReadMemBlock)(PL_Access*, PL_Word*, PL_Addr, uint32_t) = nullptr;
    int (*fn_PL_WriteMemBlock)(PL_Access*, const PL_Word*, PL_Addr, uint32_t) = nullptr;
    int (*fn_PL_Sync)(PL_Access*, int, int*) = nullptr;
    int (*fn_PL_SyncArray)(PL_Access*, int, PL_Addr, PL_Word, int*, PL_Addr*, PL_Word*) = nullptr;
    int (*fn_PL_SetTimeout)(uint32_t) = nullptr;
    int (*fn_PL_GetStatus)(PL_Access*, PL_Word*) = nullptr;
    int (*fn_PL_GetResult)(PL_Access*, PL_Word*) = nullptr;
};

} // namespace nmquad
} // namespace at
