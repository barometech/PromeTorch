// ============================================================================
// NMCardEmulator.cpp - Singleton Implementation
// ============================================================================
// Same DLL-safe singleton pattern as CUDAAllocator.cpp

#include "aten/src/ATen/nmcard/NMCardEmulator.h"

namespace at {
namespace nmcard {

static NMCardEmulator g_nmcard_emulator;

ATEN_NMCARD_API NMCardEmulator& NMCardEmulator::get() {
    return g_nmcard_emulator;
}

} // namespace nmcard
} // namespace at
