# =============================================================================
# E2KIntrinsicChecks.cmake — auto-detect E2K builtin intrinsics
# =============================================================================
#
# Не полагаемся на `__iset__` (от LCC) как единственный guard. Вместо этого
# реально пытаемся скомпилировать каждый intrinsic — определяем что компилятор
# (LCC любой версии) умеет делать на текущей машине.
#
# Source of probe codes: vliw_mission/e2k_vnni/probe_qpmaddubsh.c +
# extracted/E2K_opcodes.xlsx (МЦСТ opcode table v4-v6).
#
# Каждая проверка set'ит CMake variable, которая потом transformируется в
# define через target_compile_definitions:
#   PT_HAVE_QPMADDUBSH    — v5+ (8C2, 16C): 4-lane integer SIMD mul-add
#   PT_HAVE_PMADDUBSH     — v4+ (8C, 8C2, 16C): 2-lane integer SIMD mul-add
#   PT_HAVE_QPFMULS       — v5+: 4-lane FP32 SIMD multiply
#   PT_HAVE_QPFADDS       — v5+: 4-lane FP32 SIMD add
#   PT_HAVE_QPFNMAS       — v5+: 4-lane FP32 fused negate-multiply-add
#   PT_HAVE_QPFMAS        — v5+: 4-lane FP32 fused multiply-add
#   PT_HAVE_PFMULS        — v4+: 2-lane FP32 SIMD multiply
#   PT_HAVE_PFADDS        — v4+: 2-lane FP32 SIMD add
#   PT_HAVE_QPMADDH       — v5+: 4×i16 → 2×i32 pairwise mul-add
#   PT_HAVE_PMADDH        — v4+: 2×i16 → 1×i32 pairwise mul-add
#   PT_HAVE_QPADDW        — v5+: packed i32 add (4 lanes)
#   PT_HAVE_PADDW         — v4+: packed i32 add (2 lanes)
#   PT_HAVE_QPISTOFS      — v5+: pack i32 → f32
#   PT_HAVE_QPSUBW        — v5+: packed i32 sub (4 lanes)
#   PT_HAVE_QPMULLW       — v5+: packed i32 multiply low
#   PT_HAVE_QPFSUBS       — v5+: 4-lane FP32 SIMD sub
#
# Все checks сделаны через `try_compile` с реальным `-march` который передан
# юзером (или auto-detected). Если -march=elbrus-v3, qpmaddubsh не
# скомпилируется — define НЕ выставится, код пойдёт на v4-fallback / scalar.
#
# Использование:
#   include(cmake/E2KIntrinsicChecks.cmake)
#   pt_detect_e2k_intrinsics(target_lib)
# =============================================================================

include(CheckCXXSourceCompiles)

function(_pt_check_e2k_intrinsic varname src_body)
    # Сохраняем существующие flags, временно нагружаем -march=current
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS}")
    check_cxx_source_compiles("
        typedef long long v2di __attribute__((vector_size(16)));
        typedef long long v1di __attribute__((vector_size(8)));
        int main() {
            ${src_body}
            return 0;
        }
    " ${varname})
endfunction()

function(pt_detect_e2k_intrinsics)
    # Защита: имеет смысл только на E2K
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")
        return()
    endif()
    message(STATUS "E2K intrinsic auto-detection (try_compile, current -march):")

    # --- 4-lane (v5+) ---
    _pt_check_e2k_intrinsic(PT_HAVE_QPMADDUBSH "
        v2di a={1,1}, b={1,1};
        v2di r = __builtin_e2k_qpmaddubsh(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPMADDH "
        v2di a={1,1}, b={1,1};
        v2di r = __builtin_e2k_qpmaddh(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPADDW "
        v2di a={1,1}, b={1,1};
        v2di r = __builtin_e2k_qpaddw(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPSUBW "
        v2di a={1,1}, b={1,1};
        v2di r = __builtin_e2k_qpsubw(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPMULLW "
        v2di a={1,1}, b={1,1};
        v2di r = __builtin_e2k_qpmullw(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPISTOFS "
        v2di a={1,1};
        v2di r = __builtin_e2k_qpistofs(a);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPFMULS "
        v2di a={1,1}, b={1,1};
        v2di r = __builtin_e2k_qpfmuls(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPFADDS "
        v2di a={1,1}, b={1,1};
        v2di r = __builtin_e2k_qpfadds(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPFSUBS "
        v2di a={1,1}, b={1,1};
        v2di r = __builtin_e2k_qpfsubs(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPFNMAS "
        v2di a={1,1}, b={1,1}, c={1,1};
        v2di r = __builtin_e2k_qpfnmas(a, b, c);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_QPFMAS "
        v2di a={1,1}, b={1,1}, c={1,1};
        v2di r = __builtin_e2k_qpfmas(a, b, c);
        (void)r;
    ")

    # --- 2-lane (v4+) ---
    _pt_check_e2k_intrinsic(PT_HAVE_PMADDUBSH "
        v1di a=1LL, b=1LL;
        v1di r = __builtin_e2k_pmaddubsh(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_PMADDH "
        v1di a=1LL, b=1LL;
        v1di r = __builtin_e2k_pmaddh(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_PADDW "
        v1di a=1LL, b=1LL;
        v1di r = __builtin_e2k_paddw(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_PFMULS "
        v1di a=1LL, b=1LL;
        v1di r = __builtin_e2k_pfmuls(a, b);
        (void)r;
    ")
    _pt_check_e2k_intrinsic(PT_HAVE_PFADDS "
        v1di a=1LL, b=1LL;
        v1di r = __builtin_e2k_pfadds(a, b);
        (void)r;
    ")

    # --- Сводный отчёт ---
    set(_found "")
    set(_missing "")
    foreach(intr
        QPMADDUBSH QPMADDH QPADDW QPSUBW QPMULLW QPISTOFS
        QPFMULS QPFADDS QPFSUBS QPFNMAS QPFMAS
        PMADDUBSH PMADDH PADDW PFMULS PFADDS)
        if(PT_HAVE_${intr})
            list(APPEND _found "${intr}")
        else()
            list(APPEND _missing "${intr}")
        endif()
    endforeach()
    message(STATUS "  available: ${_found}")
    if(_missing)
        message(STATUS "  missing  : ${_missing}")
    endif()

    # --- Inference об ISA-level по реально available intrinsics ---
    if(PT_HAVE_QPMADDUBSH AND PT_HAVE_QPFMAS)
        set(PT_E2K_ISA_LEVEL "v5+" CACHE INTERNAL "Detected E2K ISA capability")
        message(STATUS "  inferred ISA level: v5+ (full VNNI + FMA)")
    elseif(PT_HAVE_PMADDUBSH AND PT_HAVE_PFMULS)
        set(PT_E2K_ISA_LEVEL "v4" CACHE INTERNAL "Detected E2K ISA capability")
        message(STATUS "  inferred ISA level: v4 (half-width VNNI + 2-lane FP)")
    elseif(PT_HAVE_PFMULS)
        set(PT_E2K_ISA_LEVEL "v3" CACHE INTERNAL "Detected E2K ISA capability")
        message(STATUS "  inferred ISA level: v3 (FP-only SIMD)")
    else()
        set(PT_E2K_ISA_LEVEL "unknown" CACHE INTERNAL "Detected E2K ISA capability")
        message(STATUS "  inferred ISA level: unknown (no E2K intrinsics worked)")
    endif()
endfunction()

# =============================================================================
# pt_apply_e2k_intrinsic_defines(target)
#
# Применяет detect'нутые define'ы к target. После include(E2KIntrinsicChecks)
# + pt_detect_e2k_intrinsics() — вызови это для каждого target который должен
# использовать intrinsics:
#   pt_apply_e2k_intrinsic_defines(aten_cpu)
#   pt_apply_e2k_intrinsic_defines(promeserve)
# =============================================================================
function(pt_apply_e2k_intrinsic_defines target)
    foreach(intr
        QPMADDUBSH QPMADDH QPADDW QPSUBW QPMULLW QPISTOFS
        QPFMULS QPFADDS QPFSUBS QPFNMAS QPFMAS
        PMADDUBSH PMADDH PADDW PFMULS PFADDS)
        if(PT_HAVE_${intr})
            target_compile_definitions(${target} PRIVATE PT_HAVE_${intr}=1)
        endif()
    endforeach()
endfunction()
