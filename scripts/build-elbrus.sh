#!/bin/bash
# ============================================================================
# Build PromeTorch на Эльбрусе (локально, без SSH, без PyTorch)
# ============================================================================
# Запускается прямо на Эльбрус-машине из любой директории внутри репо.
# НЕ требует PyTorch (наш namespace torch:: — это PromeTorch собственный,
# не имеет отношения к pytorch.org).
# НЕ требует Python (Python bindings отключены через PT_BUILD_PYTHON=OFF).
#
# Поддерживаемые ОС:
#   - Альт Линукс под Эльбрус
#   - Эльбрус Linux от МЦСТ (PDK)
#   - Astra Linux SE for Elbrus
#
# Поддерживаемые компиляторы (auto-detect):
#   1. LCC 1.29+ (MCST, default на Эльбрус Linux)
#   2. gcc-elbrus (новые ядра, Альт)
#   3. system gcc если e2k-aware
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PT_BUILD_DIR:-$PROJECT_DIR/build_elbrus}"
TOOLCHAIN="$PROJECT_DIR/cmake/toolchains/e2k-elbrus.cmake"

# Targets по умолчанию — минимальный рабочий набор без Python bindings
# и без всех examples/. Полный набор тянет gguf, promeserve, etc — overhead.
DEFAULT_TARGETS="aten_cpu torch_autograd tuda_tests_standalone tuda_tests test_gguf_inference"
TARGETS="${PT_TARGETS:-$DEFAULT_TARGETS}"

echo "================================================================"
echo " PromeTorch: native build для Эльбрус E2K (без SSH, без PyTorch)"
echo "================================================================"
echo "  Project: $PROJECT_DIR"
echo "  Build:   $BUILD_DIR"
echo "  Targets: $TARGETS"
echo

# ============================================================================
# 1. Проверка системных зависимостей
# ============================================================================
MISSING=()

check_header() {
    # Реальный compile-test через текущий $CXX. Это надёжнее find+grep
    # потому что компилятор сам знает свои include paths (особенно когда
    # omp.h лежит внутри /opt/mcst/lcc-home/.../include и подтягивается
    # автоматически только этим компилятором).
    local header="$1"; local pkg="$2"
    local compiler="${CXX:-${CC:-gcc}}"
    if ! echo "#include <$header>
int main(){return 0;}" | "$compiler" -fsyntax-only -x c++ - 2>/dev/null; then
        MISSING+=("$pkg ($header)")
    fi
}

check_lib() {
    # NB: set -o pipefail + grep -q | echo "$var" — pipe закрывается рано,
    # echo получает SIGPIPE → pipeline=141. Поэтому используем bash-builtin
    # pattern match (без pipe вообще, мгновенно).
    local lib="$1"; local pkg="$2"
    local ld_cache; ld_cache="$(ldconfig -p 2>/dev/null || true)"
    if [[ "$ld_cache" == *"lib$lib.so"* ]]; then
        return 0
    fi
    # Fallback: проверяем напрямую конкретные пути без рекурсии find.
    for d in /usr/lib64 /usr/lib /opt/mcst/lib64 /opt/mcst/lib /opt/mcst/eml/lib; do
        for ext in "" ".0" ".1" ".2"; do
            if [ -e "$d/lib$lib.so$ext" ]; then
                return 0
            fi
        done
    done
    MISSING+=("$pkg (lib$lib.so)")
}

check_bin() {
    if ! command -v "$1" >/dev/null 2>&1; then
        MISSING+=("$2 (executable: $1)")
    fi
}

# Auto-detect компилятор. Возможные варианты:
#   - LCC + lcc++  (Альт под Эльбрус: pkg lcc-c++)
#   - LCC + l++    (MCST PDK: /opt/mcst/bin/lcc, /opt/mcst/bin/l++)
#   - gcc-elbrus / g++-elbrus
#   - system gcc / g++ если поддерживает -march=elbrus-v4
detect_compiler() {
    # CMake требует FULL PATH в -DCMAKE_CXX_COMPILER, иначе ругается
    # "lcc++ is not a full path and was not found in the PATH" при первом
    # configure. Поэтому возвращаем абсолютные пути через command -v.
    local cc="" cxx=""
    if command -v lcc >/dev/null 2>&1; then
        cc="$(command -v lcc)"
        if command -v lcc++ >/dev/null 2>&1; then
            cxx="$(command -v lcc++)"
        elif command -v l++ >/dev/null 2>&1; then
            cxx="$(command -v l++)"
        fi
    fi
    if [ -z "$cxx" ] && command -v gcc-elbrus >/dev/null 2>&1; then
        cc="$(command -v gcc-elbrus)"
        cxx="$(command -v g++-elbrus)"
    fi
    if [ -z "$cxx" ] && command -v gcc >/dev/null 2>&1 && \
         echo "int main(){return 0;}" | gcc -march=elbrus-v4 -x c - -o /dev/null 2>/dev/null; then
        cc="$(command -v gcc)"
        cxx="$(command -v g++)"
    fi
    if [ -z "$cxx" ]; then
        return 1
    fi
    export CC="$cc"
    export CXX="$cxx"
    echo "[deps] Компилятор: $cc / $cxx ($("$cc" --version 2>&1 | head -1))"
}

echo "[deps] Проверка зависимостей..."

# Бинарники
check_bin cmake cmake
check_bin make "make или ninja-build"

# Компилятор
if ! detect_compiler; then
    cat <<'EOF'

ERROR: не найден ни LCC, ни gcc-elbrus, ни system gcc с поддержкой e2k.

Установи компилятор:

  # Альт Линукс под Эльбрус:
  apt-get install -y lcc lcc-c++
  # или
  apt-get install -y gcc-elbrus

  # Эльбрус Linux МЦСТ:
  # LCC обычно уже предустановлен (в составе PDK)

EOF
    exit 1
fi

# Headers + libs (E2K-only)
check_header "eml/cblas.h"  "eml-devel"
check_lib    eml_mt          "eml-devel"
check_header "omp.h"         "libomp11-devel или libomp-devel"
check_header "numa.h"        "libnuma-devel"
check_lib    numa            "libnuma-devel"

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo
    echo "ERROR: отсутствуют системные зависимости:"
    for m in "${MISSING[@]}"; do echo "    - $m"; done
    cat <<'EOF'

Установи одной командой (под root или через sudo):

  # Альт Линукс под Эльбрус:
  apt-get install -y eml-devel libomp11-devel libnuma-devel cmake make

  # Эльбрус Linux МЦСТ / Astra SE for Elbrus:
  apt-get install -y eml-devel libomp-devel libnuma-devel cmake make

Затем перезапусти этот скрипт.
EOF
    exit 1
fi

echo "[deps] OK — все зависимости найдены."
echo

# ============================================================================
# 2. Host CPU detection — выбираем -mtune под текущую машину
# ============================================================================
# <contributor>ов баг (2026-05-08): на E8C (v4) LCC принимает -mtune=elbrus-8c2
# как WARNING, не error. CMake check_cxx_compiler_flag тогда возвращает true
# и бинарь генерируется под v5 → "ошибка формата выполняемого файла" на v4.
# Чиним через явный host CPU detection:
#   - lscpu / cpuinfo / lcc --target → определяем номер ISA
#   - подаём -DPT_E2K_MTUNE=elbrus-vN
# Override: PT_E2K_MTUNE=elbrus-v6 ./build-elbrus.sh
detect_e2k_mtune() {
    # Allow user override via env
    if [ -n "${PT_E2K_MTUNE:-}" ]; then
        echo "$PT_E2K_MTUNE"; return
    fi
    # 1. Try lcc --target-name (LCC >= 1.27)
    if command -v lcc >/dev/null 2>&1; then
        local t
        t="$(lcc --target-name 2>/dev/null || true)"
        case "$t" in
            *elbrus-v6*|*-16c*|*-16С*) echo "elbrus-v6"; return ;;
            *elbrus-v5*|*-8c2*|*-8СВ*) echo "elbrus-v5"; return ;;
            *elbrus-v4*|*-8c*|*-8С*)   echo "elbrus-v4"; return ;;
            *elbrus-v3*|*-4c*|*-4С*)   echo "elbrus-v3"; return ;;
        esac
    fi
    # 2. lscpu (Alt Linux под Эльбрус добавляет model name "E2C+/E4C/E8C/E16C")
    local model
    model="$(lscpu 2>/dev/null | awk -F: '/Model name|Имя модели/ {gsub(/^ +/,"",$2); print $2; exit}')"
    case "$model" in
        *E16C*|*16С*)            echo "elbrus-v6"; return ;;
        *E8CB*|*8СВ*|*E8C2*)     echo "elbrus-v5"; return ;;
        *E8C*|*8С*)              echo "elbrus-v4"; return ;;
        *E4C*|*4С*|*E2C*|*2С*)   echo "elbrus-v3"; return ;;
    esac
    # 3. /proc/cpuinfo fallback (некоторые ОС не имеют lscpu)
    if [ -r /proc/cpuinfo ]; then
        if grep -qiE "E16C|elbrus-v6"        /proc/cpuinfo; then echo "elbrus-v6"; return; fi
        if grep -qiE "E8CB|E8C2|elbrus-v5"   /proc/cpuinfo; then echo "elbrus-v5"; return; fi
        if grep -qiE "E8C|elbrus-v4"         /proc/cpuinfo; then echo "elbrus-v4"; return; fi
        if grep -qiE "E4C|E2C|elbrus-v3"     /proc/cpuinfo; then echo "elbrus-v3"; return; fi
    fi
    # Safe default: v4 supports E8C+ (2019+) — runs everywhere modern, no v5+ intrinsics.
    echo "elbrus-v4"
}

E2K_MTUNE="$(detect_e2k_mtune)"
# Toolchain читает PT_E2K_MARCH через env (cmake/toolchains/e2k-elbrus.cmake).
# По умолчанию march = mtune (одинаковая версия), переопределяемо отдельно
# через `PT_E2K_MARCH=elbrus-v4 PT_E2K_MTUNE=elbrus-v5 ./build-elbrus.sh`
# (например для cross-build бинаря который должен работать на v4 но
# оптимизирован под v5).
export PT_E2K_MARCH="${PT_E2K_MARCH:-$E2K_MTUNE}"
echo "[deps] E2K target ISA: -march=$PT_E2K_MARCH -mtune=$E2K_MTUNE"
echo "       (override через PT_E2K_MARCH / PT_E2K_MTUNE)"
echo

# ============================================================================
# 3. CMake configure
# ============================================================================
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

NEED_RECONFIGURE=0
[ ! -f CMakeCache.txt ] && NEED_RECONFIGURE=1
[ "${PT_RECONFIGURE:-0}" = "1" ] && NEED_RECONFIGURE=1

if [ "$NEED_RECONFIGURE" = "1" ]; then
    echo "[cmake] Configuring..."
    cmake "$PROJECT_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DPT_E2K_MTUNE="$E2K_MTUNE" \
        -DPT_USE_TUDA=ON \
        -DPT_USE_LINQ=OFF \
        -DPT_USE_CUDA=OFF \
        -DPT_USE_NMCARD=OFF \
        -DPT_BUILD_TESTS=ON \
        -DPT_BUILD_PYTHON=OFF \
        -DPT_BUILD_SHARED_LIBS=ON 2>&1 | tee /tmp/pt_cmake.log

    echo
    echo "[cmake] Результат поиска зависимостей:"
    grep -E "EML headers|libnuma found|EML BLAS|NUMA-aware|Found OpenMP|TUDA: " /tmp/pt_cmake.log || true
    echo
fi

# ============================================================================
# 3. Сборка только нужных targets (поштучно — чтобы видеть на каком упало)
# ============================================================================
JOBS="${PT_JOBS:-$(nproc)}"
echo "[build] Targets: $TARGETS · jobs=$JOBS"

for tgt in $TARGETS; do
    echo
    echo "  → $tgt"
    if ! cmake --build . --target "$tgt" -j "$JOBS"; then
        echo
        echo "ERROR: target '$tgt' не собрался."
        echo "       Логи: $BUILD_DIR/CMakeFiles/${tgt}.dir/"
        echo "       Полный вывод CMake: /tmp/pt_cmake.log"
        exit 1
    fi
done

# ============================================================================
# 4. Quick sanity test (без падения скрипта)
# ============================================================================
echo
echo "================================================================"
echo " Build complete: $BUILD_DIR"
echo "================================================================"
echo

if [ -x "$BUILD_DIR/tuda_tests_standalone" ]; then
    echo "[sanity] Запуск tuda_tests_standalone (без autograd)..."
    if "$BUILD_DIR/tuda_tests_standalone"; then
        echo "[sanity] ✓ TUDA primitives — PASS"
    else
        echo "[sanity] ✗ TUDA primitives — FAIL"
        exit 2
    fi
fi

cat <<EOF

Дальнейшие шаги:

  ./build_elbrus/tuda_tests              # full test с autograd
  ./scripts/run_tp_elbrus.sh prompt      # GGUF inference (TP, auto-detect NUMA)
  ./scripts/run_1proc_elbrus.sh prompt   # single-process inference

Override через env:
  PT_BUILD_DIR=...   путь до build dir (default: build_elbrus/)
  PT_TARGETS="..."   список targets
  PT_JOBS=N          параллельность сборки
  PT_RECONFIGURE=1   принудительный re-configure

EOF
