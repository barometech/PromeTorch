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
    local header="$1"; local pkg="$2"
    if ! find /usr/include /usr/local/include /opt/mcst/eml/include /opt/eml/include \
            -name "$(basename "$header")" 2>/dev/null \
            | grep -q "/$header\$\|/$(basename "$header")\$"; then
        MISSING+=("$pkg ($header)")
    fi
}

check_lib() {
    local lib="$1"; local pkg="$2"
    if ! ldconfig -p 2>/dev/null | grep -q "lib$lib\\.so"; then
        if ! find /usr/lib /usr/lib64 /opt/mcst/eml/lib 2>/dev/null \
                | grep -qE "lib$lib\\.so"; then
            MISSING+=("$pkg (lib$lib.so)")
        fi
    fi
}

check_bin() {
    if ! command -v "$1" >/dev/null 2>&1; then
        MISSING+=("$2 (executable: $1)")
    fi
}

# Auto-detect компилятор
detect_compiler() {
    if command -v lcc++ >/dev/null 2>&1; then
        export CC="lcc"
        export CXX="lcc++"
        echo "[deps] Компилятор: LCC ($(lcc --version 2>&1 | head -1))"
    elif command -v gcc-elbrus >/dev/null 2>&1; then
        export CC="gcc-elbrus"
        export CXX="g++-elbrus"
        echo "[deps] Компилятор: gcc-elbrus ($(gcc-elbrus --version 2>&1 | head -1))"
    elif command -v gcc >/dev/null 2>&1 && \
         echo "int main(){return 0;}" | gcc -march=elbrus-v4 -x c - -o /dev/null 2>/dev/null; then
        export CC="gcc"
        export CXX="g++"
        echo "[deps] Компилятор: system gcc с e2k-поддержкой ($(gcc --version 2>&1 | head -1))"
    else
        return 1
    fi
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
# 2. CMake configure
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
