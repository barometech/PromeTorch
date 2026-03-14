#!/bin/bash
# ============================================================================
# Cross-compile PromeTorch for Baikal-M / Baikal-S (ARM aarch64)
# ============================================================================
# Run from x86_64 host (Astra/Ubuntu/Debian)
# Prerequisites: apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake
#
# Usage:
#   ./scripts/build-baikal-cross.sh          # Baikal-M (Cortex-A57)
#   ./scripts/build-baikal-cross.sh baikal-s # Baikal-S (Cortex-A75)
# ============================================================================

set -e

VARIANT="${1:-baikal-m}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

case "$VARIANT" in
    baikal-m|m)
        TOOLCHAIN="$PROJECT_DIR/cmake/toolchains/aarch64-baikal-m.cmake"
        BUILD_DIR="$PROJECT_DIR/build_baikal_m"
        echo "=== PromeTorch: Cross-compiling for Baikal-M (Cortex-A57) ==="
        ;;
    baikal-s|s)
        TOOLCHAIN="$PROJECT_DIR/cmake/toolchains/aarch64-baikal-s.cmake"
        BUILD_DIR="$PROJECT_DIR/build_baikal_s"
        echo "=== PromeTorch: Cross-compiling for Baikal-S (Cortex-A75) ==="
        ;;
    *)
        echo "Usage: $0 [baikal-m|baikal-s]"
        exit 1
        ;;
esac

# Check cross-compiler
if ! command -v aarch64-linux-gnu-gcc &>/dev/null; then
    echo "ERROR: Cross-compiler not found. Install with:"
    echo "  sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
    exit 1
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
cmake "$PROJECT_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPT_USE_TUDA=ON \
    -DPT_USE_LINQ=OFF \
    -DPT_BUILD_TESTS=ON \
    -DPT_BUILD_SHARED_LIBS=ON

# Build
cmake --build . -j$(nproc)

echo ""
echo "=== Cross-build complete: $BUILD_DIR ==="
echo "Copy to target and run: scp -r $BUILD_DIR user@baikal:/opt/promethorch/"
