#!/bin/bash
# ============================================================================
# Build PromeTorch on Astra Linux SE (x86_64, Debian-based)
# ============================================================================
# Prerequisites: apt install build-essential cmake git
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build_astra"
TOOLCHAIN="$PROJECT_DIR/cmake/toolchains/x86_64-astra.cmake"

echo "=== PromeTorch: Building for Astra Linux SE (x86_64) ==="

# Check dependencies
for cmd in gcc g++ cmake make; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install with:"
        echo "  sudo apt install build-essential cmake"
        exit 1
    fi
done

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
echo "=== Build complete: $BUILD_DIR ==="
echo "Run tests: cd $BUILD_DIR && ctest --output-on-failure"
