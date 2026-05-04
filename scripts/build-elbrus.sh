#!/bin/bash
# ============================================================================
# Build PromeTorch on Elbrus OS (E2K, LCC compiler)
# ============================================================================
# Run natively on Elbrus hardware (8C/16C/8SV)
# OS: Elbrus OS (PDK LE) or Astra Linux SE for Elbrus
# Compiler: LCC (MCST C/C++ compiler)
# Prerequisites: install cmake from PDK repository
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build_elbrus"
TOOLCHAIN="$PROJECT_DIR/cmake/toolchains/e2k-elbrus.cmake"

echo "=== PromeTorch: Building for Elbrus E2K ==="

# Check LCC compiler
if ! command -v lcc &>/dev/null; then
    echo "ERROR: LCC compiler not found."
    echo "Ensure you are running on Elbrus OS with MCST SDK installed."
    exit 1
fi

# Check cmake
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install from PDK repository."
    exit 1
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure (native build — no cross-compilation needed)
cmake "$PROJECT_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPT_USE_TUDA=ON \
    -DPT_USE_LINQ=OFF \
    -DPT_BUILD_TESTS=ON \
    -DPT_BUILD_SHARED_LIBS=ON

# Build (Elbrus 8C has 8 cores)
cmake --build . -j$(nproc)

echo ""
echo "=== Build complete: $BUILD_DIR ==="
echo "Run tests: cd $BUILD_DIR && ctest --output-on-failure"
