#!/bin/bash
# ============================================================================
# Build PromeTorch on RED OS (x86_64, RHEL-based)
# ============================================================================
# Prerequisites: dnf group install "Development Tools" && dnf install cmake
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build_redos"
TOOLCHAIN="$PROJECT_DIR/cmake/toolchains/x86_64-redos.cmake"

echo "=== PromeTorch: Building for RED OS (x86_64) ==="

# Check dependencies
for cmd in gcc g++ cmake make; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install with:"
        echo "  sudo dnf group install 'Development Tools'"
        echo "  sudo dnf install cmake"
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
