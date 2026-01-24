FROM gcc:13

# Install CMake and other build tools
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    libomp-dev \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source files
COPY . .

# Convert line endings from CRLF to LF
RUN find . -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.cmake" -o -name "CMakeLists.txt" \) -exec dos2unix {} \;

# Create build directory and build
RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DPT_USE_CUDA=OFF \
        -DPT_BUILD_TESTS=OFF && \
    ninja

# Run tests
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
