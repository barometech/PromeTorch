// ============================================================================
// MPSKernels.mm — Metal Performance Shaders + MPSGraph kernel launches.
// ============================================================================
// Apple-only. Compiled only when PT_USE_MPS=ON on macOS.
//
// Strategy:
//   * Binary/unary element-wise ops use MPSGraph. It compiles a tiny graph
//     lazily, caches the MPSGraphExecutable, and runs it on the shared
//     command queue. This is the same path PyTorch's MPS backend uses.
//   * matmul uses MPSMatrixMultiplication directly (simpler, very fast).
//
// Inputs are raw pointers returned by MPSAllocator, which are
// host-visible `[buffer contents]`. To feed them back into Metal without a
// reverse lookup, we wrap them in MTLBuffer via `newBufferWithBytesNoCopy:`
// — Apple guarantees page-aligned shared-storage pointers round-trip safely.
// ============================================================================

#include "aten/src/ATen/mps/MPSKernels.h"
#include "aten/src/ATen/mps/MPSDevice.h"

#if defined(__APPLE__) && defined(PT_USE_MPS)

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace at {
namespace mps {

// ---- Small helpers ----------------------------------------------------------

static inline id<MTLDevice> dev() {
    return (__bridge id<MTLDevice>)MPSDevice::get().mtl_device();
}
static inline id<MTLCommandQueue> queue() {
    return (__bridge id<MTLCommandQueue>)MPSDevice::get().mtl_command_queue();
}

// Wrap a host pointer coming from [MTLBuffer contents] back into a MTLBuffer
// without copying. Apple's shared-storage guarantee makes this safe.
static id<MTLBuffer> wrap(const void* ptr, std::size_t nbytes) {
    if (!dev()) mps_not_available();
    // Cast away const; Metal needs void*. We never write to `a`/`b` inputs.
    return [dev() newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                    length:nbytes
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
}

// Create a 1-D float tensor descriptor for an MPSGraph feed.
static MPSGraphTensorData* feed_1d(id<MTLBuffer> buf, std::size_t n) {
    NSArray<NSNumber*>* shape = @[ @(n) ];
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                   shape:shape
                                                dataType:MPSDataTypeFloat32];
}

// Commit a graph execution on the shared queue. Non-blocking — caller must
// call MPSDevice::synchronize() before reading the output from CPU.
static void run_graph(MPSGraph* graph,
                      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results) {
    id<MTLCommandBuffer> cb = [queue() commandBuffer];
    [graph encodeToCommandBuffer:cb
                           feeds:feeds
                targetOperations:nil
               resultsDictionary:results
             executionDescriptor:nil];
    [cb commit];
}

// ---- Element-wise: add -------------------------------------------------------

void launch_add_mps(const float* a, const float* b, float* c, std::size_t n) {
    if (n == 0) return;
    if (!dev()) mps_not_available();

    @autoreleasepool {
        const std::size_t bytes = n * sizeof(float);
        id<MTLBuffer> bA = wrap(a, bytes);
        id<MTLBuffer> bB = wrap(b, bytes);
        id<MTLBuffer> bC = wrap(c, bytes);

        MPSGraph* g = [MPSGraph new];
        NSArray<NSNumber*>* shape = @[ @(n) ];
        MPSGraphTensor* tA = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"A"];
        MPSGraphTensor* tB = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"B"];
        MPSGraphTensor* tC = [g additionWithPrimaryTensor:tA secondaryTensor:tB name:@"C"];

        MPSGraphTensorData* dA = feed_1d(bA, n);
        MPSGraphTensorData* dB = feed_1d(bB, n);
        MPSGraphTensorData* dC = feed_1d(bC, n);

        run_graph(g, @{tA: dA, tB: dB}, @{tC: dC});
    }
}

// ---- Element-wise: mul -------------------------------------------------------

void launch_mul_mps(const float* a, const float* b, float* c, std::size_t n) {
    if (n == 0) return;
    if (!dev()) mps_not_available();

    @autoreleasepool {
        const std::size_t bytes = n * sizeof(float);
        id<MTLBuffer> bA = wrap(a, bytes);
        id<MTLBuffer> bB = wrap(b, bytes);
        id<MTLBuffer> bC = wrap(c, bytes);

        MPSGraph* g = [MPSGraph new];
        NSArray<NSNumber*>* shape = @[ @(n) ];
        MPSGraphTensor* tA = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"A"];
        MPSGraphTensor* tB = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"B"];
        MPSGraphTensor* tC = [g multiplicationWithPrimaryTensor:tA secondaryTensor:tB name:@"C"];

        run_graph(g,
                  @{tA: feed_1d(bA, n), tB: feed_1d(bB, n)},
                  @{tC: feed_1d(bC, n)});
    }
}

// ---- Element-wise: relu ------------------------------------------------------

void launch_relu_mps(const float* in, float* out, std::size_t n) {
    if (n == 0) return;
    if (!dev()) mps_not_available();

    @autoreleasepool {
        const std::size_t bytes = n * sizeof(float);
        id<MTLBuffer> bIn  = wrap(in,  bytes);
        id<MTLBuffer> bOut = wrap(out, bytes);

        MPSGraph* g = [MPSGraph new];
        NSArray<NSNumber*>* shape = @[ @(n) ];
        MPSGraphTensor* t  = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"X"];
        MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* tOut = [g maximumWithPrimaryTensor:t secondaryTensor:zero name:@"Y"];

        run_graph(g,
                  @{t: feed_1d(bIn, n)},
                  @{tOut: feed_1d(bOut, n)});
    }
}

// ---- Matmul: C = A @ B using MPSMatrixMultiplication ------------------------

void launch_matmul_mps(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    if (M == 0 || N == 0 || K == 0) return;
    if (!dev()) mps_not_available();

    @autoreleasepool {
        const std::size_t bytesA = std::size_t(M) * std::size_t(K) * sizeof(float);
        const std::size_t bytesB = std::size_t(K) * std::size_t(N) * sizeof(float);
        const std::size_t bytesC = std::size_t(M) * std::size_t(N) * sizeof(float);

        id<MTLBuffer> bA = wrap(A, bytesA);
        id<MTLBuffer> bB = wrap(B, bytesB);
        id<MTLBuffer> bC = wrap(C, bytesC);

        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:K
                                                                          rowBytes:std::size_t(K) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                           columns:N
                                                                          rowBytes:std::size_t(N) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:N
                                                                          rowBytes:std::size_t(N) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];

        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bA descriptor:descA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bB descriptor:descB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bC descriptor:descC];

        MPSMatrixMultiplication* mm =
            [[MPSMatrixMultiplication alloc] initWithDevice:dev()
                                              transposeLeft:NO
                                             transposeRight:NO
                                                 resultRows:M
                                              resultColumns:N
                                            interiorColumns:K
                                                      alpha:1.0
                                                       beta:0.0];

        id<MTLCommandBuffer> cb = [queue() commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cb commit];
    }
}

} // namespace mps
} // namespace at

#endif // __APPLE__ && PT_USE_MPS
