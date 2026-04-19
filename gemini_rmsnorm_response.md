This is an excellent, textbook example of how Undefined Behavior (UB) manifests in High-Performance Computing, particularly on VLIW architectures like the Elbrus E8C2. 

Here is a thorough, critical review of your hypothesis, the LCC compiler's behavior, and the surrounding code.

---

### 1. Is the `__restrict` aliasing UB hypothesis CORRECT?
**Yes, absolutely.** 

In C++, `__restrict` is a strict promise to the compiler: *“For the lifetime of this pointer, the memory it points to will not be accessed by any other pointer.”* By passing the same buffer to `x` and `out`, you violated this contract, invoking Undefined Behavior.

While a standard x86 Out-of-Order (OoO) processor with GCC/Clang *might* accidentally survive this read-before-write pattern, **VLIW architectures like Elbrus will almost certainly break.** VLIW relies entirely on the compiler (LCC) to schedule instructions. Because you promised `x` and `out` do not alias, LCC is legally allowed to assume that stores to `out` have zero side effects on loads from `x`. It will aggressively reorder, pipeline, and overlap these memory operations, leading to data corruption.

### 2. Could there be a DIFFERENT bug that the fix accidentally addressed?
Given the evidence, the `__restrict` violation is the primary culprit. We can rule out the others:
*   **Cache coherency / OpenMP race:** Unlikely. Your OpenMP loop schedules over `i` (the batch/sequence dimension). Each thread writes to a completely independent chunk of memory (`out + i * D`). There is no cross-thread data sharing here.
*   **Memory bandwidth:** Contention causes slowdowns, not deterministic garbage text.
*   **LCC 1.29 specific issue:** It's not a "bug" in LCC; it is LCC working exactly as designed. LCC is an incredibly aggressive compiler because Elbrus hardware requires it. It exploited your `__restrict` promise to the maximum extent.

### 3. How exactly did LCC break `rmsnorm`? (Prefetch, SIMD, Loop Fusion?)
On Elbrus/LCC, the most likely failure mode is **Software Pipelining (SWP) combined with Asynchronous Prefetching.**

Even though the algorithm looks like a safe element-wise operation (`oi[d] = xi[d] * ...`), LCC does not execute one `d` iteration at a time. To keep the 6 FPU channels fed, LCC overlaps iterations. 
*   **Cycle N:** Load `xi[d+12]` to `xi[d+17]` (Prefetching future iterations).
*   **Cycle N+1:** Multiply `xi[d+6]` to `xi[d+11]` (Computing current iterations).
*   **Cycle N+2:** Store `oi[d]` to `oi[d+5]` (Writing past iterations).

Because of `__restrict`, LCC omits hardware memory-dependency barriers between the prefetch unit and the store unit. If `x` and `out` are the same pointer, the delayed stores to `oi` can physically overwrite the memory in the L1 cache *while* the asynchronous prefetcher is trying to load `xi` for a future iteration, or the prefetcher might pull stale data before the store completes. The read-before-write safety is destroyed by the compiler's temporal reordering.

**Your fix (using separate buffers) is the 100% correct solution.**

### 4. Is `silu_fwd` in-place actually safe?
**No, it is not safe.** 

Even though `silu_fwd` is purely element-wise (`out[i] = x[i] * sigmoid(x[i])`), the exact same Software Pipelining hazard applies. LCC will issue wide, asynchronous vector loads for `x[i+12]` while simultaneously issuing vector stores for `out[i]`. Without alias checking, the store queue and load queue at the hardware level can step on each other. 

The fact that `silu_fwd` was in-place was absolutely contributing to the garbage generation. Your fix to separate the buffers was necessary here as well.

### 5. Should we remove `__restrict` to be safe?
**DO NOT remove `__restrict`.** 

On Elbrus, removing `__restrict` is a death sentence for performance. Without it, LCC must assume *every* pointer might alias. It will insert pipeline flushes and memory barriers inside your innermost loops, completely destroying Software Pipelining and VLIW packing. Your 6-wide FPU utilization will plummet.

**The correct HPC practice:**
1. Keep `__restrict` in the signatures.
2. Enforce the contract at the API level.
3. Add debug assertions to prevent this from ever happening again:
   ```cpp
   static void rmsnorm_fwd(const float* __restrict x, ..., float* __restrict out, ...) {
       assert(x != out && "rmsnorm_fwd cannot be done in-place!");
       // ...
   }
   ```

### 6. OTHER BUGS: The "Gotcha" in your new code
I strongly advise you to look at this line in your **NEW (fixed)** code:

```cpp
fused::add_fwd(buf_pir.data(), buf_val.data(), buf_pir.data(), seq_len * D);
```
And this one:
```cpp
fused::add_fwd(buf_x.data(), buf_pir.data(), buf_x.data(), seq_len * D);
```

If your `fused::add_fwd` signature looks anything like this:
`void add_fwd(const float* __restrict a, const float* __restrict b, float* __restrict out)`

**You are still invoking Undefined Behavior.** You are passing `buf_pir` as both an input and an output. Residual connections (`x = x + val`) are the #1 place where developers accidentally violate `__restrict`. If `add_fwd` uses `__restrict`, you must either:
1. Create a specific `add_fwd_inplace(float* x, const float* __restrict val)` that does not use `__restrict` on the mutated pointer.
2. Use a third buffer.

#### Additional structural risk: Loop bounds
In `silu_fwd`, you have:
```cpp
for (int64_t i = 0; i < n; i += 6)
```
If `n` (which is `seq_len * H`) is **not an exact multiple of 6**, this loop will read and write out of bounds, causing memory corruption. 
*   For a 189M model, `H` is likely a power of 2 (e.g., 2048, 4096) or a multiple of 768 (e.g., 3072). 
*   3072 is divisible by 6, so you are safe. 
*   However, if you ever change the model architecture such that `H` is, say, 4096 (which is *not* divisible by 6; 4096 % 6 = 4), this loop will crash or corrupt memory during generation. You should add a static or runtime assertion: `assert(n % 6 == 0);`.

### Summary
Your hypothesis was spot on. You successfully identified a classic VLIW/C++ aliasing bug. Keep `__restrict` for performance, keep the buffers separate, add assertions to protect the pointers, and **check your `add_fwd` implementation immediately.**