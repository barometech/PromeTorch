"""
Consult Gemini 3.1 Pro on the in-place rmsnorm fix for PromeTorch/PIR generate_text.
"""
import httpx
import json
import sys

API_KEY = "AIzaSyB5HQtg_LhQI8CdOmMdvuk44D9_wEDUcYY"
PROXY = "http://D9xMFRSp:7WR1WPbB@172.120.255.225:64838"

# Read relevant code sections
fused_step_path = "/c/Users/paper/Desktop/promethorch/aten/src/ATen/native/cpu/hot_loops_rmsnorm.h"

# Actual rmsnorm_fwd from fused_step.h
RMSNORM_CODE = """
// From examples/pir/fused_step.h
static void rmsnorm_fwd(const float* __restrict x, const float* __restrict weight, float* __restrict out,
                        float* __restrict rms_cache,
                        int64_t BT, int64_t D, float eps = 1e-6f) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < BT; i++) {
        const float* __restrict xi = x + i * D;
        float* __restrict oi = out + i * D;
        // 6-wide accumulation for VLIW (6 FPU channels)
        float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0;
        #pragma loop count(768)
        for (int64_t d = 0; d < D; d += 6) {
            s0 += xi[d]*xi[d]; s1 += xi[d+1]*xi[d+1]; s2 += xi[d+2]*xi[d+2];
            s3 += xi[d+3]*xi[d+3]; s4 += xi[d+4]*xi[d+4]; s5 += xi[d+5]*xi[d+5];
        }
        float inv_rms = 1.0f / std::sqrt((s0+s1+s2+s3+s4+s5) / D + eps);
        rms_cache[i] = inv_rms;
        #pragma loop count(768)
        for (int64_t d = 0; d < D; d += 6) {
            oi[d]=xi[d]*inv_rms*weight[d]; oi[d+1]=xi[d+1]*inv_rms*weight[d+1];
            oi[d+2]=xi[d+2]*inv_rms*weight[d+2]; oi[d+3]=xi[d+3]*inv_rms*weight[d+3];
            oi[d+4]=xi[d+4]*inv_rms*weight[d+4]; oi[d+5]=xi[d+5]*inv_rms*weight[d+5];
        }
    }
}
"""

OLD_CODE = """
// OLD (buggy) — generate_text used IN-PLACE rmsnorm_fwd:
fused::linear_fwd(buf_out.data(), pw.W_out, buf_gate.data(), seq_len, D, D);
fused::rmsnorm_fwd(buf_gate.data(), pw.norm_w, buf_gate.data(),  // <-- SAME buffer in and out
                   buf_rms.data(), seq_len, D);
fused::add_fwd(buf_pir.data(), buf_gate.data(), buf_pir.data(), seq_len * D);

// Also:
fused::linear_fwd(buf_pir.data(), bw.W_mix, buf_mix.data(), seq_len, D, D);
fused::rmsnorm_fwd(buf_mix.data(), bw.norm_pir_w, buf_mix.data(),  // <-- SAME buffer in and out
                   buf_rms.data(), seq_len, D);

// Also:
fused::silu_fwd(buf_ffn1.data(), buf_ffn1.data(), seq_len * H);  // <-- in-place silu
"""

NEW_CODE = """
// NEW (fixed) — use separate buffers:
fused::linear_fwd(buf_out.data(), pw.W_out, buf_gate.data(), seq_len, D, D);
fused::rmsnorm_fwd(buf_gate.data(), pw.norm_w, buf_val.data(),   // <-- OUT = buf_val (different)
                   buf_rms.data(), seq_len, D);
fused::add_fwd(buf_pir.data(), buf_val.data(), buf_pir.data(), seq_len * D);

// Also:
fused::linear_fwd(buf_pir.data(), bw.W_mix, buf_mix.data(), seq_len, D, D);
fused::rmsnorm_fwd(buf_mix.data(), bw.norm_pir_w, buf_pir.data(), // <-- OUT = buf_pir (reuse, PIR done)
                   buf_rms.data(), seq_len, D);
fused::add_fwd(buf_x.data(), buf_pir.data(), buf_x.data(), seq_len * D);

// Also:
fused::silu_fwd(buf_ffn1.data(), buf_ffn_gated.data(), seq_len * H); // <-- separate buffer
"""

SILU_CODE = """
// From fused_step.h — note __restrict in signature:
static void silu_fwd(const float* __restrict x, float* __restrict out, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 6) {
        float s0 = 1.0f/(1.0f+std::exp(-x[i]));   out[i]   = x[i]*s0;
        float s1 = 1.0f/(1.0f+std::exp(-x[i+1])); out[i+1] = x[i+1]*s1;
        // ... 6-wide unroll
    }
}
"""

PROMPT = f"""You are a senior C++/HPC expert reviewing a bug fix in a deep learning framework written in C++ for Elbrus E8C2 (VLIW architecture) processors using LCC compiler 1.29 (based on GCC).

CONTEXT:
- PromeTorch framework, PIR 250M language model
- Char-level training on Russian text, 189M parameters
- Training forward pass produces correct loss (1.41 at step 200)
- Generation with generate_text() was producing garbage text
- Same checkpoint, same weights — training worked, generation didn't

HYPOTHESIS THAT WAS APPLIED:
The bug was that generate_text() called rmsnorm_fwd with the same buffer for input and output (in-place), while training used separate buffers. Since rmsnorm_fwd declares all pointer parameters as __restrict, the compiler was allowed to assume no aliasing. On LCC/Elbrus, this likely caused incorrect code generation for the in-place case.

RMSNORM IMPLEMENTATION:
```cpp
{RMSNORM_CODE}
```

SILU IMPLEMENTATION:
```cpp
{SILU_CODE}
```

OLD (BUGGY) CALLS IN generate_text():
```cpp
{OLD_CODE}
```

NEW (FIXED) CALLS:
```cpp
{NEW_CODE}
```

TRAINING FORWARD (always used separate buffers — never had the bug):
```cpp
// rmsnorm_fwd(pa.out_proj_out, pw.norm_w, pa.norm_out, ...) — separate buffers
// silu_fwd(la.ffn1_out, la.ffn1_silu, ...) — separate buffers
```

EVIDENCE THE FIX WORKED:
Before fix, generate produced: "Лиса Во саматились автобус и теплохибиальные основания в Троились"
After fix: "Шасть 27 Петратинам, профевианского захватель. Вседких по для в делару командуюля от социальяния"
(Still char-level garbage but grammatically coherent — dates, cases, structure appear correct)

QUESTIONS FOR YOU:
1. Is the __restrict aliasing UB hypothesis CORRECT? With aliasing but the code pattern (read all xi for sum → write oi), would a correct compiler actually break? Or is this theoretically safe because reads happen before writes?

2. Could there be a DIFFERENT bug that the fix accidentally addressed? For example:
   - Cache coherency issues on VLIW
   - Memory bandwidth contention
   - OpenMP parallel race between threads writing to same buffer
   - Some specific LCC 1.29 compiler issue

3. For rmsnorm specifically, the algorithm reads x[d] for d=0..D TWICE (once for sum-of-squares, once for normalization). In the second loop, it reads x[d] and writes to out[d]. With aliasing and __restrict, the compiler might:
   - Speculatively prefetch x[d+1] while storing out[d]
   - Use SIMD/vectorization assuming 6 consecutive elements can be loaded without interference
   - Reorder the two loops entirely (fuse them)

   Which of these is most likely on LCC 1.29? Is my fix the correct solution?

4. For silu_fwd, the operation is purely element-wise: out[i] = x[i] * sigmoid(x[i]). Read x[i], write out[i] — even with aliasing, this seems safe per element. Do you agree that silu in-place is NOT actually the bug, but the rmsnorm fix is sufficient?

5. Should we remove __restrict from rmsnorm_fwd signature to be safe? Or keep it and require callers to use separate buffers?

6. Any OTHER bug you can spot in the generate_text code pattern that might cause "training works, inference doesn't"?

Please be thorough and critical. This is production code on a Russian national processor (Elbrus), no second chances for bugs.
"""

async def consult():
    async with httpx.AsyncClient(
        proxy=PROXY,
        timeout=600.0,
        verify=False,
    ) as client:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": PROMPT}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 16384,
            },
        }
        print("Sending to Gemini 3.1 Pro...")
        print(f"Prompt length: {len(PROMPT)} chars")
        resp = await client.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text[:2000])
            return
        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            print("Unexpected response structure:")
            print(json.dumps(data, indent=2)[:2000])
            return
        out_path = "gemini_rmsnorm_response.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Response saved to {out_path}")
        print("=" * 60)
        print(text)

import asyncio
asyncio.run(consult())
