# Gemini 3.1 Pro consultation — gemini speculative decode

### 1. KV-Cache and Attention Mask for Batched Verify

In a standard single-token decode (batch=1, seqlen=1), the forward pass is memory-bandwidth bound. The attention operation is a matrix-vector multiplication (GEMV), and the KV cache is extended by exactly 1 position per layer. 

For batched speculative verification (batch=1, seqlen=$K$), the main model evaluates the last accepted token $x_0$ and the $K-1$ draft tokens $x_1 \dots x_{K-1}$ simultaneously. This transforms the attention operation from GEMV to a matrix-matrix multiplication (GEMM), which is highly advantageous for the Elbrus E8C2 VLIW architecture due to vastly improved arithmetic intensity and register reuse.

**Attention Mask Changes:**
Instead of a `[1, ctx]` vector of ones, the attention mask becomes a `[K, ctx + K]` matrix. 
- The first `ctx` columns (past tokens) are fully unmasked (e.g., `0.0` in log-space).
- The last `K` columns form a causal lower-triangular matrix. Token $j$ (where $0 \le j < K$) can attend to itself and previous draft tokens, but not future draft tokens.

**KV-Cache Changes (GQA):**
For Qwen3 (which uses Grouped Query Attention), the main model generates Key and Value tensors of shape `[K, n_kv_heads, head_dim]` at each layer. 
- **Standard decode:** Appends `[1, n_kv_heads, head_dim]` to the persistent cache.
- **Batched verify:** Concatenates the entire `[K, n_kv_heads, head_dim]` block to the persistent cache at once. The query tensor `[K, n_q_heads, head_dim]` attends to the newly formed `[ctx + K, n_kv_heads, head_dim]` cache.

### 2. The Verify Algorithm and Acceptance Rule

Given $K$ draft tokens $x_1 \dots x_K$ generated autoregressively by the draft model, we prepend the last accepted token $x_0$. The main model runs a single forward pass on `[x_0, x_1, ..., x_{K-1}]` to produce logits for positions `ctx` through `ctx + K - 1`.

For an FP32 CPU stack where RNG overhead and complex sampling can stall the pipeline, **Greedy Argmax Match** is the most efficient acceptance rule. 

**The Rule:** For each position $i \in [0, K-1]$, compare the draft token $x_{i+1}$ to the main model's argmax token at that position. If they match, accept $x_{i+1}$ and continue to $i+1$. If they mismatch, reject $x_{i+1}$, accept the main model's argmax token as the true token for that position, and discard all subsequent draft tokens.

```python
def speculative_decode_step(main_model, draft_model, prompt, K, ctx_len):
    # 1. Draft phase: generate K tokens sequentially
    draft_tokens = []
    x_curr = prompt[-1]
    for _ in range(K):
        draft_logits = draft_model.forward(x_curr)
        x_curr = argmax(draft_logits)
        draft_tokens.append(x_curr)
        
    # 2. Verify phase: main model forward pass on [x_0, x_1, ..., x_{K-1}]
    # Returns logits of shape [K, vocab_size]
    verify_inputs = [prompt[-1]] + draft_tokens[:-1]
    main_logits = main_model.forward_batched(verify_inputs)
    
    accepted_tokens = []
    mismatch_idx = K # Default to all accepted
    
    # 3. Acceptance Loop (Greedy)
    for i in range(K):
        main_tok = argmax(main_logits[i])
        if main_tok == draft_tokens[i]:
            accepted_tokens.append(draft_tokens[i])
        else:
            # First mismatch!
            accepted_tokens.append(main_tok)
            mismatch_idx = i
            break
            
    return accepted_tokens, mismatch_idx
```

### 3. The Reject-at-Position-$i$ Path and KV-Cache Rollback

When a mismatch occurs at index $i$ (meaning $x_{i+1}$ is rejected):
1. **Draft Model KV-Cache:** Must be physically or logically rewound. The draft model generated $K$ tokens, advancing its cache by $K$. We must move its sequence length pointer back by $K - i - 1$ positions so it aligns with the newly corrected token.
2. **Main Model KV-Cache:** **Nothing needs to be rolled back.** 

*Why?* During the batched forward pass, the main model computes KV states for $x_0 \dots x_{K-1}$. If a rejection happens at $i$, it means $x_0 \dots x_i$ were valid, and $x_{i+1}$ was invalid. The main model *never computed* the KV states for the rejected token $x_{i+1}$ or anything after it, because the input to the main model was shifted by one (`[x_0 ... x_{K-1}]`). 
The KV states appended during this forward pass correspond exactly to the accepted prefix plus the corrected token. We simply update the main model's persistent sequence length pointer to `ctx + i + 1`. The memory allocated for positions beyond `i + 1` in the pre-allocated KV buffer is simply overwritten on the next step.

### 4. Concrete $K$ Calculation

The theoretical speedup multiplier is:
$$ M = \frac{\sum_{i=0}^K p^i}{1 + K \cdot c} $$
Given:
- Acceptance rate $p = 0.8$
- Cost ratio $c = t_{draft} / t_{main} = 1/7 \approx 0.1428$

Let's evaluate the expected tokens per step (Numerator) and the cost per step relative to a single main model pass (Denominator) for various $K$:

*   **$K=2$**: 
    *   Num: $1 + 0.8 + 0.64 = 2.44$
    *   Denom: $1 + 2(0.1428) = 1.285$
    *   **$M = 1.898\times$**
*   **$K=3$**: 
    *   Num: $2.44 + 0.512 = 2.952$
    *   Denom: $1 + 3(0.1428) = 1.428$
    *   **$M = 2.067\times$**
*   **$K=4$**: 
    *   Num: $2.952 + 0.4096 = 3.3616$
    *   Denom: $1 + 4(0.1428) = 1.571$
    *   **$M = 2.139\times$**
*   **$K=5$**: 
    *   Num: $3.3616 + 0.32768 = 3.689$
    *   Denom: $1 + 5(0.1428) = 1.714$
    *   **$M = 2.152\times$**  *(Optimal)*
*   **$K=6$**: 
    *   Num: $3.689 + 0.26214 = 3.951$
    *   Denom: $1 + 6(0.1428) = 1.857$
    *   **$M = 2.127\times$**

**Optimal $K = 5$.** At $K=5$, the system yields a ~2.15x throughput multiplier. If the main model currently projects to 7.5 tok/s, speculative decoding will push it to **~16.1 tok/s**.

### 5. NUMA Layout on Elbrus E8C2

The E8C2 has 4 NUMA nodes. Cross-node memory access incurs significant latency, which destroys VLIW pipeline efficiency if not handled carefully.

**Layout Strategy:**
- **Main Model Weights & Compute:** Tensor Parallel across all 4 nodes (TP-4). Each node holds 1/4 of the attention heads and MLP weights.
- **Main KV Cache:** Distributed. Each node holds the KV cache for its specific TP shard of the attention heads.
- **Draft Model Weights & Compute:** Pinned entirely to **Node 0**. (0.6B params in FP32 is ~2.4GB, fitting easily in Node 0's 30GB RAM alongside its 1B param main-model shard).
- **Draft KV Cache:** Pinned entirely to **Node 0**.

**Process Topology:**
Do not run the draft model inside the main TP-4 Rank 0 process sequentially. Doing so forces Ranks 1, 2, and 3 to idle completely while Rank 0 computes the draft tokens. 
Instead, run the draft model in a **5th process** (or a dedicated pthread pinned to isolated cores on Node 0). 
1. The Draft Process generates $K$ tokens.
2. It writes them to a Shared Memory (shm) buffer.
3. Ranks 0-3 read the draft tokens from shm, synchronize via MPI/barrier, and execute the TP-4 batched verify pass.
4. Rank 0 writes the accepted token count back to shm so the Draft Process can rewind its KV cache and begin the next draft phase.

### 6. Failure Modes

**A. Draft acceptance drops < 50% — does spec still pay?**
If $p = 0.4$ and $K=5$:
- Numerator: $1 + 0.4 + 0.16 + 0.064 + 0.025 + 0.01 = 1.659$ expected tokens.
- Denominator: $1.714$ cost.
- Multiplier: $1.659 / 1.714 = 0.96\times$.
**No, it becomes a penalty.** The overhead of running the draft model exceeds the time saved by the occasional accepted token. The system will run slower than standard autoregressive decoding. A dynamic $K$ controller should monitor the moving average of $p$ and disable spec-decode (set $K=0$) if $p$ drops below ~0.55.

**B. $K$ is too large — what's the worst-case cost?**
If you set $K=15$ for a model with $p=0.8$:
The numerator (expected tokens) asymptotes to $\frac{1}{1-0.8} = 5.0$. 
The denominator (cost) grows linearly: $1 + 15(0.1428) = 3.14$.
Multiplier drops to $5.0 / 3.14 = 1.59\times$ (down from the 2.15x peak). 
The worst-case cost is wasting CPU cycles and memory bandwidth computing draft tokens and main-model GEMMs for sequence positions that are statistically guaranteed to be discarded.

**C. Different tokenizers — would a Mistral-draft work on a Qwen3-main?**
**No.** Speculative decoding fundamentally relies on a 1:1 mapping of the vocabulary space. The acceptance step directly compares token IDs (or their probability distributions). If the draft model predicts token ID `28705` (which might be `<s>` in Mistral) and the main model evaluates it against its own vocabulary (where `28705` might be ` apple`), the main model will reject it immediately. Even if it accidentally accepts it based on logits, the semantic meaning is corrupted, leading to garbage generation. Draft and main models *must* share the exact same tokenizer.