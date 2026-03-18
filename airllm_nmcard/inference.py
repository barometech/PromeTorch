"""
inference.py -- AirLLM-style layer-streaming inference on NM Card Mini.

Core design (adapted from AirLLM):
  1. Load one transformer layer's weights from disk (safetensors)
  2. Upload weight matrices to NM Card DDR
  3. Dispatch matmul to 16 NMC4 cores (4 primary vectorized, 12 secondary scalar)
  4. Run element-wise ops (RMSNorm, SiLU, RoPE) on CPU or card
  5. Discard weights, load next layer
  6. Repeat for all 36 layers

Memory budget:
  - NM Card DDR: 5 GB total (~4.5 GB usable after firmware)
  - Largest single layer (FP32): ~430 MB (BF16 original ~215 MB)
  - With INT8 quantization: ~108 MB per layer
  - lm_head: 151936*2560 = ~1.49 GB FP32 -> needs tiling (6 tiles of ~256 MB)
  - KV cache per layer: 2 * 8 * seq_len * 128 * 4 bytes
    - At seq_len=2048: ~16 MB per layer (stored on host, loaded as needed)

Bandwidth:
  - PCIe x4 DDR3L: ~1.2 GB/s theoretical
  - Per layer upload (INT8): ~108 MB -> ~90ms
  - Per layer upload (FP32): ~430 MB -> ~358ms
  - Expected throughput: ~0.5-1.5 tok/s (layer streaming bottleneck)
"""

import os
import gc
import time
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from .ops import (
    rmsnorm, precompute_freqs_cis, apply_rope, silu,
    softmax, gqa_attention, linear, embedding, sample_top_p
)
from .layer_loader import (
    LayerLoader, ModelConfigLoader,
    quantize_int8_per_channel, dequantize_int8
)
from .tokenizer import BPETokenizer


# ============================================================================
# NM Card Hardware Interface (ctypes-based)
# ============================================================================

class NMCardInterface:
    """Python interface to NM Card Mini via nm_card_load.dll.

    Mirrors the C++ NMCardHardware class from PromeTorch.
    All operations follow the protocol:
      write A,B to DDR -> set opcode -> wait STATUS=DONE -> read C

    SAFETY RULES:
      - NEVER run 16 concurrent nmpp calls (max 4 primary cores)
      - Always use timeout on card operations
      - Always close board descriptor on shutdown
    """

    # DDR Memory Layout (matches NMCardHardware.h)
    DDR_BASE        = 0x00340000
    CMD_BLOCK_SIZE  = 32        # 32 words per core
    STATUS_OFFSET   = 30
    WATCHDOG_OFFSET = 31
    DATA_START      = DDR_BASE + 512  # after 16 cmd blocks
    DDR_END         = 0x1FF00000      # ~500MB usable

    # Opcodes (match dispatcher_suda_mc.abs)
    OP_MATMUL       = 1
    OP_RMSNORM      = 2
    OP_SOFTMAX      = 3
    OP_SILU         = 4
    OP_ROPE         = 5
    OP_ELEM_ADD     = 10
    OP_ELEM_MUL     = 11
    OP_GATE_MUL     = 13
    OP_MATMUL_PARTIAL = 22
    OP_EXIT         = 255

    # Status codes
    STATUS_IDLE     = 0
    STATUS_BUSY     = 1
    STATUS_DONE     = 2

    def __init__(self):
        self._dll = None
        self._board = None
        self._access = [None] * 16
        self._num_cores = 0
        self._initialized = False
        self._ddr_next = self.DATA_START  # bump allocator

    def init(self, dll_path: str = "nm_card_load.dll",
             dispatcher_path: str = "dispatcher_suda_mc.abs") -> bool:
        """Initialize NM Card: load DLL, detect board, load dispatcher.

        Returns True if hardware is ready.
        """
        try:
            import ctypes
            self._ctypes = ctypes

            # Load DLL
            self._dll = ctypes.WinDLL(dll_path)

            # Get board count
            count = ctypes.c_uint(0)
            rc = self._dll.PL_GetBoardCount(ctypes.byref(count))
            if rc != 0 or count.value == 0:
                print("[NMCard] No board detected")
                return False

            # Get board descriptor
            self._board = ctypes.c_void_p()
            rc = self._dll.PL_GetBoardDesc(0, ctypes.byref(self._board))
            if rc != 0:
                print("[NMCard] Failed to get board descriptor")
                return False

            # Reset board
            self._dll.PL_ResetBoard(self._board)
            self._dll.PL_LoadInitCode(self._board)

            # Get access to cores (4 clusters x 4 cores)
            self._num_cores = 0
            for cluster in range(4):
                for nm in range(4):
                    core_no = (ctypes.c_int * 2)(nm, cluster)
                    access = ctypes.c_void_p()
                    rc = self._dll.PL_GetAccess(
                        self._board, core_no, ctypes.byref(access))
                    if rc == 0:
                        idx = cluster * 4 + nm
                        self._access[idx] = access
                        self._num_cores += 1

            if self._num_cores == 0:
                print("[NMCard] No cores accessible")
                return False

            # Load dispatcher on core 0 (it broadcasts to others)
            if os.path.exists(dispatcher_path):
                path_bytes = dispatcher_path.encode("utf-8")
                self._dll.PL_LoadProgramFile(self._access[0], path_bytes)

            self._initialized = True
            print(f"[NMCard] Initialized: {self._num_cores} cores")
            return True

        except Exception as e:
            print(f"[NMCard] Init failed: {e}")
            return False

    def is_available(self) -> bool:
        return self._initialized

    def shutdown(self):
        """Graceful shutdown: send EXIT, close handles."""
        if not self._initialized:
            return
        try:
            # Send EXIT to core 0
            self._dispatch_op(0, self.OP_EXIT, [])

            # Close access handles
            for i in range(16):
                if self._access[i] is not None:
                    self._dll.PL_CloseAccess(self._access[i])
                    self._access[i] = None

            # Close board
            if self._board is not None:
                self._dll.PL_CloseBoardDesc(self._board)
                self._board = None

            self._initialized = False
        except Exception as e:
            print(f"[NMCard] Shutdown error: {e}")

    def ddr_reset(self):
        """Reset DDR bump allocator (between layers)."""
        self._ddr_next = self.DATA_START

    def ddr_alloc(self, nbytes: int) -> int:
        """Allocate DDR space, returns word address."""
        words = (nbytes + 3) // 4
        words = (words + 15) & ~15  # align to 64 bytes
        if self._ddr_next + words > self.DDR_END:
            raise MemoryError(f"NMCard DDR OOM: need {words} words, "
                            f"have {self.DDR_END - self._ddr_next}")
        addr = self._ddr_next
        self._ddr_next += words
        return addr

    def write_ddr(self, data: np.ndarray, ddr_addr: int):
        """Write numpy array to DDR at given word address."""
        ctypes = self._ctypes
        raw = data.tobytes()
        word_count = len(raw) // 4
        buf = (ctypes.c_uint32 * word_count).from_buffer_copy(raw)
        self._dll.PL_WriteMemBlock(
            self._access[0], buf, ctypes.c_uint32(ddr_addr),
            ctypes.c_uint32(word_count))

    def read_ddr(self, ddr_addr: int, nbytes: int) -> np.ndarray:
        """Read from DDR, return as float32 array."""
        ctypes = self._ctypes
        word_count = nbytes // 4
        buf = (ctypes.c_uint32 * word_count)()
        self._dll.PL_ReadMemBlock(
            self._access[0], buf, ctypes.c_uint32(ddr_addr),
            ctypes.c_uint32(word_count))
        return np.frombuffer(buf, dtype=np.float32).copy()

    def _dispatch_op(self, core: int, opcode: int, args: List[int]):
        """Send opcode + args to a core's command block."""
        ctypes = self._ctypes
        cmd_addr = self.DDR_BASE + core * self.CMD_BLOCK_SIZE

        # Build command block: [opcode, arg0, arg1, ..., padding..., STATUS=0, WATCHDOG=0]
        cmd = [opcode] + args
        while len(cmd) < 30:
            cmd.append(0)
        cmd.append(0)  # STATUS = IDLE (triggers execution)
        cmd.append(0)  # WATCHDOG

        buf = (ctypes.c_uint32 * 32)(*cmd)
        self._dll.PL_WriteMemBlock(
            self._access[core] if self._access[core] else self._access[0],
            buf, ctypes.c_uint32(cmd_addr), ctypes.c_uint32(32))

    def _wait_done(self, core: int, timeout_sec: float = 10.0) -> bool:
        """Poll STATUS word until DONE or timeout."""
        ctypes = self._ctypes
        status_addr = self.DDR_BASE + core * self.CMD_BLOCK_SIZE + self.STATUS_OFFSET
        buf = (ctypes.c_uint32 * 1)()

        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            self._dll.PL_ReadMemBlock(
                self._access[core] if self._access[core] else self._access[0],
                buf, ctypes.c_uint32(status_addr), ctypes.c_uint32(1))
            if buf[0] == self.STATUS_DONE:
                return True
            time.sleep(0.0001)  # 100us poll interval

        print(f"[NMCard] TIMEOUT on core {core} after {timeout_sec}s")
        return False

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Dispatch matmul C = A @ B to NM Card.

        A: (M, K) float32
        B: (K, N) float32
        Returns: C (M, N) float32

        For N > 16: splits columns across available cores (OP_MATMUL_PARTIAL).
        For N <= 16 or single core: uses OP_MATMUL on core 0.
        """
        M, K = A.shape
        _, N = B.shape

        self.ddr_reset()

        # Upload A and B
        addr_A = self.ddr_alloc(M * K * 4)
        self.write_ddr(A.astype(np.float32), addr_A)

        addr_B = self.ddr_alloc(K * N * 4)
        self.write_ddr(B.astype(np.float32), addr_B)

        addr_C = self.ddr_alloc(M * N * 4)

        if self._num_cores > 1 and N >= self._num_cores:
            # Multi-core: split columns
            num_cores = min(self._num_cores, N)
            cols_per_core = N // num_cores
            remainder = N % num_cores

            col_start = 0
            for core in range(num_cores):
                cols = cols_per_core + (1 if core < remainder else 0)
                col_end = col_start + cols
                self._dispatch_op(core, self.OP_MATMUL_PARTIAL, [
                    M, K, N, addr_A, addr_B, addr_C, col_start, col_end
                ])
                col_start = col_end

            # Wait for all cores
            for core in range(num_cores):
                if not self._wait_done(core, timeout_sec=30.0):
                    raise RuntimeError(f"Matmul timeout on core {core}")
        else:
            # Single-core matmul
            self._dispatch_op(0, self.OP_MATMUL, [
                M, K, N, addr_A, addr_B, addr_C
            ])
            if not self._wait_done(0, timeout_sec=30.0):
                raise RuntimeError("Matmul timeout on core 0")

        # Download result
        C = self.read_ddr(addr_C, M * N * 4).reshape(M, N)
        return C

    def matmul_int8(self, A_fp32: np.ndarray, B_int8: np.ndarray,
                     B_scales: np.ndarray) -> np.ndarray:
        """INT8 matmul with dequantization: C = A @ (B_int8 * B_scales)^T

        For now, dequantize on CPU and dispatch FP32 matmul to card.
        Future: implement INT8 matmul kernel on NMC4.
        """
        B_fp32 = dequantize_int8(B_int8, B_scales)
        return self.matmul(A_fp32, B_fp32.T)


# ============================================================================
# CPU Fallback for matmul (when NM Card not available)
# ============================================================================

class CPUFallback:
    """CPU-only matmul fallback using numpy."""

    def is_available(self) -> bool:
        return True

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.matmul(A, B)

    def matmul_int8(self, A_fp32: np.ndarray, B_int8: np.ndarray,
                     B_scales: np.ndarray) -> np.ndarray:
        B_fp32 = dequantize_int8(B_int8, B_scales)
        return np.matmul(A_fp32, B_fp32.T)

    def ddr_reset(self):
        pass

    def shutdown(self):
        pass


# ============================================================================
# AirLLMNMCard -- Main Inference Engine
# ============================================================================

class AirLLMNMCard:
    """Layer-streaming LLM inference on NM Card Mini.

    Inspired by AirLLM: loads one transformer layer at a time,
    runs computation, discards weights, loads next layer.

    Usage:
        model = AirLLMNMCard(
            model_dir="path/to/split_model",
            use_nmcard=True,
            quantize_int8=True
        )

        output_ids = model.generate(
            "Once upon a time",
            max_new_tokens=100,
            temperature=0.7
        )
        print(model.tokenizer.decode(output_ids))
    """

    def __init__(self,
                 model_dir: str,
                 tokenizer_dir: Optional[str] = None,
                 use_nmcard: bool = True,
                 quantize_int8: bool = False,
                 prefetch: bool = True,
                 max_seq_len: int = 2048,
                 verbose: bool = True):
        """
        Args:
            model_dir: directory with per-layer .safetensors files (from ModelSplitter)
            tokenizer_dir: directory with tokenizer.json (defaults to model_dir)
            use_nmcard: if True, try to use NM Card hardware for matmul
            quantize_int8: if True, quantize linear weights to INT8 on load
            prefetch: if True, prefetch next layer while current executes
            max_seq_len: maximum sequence length for KV cache
            verbose: print timing info
        """
        self.model_dir = model_dir
        self.quantize_int8 = quantize_int8
        self.prefetch = prefetch
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        # Load config
        self.config = ModelConfigLoader(model_dir)
        if verbose:
            print(f"[AirLLM-NMCard] Model: {self.config}")

        # Load tokenizer
        tok_dir = tokenizer_dir or model_dir
        try:
            self.tokenizer = BPETokenizer(tok_dir)
            if verbose:
                print(f"[AirLLM-NMCard] Tokenizer: {self.tokenizer}")
        except FileNotFoundError:
            self.tokenizer = None
            if verbose:
                print("[AirLLM-NMCard] Warning: no tokenizer.json found")

        # Layer loader
        self.loader = LayerLoader(model_dir, quantize_int8=quantize_int8)
        self.layer_names = self.loader.layer_names()
        if verbose:
            print(f"[AirLLM-NMCard] Layers: {len(self.layer_names)}")
            for name in self.layer_names:
                est = self.loader.estimate_layer_bytes(name)
                print(f"  {name}: {est / (1024*1024):.1f} MB")

        # Initialize compute backend
        if use_nmcard:
            self.card = NMCardInterface()
            if not self.card.init():
                if verbose:
                    print("[AirLLM-NMCard] NM Card not available, using CPU fallback")
                self.card = CPUFallback()
        else:
            self.card = CPUFallback()

        # Precompute RoPE tables
        self.cos_table, self.sin_table = precompute_freqs_cis(
            self.config.head_dim, max_seq_len, self.config.rope_theta)

        # KV cache: pre-allocated ring buffers on host (CPU)
        # Shape per layer: (batch=1, num_kv_heads, max_seq_len, head_dim)
        self.kv_cache: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.cache_len = 0  # current filled length in KV cache

    def _reset_kv_cache(self):
        """Reset KV cache for new generation — pre-allocate ring buffers."""
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        # Pre-allocate: (1, num_kv_heads, max_seq_len, head_dim) per layer
        self.kv_cache = [
            (np.zeros((1, num_kv_heads, self.max_seq_len, head_dim), dtype=np.float32),
             np.zeros((1, num_kv_heads, self.max_seq_len, head_dim), dtype=np.float32))
            for _ in range(num_layers)
        ]
        self.cache_len = 0

    def _do_linear(self, x: np.ndarray, weights: Dict[str, np.ndarray],
                    name: str) -> np.ndarray:
        """Dispatch a linear operation (matmul) to the card or CPU.

        Handles both FP32 and INT8 quantized weights.
        """
        qweight_key = name + ".qweight"
        scales_key = name + ".scales"
        weight_key = name

        if qweight_key in weights:
            # INT8 path
            return self.card.matmul_int8(
                x, weights[qweight_key], weights[scales_key])
        elif weight_key in weights:
            # FP32 path: y = x @ W^T
            W = weights[weight_key]
            return self.card.matmul(x, W.T)
        else:
            raise KeyError(f"Weight '{name}' not found. Available: {list(weights.keys())}")

    def _forward_embed(self, input_ids: np.ndarray,
                        weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Embedding lookup: (batch, seq_len) -> (batch, seq_len, hidden)."""
        return embedding(input_ids, weights["weight"])

    def _forward_transformer_layer(self, hidden: np.ndarray, layer_idx: int,
                                     weights: Dict[str, np.ndarray],
                                     position_offset: int = 0) -> np.ndarray:
        """Forward pass through a single transformer layer.

        Qwen3-4B layer structure:
          - input_layernorm (RMSNorm)
          - self_attn: q_proj, k_proj, v_proj, o_proj (GQA)
            - Qwen3 has qk_norm (q_norm, k_norm) per head
          - post_attention_layernorm (RMSNorm)
          - mlp: gate_proj, up_proj, down_proj (SwiGLU)
        """
        batch, seq_len, hidden_size = hidden.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        # ---- Self Attention ----

        # 1. Input LayerNorm
        normed = rmsnorm(hidden, weights["input_layernorm.weight"],
                        self.config.rms_norm_eps)

        # 2. QKV projections
        # q: (batch, seq_len, num_heads * head_dim)
        q = self._do_linear(normed.reshape(-1, hidden_size),
                           weights, "self_attn.q_proj.weight")
        k = self._do_linear(normed.reshape(-1, hidden_size),
                           weights, "self_attn.k_proj.weight")
        v = self._do_linear(normed.reshape(-1, hidden_size),
                           weights, "self_attn.v_proj.weight")

        # Handle bias if present
        if "self_attn.q_proj.bias" in weights:
            q = q + weights["self_attn.q_proj.bias"]
        if "self_attn.k_proj.bias" in weights:
            k = k + weights["self_attn.k_proj.bias"]
        if "self_attn.v_proj.bias" in weights:
            v = v + weights["self_attn.v_proj.bias"]

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # 3. QK normalization (Qwen3 specific) — vectorized across all heads
        if "self_attn.q_norm.weight" in weights:
            q_norm_w = weights["self_attn.q_norm.weight"]  # (head_dim,)
            k_norm_w = weights["self_attn.k_norm.weight"]  # (head_dim,)
            eps = self.config.rms_norm_eps
            # q: (batch, num_heads, seq_len, head_dim) — RMSNorm along last axis
            q_var = np.mean(q.astype(np.float32) ** 2, axis=-1, keepdims=True)
            q = (q * (1.0 / np.sqrt(q_var + eps)) * q_norm_w).astype(q.dtype)
            k_var = np.mean(k.astype(np.float32) ** 2, axis=-1, keepdims=True)
            k = (k * (1.0 / np.sqrt(k_var + eps)) * k_norm_w).astype(k.dtype)

        # 4. RoPE
        q = apply_rope(q, self.cos_table, self.sin_table, position_offset)
        k = apply_rope(k, self.cos_table, self.sin_table, position_offset)

        # 5. GQA Attention (with pre-allocated KV cache ring buffer)
        cache_k_buf, cache_v_buf = self.kv_cache[layer_idx]
        # Write new k,v into the pre-allocated buffer at current position
        cache_k_buf[:, :, self.cache_len:self.cache_len + seq_len, :] = k
        cache_v_buf[:, :, self.cache_len:self.cache_len + seq_len, :] = v
        # Slice the valid portion of the cache for attention
        total_len = self.cache_len + seq_len
        valid_k = cache_k_buf[:, :, :total_len, :]
        valid_v = cache_v_buf[:, :, :total_len, :]

        attn_out, _, _ = gqa_attention(
            q, valid_k, valid_v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            causal=True,
            kv_cache_k=None,  # cache already merged
            kv_cache_v=None
        )

        # 6. Output projection
        # attn_out: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        attn_out = self._do_linear(attn_out.reshape(-1, num_heads * head_dim),
                                   weights, "self_attn.o_proj.weight")
        attn_out = attn_out.reshape(batch, seq_len, hidden_size)

        # 7. Residual connection
        hidden = hidden + attn_out

        # ---- MLP (SwiGLU) ----

        # 8. Post-attention LayerNorm
        normed = rmsnorm(hidden, weights["post_attention_layernorm.weight"],
                        self.config.rms_norm_eps)

        # 9. Gate + Up projections
        gate = self._do_linear(normed.reshape(-1, hidden_size),
                              weights, "mlp.gate_proj.weight")
        up = self._do_linear(normed.reshape(-1, hidden_size),
                            weights, "mlp.up_proj.weight")

        # 10. SiLU(gate) * up
        gate = silu(gate)
        mlp_out = gate * up

        # 11. Down projection
        down = self._do_linear(mlp_out, weights, "mlp.down_proj.weight")
        down = down.reshape(batch, seq_len, hidden_size)

        # 12. Residual connection
        hidden = hidden + down

        return hidden

    def _forward_norm(self, hidden: np.ndarray,
                       weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Final RMSNorm."""
        return rmsnorm(hidden, weights["weight"], self.config.rms_norm_eps)

    def _forward_lm_head(self, hidden: np.ndarray,
                          weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Language model head: (batch, seq_len, hidden) -> (batch, seq_len, vocab).

        For Qwen3-4B: weight is (151936, 2560) = ~1.49 GB FP32.
        Must tile to fit in NM Card DDR:
          - Split vocab into tiles of ~25000 tokens each (~256 MB per tile)
          - 6 tiles for 151936 vocab
        """
        batch_seq = hidden.shape[0] * hidden.shape[1] if hidden.ndim == 3 else hidden.shape[0]
        hidden_flat = hidden.reshape(-1, self.config.hidden_size)
        vocab_size = weights["weight"].shape[0]

        # Determine tile size based on available DDR
        # Each tile: hidden_flat (batch_seq * hidden * 4) + tile_weight (tile_vocab * hidden * 4) + output
        # Budget: ~4 GB usable DDR
        max_tile_vocab = min(vocab_size, 25000)  # ~256 MB per tile for hidden=2560

        logits_parts = []
        for start in range(0, vocab_size, max_tile_vocab):
            end = min(start + max_tile_vocab, vocab_size)
            W_tile = weights["weight"][start:end]  # (tile_vocab, hidden)

            if self.quantize_int8 and "weight.qweight" in weights:
                tile_logits = self.card.matmul_int8(
                    hidden_flat,
                    weights["weight.qweight"][start:end],
                    weights["weight.scales"][start:end]
                )
            else:
                # y = x @ W^T  -> (batch_seq, tile_vocab)
                tile_logits = self.card.matmul(hidden_flat, W_tile.T)

            logits_parts.append(tile_logits)
            # NOTE: ddr_reset removed here — avoid expensive DDR reset between
            # tiles. The bump allocator resets once after all tiles complete.

        # Single ddr_reset after all tiles (caller does ddr_reset after lm_head)
        logits = np.concatenate(logits_parts, axis=-1)
        return logits.reshape(hidden.shape[:-1] + (vocab_size,))

    def forward(self, input_ids: np.ndarray,
                position_offset: int = 0) -> np.ndarray:
        """Full forward pass with layer streaming.

        Args:
            input_ids: (batch, seq_len) int64
            position_offset: position offset for RoPE (for generation with KV cache)

        Returns:
            logits: (batch, seq_len, vocab_size) float32
        """
        t_total = time.monotonic()
        layer_idx = 0  # transformer layer counter

        if self.prefetch:
            return self._forward_prefetch(input_ids, position_offset)

        # Sequential (no prefetch) path
        hidden = None
        for layer_name in self.layer_names:
            t_load = time.monotonic()
            weights = self.loader.load_layer(layer_name)
            load_time = time.monotonic() - t_load

            t_compute = time.monotonic()
            if layer_name == "embed":
                hidden = self._forward_embed(input_ids, weights)
            elif layer_name.startswith("layer_"):
                hidden = self._forward_transformer_layer(
                    hidden, layer_idx, weights, position_offset)
                layer_idx += 1
            elif layer_name == "norm":
                hidden = self._forward_norm(hidden, weights)
            elif layer_name == "lm_head":
                hidden = self._forward_lm_head(hidden, weights)
            compute_time = time.monotonic() - t_compute

            if self.verbose:
                print(f"  {layer_name}: load={load_time:.2f}s compute={compute_time:.2f}s")

            # Free weights
            self.loader.unload_layer(weights)
            self.card.ddr_reset()
            gc.collect()

        # Advance KV cache position after all layers processed these tokens
        if self.kv_cache is not None:
            self.cache_len += input_ids.shape[1]

        if self.verbose:
            total = time.monotonic() - t_total
            print(f"  Total forward: {total:.2f}s")

        return hidden

    def _forward_prefetch(self, input_ids: np.ndarray,
                           position_offset: int = 0) -> np.ndarray:
        """Forward pass with asynchronous layer prefetching.

        While the current layer is computing, the next layer's weights
        are being loaded from disk in a background thread.
        This overlaps ~90ms disk I/O with ~50ms compute per layer.
        """
        hidden = None
        layer_idx = 0

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Start loading first layer
            future = executor.submit(self.loader.load_layer, self.layer_names[0])

            for i, layer_name in enumerate(self.layer_names):
                # Wait for current layer to finish loading
                weights = future.result()

                # Start prefetching next layer
                if i + 1 < len(self.layer_names):
                    future = executor.submit(
                        self.loader.load_layer, self.layer_names[i + 1])

                # Compute current layer
                t_compute = time.monotonic()
                if layer_name == "embed":
                    hidden = self._forward_embed(input_ids, weights)
                elif layer_name.startswith("layer_"):
                    hidden = self._forward_transformer_layer(
                        hidden, layer_idx, weights, position_offset)
                    layer_idx += 1
                elif layer_name == "norm":
                    hidden = self._forward_norm(hidden, weights)
                elif layer_name == "lm_head":
                    hidden = self._forward_lm_head(hidden, weights)
                compute_time = time.monotonic() - t_compute

                if self.verbose:
                    print(f"  {layer_name}: compute={compute_time:.3f}s")

                # Free weights
                self.loader.unload_layer(weights)
                self.card.ddr_reset()
                gc.collect()

        # Advance KV cache position after all layers processed these tokens
        if self.kv_cache is not None:
            self.cache_len += input_ids.shape[1]

        return hidden

    def generate(self, prompt: str,
                 max_new_tokens: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 stop_tokens: Optional[List[int]] = None) -> List[int]:
        """Generate text from a prompt.

        Args:
            prompt: input text
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling threshold
            top_k: top-k filtering
            stop_tokens: list of token IDs that stop generation

        Returns:
            list of generated token IDs (including prompt)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if self.verbose:
            print(f"[Generate] Prompt: {len(input_ids)} tokens")

        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token_id]

        # Reset KV cache
        self._reset_kv_cache()

        # Prefill: process all prompt tokens at once
        t_prefill = time.monotonic()
        ids_array = np.array([input_ids], dtype=np.int64)  # (1, seq_len)
        logits = self.forward(ids_array, position_offset=0)
        prefill_time = time.monotonic() - t_prefill

        if self.verbose:
            print(f"[Generate] Prefill: {prefill_time:.2f}s "
                  f"({len(input_ids) / prefill_time:.1f} tok/s)")

        # Sample first new token
        next_logits = logits[0, -1, :]  # (vocab,)
        next_token = sample_top_p(next_logits, temperature, top_p, top_k)
        generated = list(input_ids) + [next_token]

        # Autoregressive generation: one token at a time
        for step in range(1, max_new_tokens):
            if next_token in stop_tokens:
                if self.verbose:
                    print(f"[Generate] Stop token at step {step}")
                break

            t_step = time.monotonic()

            # Forward pass with single token + KV cache
            token_array = np.array([[next_token]], dtype=np.int64)  # (1, 1)
            position = len(generated) - 1
            logits = self.forward(token_array, position_offset=position)

            step_time = time.monotonic() - t_step

            # Sample
            next_logits = logits[0, -1, :]
            next_token = sample_top_p(next_logits, temperature, top_p, top_k)
            generated.append(next_token)

            if self.verbose and step % 10 == 0:
                print(f"  Step {step}: {1.0/step_time:.2f} tok/s")

        return generated

    def generate_chat(self, messages: List[Dict[str, str]],
                      max_new_tokens: int = 256,
                      **kwargs) -> str:
        """Generate response for chat messages.

        Args:
            messages: [{"role": "user", "content": "Hello"}]
            max_new_tokens: max tokens to generate

        Returns:
            assistant response text
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        input_ids = self.tokenizer.apply_chat_template(messages)

        if self.verbose:
            print(f"[Chat] Input: {len(input_ids)} tokens")

        # Generate
        self._reset_kv_cache()
        ids_array = np.array([input_ids], dtype=np.int64)
        logits = self.forward(ids_array, position_offset=0)

        # Decode
        stop_tokens = [self.tokenizer.eos_token_id]
        im_end = self.tokenizer.token_to_id.get("<|im_end|>")
        if im_end is not None:
            stop_tokens.append(im_end)

        generated = []
        next_logits = logits[0, -1, :]
        next_token = sample_top_p(next_logits, kwargs.get("temperature", 0.7),
                                   kwargs.get("top_p", 0.9),
                                   kwargs.get("top_k", 50))

        for step in range(max_new_tokens):
            if next_token in stop_tokens:
                break
            generated.append(next_token)

            token_array = np.array([[next_token]], dtype=np.int64)
            position = len(input_ids) + len(generated) - 1
            logits = self.forward(token_array, position_offset=position)
            next_logits = logits[0, -1, :]
            next_token = sample_top_p(next_logits, kwargs.get("temperature", 0.7),
                                       kwargs.get("top_p", 0.9),
                                       kwargs.get("top_k", 50))

        return self.tokenizer.decode(generated)

    def __del__(self):
        """Cleanup: shutdown card."""
        if hasattr(self, "card"):
            self.card.shutdown()


# ============================================================================
# Convenience function
# ============================================================================

def run_inference(model_dir: str, prompt: str,
                  max_new_tokens: int = 50,
                  use_nmcard: bool = True,
                  quantize_int8: bool = False) -> str:
    """One-shot inference helper.

    Usage:
        text = run_inference("path/to/split_qwen3", "Hello, world!")
        print(text)
    """
    model = AirLLMNMCard(
        model_dir=model_dir,
        use_nmcard=use_nmcard,
        quantize_int8=quantize_int8
    )
    token_ids = model.generate(prompt, max_new_tokens=max_new_tokens)
    return model.tokenizer.decode(token_ids[len(model.tokenizer.encode(prompt)):])
