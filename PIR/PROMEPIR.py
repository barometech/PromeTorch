"""
PROMEPIR — PIR 250M running on PromeTorch
==========================================

This is the PIR 250M model (Next Concept Prediction) ported from PyTorch
to PromeTorch, our custom deep learning framework built from scratch.

All torch.* calls are replaced with promethorch.* equivalents.
The model architecture is IDENTICAL to PIR 270M.py — no math changes.

Original: PIR/20 MARCH MODEL/PIR 270M.py
Framework: PromeTorch (C++/CUDA, ~48K lines, 108+ files)

Architecture: Pure PIR (no attention)
- 4 decay layers per block (multi-scale concept compression)
- Layer efficiency tracking during eval
- SwiGLU FFN, RoPE, RMSNorm

Usage:
    python PROMEPIR.py
    python PROMEPIR.py --resume
"""

import os
import sys
import math
import time
import json
import random
import gc
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import promethorch as torch
import promethorch.nn as nn
import promethorch.nn.functional as F
from promethorch.amp import autocast, GradScaler


# ============================================================================
# GOOGLE DRIVE SETUP
# ============================================================================

def setup_google_drive():
    """Mount Google Drive and setup paths"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        base_path = Path('/content/drive/MyDrive/PIR_250M')
        print(f"Google Drive mounted at {base_path}")
    except:
        base_path = Path('./PIR_250M')
        print(f"Local mode: {base_path}")

    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / 'checkpoints').mkdir(exist_ok=True)
    (base_path / 'logs').mkdir(exist_ok=True)
    (base_path / 'data').mkdir(exist_ok=True)

    return base_path


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class PIR250MConfig:
    """
    PIR 250M Model Configuration

    ~250M parameters:
    - Embedding: 50257 x 768 = 38.6M
    - 16 blocks x ~13M each = 208M
    - Total: ~247M
    """
    vocab_size: int = 50257
    n_embd: int = 768
    n_layers: int = 16
    n_pir_layers: int = 4  # 4 decay scales per block
    block_size: int = 2048  # 2K context
    ffn_mult: float = 3.5
    dropout: float = 0.0
    tie_weights: bool = True

    # MULTI-SCALE DECAY RANGES FOR 2K CONTEXT
    # Adjusted half-lives for smaller context
    # Layer 0: ~5 tokens (words, morphology)
    # Layer 1: ~25 tokens (phrases, clauses)
    # Layer 2: ~125 tokens (paragraphs)
    # Layer 3: ~600 tokens (document sections)
    decay_ranges: tuple = (
        (0.80, 0.95),    # L0: half-life 5-13 tok
        (0.95, 0.99),    # L1: half-life 14-69 tok
        (0.99, 0.998),   # L2: half-life 69-346 tok
        (0.998, 0.9995), # L3: half-life 346-1386 tok
    )


@dataclass
class TrainConfig:
    """Training Configuration - Chinchilla Optimal for 250M"""
    # CHINCHILLA: 250M params x 20 = 5B tokens
    # 5B / (8 x 8 x 2048) = 38,147 steps -> round to 40,000
    batch_size: int = 8
    gradient_accumulation: int = 8
    seq_len: int = 2048
    max_steps: int = 40000
    warmup_steps: int = 1000
    learning_rate: float = 6e-4  # Higher LR for smaller model
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Precision
    dtype: str = "bfloat16"
    use_compile: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2000

    # Dataset mix (80% text for pretrain, dialogs are secondary)
    text_ratio: float = 0.8  # 80% text, 20% dialogs

    # Efficiency loss (forces L2-L3 to learn long-range patterns)
    efficiency_lambda: float = 0.01  # Weight for efficiency penalty
    efficiency_warmup_steps: int = 1000  # Steps before enabling efficiency loss

    @property
    def tokens_per_step(self):
        return self.batch_size * self.gradient_accumulation * self.seq_len

    @property
    def total_tokens(self):
        return self.max_steps * self.tokens_per_step


# ============================================================================
# PARALLEL SCAN (CORE PIR MECHANISM)
# ============================================================================

def pytorch_parallel_scan(gates, x):
    """
    O(T) parallel scan via cumsum trick
    out[t] = sum_{s<=t} prod_{s<k<=t}(gate[k]) * x[s]
    """
    log_gates = torch.log(gates.clamp(min=1e-6))
    cumsum_log = torch.cumsum(log_gates, dim=1).clamp(-20, 20)
    scale = torch.exp(-cumsum_log)
    scaled = x * scale
    cumsum = torch.cumsum(scaled, dim=1)
    result = cumsum * torch.exp(cumsum_log)
    return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)


def dynamic_parallel_scan(x, gate_logits, base_decay, segment_size=64):
    """
    Parallel scan with dynamic gating and segmented processing
    """
    modulation = torch.tanh(gate_logits) * 0.1
    gates = base_decay * (1 + modulation)
    gates = gates.clamp(0.5, 0.9999)

    B, T, D = x.shape
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    if T <= segment_size:
        return pytorch_parallel_scan(gates, x)

    # Segmented scan for long sequences
    num_seg = (T + segment_size - 1) // segment_size
    padded_T = num_seg * segment_size
    pad = padded_T - T

    if pad > 0:
        gates = F.pad(gates, (0, 0, 0, pad), value=1.0)
        x = F.pad(x, (0, 0, 0, pad), value=0.0)

    gates_s = gates.view(B, num_seg, segment_size, D)
    x_s = x.view(B, num_seg, segment_size, D)

    # Local scan
    log_g = torch.log(gates_s.clamp(min=1e-6))
    cumsum_log = torch.cumsum(log_g, dim=2).clamp(-20, 20)
    exp_neg = torch.exp(-cumsum_log)
    exp_pos = torch.exp(cumsum_log)
    local_h = torch.cumsum(x_s * exp_neg, dim=2) * exp_pos
    local_h = torch.nan_to_num(local_h, nan=0.0, posinf=1e4, neginf=-1e4)

    # Inter-segment carries
    total_log = cumsum_log[:, :, -1, :]
    A_seg = torch.exp(total_log.clamp(-20, 20))
    h_end = local_h[:, :, -1, :]
    h_shifted = F.pad(h_end[:, :-1, :], (0, 0, 1, 0), value=0.0)

    log_A = torch.log(A_seg.clamp(min=1e-6)).clamp(-20, 20)
    cumsum_log_A = torch.cumsum(log_A, dim=1).clamp(-20, 20)
    scale_neg = torch.exp(-cumsum_log_A)
    scale_pos = torch.exp(cumsum_log_A)
    carries = torch.cumsum(h_shifted * scale_neg, dim=1) * scale_pos
    carries = torch.nan_to_num(carries, nan=0.0, posinf=1e4, neginf=-1e4)

    result = local_h + exp_pos * carries.unsqueeze(2)
    result = result.view(B, padded_T, D)

    if pad > 0:
        result = result[:, :T, :].contiguous()

    return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones([dim]))

    def forward(self, x):
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(0, max_seq_len, 1.0)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len].unsqueeze(0), self.sin_cached[:seq_len].unsqueeze(0)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    x1_rotated = (x1 * cos) + (rotate_half(x1) * sin)
    return torch.cat([x1_rotated, x2], dim=-1)


# ============================================================================
# PIR LAYERS (Next Concept Prediction)
# ============================================================================

class PIRLayer(nn.Module):
    """
    Single PIR Layer - compresses information at one temporal scale

    This is where CONCEPT COMPRESSION happens:
    - value_proj: extracts WHAT concept to store
    - gate_proj: controls HOW MUCH to modulate decay
    - out_proj: reads FROM the compressed state
    """
    def __init__(self, n_embd: int, decay_min: float, decay_max: float, layer_idx: int = 0):
        super().__init__()
        self.n_embd = n_embd
        self.layer_idx = layer_idx

        decay = torch.linspace(decay_min, decay_max, n_embd)
        self.register_buffer("base_decay", decay)
        self.decay_avg = (decay_min + decay_max) / 2

        self.gate_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.value_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.norm = RMSNorm(n_embd)

        # Storage for efficiency measurement
        self.last_states = None

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.gate_proj.weight, gain=0.1)
        nn.init.orthogonal_(self.value_proj.weight, gain=1.0)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.5)

    def forward(self, x, store_states: bool = False):
        gate_logits = self.gate_proj(x)
        values = self.value_proj(x)
        value_gate = torch.sigmoid(gate_logits)
        gated_values = values * value_gate

        scanned = dynamic_parallel_scan(gated_values, gate_logits, self.base_decay)

        # Store for efficiency measurement (only during eval)
        if store_states:
            self.last_states = scanned.detach()

        out = self.out_proj(scanned)
        return self.norm(out)


class PIRBlock(nn.Module):
    """
    4 PIR layers with different decay scales
    Multi-scale concept compression
    """
    def __init__(self, n_embd: int, n_pir_layers: int = 4, decay_ranges: list = None):
        super().__init__()
        if decay_ranges is None:
            decay_ranges = [(0.80, 0.95), (0.95, 0.99), (0.99, 0.998), (0.998, 0.9995)]

        self.layers = nn.ModuleList([
            PIRLayer(n_embd, *decay_ranges[i % len(decay_ranges)], layer_idx=i)
            for i in range(n_pir_layers)
        ])

        self.mix_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.norm = RMSNorm(n_embd)

        nn.init.orthogonal_(self.mix_proj.weight, gain=0.5)

    def forward(self, x, store_states: bool = False):
        h = x
        for layer in self.layers:
            h = h + layer(h, store_states=store_states)
        out = self.mix_proj(h)
        return self.norm(out)

    def get_layer_states(self) -> List:
        """Get stored states from all PIR layers"""
        return [layer.last_states for layer in self.layers if layer.last_states is not None]


class FeedForward(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, n_embd: int, mult: float = 3.5):
        super().__init__()
        hidden = int(n_embd * mult * 2 / 3)
        hidden = (hidden + 63) // 64 * 64

        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.w1.weight, gain=1.0)
        nn.init.orthogonal_(self.w2.weight, gain=0.5)
        nn.init.orthogonal_(self.w3.weight, gain=1.0)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """PIR + FFN block"""
    def __init__(self, n_embd: int, n_pir_layers: int = 4, ffn_mult: float = 3.5,
                 dropout: float = 0.0, decay_ranges: list = None):
        super().__init__()
        self.pir = PIRBlock(n_embd, n_pir_layers, decay_ranges)
        self.ffn = FeedForward(n_embd, ffn_mult)
        self.norm1 = RMSNorm(n_embd)
        self.norm2 = RMSNorm(n_embd)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, store_states: bool = False):
        x = x + self.dropout(self.pir(self.norm1(x), store_states=store_states))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ============================================================================
# PIR 250M MODEL
# ============================================================================

class PIR250M(nn.Module):
    """PIR 250M Language Model"""
    def __init__(self, config: PIR250MConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryEmbedding(config.n_embd // 2, config.block_size)

        decay_ranges = list(config.decay_ranges)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.n_embd,
                config.n_pir_layers,
                config.ffn_mult,
                config.dropout,
                decay_ranges
            )
            for _ in range(config.n_layers)
        ])

        self.norm_out = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"PIR 250M initialized: {n_params/1e6:.1f}M parameters")

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        if self.lm_head.weight is not self.tok_emb.weight:
            nn.init.normal_(self.lm_head.weight, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(self, idx, targets=None, store_states: bool = False):
        B, T = idx.shape

        x = self.tok_emb(idx)
        cos, sin = self.rope(T)
        x = apply_rotary_pos_emb(x, cos, sin)

        for block in self.blocks:
            x = block(x, store_states=store_states)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def get_layer_states_from_last_block(self) -> List:
        """Get PIR states from last block for efficiency measurement"""
        return self.blocks[-1].pir.get_layer_states()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.2):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == 50256:
                break

            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ============================================================================
# LAYER EFFICIENCY METRICS
# ============================================================================

@torch.no_grad()
def measure_layer_efficiency(model: PIR250M, decay_ranges: tuple) -> Dict[str, Dict]:
    """
    Measure how well each layer uses its decay capacity

    Ratio = actual_similarity / expected_similarity
    - ratio < 1.0: layer NOT using capacity (empty)
    - ratio ~ 1.0: layer = pure decay, no learned patterns
    - ratio > 1.0: layer CAPTURING structure (learned concepts!)
    """
    states_list = model.get_layer_states_from_last_block()

    if not states_list or states_list[0] is None:
        return {}

    # Test distances appropriate for each layer's half-life
    test_distances = {
        0: [5, 10, 20],      # L0: half-life ~8
        1: [20, 40, 80],     # L1: half-life ~35
        2: [50, 100, 200],   # L2: half-life ~150
        3: [100, 200, 400],  # L3: half-life ~700
    }

    results = {}

    for layer_idx, states in enumerate(states_list):
        if states is None:
            continue

        # Handle batched states: [B, T, D]
        if states.dim() == 3:
            states = states[0]  # Take first batch item

        T, D = states.shape
        decay_min, decay_max = decay_ranges[layer_idx]
        decay_avg = (decay_min + decay_max) / 2

        # Normalize
        norms = torch.norm(states, dim=1, keepdim=True) + 1e-8
        states_norm = states / norms

        layer_results = {}
        for d in test_distances.get(layer_idx, [50, 100]):
            if d >= T - 10:
                continue

            # Compute similarity at distance d
            sims = torch.sum(states_norm[:-d] * states_norm[d:], dim=1)
            actual_sim = sims.mean().item()

            # Expected from pure decay
            expected_sim = decay_avg ** d

            # Efficiency ratio
            ratio = actual_sim / (expected_sim + 1e-8)

            layer_results[f'd{d}'] = {
                'actual': round(actual_sim, 4),
                'expected': round(expected_sim, 4),
                'ratio': round(ratio, 3)
            }

        # Compute average ratio for this layer
        ratios = [v['ratio'] for v in layer_results.values()]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0

        results[f'L{layer_idx}'] = {
            'distances': layer_results,
            'avg_ratio': round(avg_ratio, 3),
            'half_life': round(-1 / math.log(decay_avg + 1e-8), 1)
        }

    return results


def format_efficiency_report(eff_results: Dict) -> str:
    """Format efficiency results for logging"""
    if not eff_results:
        return "No efficiency data"

    lines = ["Layer Efficiency (ratio = actual/expected, >1.0 = learning structure):"]
    for layer, data in eff_results.items():
        hl = data.get('half_life', '?')
        avg = data.get('avg_ratio', 0)
        status = "ACTIVE" if avg > 1.1 else ("FILLING" if avg > 0.8 else "EMPTY")
        lines.append(f"  {layer} (half-life={hl}): avg_ratio={avg:.2f} {status}")
        for d_name, metrics in data.get('distances', {}).items():
            lines.append(f"    {d_name}: actual={metrics['actual']:.3f} expected={metrics['expected']:.3f} ratio={metrics['ratio']:.2f}")

    return "\n".join(lines)


def compute_efficiency_loss(model: PIR250M, decay_ranges: tuple,
                            target_layers: List[int] = [2, 3]):
    """
    Compute efficiency penalty for training.

    Penalty when Layer 2-3 have ratio < 1.0 (not using their capacity).
    This FORCES the model to use long-range decay layers.

    Args:
        model: PIR model with stored states
        decay_ranges: tuple of (min, max) decay for each layer
        target_layers: which layers to penalize (default: L2, L3)

    Returns:
        penalty: scalar tensor (0 if layers are efficient, >0 if not)
        l2_ratio: float, ratio for layer 2
        l3_ratio: float, ratio for layer 3
    """
    states_list = model.get_layer_states_from_last_block()
    device = next(model.parameters()).device

    if not states_list or len(states_list) < max(target_layers) + 1:
        return torch.full([1], 0.0).squeeze(), 0.0, 0.0

    # Test distances for L2 and L3
    test_distances = {
        2: 100,   # L2 half-life ~150
        3: 300,   # L3 half-life ~700
    }

    penalties = []
    ratios = {2: 0.0, 3: 0.0}

    for layer_idx in target_layers:
        states = states_list[layer_idx]
        if states is None:
            continue

        # Handle batched states
        if states.dim() == 3:
            states = states[0]

        T, D = states.shape
        d = test_distances.get(layer_idx, 100)

        if d >= T - 10:
            continue

        decay_min, decay_max = decay_ranges[layer_idx]
        decay_avg = (decay_min + decay_max) / 2

        # Normalize states
        norms = torch.norm(states, dim=1, keepdim=True) + 1e-8
        states_norm = states / norms

        # Actual similarity at distance d
        sims = torch.sum(states_norm[:-d] * states_norm[d:], dim=1)
        actual_sim = sims.mean()

        # Expected from decay
        expected_sim = decay_avg ** d

        # Ratio
        ratio = actual_sim / (expected_sim + 1e-8)
        ratios[layer_idx] = ratio.item()

        # Penalty: max(0, 1.0 - ratio)
        penalty = torch.clamp(1.0 - ratio, min=0.0)
        penalties.append(penalty)

    if not penalties:
        return torch.full([1], 0.0).squeeze(), 0.0, 0.0

    total_penalty = torch.stack(penalties).mean()

    return total_penalty, ratios.get(2, 0.0), ratios.get(3, 0.0)


# ============================================================================
# DATASET LOADER (80% Text + 20% Dialogs)
# ============================================================================

class MixedDataLoader:
    """
    Loads and mixes:
    - 80% FineWeb-Edu (high-quality web text)
    - 20% UltraChat/OpenAssistant (dialogs)
    """
    def __init__(self, config: TrainConfig, base_path: Path):
        self.config = config
        self.base_path = base_path
        self.seq_len = config.seq_len
        self.text_ratio = config.text_ratio

        # Tokenizer
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")

        # Load datasets
        self._load_datasets()

        # Current positions
        self.text_pos = 0
        self.dialog_pos = 0

    def _load_datasets(self):
        """Load and tokenize both datasets - streaming to disk, not RAM"""
        from datasets import load_dataset

        cache_text = self.base_path / 'data' / 'fineweb_tokens.bin'
        cache_dialog = self.base_path / 'data' / 'ultrachat_tokens.bin'

        # For 250M Chinchilla: 5B tokens total (50/50 split)
        target_text = 2_500_000_000   # 2.5B tokens text
        target_dialog = 2_500_000_000 # 2.5B tokens dialog
        chunk_size = 10_000_000       # Write 10M tokens at a time (keeps RAM low)

        # Load text data (FineWeb-Edu sample)
        if cache_text.exists():
            print(f"Loading cached text tokens from {cache_text}")
            self.text_tokens = np.memmap(cache_text, dtype=np.uint16, mode='r')
            print(f"  Loaded {len(self.text_tokens)/1e6:.1f}M tokens")
        else:
            print("Downloading FineWeb-Edu (streaming to disk)...")
            try:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                                  split="train", streaming=True)

                # Pre-allocate memmap
                mm = np.memmap(cache_text, dtype=np.uint16, mode='w+', shape=(target_text,))

                pos = 0
                chunk = []
                for i, item in enumerate(ds):
                    text = item.get('text', '')
                    if text:
                        toks = self.enc.encode_ordinary(text)
                        chunk.extend(toks)

                        # Write chunk to disk when big enough
                        if len(chunk) >= chunk_size:
                            end = min(pos + len(chunk), target_text)
                            mm[pos:end] = np.array(chunk[:end-pos], dtype=np.uint16)
                            pos = end
                            chunk = chunk[end-pos:] if end < pos + len(chunk) else []
                            mm.flush()

                            if i % 5000 == 0:
                                print(f"  {pos/1e6:.1f}M tokens written")

                    if pos >= target_text:
                        break

                # Write remaining
                if chunk and pos < target_text:
                    end = min(pos + len(chunk), target_text)
                    mm[pos:end] = np.array(chunk[:end-pos], dtype=np.uint16)
                    pos = end

                mm.flush()
                del mm

                # Reopen as read-only with actual size
                self.text_tokens = np.memmap(cache_text, dtype=np.uint16, mode='r')[:pos]
                print(f"Cached {pos/1e6:.1f}M text tokens")

            except Exception as e:
                print(f"FineWeb failed: {e}, falling back to OpenWebText")
                ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

                mm = np.memmap(cache_text, dtype=np.uint16, mode='w+', shape=(target_text,))
                pos = 0
                chunk = []

                for i, item in enumerate(ds):
                    text = item.get('text', '')
                    if text:
                        chunk.extend(self.enc.encode_ordinary(text))
                        if len(chunk) >= chunk_size:
                            end = min(pos + len(chunk), target_text)
                            mm[pos:end] = np.array(chunk[:end-pos], dtype=np.uint16)
                            pos = end
                            chunk = []
                            mm.flush()
                            if i % 5000 == 0:
                                print(f"  {pos/1e6:.1f}M tokens")
                    if pos >= target_text:
                        break

                if chunk and pos < target_text:
                    end = min(pos + len(chunk), target_text)
                    mm[pos:end] = np.array(chunk[:end-pos], dtype=np.uint16)

                mm.flush()
                del mm
                self.text_tokens = np.memmap(cache_text, dtype=np.uint16, mode='r')

        # Load dialog data (UltraChat) WITH MASKS
        # Mask: 0 = User (don't learn), 1 = Assistant (learn)
        if cache_dialog.exists():
            print(f"Loading cached dialog tokens from {cache_dialog}")
            self.dialog_tokens = np.memmap(cache_dialog, dtype=np.uint16, mode='r')

            # Load masks
            cache_mask = self.base_path / 'data' / 'ultrachat_masks.bin'
            if cache_mask.exists():
                self.dialog_masks = np.memmap(cache_mask, dtype=np.uint8, mode='r')
                print(f"  Loaded {len(self.dialog_tokens)/1e6:.1f}M tokens with masks")
            else:
                print("  WARNING: No masks found, will learn on everything")
                self.dialog_masks = None
        else:
            print("Downloading UltraChat (streaming to disk with masks)...")
            cache_mask = self.base_path / 'data' / 'ultrachat_masks.bin'

            try:
                ds = load_dataset("stingning/ultrachat", split="train", streaming=True)

                # Pre-allocate memmaps for tokens AND masks
                mm_tok = np.memmap(cache_dialog, dtype=np.uint16, mode='w+', shape=(target_dialog,))
                mm_mask = np.memmap(cache_mask, dtype=np.uint8, mode='w+', shape=(target_dialog,))

                pos = 0
                chunk_tok = []
                chunk_mask = []

                # Tokenize markers once
                user_marker = self.enc.encode_ordinary("User:")
                asst_marker = self.enc.encode_ordinary("Assistant:")

                for i, item in enumerate(ds):
                    messages = item.get('data', [])
                    if messages:
                        for j, msg in enumerate(messages):
                            if j % 2 == 0:
                                # User turn - mask = 0 (don't learn)
                                text = f"User: {msg}\n\n"
                                toks = self.enc.encode_ordinary(text)
                                chunk_tok.extend(toks)
                                chunk_mask.extend([0] * len(toks))
                            else:
                                # Assistant turn - mask = 1 (learn)
                                text = f"Assistant: {msg}\n\n"
                                toks = self.enc.encode_ordinary(text)
                                chunk_tok.extend(toks)
                                chunk_mask.extend([1] * len(toks))

                        # Write chunks when big enough
                        if len(chunk_tok) >= chunk_size:
                            end = min(pos + len(chunk_tok), target_dialog)
                            write_len = end - pos
                            mm_tok[pos:end] = np.array(chunk_tok[:write_len], dtype=np.uint16)
                            mm_mask[pos:end] = np.array(chunk_mask[:write_len], dtype=np.uint8)
                            pos = end
                            chunk_tok = chunk_tok[write_len:]
                            chunk_mask = chunk_mask[write_len:]
                            mm_tok.flush()
                            mm_mask.flush()
                            if i % 5000 == 0:
                                print(f"  {pos/1e6:.1f}M tokens")

                    if pos >= target_dialog:
                        break

                # Write remaining
                if chunk_tok and pos < target_dialog:
                    end = min(pos + len(chunk_tok), target_dialog)
                    write_len = end - pos
                    mm_tok[pos:end] = np.array(chunk_tok[:write_len], dtype=np.uint16)
                    mm_mask[pos:end] = np.array(chunk_mask[:write_len], dtype=np.uint8)
                    pos = end

                mm_tok.flush()
                mm_mask.flush()
                del mm_tok, mm_mask

                self.dialog_tokens = np.memmap(cache_dialog, dtype=np.uint16, mode='r')[:pos]
                self.dialog_masks = np.memmap(cache_mask, dtype=np.uint8, mode='r')[:pos]
                print(f"Cached {pos/1e6:.1f}M dialog tokens with masks")

            except Exception as e:
                print(f"UltraChat failed: {e}, trying OpenAssistant...")
                try:
                    ds = load_dataset("OpenAssistant/oasst1", split="train")
                    tokens = []
                    masks = []
                    for item in ds:
                        text = item.get('text', '')
                        role = item.get('role', 'user')
                        if text:
                            formatted = f"{'User' if role == 'user' else 'Assistant'}: {text}\n\n"
                            toks = self.enc.encode_ordinary(formatted)
                            tokens.extend(toks)
                            # Mask: 0 for user, 1 for assistant
                            masks.extend([0 if role == 'user' else 1] * len(toks))
                        if len(tokens) >= 50_000_000:
                            break

                    self.dialog_tokens = np.array(tokens, dtype=np.uint16)
                    self.dialog_masks = np.array(masks, dtype=np.uint8)

                    mm_tok = np.memmap(cache_dialog, dtype=np.uint16, mode='w+', shape=self.dialog_tokens.shape)
                    mm_tok[:] = self.dialog_tokens
                    mm_tok.flush()

                    mm_mask = np.memmap(cache_mask, dtype=np.uint8, mode='w+', shape=self.dialog_masks.shape)
                    mm_mask[:] = self.dialog_masks
                    mm_mask.flush()
                    print(f"Cached {len(self.dialog_tokens)/1e6:.1f}M dialog tokens with masks")
                except Exception as e2:
                    print(f"OpenAssistant failed: {e2}, using text as fallback")
                    self.dialog_tokens = self.text_tokens
                    self.dialog_masks = None

        print(f"\nDataset loaded:")
        print(f"  Text tokens: {len(self.text_tokens)/1e9:.2f}B")
        print(f"  Dialog tokens: {len(self.dialog_tokens)/1e9:.2f}B")
        print(f"  Mix ratio: {self.text_ratio*100:.0f}% text / {(1-self.text_ratio)*100:.0f}% dialog")

    def get_batch(self, device) -> Tuple:
        """
        Get mixed batch with MASKING for dialogs.

        Text: learn on everything (standard LM)
        Dialog: learn ONLY on Assistant responses (masked LM)
        """
        batch_x = []
        batch_y = []

        for _ in range(self.config.batch_size):
            # Decide source based on ratio
            if random.random() < self.text_ratio:
                # TEXT: standard next-token prediction
                tokens = self.text_tokens
                pos = self.text_pos
                self.text_pos = (self.text_pos + self.seq_len) % (len(tokens) - self.seq_len - 1)

                x = torch.from_numpy(tokens[pos:pos + self.seq_len].astype(np.int64))
                y = torch.from_numpy(tokens[pos + 1:pos + self.seq_len + 1].astype(np.int64))
            else:
                # DIALOG: masked - only learn on Assistant responses
                tokens = self.dialog_tokens
                pos = self.dialog_pos
                self.dialog_pos = (self.dialog_pos + self.seq_len) % (len(tokens) - self.seq_len - 1)

                x = torch.from_numpy(tokens[pos:pos + self.seq_len].astype(np.int64))
                y = torch.from_numpy(tokens[pos + 1:pos + self.seq_len + 1].astype(np.int64))

                # Apply mask: -100 for User tokens (model sees but doesn't learn)
                if self.dialog_masks is not None:
                    mask = torch.from_numpy(self.dialog_masks[pos + 1:pos + self.seq_len + 1].astype(np.int64))
                    # Where mask = 0 (User), set y = -100 (ignored in loss)
                    ignore_val = torch.full([1], -100, dtype=torch.int64).squeeze()
                    y = torch.where(mask == 1, y, ignore_val)

            batch_x.append(x)
            batch_y.append(y)

        return torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

EVAL_PROMPTS = [
    # ===== GPT-2 STYLE (text completion) =====
    # Short factual
    "The capital of France is",
    "Water boils at",
    "The largest planet in our solar system is",
    "Albert Einstein was born in",

    # Article openings
    "The human brain contains approximately",
    "Climate change refers to long-term shifts in",
    "Artificial intelligence, often abbreviated as AI, is",
    "The Industrial Revolution began in",
    "Photosynthesis is the process by which plants",

    # News style
    "Scientists have discovered a new species of",
    "According to recent research published in Nature,",
    "The stock market experienced significant volatility today as",
    "A new study suggests that regular exercise can",

    # Technical/educational
    "In computer science, an algorithm is defined as",
    "The Pythagorean theorem states that",
    "DNA, or deoxyribonucleic acid, carries",
    "Machine learning is a subset of artificial intelligence that",

    # ===== DIALOG STYLE =====
    "User: What is machine learning?\n\nAssistant:",
    "User: How does photosynthesis work?\n\nAssistant:",
    "User: Can you explain the theory of relativity?\n\nAssistant:",
    "User: What are the benefits of regular exercise?\n\nAssistant:",
    "User: How do computers store information?\n\nAssistant:",
    "User: What causes climate change?\n\nAssistant:",
]


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_lr(step: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with warmup"""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    if step >= config.max_steps:
        return config.min_lr

    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def generate_sample(model, enc, prompt: str, max_tokens: int = 150,
                    temperature: float = 0.7, device: str = "cuda") -> str:
    """Generate text from prompt"""
    model.eval()
    tokens = enc.encode(prompt)
    x = torch.from_numpy(np.array([tokens], dtype=np.int64))

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature,
                          top_k=50, top_p=0.9, repetition_penalty=1.2)

    return enc.decode(y[0].tolist())


def save_checkpoint(model, optimizer, scaler, step, loss, tokens_seen,
                    train_config, model_config, efficiency_history, path):
    """Save training checkpoint"""
    # Handle compiled model
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

    checkpoint = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'tokens_seen': tokens_seen,
        'train_config': asdict(train_config),
        'model_config': asdict(model_config),
        'efficiency_history': efficiency_history,
    }

    import pickle
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {path} (step {step}, loss {loss:.4f})")


def load_checkpoint(path, model, optimizer, scaler):
    """Load checkpoint"""
    import pickle
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    efficiency_history = checkpoint.get('efficiency_history', [])

    return checkpoint['step'], checkpoint['loss'], checkpoint['tokens_seen'], efficiency_history


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(train_config: TrainConfig, model_config: PIR250MConfig,
          base_path: Path, resume: bool = False):
    """Main training function"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # PromeTorch uses float32 by default (bfloat16/float16 via AMP shim)
    use_amp = train_config.dtype in ("bfloat16", "float16")

    print("=" * 70)
    print("PROMEPIR 250M PRETRAINING - NEXT CONCEPT PREDICTION")
    print("Running on PromeTorch (custom framework)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Chinchilla target: {train_config.total_tokens/1e9:.1f}B tokens")
    print(f"Max steps: {train_config.max_steps}")
    print(f"Tokens per step: {train_config.tokens_per_step:,}")
    print(f"Context length: {model_config.block_size}")
    print(f"Dataset mix: {train_config.text_ratio*100:.0f}% text + {(1-train_config.text_ratio)*100:.0f}% dialogs")
    print(f"Decay ranges: {model_config.decay_ranges}")
    print("=" * 70)

    # Data
    print("\nLoading data...")
    data_loader = MixedDataLoader(train_config, base_path)

    # Model
    print("\nInitializing model...")
    model = PIR250M(model_config)
    if device == "cuda":
        model.to(device)

    # Optimizer — PromeTorch AdamW
    # NOTE: PromeTorch AdamW takes flat param list, not param_groups
    # For now, pass all parameters with single weight_decay
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=train_config.learning_rate,
                                   betas=(train_config.beta1, train_config.beta2),
                                   weight_decay=train_config.weight_decay)
    scaler = GradScaler()

    # Resume
    start_step = 0
    tokens_seen = 0
    efficiency_history = []
    checkpoint_dir = base_path / 'checkpoints'
    latest_path = checkpoint_dir / 'latest.pt'

    if resume and latest_path.exists():
        print(f"Resuming from {latest_path}...")
        start_step, _, tokens_seen, efficiency_history = load_checkpoint(
            latest_path, model, optimizer, scaler)

    # Compile (no-op in PromeTorch)
    if train_config.use_compile:
        print("torch.compile: no-op in PromeTorch (native C++ execution)")
        model = torch.compile(model)

    # Tokenizer for generation
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # Log file
    log_file = base_path / 'logs' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + "\n")

    # Training
    log(f"\nStarting training from step {start_step}...")
    model.train()

    start_time = time.time()
    running_loss = 0.0
    best_loss = float("inf")

    for step in range(start_step, train_config.max_steps):
        step_start = time.time()

        # Learning rate — manual schedule (PromeTorch optimizer has .lr attribute)
        lr = get_lr(step, train_config)
        # NOTE: PromeTorch AdamW doesn't support param_groups yet
        # Set lr directly on optimizer options
        optimizer.lr = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_eff_penalty = 0.0
        L2_ratio_accum = 0.0
        L3_ratio_accum = 0.0
        valid_microbatches = 0

        # Enable efficiency loss after warmup
        use_efficiency = step >= train_config.efficiency_warmup_steps

        for _ in range(train_config.gradient_accumulation):
            x, y = data_loader.get_batch(device)

            # AMP autocast is no-op in PromeTorch
            with autocast(device):
                # Store states only when using efficiency loss
                logits, ce_loss = model(x, y, store_states=use_efficiency)

                # Compute efficiency penalty (only after warmup)
                if use_efficiency:
                    eff_penalty, l2_ratio, l3_ratio = compute_efficiency_loss(model, model_config.decay_ranges)
                    loss = (ce_loss + train_config.efficiency_lambda * eff_penalty) / train_config.gradient_accumulation
                    accum_eff_penalty += eff_penalty.item()
                    L2_ratio_accum += l2_ratio
                    L3_ratio_accum += l3_ratio
                else:
                    loss = ce_loss / train_config.gradient_accumulation

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            accum_loss += ce_loss.item()
            valid_microbatches += 1

        if valid_microbatches == 0:
            scaler.update()
            continue

        accum_loss /= valid_microbatches
        accum_eff_penalty /= max(valid_microbatches, 1)
        L2_ratio_avg = L2_ratio_accum / max(valid_microbatches, 1)
        L3_ratio_avg = L3_ratio_accum / max(valid_microbatches, 1)

        # Gradient clipping and step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Stats
        running_loss = 0.99 * running_loss + 0.01 * accum_loss if running_loss else accum_loss
        tokens_seen += train_config.tokens_per_step
        step_time = time.time() - step_start
        tokens_per_sec = train_config.tokens_per_step / step_time

        if running_loss < best_loss:
            best_loss = running_loss

        # Logging
        if step % train_config.log_interval == 0:
            elapsed = time.time() - start_time
            eta = (train_config.max_steps - step) * (elapsed / max(step - start_step + 1, 1))
            ppl = math.exp(min(accum_loss, 20))

            log_msg = (f"Step {step:>6}/{train_config.max_steps} | "
                      f"Loss: {accum_loss:.4f} (avg: {running_loss:.4f}) | "
                      f"PPL: {ppl:.1f} | "
                      f"LR: {lr:.2e} | "
                      f"Grad: {grad_norm:.2f} | "
                      f"Tokens: {tokens_seen/1e9:.2f}B | "
                      f"Speed: {tokens_per_sec/1e3:.1f}K tok/s | "
                      f"ETA: {timedelta(seconds=int(eta))}")

            # Add efficiency info if active
            if use_efficiency:
                log_msg += f" | L2:{L2_ratio_avg:.2f} L3:{L3_ratio_avg:.2f}"

            log(log_msg)

        # Evaluation
        if step > 0 and step % train_config.eval_interval == 0 and step != start_step:
            model.eval()
            log("\n" + "=" * 60)
            log("EVALUATION")
            log("=" * 60)

            # Run forward pass with state storage for efficiency measurement
            with torch.no_grad():
                x_eval, y_eval = data_loader.get_batch(device)
                _, _ = model(x_eval, y_eval, store_states=True)

                # Measure layer efficiency
                eff_results = measure_layer_efficiency(model, model_config.decay_ranges)
                log("\n" + format_efficiency_report(eff_results))

                # Store in history
                efficiency_history.append({
                    'step': step,
                    'tokens': tokens_seen,
                    'efficiency': {k: v['avg_ratio'] for k, v in eff_results.items()}
                })

            # Generation samples
            log("\nGeneration Samples:")
            num_prompts = 15
            selected_prompts = random.sample(EVAL_PROMPTS, num_prompts)

            for i, prompt in enumerate(selected_prompts):
                try:
                    sample = generate_sample(model, enc, prompt, max_tokens=500,
                                            temperature=0.7, device=device)
                    log(f"\n[{i+1}] {sample}")
                except Exception as e:
                    log(f"\n[{i+1}] [{prompt[:30]}...] Failed: {e}")

            log("\n" + "=" * 60)
            model.train()

        # Save checkpoint
        if step > 0 and step % train_config.save_interval == 0 and step != start_step:
            save_checkpoint(model, optimizer, scaler, step, running_loss, tokens_seen,
                          train_config, model_config, efficiency_history,
                          checkpoint_dir / f'step_{step:06d}.pt')
            save_checkpoint(model, optimizer, scaler, step, running_loss, tokens_seen,
                          train_config, model_config, efficiency_history, latest_path)

            if running_loss <= best_loss:
                save_checkpoint(model, optimizer, scaler, step, running_loss, tokens_seen,
                              train_config, model_config, efficiency_history,
                              checkpoint_dir / 'best.pt')

    # Final save
    save_checkpoint(model, optimizer, scaler, train_config.max_steps, running_loss, tokens_seen,
                   train_config, model_config, efficiency_history, checkpoint_dir / 'final.pt')

    log("\n" + "=" * 70)
    log("TRAINING COMPLETE!")
    log("=" * 70)
    log(f"Total time: {timedelta(seconds=int(time.time() - start_time))}")
    log(f"Final loss: {running_loss:.4f}")
    log(f"Best loss: {best_loss:.4f}")
    log(f"Tokens processed: {tokens_seen/1e9:.2f}B")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PROMEPIR 250M Pretraining (PromeTorch)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--text_ratio", type=float, default=0.8, help="Ratio of text vs dialogs (0.8 = 80%% text, 20%% dialogs)")

    args, _ = parser.parse_known_args()

    base_path = setup_google_drive()

    model_config = PIR250MConfig()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        use_compile=not getattr(args, 'no_compile', False),
        text_ratio=args.text_ratio
    )

    print("\n" + "=" * 70)
    print("PROMEPIR 250M - NEXT CONCEPT PREDICTION")
    print("Powered by PromeTorch (custom framework, not PyTorch)")
    print("=" * 70)
    print(f"Parameters: ~250M")
    print(f"Context: {model_config.block_size} tokens")
    print(f"Chinchilla optimal: 5B tokens")
    print(f"Planned tokens: {train_config.total_tokens/1e9:.1f}B")
    print(f"Dataset: {train_config.text_ratio*100:.0f}% FineWeb-Edu + {(1-train_config.text_ratio)*100:.0f}% UltraChat")
    print(f"Decay ranges: {model_config.decay_ranges}")
    print(f"Layer efficiency monitoring enabled")
    print("=" * 70 + "\n")

    train(train_config, model_config, base_path, resume=args.resume)
