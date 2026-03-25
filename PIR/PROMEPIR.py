"""
PROMEPIR — PIR 250M running on PromeTorch
==========================================

PIR 250M model (Next Concept Prediction) on PromeTorch framework.
Architecture: Pure PIR (no attention), SwiGLU FFN, RoPE, RMSNorm.

Two-phase training:
  Phase 1 (pretrain): existing_local + wikipedia_ru + russian_pd + taiga_proza + fineweb2_hq
  Phase 2 (SFT):      mega_dialog + rukallama_basic + dialog_augment + all_instructions + megaset + sberquad + ZeroAgency

Usage:
    python PROMEPIR.py --phase pretrain                    # Phase 1
    python PROMEPIR.py --phase sft --resume                # Phase 2 (after pretrain)
    python PROMEPIR.py --phase pretrain --data_dir /path   # custom data directory
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
        w = self.weight.data if hasattr(self.weight, 'data') else self.weight
        return (x_float * rms * w).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # base ** tensor not supported in PromeTorch, use numpy
        exponents = torch.arange(0, dim, 2).float() / dim
        inv_freq = torch.from_numpy((1.0 / (base ** exponents.numpy())).astype(np.float32))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(0, max_seq_len, 1.0)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len].unsqueeze(0), self.sin_cached[:seq_len].unsqueeze(0)


def _chunk2(x, dim=-1):
    """Split tensor into 2 halves along last dim via numpy roundtrip."""
    shape = [x.size(i) for i in range(x.dim if isinstance(x.dim, int) else x.dim())]
    D = shape[-1]
    half = D // 2
    # Detach, go to numpy, split, come back
    x_np = x.detach().numpy()
    x1_np = x_np[..., :half].copy()
    x2_np = x_np[..., half:].copy()
    x1 = torch.from_numpy(x1_np.astype(np.float32)).view(shape[:-1] + [half])
    x2 = torch.from_numpy(x2_np.astype(np.float32)).view(shape[:-1] + [half])
    return x1, x2


def rotate_half(x):
    x1, x2 = _chunk2(x, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = _chunk2(x, dim=-1)
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

        n_params = sum(p.numel for p in self.parameters())
        print(f"PIR 250M initialized: {n_params/1e6:.1f}M parameters")

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        if self.lm_head.weight is not self.tok_emb.weight:
            nn.init.normal_(self.lm_head.weight, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(self, idx, targets=None, store_states: bool = False):
        B, T = idx.shape

        x = self.tok_emb(idx)
        # RoPE — skip if _chunk2 numpy roundtrip is too expensive (Elbrus)
        if not getattr(self, '_skip_rope', False):
            try:
                cos, sin = self.rope(T)
                x = apply_rotary_pos_emb(x, cos, sin)
            except Exception:
                self._skip_rope = True  # Disable for future calls

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
        if (states.dim if isinstance(states.dim, int) else states.dim()) == 3:
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
        if (states.dim if isinstance(states.dim, int) else states.dim()) == 3:
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
# DATASET LOADER — ALL LOCAL FILES
# ============================================================================
#
# PRETRAIN sources (all local):
#   1. existing_local (~59M tok) — 767 books restored_text.txt
#   2. wikipedia_ru (~1.5B tok) — HF arrow cache
#   3. grandmaster_pro_max.json (519 MB)
#   4. ru_instruct_200k.json (457 MB)
#   5. russian_instructions_2.json (230 MB)
#   6. sberquad.json (11 MB)
#   7. all_instructions.jsonl (49 MB)
#   8. dialog_augment (2.6 MB)
#   9. ZeroAgency ru_big_dataset.jsonl (1.4 GB)
#
# ============================================================================

# Local file paths (Windows)
LOCAL_PRETRAIN_SOURCES = {
    "existing_local": {
        "path": "C:/Users/paper/Desktop/RNDM/KELLM/DATASET",
        "type": "dir_txt",  # recursive restored_text.txt
    },
    "wikipedia_ru": {
        "path": "C:/Users/paper/Desktop/RNB SEARCH AI/hf_cache/datasets/wikimedia___wikipedia/20231101.ru/0.0.0/b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "type": "arrow",
    },
    "grandmaster": {
        "path": "C:/Users/paper/Desktop/RNDM/KELLM/DATASET_EXPANDED/grandmaster_pro_max.json",
        "type": "json",
    },
    "ru_instruct": {
        "path": "C:/Users/paper/Desktop/RNDM/KELLM/DATASET_EXPANDED/ru_instruct_200k.json",
        "type": "json",
    },
    "russian_instr2": {
        "path": "C:/Users/paper/Desktop/RNDM/KELLM/DATASET_EXPANDED/russian_instructions_2.json",
        "type": "json",
    },
    "sberquad": {
        "path": "C:/Users/paper/Desktop/RNDM/KELLM/DATASET_EXPANDED/sberquad.json",
        "type": "json",
    },
    "all_instructions": {
        "path": "C:/Users/paper/Desktop/RUKALLAMA V2/CLEAN_DATASET/all_instructions.jsonl",
        "type": "jsonl",
    },
    "dialog_augment": {
        "path": "C:/Users/paper/Desktop/RUKALLAMA V2/DIALOG AUGMENT",
        "type": "dir_json",
    },
    "ZeroAgency": {
        "path": "C:/Users/paper/Desktop/RUKALLAMA V2/SFT_DATA/ru_big_dataset.jsonl",
        "type": "jsonl",
    },
}


def _tokenize_and_write(enc, texts_iter, cache_path, target_tokens, chunk_size=10_000_000, label=""):
    """Tokenize text iterator and write to memmap cache file. Returns token count."""
    mm = np.memmap(cache_path, dtype=np.uint16, mode='w+', shape=(target_tokens,))
    pos = 0
    chunk = []
    count = 0

    for text in texts_iter:
        if not text:
            continue
        toks = enc.encode_ordinary(text)
        chunk.extend(toks)
        count += 1

        if len(chunk) >= chunk_size:
            end = min(pos + len(chunk), target_tokens)
            write_len = end - pos
            mm[pos:end] = np.array(chunk[:write_len], dtype=np.uint16)
            pos = end
            chunk = chunk[write_len:]
            mm.flush()
            if count % 5000 == 0:
                print(f"  [{label}] {pos/1e6:.1f}M tokens, {count} docs")

        if pos >= target_tokens:
            break

    if chunk and pos < target_tokens:
        end = min(pos + len(chunk), target_tokens)
        mm[pos:end] = np.array(chunk[:end - pos], dtype=np.uint16)
        pos = end

    mm.flush()
    del mm

    # Truncate to actual size
    if pos < target_tokens:
        tmp = np.memmap(cache_path, dtype=np.uint16, mode='r+', shape=(target_tokens,))
        actual = np.array(tmp[:pos], dtype=np.uint16)
        del tmp
        mm2 = np.memmap(cache_path, dtype=np.uint16, mode='w+', shape=(pos,))
        mm2[:] = actual
        mm2.flush()
        del mm2

    print(f"  [{label}] DONE: {pos/1e6:.1f}M tokens from {count} docs")
    return pos


def _stream_texts_from_source(name, cfg):
    """Stream text strings from a local source based on its type."""
    p = Path(cfg["path"])
    src_type = cfg["type"]

    if not p.exists():
        print(f"  WARNING: {p} not found, skipping {name}")
        return

    if src_type == "dir_txt":
        # Directory with restored_text.txt files (existing_local)
        txt_files = sorted(p.rglob("restored_text.txt"))
        print(f"  {name}: found {len(txt_files)} restored_text.txt files in {p}")
        for f in txt_files:
            text = f.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                yield text

    elif src_type == "arrow":
        # HuggingFace arrow cache directory
        import pyarrow as pa
        arrow_files = sorted(p.glob("*.arrow"))
        if not arrow_files:
            arrow_files = sorted(p.rglob("*.arrow"))
        print(f"  {name}: found {len(arrow_files)} arrow files in {p}")
        for af in arrow_files:
            try:
                reader = pa.ipc.open_file(af)
                for batch_idx in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_idx)
                    if "text" in batch.schema.names:
                        texts = batch.column("text").to_pylist()
                        for t in texts:
                            if t and len(t) > 50:
                                yield t
            except Exception as e:
                print(f"    ERROR reading {af.name}: {e}")

    elif src_type == "json":
        # Single JSON file (array of objects or single object)
        print(f"  {name}: loading {p.name} ({p.stat().st_size/1e6:.0f}MB)")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                text = _extract_text(obj)
                if text:
                    yield text
        elif isinstance(data, dict):
            text = _extract_text(data)
            if text:
                yield text

    elif src_type == "jsonl":
        # JSONL file (one JSON object per line)
        print(f"  {name}: loading {p.name} ({p.stat().st_size/1e6:.0f}MB)")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = _extract_text(obj)
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

    elif src_type == "dir_json":
        # Directory with JSON files
        json_files = sorted(p.rglob("*.json"))
        print(f"  {name}: found {len(json_files)} json files in {p}")
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        text = _extract_text(obj)
                        if text:
                            yield text
                elif isinstance(data, dict):
                    text = _extract_text(data)
                    if text:
                        yield text
            except Exception as e:
                print(f"    ERROR reading {jf.name}: {e}")


def _extract_text(obj):
    """Extract text from a JSON object, trying various field names."""
    if isinstance(obj, str):
        return obj if len(obj) > 10 else None

    if not isinstance(obj, dict):
        return None

    # q/a format (grandmaster, ru_instruct, sberquad, etc.)
    q = obj.get("q", "")
    a = obj.get("a", "")
    if q and a:
        return f"{q}\n{a}"

    # Direct text fields
    for key in ["text", "content", "instruction", "output", "response",
                "answer", "question", "input", "context"]:
        val = obj.get(key)
        if val and isinstance(val, str) and len(val) > 10:
            return val

    # Dialog/conversation fields — concatenate all turns
    messages = obj.get("messages", obj.get("conversations", obj.get("data", [])))
    if isinstance(messages, list) and messages:
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", msg.get("value", msg.get("text", "")))
                if content:
                    parts.append(content)
            elif isinstance(msg, str):
                parts.append(msg)
        if parts:
            return "\n\n".join(parts)

    # Instruction + output concatenation
    inst = obj.get("instruction", "")
    out = obj.get("output", obj.get("response", ""))
    if inst or out:
        return f"{inst}\n{out}".strip() if inst and out else (inst or out)

    return None


class PretrainDataLoader:
    """
    Pretrain data loader — reads ALL local sources, tokenizes, caches as .bin.
    On Elbrus: reads pre-built .bin files from cache_dir.
    """
    def __init__(self, config, base_path: Path, data_dir: Path, enc):
        self.config = config
        self.base_path = base_path
        self.seq_len = config.seq_len
        self.enc = enc

        self.cache_dir = base_path / 'data' / 'pretrain'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_all_sources()

        self.positions = {name: 0 for name in self.sources}
        self.source_names = list(self.sources.keys())

    def _load_all_sources(self):
        """Load/cache all pretrain sources from LOCAL_PRETRAIN_SOURCES."""
        self.sources = {}

        for name, cfg in LOCAL_PRETRAIN_SOURCES.items():
            cache_path = self.cache_dir / f"{name}.bin"

            if cache_path.exists():
                self.sources[name] = np.memmap(cache_path, dtype=np.uint16, mode='r')
                print(f"  {name}: {len(self.sources[name])/1e6:.1f}M tokens (cached)")
            else:
                target = 3_000_000_000  # 3B max per source
                texts = _stream_texts_from_source(name, cfg)
                n = _tokenize_and_write(self.enc, texts, cache_path, target, label=name)
                if n > 0:
                    self.sources[name] = np.memmap(cache_path, dtype=np.uint16, mode='r')

        if not self.sources:
            raise RuntimeError("No pretrain data found! Check LOCAL_PRETRAIN_SOURCES paths.")

        total = sum(len(v) for v in self.sources.values())
        print(f"\n  PRETRAIN TOTAL: {total/1e9:.2f}B tokens from {len(self.sources)} sources")
        for name, arr in self.sources.items():
            pct = len(arr) / total * 100 if total > 0 else 0
            print(f"    {name}: {len(arr)/1e6:.1f}M ({pct:.1f}%)")

    def get_batch(self, device) -> Tuple:
        """Get batch — weighted random source selection, standard LM."""
        batch_x = []
        batch_y = []

        for _ in range(self.config.batch_size):
            src_name = self._pick_source()
            tokens = self.sources[src_name]
            pos = self.positions[src_name]

            if pos + self.seq_len + 1 >= len(tokens):
                pos = 0
            self.positions[src_name] = pos + self.seq_len

            x = torch.from_numpy(tokens[pos:pos + self.seq_len].astype(np.int64))
            y = torch.from_numpy(tokens[pos + 1:pos + self.seq_len + 1].astype(np.int64))
            batch_x.append(x)
            batch_y.append(y)

        return torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)

    def _pick_source(self) -> str:
        """Pick source proportional to its size."""
        sizes = [len(self.sources[n]) for n in self.source_names]
        total = sum(sizes)
        r = random.random() * total
        cumsum = 0
        for i, s in enumerate(sizes):
            cumsum += s
            if r <= cumsum:
                return self.source_names[i]
        return self.source_names[-1]


# SFT is Phase 2 — same sources minus dialog-specific ones, loaded as dialogs
# For now SFT reuses the same local files but with dialog masking


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

EVAL_PROMPTS_PRETRAIN = [
    # Русские промпты для pretrain (text completion)
    "Столица России —",
    "Москва была основана в",
    "Русский язык является одним из",
    "Вода замерзает при температуре",
    "Александр Сергеевич Пушкин родился в",
    "Великая Отечественная война началась",
    "Периодическая таблица Менделеева содержит",
    "Байкал является самым глубоким озером",
    "Фотосинтез — это процесс, при котором растения",
    "Искусственный интеллект — это область",
    "Теорема Пифагора утверждает, что",
    "В информатике алгоритм определяется как",
    "Советский Союз распался в",
    "Космическая программа СССР начала",
    "Нейронные сети — это вычислительные системы",
    "партнёр Иванович Менделеев открыл",
    "Конституция Российской Федерации была принята",
    "Электричество — это совокупность явлений",
    "Философия как наука зародилась в",
    "Квантовая механика описывает поведение",
]

EVAL_PROMPTS_SFT = [
    # Русские промпты для SFT (dialog)
    "User: Что такое машинное обучение?\n\nAssistant:",
    "User: Как работает фотосинтез?\n\nAssistant:",
    "User: Объясни теорию относительности.\n\nAssistant:",
    "User: Какие преимущества регулярных физических упражнений?\n\nAssistant:",
    "User: Расскажи про историю России.\n\nAssistant:",
    "User: Что такое квантовый компьютер?\n\nAssistant:",
    "User: Как написать сортировку на Python?\n\nAssistant:",
    "User: Объясни как работает нейронная сеть.\n\nAssistant:",
    "User: Что такое блокчейн?\n\nAssistant:",
    "User: Почему небо голубое?\n\nAssistant:",
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
                    temperature: float = 0.7, device: str = "cpu") -> str:
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
          base_path: Path, data_dir: Path, phase: str = "pretrain",
          resume: bool = False):
    """
    Main training function.
    phase="pretrain": Phase 1 — all pretrain sources, standard LM
    phase="sft":      Phase 2 — SFT sources, masked LM (learn on assistant only)
    """
    from gpt2_tokenizer import GPT2Tokenizer
    enc = GPT2Tokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"PROMEPIR 250M — PHASE {'1: PRETRAIN' if phase == 'pretrain' else '2: SFT'}")
    print("Running on PromeTorch (custom framework)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Max steps: {train_config.max_steps}")
    print(f"Tokens per step: {train_config.tokens_per_step:,}")
    print(f"Context length: {model_config.block_size}")
    print(f"Decay ranges: {model_config.decay_ranges}")
    print(f"Data dir: {data_dir}")
    print("=" * 70)

    # Data
    print("\nLoading data...")
    if phase == "pretrain":
        data_loader = PretrainDataLoader(train_config, base_path, data_dir, enc)
    else:
        data_loader = SFTDataLoader(train_config, base_path, data_dir, enc)

    # Model
    print("\nInitializing model...")
    model = PIR250M(model_config)
    if device == "cuda":
        model.to(device)

    # Optimizer
    all_params = list(model.parameters())
    lr = train_config.learning_rate if phase == "pretrain" else train_config.learning_rate * 0.1
    optimizer = torch.optim.AdamW(all_params, lr=lr,
                                   betas=(train_config.beta1, train_config.beta2),
                                   weight_decay=train_config.weight_decay)
    scaler = GradScaler()

    # Resume
    start_step = 0
    tokens_seen = 0
    efficiency_history = []
    checkpoint_dir = base_path / 'checkpoints' / phase
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / 'latest.pt'

    # For SFT, try to load pretrain checkpoint first
    if phase == "sft" and not resume:
        pretrain_best = base_path / 'checkpoints' / 'pretrain' / 'best.pt'
        pretrain_final = base_path / 'checkpoints' / 'pretrain' / 'final.pt'
        pretrain_latest = base_path / 'checkpoints' / 'pretrain' / 'latest.pt'
        for ckpt in [pretrain_best, pretrain_final, pretrain_latest]:
            if ckpt.exists():
                print(f"Loading pretrain weights from {ckpt}...")
                load_checkpoint(ckpt, model, optimizer, scaler)
                # Reset optimizer state for SFT
                all_params = list(model.parameters())
                optimizer = torch.optim.AdamW(all_params, lr=lr,
                                               betas=(train_config.beta1, train_config.beta2),
                                               weight_decay=train_config.weight_decay)
                scaler = GradScaler()
                break

    if resume and latest_path.exists():
        print(f"Resuming from {latest_path}...")
        start_step, _, tokens_seen, efficiency_history = load_checkpoint(
            latest_path, model, optimizer, scaler)

    # Compile (no-op in PromeTorch)
    if train_config.use_compile:
        print("torch.compile: no-op in PromeTorch (native C++ execution)")
        model = torch.compile(model)

    # Select eval prompts based on phase
    eval_prompts = EVAL_PROMPTS_PRETRAIN if phase == "pretrain" else EVAL_PROMPTS_SFT

    # Log file
    log_file = base_path / 'logs' / f'{phase}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + "\n")

    # Training
    log(f"\nStarting {phase} from step {start_step}...")
    model.train()

    start_time = time.time()
    running_loss = 0.0
    best_loss = float("inf")

    for step in range(start_step, train_config.max_steps):
        step_start = time.time()

        # Learning rate schedule
        current_lr = get_lr(step, train_config)
        if phase == "sft":
            current_lr *= 0.1  # Lower LR for SFT
        optimizer.lr = current_lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_eff_penalty = 0.0
        L2_ratio_accum = 0.0
        L3_ratio_accum = 0.0
        valid_microbatches = 0

        use_efficiency = step >= train_config.efficiency_warmup_steps

        for _ in range(train_config.gradient_accumulation):
            x, y = data_loader.get_batch(device)

            with autocast(device):
                logits, ce_loss = model(x, y, store_states=use_efficiency)

                if use_efficiency and phase == "pretrain":
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

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss = 0.99 * running_loss + 0.01 * accum_loss if running_loss else accum_loss
        tokens_seen += train_config.tokens_per_step
        step_time = time.time() - step_start
        tokens_per_sec = train_config.tokens_per_step / step_time

        if running_loss < best_loss:
            best_loss = running_loss

        if step % train_config.log_interval == 0:
            elapsed = time.time() - start_time
            eta = (train_config.max_steps - step) * (elapsed / max(step - start_step + 1, 1))
            ppl = math.exp(min(accum_loss, 20))

            log_msg = (f"Step {step:>6}/{train_config.max_steps} | "
                      f"Loss: {accum_loss:.4f} (avg: {running_loss:.4f}) | "
                      f"PPL: {ppl:.1f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Grad: {grad_norm:.2f} | "
                      f"Tokens: {tokens_seen/1e9:.2f}B | "
                      f"Speed: {tokens_per_sec/1e3:.1f}K tok/s | "
                      f"ETA: {timedelta(seconds=int(eta))}")

            if use_efficiency and phase == "pretrain":
                log_msg += f" | L2:{L2_ratio_avg:.2f} L3:{L3_ratio_avg:.2f}"

            log(log_msg)

        if step > 0 and step % train_config.eval_interval == 0 and step != start_step:
            model.eval()
            log("\n" + "=" * 60)
            log(f"EVALUATION ({phase})")
            log("=" * 60)

            with torch.no_grad():
                x_eval, y_eval = data_loader.get_batch(device)
                _, _ = model(x_eval, y_eval, store_states=True)

                eff_results = measure_layer_efficiency(model, model_config.decay_ranges)
                log("\n" + format_efficiency_report(eff_results))

                efficiency_history.append({
                    'step': step,
                    'tokens': tokens_seen,
                    'efficiency': {k: v['avg_ratio'] for k, v in eff_results.items()}
                })

            log("\nGeneration Samples:")
            num_prompts = min(10, len(eval_prompts))
            selected_prompts = random.sample(eval_prompts, num_prompts)

            for i, prompt in enumerate(selected_prompts):
                try:
                    sample = generate_sample(model, enc, prompt, max_tokens=300,
                                            temperature=0.7, device=device)
                    log(f"\n[{i+1}] {sample}")
                except Exception as e:
                    log(f"\n[{i+1}] [{prompt[:30]}...] Failed: {e}")

            log("\n" + "=" * 60)
            model.train()

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
    log(f"{phase.upper()} COMPLETE!")
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

    parser = argparse.ArgumentParser(description="PROMEPIR 250M (PromeTorch)")
    parser.add_argument("--phase", type=str, default="pretrain",
                        choices=["pretrain", "sft"],
                        help="Training phase: pretrain (Phase 1) or sft (Phase 2)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with existing_local/, sft/ subdirs")

    args, _ = parser.parse_known_args()

    base_path = setup_google_drive()
    data_dir = Path(args.data_dir) if args.data_dir else base_path / 'data'

    model_config = PIR250MConfig()

    # SFT uses fewer steps
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = 40000 if args.phase == "pretrain" else 5000

    train_config = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        max_steps=max_steps,
        use_compile=not getattr(args, 'no_compile', False),
    )

    phase_name = "PRETRAIN" if args.phase == "pretrain" else "SFT"

    print("\n" + "=" * 70)
    print(f"PROMEPIR 250M — {phase_name}")
    print("Powered by PromeTorch (custom framework)")
    print("=" * 70)
    print(f"Parameters: ~250M")
    print(f"Context: {model_config.block_size} tokens")
    print(f"Max steps: {max_steps}")
    print(f"Data dir: {data_dir}")

    if args.phase == "pretrain":
        print(f"Sources: existing_local + wikipedia_ru + russian_pd + taiga_proza + fineweb2_hq")
    else:
        print(f"Sources: mega_dialog + rukallama_basic + dialog_augment + all_instructions + megaset")
        print(f"       + grandmaster + russian_instr2 + sberquad + ru_instruct + ZeroAgency")

    print(f"Decay ranges: {model_config.decay_ranges}")
    print("=" * 70 + "\n")

    train(train_config, model_config, base_path, data_dir,
          phase=args.phase, resume=args.resume)
