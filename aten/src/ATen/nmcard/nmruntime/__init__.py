"""
NMRuntime - Python runtime for NM Card Mini
Provides high-level API for neural network operations on NMC4
"""

from .device import Device, MultiCoreDevice
from .ops import (
    OP_NOP, OP_MATMUL, OP_RMSNORM, OP_SOFTMAX, OP_SILU,
    OP_ROPE, OP_ATTENTION, OP_ELEM_ADD, OP_ELEM_MUL, OP_GATE_MUL
)
from .quantize import (
    quantize_weights_perchannel,
    quantize_activation,
    pack_int8_to_uint32,
    float32_to_q16_16,
    q16_16_to_float32
)
from .model import TinyLlamaModel, TinyLlamaConfig, WeightLoader
from .tokenizer import Tokenizer

__version__ = "0.1.0"
__all__ = [
    # Device
    'Device', 'MultiCoreDevice',
    # Operations
    'OP_NOP', 'OP_MATMUL', 'OP_RMSNORM', 'OP_SOFTMAX',
    'OP_SILU', 'OP_ROPE', 'OP_ATTENTION', 'OP_ELEM_ADD',
    'OP_ELEM_MUL', 'OP_GATE_MUL',
    # Quantization
    'quantize_weights_perchannel', 'quantize_activation',
    'pack_int8_to_uint32', 'float32_to_q16_16', 'q16_16_to_float32',
    # Model
    'TinyLlamaModel', 'TinyLlamaConfig', 'WeightLoader', 'Tokenizer'
]
