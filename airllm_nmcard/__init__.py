# AirLLM-NMCard: Layer-streaming LLM inference on NM Card Mini 16-core
# Inspired by AirLLM (https://github.com/lyogavin/airllm)
#
# Design: Load one transformer layer at a time from disk into NM Card's 5GB DDR,
# run matmul on 16 NMC4 cores (4 primary vectorized + 12 secondary scalar),
# then discard weights before loading the next layer.
#
# No torch or transformers dependency required.

from .inference import AirLLMNMCard
from .layer_loader import LayerLoader
from .model_splitter import ModelSplitter
from .tokenizer import BPETokenizer

__version__ = "0.1.0"
__all__ = ["AirLLMNMCard", "LayerLoader", "ModelSplitter", "BPETokenizer"]
