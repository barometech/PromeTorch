"""
model_splitter.py -- Split HuggingFace Qwen3-4B into per-layer safetensors files.

Input: a HuggingFace model directory with:
  - config.json
  - model.safetensors.index.json (maps tensor names to shard files)
  - model-00001-of-00002.safetensors, model-00002-of-00002.safetensors, ...

Output: a directory with:
  - embed.safetensors          (model.embed_tokens.weight)
  - layer_00.safetensors       (model.layers.0.*)
  - layer_01.safetensors       (model.layers.1.*)
  - ...
  - layer_35.safetensors       (model.layers.35.*)
  - norm.safetensors           (model.norm.weight)
  - lm_head.safetensors        (lm_head.weight)
  - config.json                (copied from source)

No torch or transformers dependency. Uses raw safetensors parsing.
"""

import json
import os
import struct
import shutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from .layer_loader import SafetensorsFile, bf16_to_fp32, SAFETENSORS_DTYPE_MAP


def write_safetensors(filepath: str, tensors: Dict[str, np.ndarray]):
    """Write a dict of numpy arrays to safetensors format.

    All tensors are saved as F32.
    """
    # Build header
    header = {}
    current_offset = 0
    tensor_data_parts = []

    for name, arr in tensors.items():
        arr = arr.astype(np.float32)
        raw = arr.tobytes()
        size = len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + size]
        }
        tensor_data_parts.append(raw)
        current_offset += size

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    padding = (8 - len(header_json) % 8) % 8
    header_json += b" " * padding

    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for part in tensor_data_parts:
            f.write(part)


class ModelSplitter:
    """Split a multi-shard HuggingFace model into per-layer safetensors files.

    Usage:
        splitter = ModelSplitter("path/to/Qwen3-4B")
        splitter.split("path/to/output_split_dir")
    """

    # Qwen3 layer name patterns
    EMBED_PREFIX = "model.embed_tokens."
    LAYER_PREFIX = "model.layers."
    NORM_PREFIX = "model.norm."
    LM_HEAD_PREFIX = "lm_head."

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.config = self._load_config()
        self.weight_map = self._load_weight_map()
        self.num_layers = self.config.get("num_hidden_layers", 36)

    def _load_config(self) -> dict:
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_weight_map(self) -> Dict[str, str]:
        """Load tensor->shard mapping from model.safetensors.index.json.

        If the model is a single file (no index), return a synthetic mapping.
        """
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            return index["weight_map"]

        # Single-file model
        single_path = os.path.join(self.model_dir, "model.safetensors")
        if os.path.exists(single_path):
            sf = SafetensorsFile(single_path)
            return {name: "model.safetensors" for name in sf.tensor_names()}

        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {self.model_dir}")

    def _classify_tensor(self, tensor_name: str) -> Tuple[str, str]:
        """Classify a tensor name into (layer_id, short_name).

        Returns:
            layer_id: "embed", "layer_00", "norm", "lm_head"
            short_name: name relative to the layer
        """
        if tensor_name.startswith(self.EMBED_PREFIX):
            return "embed", tensor_name[len(self.EMBED_PREFIX):]
        elif tensor_name.startswith(self.LAYER_PREFIX):
            # "model.layers.0.self_attn.q_proj.weight" -> layer_id="layer_00", short="self_attn.q_proj.weight"
            rest = tensor_name[len(self.LAYER_PREFIX):]
            dot_pos = rest.index(".")
            layer_num = int(rest[:dot_pos])
            short_name = rest[dot_pos + 1:]
            return f"layer_{layer_num:02d}", short_name
        elif tensor_name.startswith(self.NORM_PREFIX):
            return "norm", tensor_name[len(self.NORM_PREFIX):]
        elif tensor_name.startswith(self.LM_HEAD_PREFIX):
            return "lm_head", tensor_name[len(self.LM_HEAD_PREFIX):]
        else:
            # Unknown -- put in a catch-all
            return "other", tensor_name

    def split(self, output_dir: str, convert_to_fp32: bool = True,
              verbose: bool = True):
        """Split the model into per-layer safetensors files.

        Args:
            output_dir: directory to write split files
            convert_to_fp32: if True, convert BF16/FP16 to FP32
            verbose: print progress
        """
        os.makedirs(output_dir, exist_ok=True)

        # Copy config.json
        src_config = os.path.join(self.model_dir, "config.json")
        dst_config = os.path.join(output_dir, "config.json")
        shutil.copy2(src_config, dst_config)

        # Group tensors by layer
        layer_tensors: Dict[str, Dict[str, Tuple[str, str]]] = {}
        # layer_tensors[layer_id][short_name] = (shard_file, full_tensor_name)
        for tensor_name, shard_file in self.weight_map.items():
            layer_id, short_name = self._classify_tensor(tensor_name)
            if layer_id not in layer_tensors:
                layer_tensors[layer_id] = {}
            layer_tensors[layer_id][short_name] = (shard_file, tensor_name)

        # Cache of opened shard files
        shard_cache: Dict[str, SafetensorsFile] = {}

        def get_shard(shard_name: str) -> SafetensorsFile:
            if shard_name not in shard_cache:
                shard_path = os.path.join(self.model_dir, shard_name)
                shard_cache[shard_name] = SafetensorsFile(shard_path)
            return shard_cache[shard_name]

        # Process each layer
        # Sort to ensure consistent order: embed, layer_00..layer_35, norm, lm_head
        sorted_layers = self._sort_layer_ids(list(layer_tensors.keys()))

        for layer_id in sorted_layers:
            if verbose:
                print(f"  Splitting: {layer_id} ({len(layer_tensors[layer_id])} tensors)...")

            tensors_out: Dict[str, np.ndarray] = {}
            for short_name, (shard_file, full_name) in layer_tensors[layer_id].items():
                sf = get_shard(shard_file)
                arr = sf.load_tensor(full_name, to_float32=convert_to_fp32)
                tensors_out[short_name] = arr

            # Write per-layer file
            out_path = os.path.join(output_dir, f"{layer_id}.safetensors")
            write_safetensors(out_path, tensors_out)

            # Report size
            if verbose:
                file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
                print(f"    -> {out_path} ({file_size_mb:.1f} MB)")

            # Free memory
            del tensors_out

        # Clear shard cache
        shard_cache.clear()

        if verbose:
            total_size = sum(
                os.path.getsize(os.path.join(output_dir, f))
                for f in os.listdir(output_dir)
                if f.endswith(".safetensors")
            )
            print(f"\nDone! Total: {total_size / (1024**3):.2f} GB in {output_dir}")

        # Write manifest
        self._write_manifest(output_dir, sorted_layers, layer_tensors)

    def _sort_layer_ids(self, layer_ids: List[str]) -> List[str]:
        """Sort layer IDs in execution order."""
        order = {"embed": 0, "norm": 9000, "lm_head": 9001, "other": 9002}

        def sort_key(lid):
            if lid.startswith("layer_"):
                return 1 + int(lid.split("_")[1])
            return order.get(lid, 9999)

        return sorted(layer_ids, key=sort_key)

    def _write_manifest(self, output_dir: str, sorted_layers: List[str],
                        layer_tensors: Dict):
        """Write a manifest.json with layer order and sizes."""
        manifest = {
            "model_type": self.config.get("model_type", "qwen3"),
            "num_layers": self.num_layers,
            "hidden_size": self.config.get("hidden_size", 2560),
            "layer_order": sorted_layers,
            "layers": {}
        }
        for layer_id in sorted_layers:
            filepath = os.path.join(output_dir, f"{layer_id}.safetensors")
            manifest["layers"][layer_id] = {
                "file": f"{layer_id}.safetensors",
                "size_bytes": os.path.getsize(filepath),
                "tensors": list(layer_tensors[layer_id].keys())
            }

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def estimate_split_size(self) -> Dict[str, int]:
        """Estimate per-layer file sizes (FP32) without actually loading."""
        sizes: Dict[str, int] = {}

        # Open shards to read headers only
        shard_headers: Dict[str, SafetensorsFile] = {}
        for shard_name in set(self.weight_map.values()):
            shard_path = os.path.join(self.model_dir, shard_name)
            if os.path.exists(shard_path):
                shard_headers[shard_name] = SafetensorsFile(shard_path)

        for tensor_name, shard_file in self.weight_map.items():
            layer_id, _ = self._classify_tensor(tensor_name)
            if layer_id not in sizes:
                sizes[layer_id] = 0

            if shard_file in shard_headers:
                info = shard_headers[shard_file].get_tensor_info(tensor_name)
                if info:
                    numel = 1
                    for s in info["shape"]:
                        numel *= s
                    sizes[layer_id] += numel * 4  # FP32

        return sizes


def main():
    """CLI entry point for splitting."""
    import argparse
    parser = argparse.ArgumentParser(description="Split HF model into per-layer safetensors")
    parser.add_argument("model_dir", help="Path to HuggingFace model directory")
    parser.add_argument("output_dir", help="Output directory for split files")
    parser.add_argument("--no-fp32", action="store_true", help="Keep original dtype (no FP32 conversion)")
    args = parser.parse_args()

    splitter = ModelSplitter(args.model_dir)

    print(f"Model: {splitter.config.get('model_type', 'unknown')}")
    print(f"Layers: {splitter.num_layers}")
    print(f"Estimating sizes...")

    est = splitter.estimate_split_size()
    for lid in splitter._sort_layer_ids(list(est.keys())):
        print(f"  {lid}: {est[lid] / (1024*1024):.1f} MB")

    print(f"\nSplitting to {args.output_dir}...")
    splitter.split(args.output_dir, convert_to_fp32=not args.no_fp32)


if __name__ == "__main__":
    main()
