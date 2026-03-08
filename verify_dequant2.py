"""Verify dequantization using gguf library."""
import numpy as np
import json
import os

# Find qwen3:4b GGUF blob
ollama_home = os.path.join(os.environ.get("USERPROFILE", ""), ".ollama", "models")
manifest_path = os.path.join(ollama_home, "manifests", "registry.ollama.ai", "library", "qwen3", "4b")
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

blob_digest = None
for layer in manifest['layers']:
    if layer.get('mediaType') == 'application/vnd.ollama.image.model':
        blob_digest = layer['digest'].replace(':', '-')
        break

blob_path = os.path.join(ollama_home, "blobs", blob_digest)
print(f"GGUF: {blob_path}")

# Use gguf library
from gguf import GGUFReader

reader = GGUFReader(blob_path)

# Print some metadata
for field in reader.fields:
    f = reader.fields[field]
    if 'bos' in field or 'add_bos' in field or 'vocab_size' in field:
        # Get value
        if hasattr(f, 'parts') and len(f.parts) > 0:
            val = f.parts[-1].tolist()
            if len(val) == 1:
                val = val[0]
            print(f"  {field} = {val}")

# Find token_embd.weight tensor
print("\nLooking for tensors...")
for tensor in reader.tensors:
    name = tensor.name
    if name in ['token_embd.weight', 'blk.0.attn_norm.weight', 'blk.0.attn_q.weight']:
        print(f"\n  {name}: shape={tensor.shape}, type={tensor.tensor_type}")

        # Dequantize
        data = tensor.data
        dtype = tensor.tensor_type

        # GGUF GGUFReader provides raw data, we need to dequantize ourselves
        # But the gguf library may have dequantization support
        # Let's check the shape and first values

        # For F32 tensors
        if dtype == 0:  # F32
            vals = np.frombuffer(data.tobytes(), dtype=np.float32)
            print(f"  First 10: {vals[:10]}")
        else:
            print(f"  Raw data size: {len(data)} bytes")
            print(f"  Tensor shape in GGUF: {tensor.shape}")

# Try using llama-cpp-python for better dequantization
try:
    from llama_cpp import Llama
    print("\nllama-cpp-python available")
except:
    print("\nllama-cpp-python not available")

# Manual Q6_K dequantization (following GGML spec exactly)
print("\n=== Manual Q6_K dequant for token_embd.weight ===")
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        data = tensor.data.tobytes()
        # Q6_K block: 210 bytes = ql[128] + qh[64] + scales[16] + d(fp16)
        # Each block = 256 values

        # Dequant first block (first 256 elements = first row partial)
        QK_K = 256
        block_size = 210

        block = data[0:block_size]
        ql = np.frombuffer(block[0:128], dtype=np.uint8)  # low 4 bits
        qh = np.frombuffer(block[128:192], dtype=np.uint8)  # high 2 bits
        scales = np.frombuffer(block[192:208], dtype=np.int8)
        d = np.frombuffer(block[208:210], dtype=np.float16)[0]

        print(f"  d = {float(d)}")
        print(f"  scales = {scales}")

        # GGML Q6_K dequant formula (from ggml-quants.c):
        # For each sub-block of 16 values (16 sub-blocks of 16 = 256 total):
        #   sub-block i: scale = scales[i]
        #   For j in range(16):
        #     idx = i * 16 + j
        #     q_lo = (ql[idx/2] >> (4*(idx%2))) & 0xF
        #     q_hi = (qh[idx/4] >> (2*(idx%4))) & 0x3
        #     q = ((q_lo | (q_hi << 4))) - 32  (6-bit value, signed)
        #     result = d * scale * q

        # Actually the ql layout is different. Let me follow ggml-quants.c exactly:
        # ql[128] stores 4 bits per value, 2 values per byte
        # For the first 128 values: ql[j] & 0xF and ql[j] >> 4
        # For the last 128 values: ql[j+64] & 0xF and ql[j+64] >> 4
        # qh[64] stores the high 2 bits, 4 values per byte

        dequant = np.zeros(256, dtype=np.float32)

        for k in range(256):
            # Get low 4 bits from ql
            ql_idx = k // 2
            if k < 128:
                q_lo = int(ql[k // 2])
                q_lo = (q_lo & 0x0F) if (k % 2 == 0) else (q_lo >> 4)
            else:
                q_lo = int(ql[(k - 128) // 2 + 64])
                q_lo = (q_lo & 0x0F) if (k % 2 == 0) else (q_lo >> 4)

            # Get high 2 bits from qh
            qh_idx = k // 4
            qh_shift = (k % 4) * 2
            q_hi = (int(qh[qh_idx]) >> qh_shift) & 0x03

            # Combine to 6-bit signed value
            q = int(q_lo | (q_hi << 4)) - 32

            # Scale index
            sc_idx = k // 16
            sc = int(scales[sc_idx])

            dequant[k] = float(d) * sc * q

        print(f"  First 20 dequantized values: {dequant[:20]}")

        # Also dequant the 2nd block to get more of row 0
        # Each row of token_embd is 2560 elements = 10 blocks of 256
        break

# Also check the norm weight
print("\n=== blk.0.attn_norm.weight (F32) ===")
for tensor in reader.tensors:
    if tensor.name == 'blk.0.attn_norm.weight':
        vals = np.frombuffer(tensor.data.tobytes(), dtype=np.float32)
        print(f"  First 10: {vals[:10]}")
        print(f"  Stats: mean={vals.mean():.6f}, min={vals.min():.6f}, max={vals.max():.6f}")
        break

print("\nDone!")
