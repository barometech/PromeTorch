"""Verify dequantization by comparing with Python gguf library."""
import numpy as np
import json
import os
import struct

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
print(f"GGUF blob: {blob_path}")
print(f"Size: {os.path.getsize(blob_path) / 1024**2:.0f} MB")

# Parse GGUF header manually to find token_embd.weight
with open(blob_path, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

    print(f"Magic: 0x{magic:08X}, Version: {version}")
    print(f"Tensors: {tensor_count}, Metadata KVs: {metadata_kv_count}")

    # Read string helper
    def read_string(f):
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    # Read value helper
    def read_value(f, vtype):
        if vtype == 0: return struct.unpack('<B', f.read(1))[0]  # UINT8
        elif vtype == 1: return struct.unpack('<b', f.read(1))[0]  # INT8
        elif vtype == 2: return struct.unpack('<H', f.read(2))[0]  # UINT16
        elif vtype == 3: return struct.unpack('<h', f.read(2))[0]  # INT16
        elif vtype == 4: return struct.unpack('<I', f.read(4))[0]  # UINT32
        elif vtype == 5: return struct.unpack('<i', f.read(4))[0]  # INT32
        elif vtype == 6: return struct.unpack('<f', f.read(4))[0]  # FLOAT32
        elif vtype == 7: return struct.unpack('?', f.read(1))[0]  # BOOL
        elif vtype == 8: return read_string(f)  # STRING
        elif vtype == 9:  # ARRAY
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [read_value(f, arr_type) for _ in range(arr_len)]
        elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]  # UINT64
        elif vtype == 11: return struct.unpack('<q', f.read(8))[0]  # INT64
        elif vtype == 12: return struct.unpack('<d', f.read(8))[0]  # FLOAT64
        else: raise ValueError(f"Unknown type {vtype}")

    # Skip metadata
    print("Skipping metadata...")
    for i in range(metadata_kv_count):
        key = read_string(f)
        vtype = struct.unpack('<I', f.read(4))[0]
        val = read_value(f, vtype)
        if 'bos' in key or 'add_bos' in key:
            print(f"  {key} = {val}")

    # Read tensor info
    print("\nReading tensor info...")
    tensor_info = []
    for i in range(tensor_count):
        name = read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensor_info.append({
            'name': name, 'dims': dims, 'type': dtype, 'offset': offset
        })

    # Find alignment point (data section start)
    # Data section is aligned to 32 bytes after all metadata + tensor info
    pos = f.tell()
    alignment = 32
    data_start = ((pos + alignment - 1) // alignment) * alignment
    print(f"Current pos: {pos}, Data start: {data_start}")

    # Find token_embd.weight and blk.0.attn_norm.weight
    for t in tensor_info:
        if t['name'] in ['token_embd.weight', 'blk.0.attn_norm.weight', 'blk.0.attn_q.weight']:
            print(f"\n  {t['name']}: dims={t['dims']}, type={t['type']}, offset={t['offset']}")

            # Read a few bytes of the tensor
            abs_offset = data_start + t['offset']
            f.seek(abs_offset)

            if t['type'] == 0:  # F32
                vals = np.frombuffer(f.read(40), dtype=np.float32)
                print(f"  First 10 values (F32): {vals[:10]}")

            elif t['type'] == 1:  # F16
                raw = np.frombuffer(f.read(20), dtype=np.float16)
                print(f"  First 10 values (F16): {raw[:10].astype(np.float32)}")

            elif t['type'] == 14:  # Q6_K
                # Q6_K block: 256 values per block, 210 bytes per block
                # Layout: ql[128] + qh[64] + scales[16] + d(fp16)
                block_data = f.read(210)
                ql = np.frombuffer(block_data[0:128], dtype=np.uint8)
                qh = np.frombuffer(block_data[128:192], dtype=np.uint8)
                scales = np.frombuffer(block_data[192:208], dtype=np.int8)
                d = np.frombuffer(block_data[208:210], dtype=np.float16)[0]

                print(f"  Q6_K block 0: d={float(d):.6f}")
                print(f"  scales: {scales}")

                # Dequantize first 16 values
                dequant = np.zeros(16, dtype=np.float32)
                for j in range(16):
                    q_lo = ql[j] & 0xF
                    q_hi = (qh[j // 4] >> (2 * (j % 4))) & 3
                    q = (q_lo | (q_hi << 4)) - 32
                    sc = scales[j // 16]
                    dequant[j] = float(d) * sc * q
                print(f"  First 16 dequantized: {dequant}")

            elif t['type'] == 12:  # Q4_K
                # Q4_K block: 256 values, 144 bytes
                block_data = f.read(144)
                d = np.frombuffer(block_data[0:2], dtype=np.float16)[0]
                dmin = np.frombuffer(block_data[2:4], dtype=np.float16)[0]
                packed_scales = block_data[4:16]
                qs = np.frombuffer(block_data[16:144], dtype=np.uint8)
                print(f"  Q4_K block 0: d={float(d):.6f}, dmin={float(dmin):.6f}")
                print(f"  packed_scales: {list(packed_scales)}")

                # Dequantize first 16 values
                # Extract scales and mins from packed format
                sc0 = packed_scales[0] & 0x3F
                mn0 = packed_scales[0 + 6] & 0x3F
                if len(packed_scales) > 6:
                    mn0 = packed_scales[6] & 0x3F

                print(f"  scale[0]={sc0}, min[0]={mn0}")
                dequant = np.zeros(16, dtype=np.float32)
                for j in range(16):
                    q_val = qs[j // 2]
                    q = (q_val & 0xF) if (j % 2 == 0) else (q_val >> 4)
                    dequant[j] = float(d) * sc0 * q - float(dmin) * mn0
                print(f"  First 16 dequantized: {dequant}")

print("\nDone!")
