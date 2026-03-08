"""Verify Q6_K dequantization using correct GGML layout."""
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

# Parse GGUF header to find token_embd.weight
with open(blob_path, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

    def read_string(f):
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def read_value(f, vtype):
        if vtype == 0: return struct.unpack('<B', f.read(1))[0]
        elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
        elif vtype == 2: return struct.unpack('<H', f.read(2))[0]
        elif vtype == 3: return struct.unpack('<h', f.read(2))[0]
        elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
        elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
        elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
        elif vtype == 7: return struct.unpack('?', f.read(1))[0]
        elif vtype == 8: return read_string(f)
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [read_value(f, arr_type) for _ in range(arr_len)]
        elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
        elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
        elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
        else: raise ValueError(f"Unknown type {vtype}")

    for i in range(metadata_kv_count):
        key = read_string(f)
        vtype = struct.unpack('<I', f.read(4))[0]
        val = read_value(f, vtype)

    tensor_info = []
    for i in range(tensor_count):
        name = read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensor_info.append({'name': name, 'dims': dims, 'type': dtype, 'offset': offset})

    pos = f.tell()
    alignment = 32
    data_start = ((pos + alignment - 1) // alignment) * alignment

    # Find token_embd.weight
    for t in tensor_info:
        if t['name'] == 'token_embd.weight':
            print(f"token_embd.weight: dims={t['dims']}, type={t['type']}")
            abs_offset = data_start + t['offset']
            f.seek(abs_offset)

            # Read first block (210 bytes for Q6_K)
            block_data = f.read(210)
            ql = np.frombuffer(block_data[0:128], dtype=np.uint8)
            qh = np.frombuffer(block_data[128:192], dtype=np.uint8)
            scales = np.frombuffer(block_data[192:208], dtype=np.int8)
            d = np.frombuffer(block_data[208:210], dtype=np.float16)[0]

            print(f"\nBlock 0: d={float(d)}")
            print(f"scales: {scales}")
            print(f"ql[0:4]: {ql[0:4]}")
            print(f"qh[0:4]: {qh[0:4]}")

            # Correct GGML Q6_K dequantization (from ggml-quants.c)
            # Layout within each 256-value block:
            # ql[128]: stores low 4 bits of quants
            # qh[64]: stores upper 2 bits of quants
            # scales[16]: 8-bit scales for 16-value sub-blocks
            # d: fp16 super-block scale
            #
            # Processing: two groups of 128 values (n=0, n=128)
            # Each group: 32 iterations producing 4 values each (at offsets 0,32,64,96)

            dequant = np.zeros(256, dtype=np.float32)

            ql_ptr = 0
            qh_ptr = 0
            y_ptr = 0

            for n_group in range(0, 256, 128):
                for l in range(32):
                    is_val = n_group // 16 + l // 16

                    q1_lo = int(ql[ql_ptr + l]) & 0x0F
                    q1_hi = (int(qh[qh_ptr + l]) >> 0) & 3
                    q1 = (q1_lo | (q1_hi << 4)) - 32

                    q2_lo = int(ql[ql_ptr + l + 32]) & 0x0F
                    q2_hi = (int(qh[qh_ptr + l]) >> 2) & 3
                    q2 = (q2_lo | (q2_hi << 4)) - 32

                    q3_lo = int(ql[ql_ptr + l]) >> 4
                    q3_hi = (int(qh[qh_ptr + l]) >> 4) & 3
                    q3 = (q3_lo | (q3_hi << 4)) - 32

                    q4_lo = int(ql[ql_ptr + l + 32]) >> 4
                    q4_hi = (int(qh[qh_ptr + l]) >> 6) & 3
                    q4 = (q4_lo | (q4_hi << 4)) - 32

                    dequant[y_ptr + l + 0] = float(d) * int(scales[is_val + 0]) * q1
                    dequant[y_ptr + l + 32] = float(d) * int(scales[is_val + 2]) * q2
                    dequant[y_ptr + l + 64] = float(d) * int(scales[is_val + 4]) * q3
                    dequant[y_ptr + l + 96] = float(d) * int(scales[is_val + 6]) * q4

                ql_ptr += 64
                qh_ptr += 32
                y_ptr += 128

            print(f"\nCorrect dequant first 20: {dequant[:20]}")

            # Print detailed trace for first 5 values
            ql_ptr = 0
            qh_ptr = 0
            for l in range(5):
                is_val = 0 + l // 16
                q1_lo = int(ql[l]) & 0x0F
                q1_hi = (int(qh[l]) >> 0) & 3
                q1 = (q1_lo | (q1_hi << 4)) - 32
                val = float(d) * int(scales[is_val]) * q1
                print(f"  pos {l}: ql[{l}]={ql[l]:02x}, qh[{l}]={qh[l]:02x}, "
                      f"q1_lo={q1_lo}, q1_hi={q1_hi}, q1={q1}, "
                      f"scale={scales[is_val]}, val={val}")

            break

print("\nDone!")
