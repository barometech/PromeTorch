"""Check tokenizer vocabulary for presence of ▁ marker."""
import json
import os
import struct

# Find qwen3:4b
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

from gguf import GGUFReader
reader = GGUFReader(blob_path)

# Get tokenizer metadata
for field_name in reader.fields:
    f = reader.fields[field_name]
    if 'tokenizer' in field_name and 'token' not in field_name or field_name == 'tokenizer.ggml.model':
        val = f.parts[-1].tolist() if hasattr(f, 'parts') and len(f.parts) > 0 else None
        if val is not None and len(val) == 1:
            val = val[0]
        if isinstance(val, list) and len(val) > 5:
            val = f"[{len(val)} items]"
        print(f"  {field_name} = {val}")

# Check tokens
tokens_field = reader.fields.get('tokenizer.ggml.tokens')
if tokens_field:
    # Read some tokens
    print(f"\nTotal tokens: {len(tokens_field.parts) - 1}")

    # Check for ▁ in vocabulary
    sp_marker = b'\xe2\x96\x81'
    has_sp = False
    hello_ids = []
    sp_hello_ids = []

    # The tokens are stored as array of strings
    # Each part after the array header contains the token
    token_data = tokens_field.parts
    # First part is array metadata, rest are tokens
    # Actually, GGUFReader stores token data differently
    # Let's just check specific tokens

    # Try to read vocab from raw data
    print("\nChecking specific tokens...")

    # Try to check if 'Hello' and '▁Hello' exist
    for check in ['Hello', '▁Hello', 'hello', '▁hello', ' Hello', 'Ġ', '▁', 'Ġhello']:
        # Search in token_data
        found = False
        for i, part in enumerate(token_data):
            try:
                s = bytes(part).decode('utf-8', errors='replace')
                if s == check:
                    print(f"  Token '{check}' found at index {i}")
                    found = True
                    break
            except:
                pass
        if not found:
            print(f"  Token '{check}' NOT found")

# Also check gemma3:4b for comparison
print("\n=== Checking gemma3:4b ===")
manifest_path_g = os.path.join(ollama_home, "manifests", "registry.ollama.ai", "library", "gemma3", "4b")
with open(manifest_path_g, 'r') as f:
    manifest_g = json.load(f)
for layer in manifest_g['layers']:
    if layer.get('mediaType') == 'application/vnd.ollama.image.model':
        blob_digest_g = layer['digest'].replace(':', '-')
        break
blob_path_g = os.path.join(ollama_home, "blobs", blob_digest_g)
reader_g = GGUFReader(blob_path_g)

for field_name in reader_g.fields:
    if field_name == 'tokenizer.ggml.model':
        val = reader_g.fields[field_name].parts[-1].tolist()
        if len(val) == 1: val = val[0]
        print(f"  {field_name} = {val}")

tokens_g = reader_g.fields.get('tokenizer.ggml.tokens')
if tokens_g:
    token_data_g = tokens_g.parts
    for check in ['Hello', '▁Hello', ' Hello', '▁']:
        found = False
        for i, part in enumerate(token_data_g):
            try:
                s = bytes(part).decode('utf-8', errors='replace')
                if s == check:
                    print(f"  Token '{check}' found at index {i}")
                    found = True
                    break
            except:
                pass
        if not found:
            print(f"  Token '{check}' NOT found")
