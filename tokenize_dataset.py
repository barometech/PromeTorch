"""
Tokenize russian_mega.txt with RUKANIZER V3 (100K vocab).
Output: russian_mega.tokens (uint32 binary, ~500MB if 4x compression).
"""
import sentencepiece as spm
import struct
import os
import sys

SRC = r"C:\Users\paper\Desktop\promethorch\data_local\russian_mega.txt"
DST = r"C:\Users\paper\Desktop\promethorch\data_local\russian_mega.tokens"
TOK = r"C:\Users\paper\Desktop\RUKALLAMA V2\TOKENIZER_TESTS\trained_models\rukanizer_100k_v3.model"

if not os.path.exists(SRC):
    SRC = r"C:\Users\paper\Desktop\promethorch\PIR\pretrain_cache\..\..\data\russian_mega.txt"  # try

# Use the file from server-side via local mirror or just generate locally
# Use any russian text file we have
candidates = [
    r"C:\Users\paper\Desktop\promethorch\data_local\russian_mega.txt",
    r"C:\Users\paper\Desktop\promethorch\data\russian_mega.txt",
    r"C:\Users\paper\Desktop\RUKALLAMA V2\data\corpus.txt",
]
for c in candidates:
    if os.path.exists(c):
        SRC = c
        break

print(f"SRC: {SRC}")
print(f"size: {os.path.getsize(SRC)/1e6:.1f} MB")

print(f"Loading tokenizer: {TOK}")
sp = spm.SentencePieceProcessor()
sp.Load(TOK)
print(f"vocab: {sp.GetPieceSize()}")

# Read in chunks (text file is 2GB)
CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks
total_tokens = 0
total_bytes = 0
chunks_done = 0

with open(SRC, 'rb') as fin, open(DST, 'wb') as fout:
    leftover = b''
    while True:
        chunk = fin.read(CHUNK_SIZE)
        if not chunk and not leftover:
            break
        chunk = leftover + chunk
        if not chunk:
            break

        # Decode safely - find last newline
        try:
            text = chunk.decode('utf-8')
            leftover = b''
        except UnicodeDecodeError as e:
            # Cut off incomplete UTF-8 sequence at end
            cut = e.start
            text = chunk[:cut].decode('utf-8', errors='replace')
            leftover = chunk[cut:]

        # Tokenize
        ids = sp.EncodeAsIds(text)
        # Write as uint32 little-endian
        fout.write(struct.pack(f'<{len(ids)}I', *ids))

        total_tokens += len(ids)
        total_bytes += len(chunk) - len(leftover)
        chunks_done += 1
        if chunks_done % 50 == 0:
            mb = total_bytes / 1e6
            mtok = total_tokens / 1e6
            ratio = total_bytes / max(1, total_tokens)
            print(f"  {mb:.1f} MB read, {mtok:.1f}M tokens, {ratio:.2f} bytes/token")
            sys.stdout.flush()

print(f"\nDONE: {total_tokens/1e6:.1f}M tokens, {os.path.getsize(DST)/1e6:.1f} MB output")
print(f"Compression: {total_bytes/total_tokens:.2f} bytes/token")
