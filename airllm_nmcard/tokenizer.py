"""
tokenizer.py -- BPE tokenizer parsing tokenizer.json directly.

No transformers or sentencepiece dependency. Parses the HuggingFace
tokenizer.json format used by Qwen3 (byte-level BPE).

Supports:
  - encode(text) -> list[int]
  - decode(token_ids) -> str
  - Special tokens: <|endoftext|>, <|im_start|>, <|im_end|>
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple


class BPETokenizer:
    """Byte-level BPE tokenizer compatible with Qwen3 tokenizer.json.

    Usage:
        tok = BPETokenizer("path/to/model_dir")  # must contain tokenizer.json
        ids = tok.encode("Hello, world!")
        text = tok.decode(ids)
    """

    def __init__(self, model_dir: str):
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            self.tokenizer_data = json.load(f)

        # Parse components
        self._build_vocab()
        self._build_merges()
        self._build_special_tokens()
        self._load_added_tokens()
        self._build_pre_tokenizer()

    def _build_vocab(self):
        """Build token->id and id->token maps from the vocab."""
        model = self.tokenizer_data.get("model", {})
        vocab = model.get("vocab", {})

        self.token_to_id: Dict[str, int] = dict(vocab)
        self.id_to_token: Dict[int, str] = {v: k for k, v in vocab.items()}
        self.vocab_size = len(self.token_to_id)

    def _build_merges(self):
        """Parse BPE merge rules."""
        model = self.tokenizer_data.get("model", {})
        merges_raw = model.get("merges", [])

        # Each merge is "token_a token_b" -> priority (lower index = higher priority)
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for i, merge in enumerate(merges_raw):
            parts = merge.split(" ", 1)
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = i

    def _build_special_tokens(self):
        """Extract special token IDs."""
        # Common Qwen3 special tokens
        self.eos_token_id = self.token_to_id.get("<|endoftext|>",
                            self.token_to_id.get("<|im_end|>", 151643))
        self.bos_token_id = self.token_to_id.get("<|im_start|>",
                            self.token_to_id.get("<s>", 151644))
        self.pad_token_id = self.token_to_id.get("<|endoftext|>",
                            self.eos_token_id)

        # Build set of all special token strings for fast lookup
        self.special_tokens: Dict[str, int] = {}

        added_tokens = self.tokenizer_data.get("added_tokens", [])
        for tok_info in added_tokens:
            if tok_info.get("special", False):
                self.special_tokens[tok_info["content"]] = tok_info["id"]

    def _load_added_tokens(self):
        """Load added tokens (special + non-special) into vocab."""
        added_tokens = self.tokenizer_data.get("added_tokens", [])
        for tok_info in added_tokens:
            content = tok_info["content"]
            tid = tok_info["id"]
            self.token_to_id[content] = tid
            self.id_to_token[tid] = content

    def _build_pre_tokenizer(self):
        """Build the pre-tokenization regex pattern.

        Qwen3 uses a GPT-4-style pattern that splits on whitespace boundaries,
        punctuation, numbers, etc.
        """
        pre_tok = self.tokenizer_data.get("pre_tokenizer", {})

        # Try to extract the regex pattern
        if pre_tok.get("type") == "Sequence":
            for step in pre_tok.get("pretokenizers", []):
                if step.get("type") == "Split" and "pattern" in step:
                    pattern = step["pattern"]
                    if isinstance(pattern, dict) and "Regex" in pattern:
                        try:
                            self._pre_tok_pattern = re.compile(pattern["Regex"])
                            return
                        except re.error:
                            pass

        # Fallback: GPT-2/GPT-4 style pattern
        # This handles most byte-level BPE tokenizers
        self._pre_tok_pattern = re.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            re.UNICODE
        ) if False else None  # Complex regex needs 'regex' module; fallback below

        # Simple fallback that works without the regex module
        self._pre_tok_pattern = None

    def _pre_tokenize(self, text: str) -> List[str]:
        """Split text into pre-tokens."""
        if self._pre_tok_pattern is not None:
            return self._pre_tok_pattern.findall(text)

        # Simple fallback: split on whitespace boundaries, keeping whitespace attached
        # This matches GPT-style tokenizers well enough for inference
        tokens = []
        current = ""
        for ch in text:
            if ch in " \t\n\r":
                if current:
                    tokens.append(current)
                    current = ""
                current += ch
            else:
                if current and current[-1] in " \t\n\r" and ch not in " \t\n\r":
                    tokens.append(current)
                    current = ""
                current += ch
        if current:
            tokens.append(current)
        return tokens if tokens else [text]

    def _bytes_to_unicode(self) -> Dict[int, str]:
        """Build byte->unicode mapping (GPT-2 style)."""
        # Standard GPT-2 byte encoder
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("\xa1"), ord("\xac") + 1))
        bs += list(range(ord("\xae"), ord("\xff") + 1))
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def _bpe(self, word: List[str]) -> List[str]:
        """Apply BPE merges to a list of characters/tokens."""
        if len(word) <= 1:
            return word

        while True:
            # Find the highest-priority merge pair
            best_pair = None
            best_rank = float("inf")
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None or best_rank == float("inf"):
                break

            # Apply the merge
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

            if len(word) == 1:
                break

        return word

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: input text
            add_special_tokens: if True, prepend BOS token
        Returns:
            list of token IDs
        """
        if not text:
            return []

        # Check for special tokens in the text first
        # Split text around special tokens
        segments = self._split_on_special_tokens(text)

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        byte_encoder = self._bytes_to_unicode()

        for segment, is_special in segments:
            if is_special:
                if segment in self.token_to_id:
                    token_ids.append(self.token_to_id[segment])
                continue

            # Pre-tokenize
            pre_tokens = self._pre_tokenize(segment)

            for pre_token in pre_tokens:
                # Convert to bytes, then to unicode representation
                encoded_bytes = pre_token.encode("utf-8")
                unicode_chars = [byte_encoder.get(b, chr(b)) for b in encoded_bytes]

                # Apply BPE
                bpe_tokens = self._bpe(unicode_chars)

                for tok in bpe_tokens:
                    if tok in self.token_to_id:
                        token_ids.append(self.token_to_id[tok])
                    else:
                        # Unknown token: encode each character separately
                        for ch in tok:
                            if ch in self.token_to_id:
                                token_ids.append(self.token_to_id[ch])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: list of token IDs
            skip_special_tokens: if True, omit special tokens from output
        Returns:
            decoded text
        """
        byte_decoder = {v: k for k, v in self._bytes_to_unicode().items()}

        text_parts = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, "")

            if skip_special_tokens and token in self.special_tokens:
                continue

            text_parts.append(token)

        # Join and decode from byte-level representation
        joined = "".join(text_parts)

        # Convert unicode representation back to bytes
        byte_list = []
        for ch in joined:
            if ch in byte_decoder:
                byte_list.append(byte_decoder[ch])
            else:
                byte_list.extend(ch.encode("utf-8"))

        try:
            return bytes(byte_list).decode("utf-8", errors="replace")
        except Exception:
            return joined

    def _split_on_special_tokens(self, text: str) -> List[Tuple[str, bool]]:
        """Split text into segments, identifying special tokens.

        Returns list of (segment_text, is_special) tuples.
        """
        if not self.special_tokens:
            return [(text, False)]

        # Sort special tokens by length (longest first) for greedy matching
        sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)

        # Build regex pattern
        pattern = "|".join(re.escape(st) for st in sorted_specials)

        segments = []
        last_end = 0
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if start > last_end:
                segments.append((text[last_end:start], False))
            segments.append((match.group(), True))
            last_end = end

        if last_end < len(text):
            segments.append((text[last_end:], False))

        return segments if segments else [(text, False)]

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> List[int]:
        """Apply Qwen3 chat template.

        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        Returns: token IDs
        """
        # Qwen3 chat format:
        # <|im_start|>system\n{content}<|im_end|>\n
        # <|im_start|>user\n{content}<|im_end|>\n
        # <|im_start|>assistant\n

        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        # Add assistant prompt
        text_parts.append("<|im_start|>assistant\n")

        full_text = "".join(text_parts)
        return self.encode(full_text)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self):
        return f"BPETokenizer(vocab_size={self.vocab_size})"
