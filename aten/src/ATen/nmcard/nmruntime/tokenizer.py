# nmruntime/tokenizer.py
# Simple tokenizer wrapper for TinyLlama

import json
import re
from pathlib import Path
from typing import List, Optional


class Tokenizer:
    """
    Simple tokenizer for TinyLlama.
    Uses the vocabulary saved by prepare_tinyllama.py
    """

    def __init__(self, weights_dir: Path):
        weights_dir = Path(weights_dir)

        # Load vocabulary
        vocab_path = weights_dir / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Load special tokens
        special_path = weights_dir / "special_tokens.json"
        with open(special_path, "r", encoding="utf-8") as f:
            special = json.load(f)

        self.bos_token = special.get("bos_token", "<s>")
        self.eos_token = special.get("eos_token", "</s>")
        self.pad_token = special.get("pad_token")
        self.unk_token = special.get("unk_token", "<unk>")

        self.bos_token_id = special.get("bos_token_id", 1)
        self.eos_token_id = special.get("eos_token_id", 2)
        self.pad_token_id = special.get("pad_token_id")
        self.unk_token_id = special.get("unk_token_id", 0)

        self.vocab_size = len(self.vocab)

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Note: This is a simplified tokenizer. For production use,
        you should use the HuggingFace tokenizer.
        """
        # Try to use HuggingFace tokenizer if available
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
            tokens = hf_tokenizer.encode(text, add_special_tokens=add_bos)
            return tokens
        except ImportError:
            pass

        # Fallback: simple character-based encoding (not ideal)
        tokens = []

        if add_bos:
            tokens.append(self.bos_token_id)

        # Very simple tokenization - just look up words/subwords
        # This won't work well for real text but is a fallback
        words = text.replace("\n", " \n ").split(" ")

        for word in words:
            if not word:
                continue

            # Try to find exact match
            if word in self.vocab:
                tokens.append(self.vocab[word])
            elif f"▁{word}" in self.vocab:  # SentencePiece style
                tokens.append(self.vocab[f"▁{word}"])
            else:
                # Character by character
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    elif f"▁{char}" in self.vocab:
                        tokens.append(self.vocab[f"▁{char}"])
                    else:
                        tokens.append(self.unk_token_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        # Try HuggingFace tokenizer first
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
            return hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special)
        except ImportError:
            pass

        # Fallback
        tokens = []
        for tid in token_ids:
            if skip_special:
                if tid in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                    continue

            token = self.id_to_token.get(tid, self.unk_token)
            # Remove SentencePiece prefix
            if token.startswith("▁"):
                token = " " + token[1:]
            tokens.append(token)

        text = "".join(tokens)
        return text.strip()

    def apply_chat_template(self, messages: List[dict]) -> str:
        """
        Apply TinyLlama chat template.

        Format:
        <|system|>
        {system_message}</s>
        <|user|>
        {user_message}</s>
        <|assistant|>
        {assistant_message}</s>
        """
        formatted = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}</s>\n"

        # Add assistant prefix for generation
        if messages[-1]["role"] != "assistant":
            formatted += "<|assistant|>\n"

        return formatted
