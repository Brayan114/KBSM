"""
Wikitext-2 Dataset & Byte-Level Tokenizer.
Provides fast streaming batches for autoregressive next-token language modeling.
"""

import os
import torch
from torch.utils.data import Dataset
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "wikitext-2")

class ByteTokenizer:
    """Byte-level tokenizer mapping UTF-8 bytes to tokens 0..255."""
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode('utf-8')), dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode('utf-8', errors='replace')

class WikitextDataset:
    def __init__(self, split: str = "train", seq_len: int = 256, max_tokens: int = 500_000):
        self.seq_len = seq_len
        self.tokenizer = ByteTokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        fname = "train.txt" if split == "train" else "valid.txt"
        file_path = os.path.join(DATA_DIR, fname)
        assert os.path.exists(file_path), f"File {file_path} not found. Run download_wikitext.py first."

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Encode to bytes
        raw_tokens = self.tokenizer.encode(text)
        if max_tokens is not None and len(raw_tokens) > max_tokens:
            self.data = raw_tokens[:max_tokens]
        else:
            self.data = raw_tokens

        print(f"Loaded Wikitext-2 [{split}]: {len(self.data):,} tokens (seq_len={seq_len})")

    def get_batch(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        max_idx = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, max_idx, (batch_size,))
        x = torch.stack([self.data[s : s + self.seq_len] for s in starts]).to(device)
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts]).to(device)
        return x, y
