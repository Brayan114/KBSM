"""
Multi-Query Associative Recall (MQAR) Benchmark.
Tests an architecture's ability to memorize non-contiguous key-value pairs
and recall the exact value when queried later in the sequence.
"""

import torch
from typing import Tuple, Dict, Any

class MQARDataset:
    def __init__(
        self,
        num_examples: int = 2000,
        seq_len: int = 128,
        num_pairs: int = 8,
        num_queries: int = 4,
        vocab_size: int = 64,
        seed: int = 42
    ):
        """
        Args:
            num_examples: Number of sequences to generate.
            seq_len: Total length of sequence.
            num_pairs: Number of distinct (key, value) pairs to insert.
            num_queries: Number of recall queries at the end of the sequence.
            vocab_size: Total vocabulary size.
                Token 0: PAD / NOISE
                Tokens 1 .. vocab_size // 2: Keys
                Tokens vocab_size // 2 + 1 .. vocab_size - 1: Values
        """
        assert seq_len >= (num_pairs * 2 + num_queries * 2), "seq_len must fit pairs and queries"
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.seed = seed

        self.key_min = 2
        self.key_max = vocab_size // 2
        self.val_min = vocab_size // 2 + 1
        self.val_max = vocab_size - 1
        self.noise_token = 1

        self.inputs, self.targets, self.masks = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(self.seed)
        
        inputs = torch.full((self.num_examples, self.seq_len), self.noise_token, dtype=torch.long)
        targets = torch.zeros((self.num_examples, self.seq_len), dtype=torch.long)
        masks = torch.zeros((self.num_examples, self.seq_len), dtype=torch.bool)

        key_pool_size = self.key_max - self.key_min + 1
        val_pool_size = self.val_max - self.val_min + 1
        assert key_pool_size >= self.num_pairs, "Key vocabulary too small for num_pairs"

        # Split sequence: first 75% for key-value storage, last 25% for queries
        storage_len = int(self.seq_len * 0.70)
        query_start = int(self.seq_len * 0.75)

        for i in range(self.num_examples):
            # 1. Sample keys and values
            perm_keys = torch.randperm(key_pool_size) + self.key_min
            keys = perm_keys[:self.num_pairs]
            vals = torch.randint(self.val_min, self.val_max + 1, (self.num_pairs,))
            
            # Map key to value for quick lookup
            kv_map = {k.item(): v.item() for k, v in zip(keys, vals)}

            # 2. Place pairs in storage region
            # We need 2 * num_pairs non-overlapping positions
            avail_pos = torch.randperm(storage_len - 1)[:self.num_pairs * 2].sort().values
            for p_idx in range(self.num_pairs):
                k_pos = avail_pos[p_idx * 2]
                v_pos = k_pos + 1
                inputs[i, k_pos] = keys[p_idx]
                inputs[i, v_pos] = vals[p_idx]

            # 3. Place queries in query region
            query_indices = torch.randperm(self.num_pairs)[:self.num_queries]
            query_keys = keys[query_indices]
            
            q_positions = torch.linspace(query_start, self.seq_len - 1, self.num_queries).long()
            for q_idx, q_pos in enumerate(q_positions):
                q_key = query_keys[q_idx].item()
                inputs[i, q_pos] = q_key
                # Target is evaluated at q_pos
                targets[i, q_pos] = kv_map[q_key]
                masks[i, q_pos] = True

        return inputs, targets, masks

    def get_batch(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.num_examples, (batch_size,))
        return (
            self.inputs[indices].to(device),
            self.targets[indices].to(device),
            self.masks[indices].to(device)
        )
