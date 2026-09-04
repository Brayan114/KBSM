"""
Algorithmic State Tracking & Extrapolation Benchmark.
Evaluates an architecture's capability to track an internal dynamic state (e.g. prefix counter modulo M)
over long sequence horizons and test out-of-distribution length extrapolation.
"""

import torch
from typing import Tuple

class StateTrackingDataset:
    def __init__(
        self,
        num_examples: int = 2000,
        seq_len: int = 64,
        modulus: int = 8,
        vocab_size: int = 32,
        seed: int = 42
    ):
        """
        Args:
            num_examples: Number of sequences to generate.
            seq_len: Length of sequence.
            modulus: Modulo base for the counter (0 to modulus-1).
            vocab_size: Total vocabulary size.
                0: PAD
                1: NOOP / noise
                2: INC (+1)
                3: DEC (-1)
                4: QUERY
                5 .. 5 + modulus - 1: Output state classes
        """
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.modulus = modulus
        self.vocab_size = vocab_size
        self.seed = seed

        self.noop_tok = 1
        self.inc_tok = 2
        self.dec_tok = 3
        self.query_tok = 4
        self.target_offset = 5

        self.inputs, self.targets, self.masks = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(self.seed)
        inputs = torch.full((self.num_examples, self.seq_len), self.noop_tok, dtype=torch.long)
        targets = torch.zeros((self.num_examples, self.seq_len), dtype=torch.long)
        masks = torch.zeros((self.num_examples, self.seq_len), dtype=torch.bool)

        for i in range(self.num_examples):
            current_state = 0
            # Generate random operations with probabilities:
            # 30% INC, 30% DEC, 20% NOOP, 20% QUERY
            probs = torch.rand(self.seq_len)
            for t in range(self.seq_len):
                p = probs[t].item()
                if p < 0.35:
                    inputs[i, t] = self.inc_tok
                    current_state = (current_state + 1) % self.modulus
                elif p < 0.70:
                    inputs[i, t] = self.dec_tok
                    current_state = (current_state - 1) % self.modulus
                elif p < 0.85:
                    inputs[i, t] = self.noop_tok
                    # state unchanged
                else:
                    inputs[i, t] = self.query_tok
                    targets[i, t] = self.target_offset + current_state
                    masks[i, t] = True

        return inputs, targets, masks

    def get_batch(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.num_examples, (batch_size,))
        return (
            self.inputs[indices].to(device),
            self.targets[indices].to(device),
            self.masks[indices].to(device)
        )
