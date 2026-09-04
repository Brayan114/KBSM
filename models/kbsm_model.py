"""
Kernelized Bound Synaptic Model (KBSM) architecture.
Combines 1D depthwise local convolution with rectified power-kernel associative memory
to conquer the MQAR associative recall barrier with strictly O(1) state memory.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from models.psan_modules import KernelizedBoundSynapticMemory

class KBSMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        head_dim: int = 16,
        conv_kernel: int = 4,
        use_power_kernel: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.kbsm = KernelizedBoundSynapticMemory(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_kernel=conv_kernel,
            use_power_kernel=use_power_kernel
        )

        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # idx: (B, T)
        x = self.drop(self.tok_emb(idx))
        out_seq, new_state = self.kbsm(x, M_prev=state)
        logits = self.head(self.ln_out(out_seq))
        
        metrics = {
            "avg_steps": 1.0,
            "active_flops_ratio": 1.0
        }
        return logits, new_state, metrics

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
