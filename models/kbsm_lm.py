"""
Multi-Layer Stacked KBSM Language Model for Natural Language Scaling.
Each block contains:
- 1D Causal Convolutional Binding + Multi-Head Synaptic Memory
- Pre-LayerNorm & Residual connection
- SwiGLU / MLP FeedForward Network
- Pre-LayerNorm & Residual connection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
from models.psan_modules import KernelizedBoundSynapticMemory

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))

class KBSMBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        d_ff: int,
        conv_kernel: int = 4,
        use_power_kernel: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.kbsm = KernelizedBoundSynapticMemory(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_kernel=conv_kernel,
            use_power_kernel=use_power_kernel
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d_model)
        kbsm_out, new_state = self.kbsm(self.ln1(x), M_prev=state)
        x = x + kbsm_out
        x = x + self.mlp(self.ln2(x))
        return x, new_state

class KBSMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 2,
        num_heads: int = 4,
        head_dim: int = 32,
        d_ff: Optional[int] = None,
        conv_kernel: int = 4,
        use_power_kernel: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            KBSMBlock(
                d_model=d_model,
                num_heads=num_heads,
                head_dim=head_dim,
                d_ff=self.d_ff,
                conv_kernel=conv_kernel,
                use_power_kernel=use_power_kernel,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        # idx: (B, T)
        B, T = idx.size()
        x = self.drop(self.tok_emb(idx))

        new_states = []
        for i, block in enumerate(self.blocks):
            layer_state = states[i] if states is not None else None
            x, new_s = block(x, state=layer_state)
            new_states.append(new_s)

        x = self.ln_f(x)
        logits = self.head(x)

        metrics = {
            "active_flops_ratio": 1.0,
            "avg_steps": 1.0
        }
        return logits, new_states, metrics

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops_per_token(self) -> float:
        """Forward + backward FLOPs per token ~= 6 * N_params."""
        return 6.0 * self.count_parameters()
