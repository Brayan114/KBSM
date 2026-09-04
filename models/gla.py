"""
Pure PyTorch implementation of Gated Linear Attention (GLA)
Reference: Yang et al. (2023) "Gated Linear Attention Transformers with Hardware-Efficient Training"

Implements:
1. Data-dependent retention gating: alpha_t = sigmoid(Linear(x_t))
2. Outer-product key-value associative state: S_t = alpha_t * S_{t-1} + k_t^T v_t
3. Linear attention query readout: O_t = q_t @ S_t
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class GLABlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        head_dim: int = 16
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, num_heads, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.group_norm = nn.LayerNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)
        gates = torch.sigmoid(self.gate_proj(x)).view(B, T, H, 1, 1)

        # Normalize q and k (standard in linear attention)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        if state is None:
            S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        else:
            S = state

        outputs = []
        for t in range(T):
            q_t = q[:, t]  # (B, H, D)
            k_t = k[:, t]  # (B, H, D)
            v_t = v[:, t]  # (B, H, D)
            g_t = gates[:, t]  # (B, H, 1, 1)

            # Readout: q_t @ S_{t-1}
            # q_t is (B, H, 1, D), S is (B, H, D, D) -> (B, H, 1, D)
            o_t = torch.matmul(q_t.unsqueeze(-2), S).squeeze(-2)  # (B, H, D)

            # State update: S_t = g_t * S_{t-1} + k_t^T @ v_t
            # k_t^T @ v_t: (B, H, D, 1) @ (B, H, 1, D) -> (B, H, D, D)
            update = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
            S = g_t * S + update

            o_norm = self.group_norm(o_t)
            outputs.append(o_norm)

        out = torch.stack(outputs, dim=1).view(B, T, self.inner_dim)
        return self.out_proj(out), S

class GLAModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        head_dim: int = 16,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'gla': GLABlock(d_model=d_model, num_heads=num_heads, head_dim=head_dim),
                'norm2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)
                )
            })
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        state: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any, Dict[str, Any]]:
        x = self.drop(self.tok_emb(idx))
        
        new_states = []
        for i, layer in enumerate(self.blocks):
            prev_s = state[i] if state is not None else None
            g_out, s = layer['gla'](layer['norm1'](x), state=prev_s)
            x = x + g_out
            x = x + layer['mlp'](layer['norm2'](x))
            new_states.append(s)

        logits = self.head(self.ln_f(x))
        return logits, new_states, {"avg_steps": 1.0}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
