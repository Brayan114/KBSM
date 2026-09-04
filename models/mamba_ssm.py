"""
Pure PyTorch implementation of Mamba (Selective State Space Model)
Reference: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

Implements the selective SSM mechanism with:
1. 1D Causal Convolution for local context
2. Input-dependent discretization: Delta, B, C projections
3. Recurrent state scan with input-dependent state decay
4. Multiplicative gating branch (SwiGLU/SiLU-style)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            groups=channels, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        out = self.conv(x)
        return out.transpose(1, 2)

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        # In-projection to 2 * d_inner (one for conv+SSM branch, one for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Causal Convolution
        self.conv1d = CausalConv1d(self.d_inner, kernel_size=d_conv)

        # Selective SSM projections: x -> (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # State transition matrix A (initialized to HIPPO-like log spaced)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Out projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d_model)
        B, T, _ = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # 1. Local Causal Conv + SiLU
        x_conv = F.silu(self.conv1d(x_in))

        # 2. Selective SSM Parameters
        ssm_proj = self.x_proj(x_conv)  # (B, T, dt_rank + 2*d_state)
        dt = ssm_proj[:, :, :self.dt_rank]
        B_ssm = ssm_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_ssm = ssm_proj[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # 3. Recurrent Scan
        if state is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h = state

        ys = []
        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            dA_t = torch.exp(dt_t * A.unsqueeze(0))  # (B, d_inner, d_state)
            B_t = B_ssm[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            dB_t = dt_t * B_t  # (B, d_inner, d_state)
            x_t = x_conv[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)

            h = dA_t * h + dB_t * x_t  # (B, d_inner, d_state)
            C_t = C_ssm[:, t, :].unsqueeze(-1)  # (B, d_state, 1)
            y_t = torch.matmul(h, C_t).squeeze(-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # 4. Multiplicative gate with z branch
        out = y * F.silu(z)
        return self.out_proj(out), h

class MambaModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
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
            m_out, s = layer['mamba'](layer['norm'](x), state=prev_s)
            x = x + m_out
            new_states.append(s)

        logits = self.head(self.ln_f(x))
        return logits, new_states, {"avg_steps": 1.0}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
