"""
Standard Causal Transformer Baseline with KV-Cache support for exact inference profiling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Register causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv_cache = (k, v)

        total_T = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal masking
        if kv_cache is None:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        else:
            # During incremental decoding (T=1), new query attends to all past_k + current_k
            pass

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y), new_kv_cache

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv

class CausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        max_seq_len: int = 1024,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
        B, T = idx.size()
        
        if kv_caches is None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
        else:
            # During incremental step, position is current cache length
            past_len = kv_caches[0][0].size(2)
            pos = torch.tensor([[past_len]], dtype=torch.long, device=idx.device)
            x = self.tok_emb(idx) + self.pos_emb(pos)

        x = self.drop(x)

        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_layer_cache = block(x, kv_cache=layer_cache)
            new_caches.append(new_layer_cache)

        x = self.ln_f(x)
        logits = self.head(x)
        
        metrics = {
            "active_flops_ratio": 1.0,
            "avg_steps": 1.0
        }
        return logits, new_caches, metrics

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, seq_len: int) -> float:
        """Estimates forward-pass FLOPs for sequence of length seq_len."""
        # Embeddings: negligible
        # Per layer:
        # 1. Self-attention: QKV proj (2 * 3 * d^2 * T), Attn matrix (2 * T^2 * d), Out proj (2 * d^2 * T)
        # 2. MLP: 2 * (2 * d * d_ff * T)
        layer_flops = (
            (6 * self.d_model**2 * seq_len) +
            (2 * seq_len**2 * self.d_model) +
            (2 * self.d_model**2 * seq_len) +
            (4 * self.d_model * self.d_ff * seq_len)
        )
        total_flops = self.n_layers * layer_flops + (2 * self.d_model * self.vocab_size * seq_len)
        return float(total_flops)
