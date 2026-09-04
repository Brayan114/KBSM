"""
Predictive Sparse Attractor Network (PSAN) with modular ablation flags.
Supports:
- M1: Dense Recurrent Baseline (all mechanisms False)
- M2: + Sparsity
- M3: + Predictive Delta Skip
- M4: + Dynamic Halting (Attractor Core)
- M5: + Fast Synaptic Trace Memory
- M6: Full PSAN (all mechanisms True)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from models.psan_modules import (
    FastSynapticTrace,
    MultiHeadSynapticMemory,
    ThalamicGatedSynapticMemory,
    TopkSparsity,
    PredictiveDeltaSkip,
    DynamicAttractorCore
)

class PSAN(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        mem_dim: int = 32,
        max_steps: int = 4,
        topk_ratio: float = 0.25,
        skip_threshold: float = 0.15,
        use_sparsity: bool = True,
        use_predictive: bool = True,
        use_dynamic_halting: bool = True,
        use_synaptic_memory: bool = True,
        synaptic_memory_type: str = "single_head",
        num_heads: int = 4,
        head_dim: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mem_dim = mem_dim
        self.max_steps = max_steps
        self.topk_ratio = topk_ratio
        self.skip_threshold = skip_threshold
        self.synaptic_memory_type = synaptic_memory_type
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Ablation Flags
        self.use_sparsity = use_sparsity
        self.use_predictive = use_predictive
        self.use_dynamic_halting = use_dynamic_halting
        self.use_synaptic_memory = use_synaptic_memory

        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        # Mechanism 1: Predictive Delta Skip
        if self.use_predictive:
            self.predictor = PredictiveDeltaSkip(d_model, skip_threshold=skip_threshold)
        else:
            self.predictor = None

        # Mechanism 2 & 3: Dynamic Attractor Core + Lateral Inhibition
        if self.use_dynamic_halting:
            self.attractor = DynamicAttractorCore(
                d_model,
                max_steps=max_steps,
                topk_ratio=topk_ratio,
                use_sparsity=use_sparsity
            )
        else:
            # Fixed single-step recurrent transition
            self.w_h = nn.Linear(d_model, d_model)
            self.w_x = nn.Linear(d_model, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.sparsity = TopkSparsity(topk_ratio) if use_sparsity else None

        # Mechanism 4: Gated Delta-Rule Synaptic Memory
        if self.use_synaptic_memory:
            if self.synaptic_memory_type == "thalamic":
                self.synaptic_mem = ThalamicGatedSynapticMemory(
                    d_model, num_heads=num_heads, head_dim=head_dim
                )
            elif self.synaptic_memory_type == "multi_head":
                self.synaptic_mem = MultiHeadSynapticMemory(
                    d_model, num_heads=num_heads, head_dim=head_dim
                )
            else:
                self.synaptic_mem = FastSynapticTrace(d_model, mem_dim=mem_dim)
        else:
            self.synaptic_mem = None

        # Readout head
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Dict[str, Any]]:
        """
        Processes a sequence of tokens autoregressively through the recurrent state.
        Args:
            idx: (B, T) token indices
            state: Optional tuple (h_prev, M_prev)
        """
        B, T = idx.size()
        device = idx.device

        if state is None:
            h = torch.zeros(B, self.d_model, device=device)
            M = None
        else:
            h, M = state

        outputs = []
        step_counts_history = []
        active_flops_list = []
        ponder_losses = []

        for t in range(T):
            x_t = self.tok_emb(idx[:, t]) # (B, d_model)
            x_t = self.drop(x_t)

            # 1. Predictive Delta Check
            if self.use_predictive and self.predictor is not None:
                delta_x, skip_mask, _ = self.predictor(x_t, h)
            else:
                delta_x = x_t
                skip_mask = None

            # 2. Recurrent Transition (Dynamic Attractor vs Single Step)
            if self.use_dynamic_halting:
                h_next, ponder_cost, stats = self.attractor(delta_x, h, skip_mask=skip_mask)
                step_counts_history.append(stats["step_counts"])
                active_flops_list.append(stats["active_flops_ratio"])
                ponder_losses.append(ponder_cost)
            else:
                pre_act = self.w_h(h) + self.w_x(delta_x)
                if self.use_sparsity and self.sparsity is not None:
                    pre_act, act_ratio = self.sparsity(pre_act)
                    active_flops_list.append(act_ratio)
                else:
                    active_flops_list.append(1.0)

                h_next = h + torch.tanh(self.ln(pre_act))
                step_counts_history.append(torch.ones(B, device=device))
                ponder_losses.append(torch.tensor(1.0, device=device))

            # 3. Fast Synaptic Associative Memory Update & Readout
            if self.use_synaptic_memory and self.synaptic_mem is not None:
                mem_readout, M = self.synaptic_mem(h_next, M_prev=M)
                combined = h_next + mem_readout
            else:
                combined = h_next

            h = h_next
            outputs.append(combined)

        out_stack = torch.stack(outputs, dim=1) # (B, T, d_model)
        logits = self.head(self.ln_out(out_stack)) # (B, T, vocab_size)

        all_steps = torch.cat(step_counts_history)
        avg_steps = all_steps.mean().item()
        median_steps = all_steps.median().item()
        min_steps = all_steps.min().item()
        max_steps = all_steps.max().item()
        avg_active_flops = sum(active_flops_list) / max(1, len(active_flops_list))
        total_ponder_cost = sum(ponder_losses) / max(1, len(ponder_losses))

        metrics = {
            "avg_steps": avg_steps,
            "median_steps": median_steps,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "active_flops_ratio": avg_active_flops,
            "ponder_cost": total_ponder_cost
        }

        return logits, (h, M), metrics

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, seq_len: int) -> float:
        """Estimates total theoretical FLOPs per sequence assuming mean active steps."""
        # Base token embedding + head: 2 * d * vocab * seq_len
        base_flops = 2 * self.d_model * self.vocab_size * seq_len
        
        # Recurrent core per step:
        # W_h: 2 * d^2, W_x: 2 * d^2
        core_step_flops = 4 * self.d_model**2
        if self.use_sparsity:
            core_step_flops *= self.topk_ratio

        # Memory per step (if active):
        # 3 projections (3 * 2 * d * mem_dim) + 2 bmm (2 * 2 * mem_dim^2) + out_proj (2 * mem_dim * d)
        if self.use_synaptic_memory:
            mem_flops = (8 * self.d_model * self.mem_dim) + (4 * self.mem_dim**2)
        else:
            mem_flops = 0

        # Predictor: 2 * d^2
        pred_flops = 2 * self.d_model**2 if self.use_predictive else 0

        # Assume average of 2 steps if dynamic halting is on, else 1 step
        steps = 2.0 if self.use_dynamic_halting else 1.0
        step_flops = pred_flops + (steps * core_step_flops) + mem_flops
        
        total = base_flops + (seq_len * step_flops)
        return float(total)
