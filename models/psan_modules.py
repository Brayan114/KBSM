"""
Core building blocks for Predictive Sparse Attractor Network (PSAN):
1. Gated Delta-Rule Synaptic Memory (Fixed O(1) state, interference-resistant)
2. Lateral Inhibition (Top-k activation sparsity)
3. Predictive Delta Skip (Work-skipping on low surprise)
4. Dynamic Attractor Core (Adaptive Computation Time with ponder penalty)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class FastSynapticTrace(nn.Module):
    """
    Fixed-size associative working memory using a gated Delta-rule:
    M_t = gamma_t * M_{t-1} + alpha_t * ( (v_t - M_{t-1} k_t) (x) k_t ) / (||k_t||^2 + eps)
    
    The delta-rule cancels out interference by computing the retrieval error
    before writing, preventing catastrophic crosstalk in fixed-rank states.
    """
    def __init__(self, d_model: int, mem_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.mem_dim = mem_dim

        self.q_proj = nn.Linear(d_model, mem_dim, bias=False)
        self.k_proj = nn.Linear(d_model, mem_dim, bias=False)
        self.v_proj = nn.Linear(d_model, mem_dim, bias=False)
        self.out_proj = nn.Linear(mem_dim, d_model, bias=False)

        # Gating projections
        self.gate_proj = nn.Linear(d_model, 2)  # [decay_gate, write_gate]

    def forward(
        self,
        x: torch.Tensor,
        M_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, d_model) - current token representation
            M_prev: (B, mem_dim, mem_dim) - previous associative memory matrix
        Returns:
            out: (B, d_model)
            M_t: (B, mem_dim, mem_dim)
        """
        B = x.size(0)
        if M_prev is None:
            M_prev = torch.zeros(B, self.mem_dim, self.mem_dim, device=x.device)

        q = F.normalize(self.q_proj(x), dim=-1)  # (B, d_m)
        k = F.normalize(self.k_proj(x), dim=-1)  # (B, d_m)
        v = self.v_proj(x)                      # (B, d_m)

        gates = torch.sigmoid(self.gate_proj(x))
        gamma = gates[:, 0:1].unsqueeze(-1)      # retention gate (B, 1, 1)
        alpha = gates[:, 1:2].unsqueeze(-1)      # write gate (B, 1, 1)

        # 1. Read before write
        readout = torch.bmm(M_prev, q.unsqueeze(-1)).squeeze(-1)  # (B, d_m)

        # 2. Compute prediction error on this key (Delta rule)
        pred_v = torch.bmm(M_prev, k.unsqueeze(-1)).squeeze(-1)   # (B, d_m)
        error = v - pred_v                                        # (B, d_m)

        # 3. Normalized outer-product update
        delta_write = torch.bmm(error.unsqueeze(-1), k.unsqueeze(1)) # (B, d_m, d_m)
        M_t = gamma * M_prev + alpha * delta_write

        out = self.out_proj(readout)
        return out, M_t


class MultiHeadSynapticMemory(nn.Module):
    """
    Multi-Head Gated Synaptic Associative Memory.
    Partitions the working memory state into H parallel heads with multi-scale decay rates
    to eliminate associative crosstalk and provide distinct storage channels.
    
    State footprint: (B, H, head_dim, head_dim) -> Strictly O(1) across sequence length!
    """
    def __init__(self, d_model: int, num_heads: int = 4, head_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        # Multi-scale learned decay timescales (initialized geometrically)
        # Heads span from long-range retention (gamma ~ 0.99) to fast local updates (gamma ~ 0.8)
        init_decays = torch.linspace(-4.0, -1.0, num_heads)
        self.decay_log = nn.Parameter(init_decays)

        # Dynamic write and modulation gates
        self.gate_proj = nn.Linear(d_model, num_heads * 2)
        self.head_norm = nn.LayerNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        M_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, d_model)
            M_prev: (B, H, head_dim, head_dim)
        Returns:
            out: (B, d_model)
            M_t: (B, H, head_dim, head_dim)
        """
        B = x.size(0)
        H = self.num_heads
        D = self.head_dim

        if M_prev is None:
            M_prev = torch.zeros(B, H, D, D, device=x.device)

        # Projections & reshape to heads: (B, H, D)
        q = F.normalize(self.q_proj(x).view(B, H, D), dim=-1)
        k = F.normalize(self.k_proj(x).view(B, H, D), dim=-1)
        v = self.v_proj(x).view(B, H, D)

        # Gates
        gates = torch.sigmoid(self.gate_proj(x)).view(B, H, 2)
        base_gamma = torch.exp(-torch.exp(self.decay_log)).view(1, H, 1, 1) # multi-scale decay
        gamma = base_gamma * gates[:, :, 0].view(B, H, 1, 1)
        alpha = gates[:, :, 1].view(B, H, 1, 1)

        # 1. Readout per head before writing: M_prev @ q
        # M_prev: (B, H, D, D), q: (B, H, D, 1) -> (B, H, D)
        readout = torch.matmul(M_prev, q.unsqueeze(-1)).squeeze(-1)

        # 2. Delta error computation: v - M_prev @ k
        pred_v = torch.matmul(M_prev, k.unsqueeze(-1)).squeeze(-1)
        error = v - pred_v

        # 3. Parallel outer-product update per head: error (x) k
        delta_write = torch.matmul(error.unsqueeze(-1), k.unsqueeze(-2)) # (B, H, D, D)
        M_t = gamma * M_prev + alpha * delta_write

        # 4. Normalize each head readout and project out
        readout_norm = self.head_norm(readout)
        out = self.out_proj(readout_norm.view(B, self.inner_dim))

        return out, M_t


class ThalamicGatedSynapticMemory(nn.Module):
    """
    Thalamic-Gated Synaptic Working Memory.
    Mimics the biological thalamus by actively filtering sensory noise:
    - On uninformative/distractor tokens: Salience s_t -> 0.
      Memory is frozen: gamma_t = 1.0 (no forgetting), alpha_t = 0.0 (no noise written).
      M_t = M_{t-1} precisely.
    - On salient tokens (key/value events): Salience s_t -> 1.
      Delta-rule associative write is executed.
    """
    def __init__(self, d_model: int, num_heads: int = 4, head_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        # Thalamic salience / sensory filter gate: initialized with negative bias so default is ignore
        self.salience_gate = nn.Linear(d_model, num_heads)
        nn.init.constant_(self.salience_gate.bias, -1.0)

        # Content-dependent write and overwrite gates
        self.write_gate = nn.Linear(d_model, num_heads)
        self.decay_gate = nn.Linear(d_model, num_heads)

        self.head_norm = nn.LayerNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        M_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        H = self.num_heads
        D = self.head_dim

        if M_prev is None:
            M_prev = torch.zeros(B, H, D, D, device=x.device)

        q = F.normalize(self.q_proj(x).view(B, H, D), dim=-1)
        k = F.normalize(self.k_proj(x).view(B, H, D), dim=-1)
        v = self.v_proj(x).view(B, H, D)

        # 1. Readout per head before writing: M_prev @ q
        readout = torch.matmul(M_prev, q.unsqueeze(-1)).squeeze(-1) # (B, H, D)

        # 2. Thalamic Filtering
        s = torch.sigmoid(self.salience_gate(x)).view(B, H, 1, 1) # salience in [0, 1]
        raw_alpha = torch.sigmoid(self.write_gate(x)).view(B, H, 1, 1)
        raw_beta = torch.sigmoid(self.decay_gate(x)).view(B, H, 1, 1)

        # Crucial Thalamic Gating Property:
        # If salience is near 0: alpha -> 0 (no write), gamma -> 1.0 (no decay / perfect retention)
        alpha = s * raw_alpha
        gamma = 1.0 - (s * raw_beta * 0.5) # When s=0, gamma is exactly 1.0!

        # 3. Delta-rule error: v - M_prev @ k
        pred_v = torch.matmul(M_prev, k.unsqueeze(-1)).squeeze(-1)
        error = v - pred_v

        # 4. Gated update
        delta_write = torch.matmul(error.unsqueeze(-1), k.unsqueeze(-2)) # (B, H, D, D)
        M_t = gamma * M_prev + alpha * delta_write

        # 5. Output projection
        readout_norm = self.head_norm(readout)
        out = self.out_proj(readout_norm.view(B, self.inner_dim))

        return out, M_t


class TopkSparsity(nn.Module):
    """
    Lateral Inhibition / Winner-Take-All sparsity.
    Keeps only top-k proportion of activations, setting the rest to zero.
    Tracks theoretical active FLOP ratio.
    """
    def __init__(self, topk_ratio: float = 0.25):
        super().__init__()
        self.topk_ratio = topk_ratio

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.topk_ratio >= 1.0 or not self.training and self.topk_ratio <= 0.0:
            return x, 1.0

        k = max(1, int(x.size(-1) * self.topk_ratio))
        val, idx = torch.topk(torch.abs(x), k=k, dim=-1)
        threshold = val[..., -1:].expand_as(x)
        
        # Hard thresholding with straight-through or smooth mask
        mask = (torch.abs(x) >= threshold).float()
        sparse_x = x * mask
        active_ratio = k / x.size(-1)
        return sparse_x, active_ratio


class PredictiveDeltaSkip(nn.Module):
    """
    Predictive coding module:
    Generates top-down expectation x_hat from h_{t-1}.
    If surprise ||x_t - x_hat|| is below threshold, skips the heavy recurrent attractor loop.
    """
    def __init__(self, d_model: int, skip_threshold: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.skip_threshold = skip_threshold
        self.predictor = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_x: prediction error (B, d_model)
            skip_mask: boolean mask (B,) indicating which samples skip heavy compute
            surprise: (B,) scalar surprise value
        """
        x_hat = self.predictor(h_prev)
        delta_x = x_t - x_hat
        
        norm_delta = torch.norm(delta_x, dim=-1)
        norm_x = torch.norm(x_t, dim=-1) + 1e-6
        surprise = norm_delta / norm_x  # relative surprise

        skip_mask = surprise < self.skip_threshold
        return delta_x, skip_mask, surprise


class DynamicAttractorCore(nn.Module):
    """
    Recurrent Cortical Core with Adaptive Computation Time (ACT).
    Iterates dynamically until halting probability accumulates to ~1.0 or max_steps reached.
    """
    def __init__(
        self,
        d_model: int,
        max_steps: int = 4,
        topk_ratio: float = 0.25,
        use_sparsity: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        self.use_sparsity = use_sparsity

        # Recurrent cell transitions
        self.w_h = nn.Linear(d_model, d_model)
        self.w_x = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        
        # Sparsity
        self.sparsity = TopkSparsity(topk_ratio) if use_sparsity else None

        # Halting gate
        self.halt_gate = nn.Linear(d_model, 1)

    def forward(
        self,
        delta_x: torch.Tensor,
        h_prev: torch.Tensor,
        skip_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            delta_x: (B, d_model)
            h_prev: (B, d_model)
            skip_mask: (B,) boolean mask of samples to skip
        Returns:
            h_next: (B, d_model)
            ponder_cost: (B,) scalar penalty for number of steps used
            stats: step counts and active FLOPs
        """
        B = delta_x.size(0)
        device = delta_x.device

        h = h_prev
        accum_p = torch.zeros(B, 1, device=device)
        h_acc = torch.zeros(B, self.d_model, device=device)
        step_counts = torch.zeros(B, device=device)
        active_flops_sum = 0.0

        for step in range(1, self.max_steps + 1):
            # Compute halting probability for this iteration
            p_step = torch.sigmoid(self.halt_gate(h)) # (B, 1)
            
            # Check if this is the final step
            is_last = (step == self.max_steps)
            still_running = (accum_p < 0.99).float()

            if is_last:
                # Remainder probability
                p = 1.0 - accum_p
            else:
                p = torch.min(p_step, 1.0 - accum_p)

            # Weight current state
            weight = p * still_running

            # Transition step
            pre_act = self.w_h(h) + self.w_x(delta_x)
            if self.use_sparsity and self.sparsity is not None:
                pre_act, act_ratio = self.sparsity(pre_act)
                active_flops_sum += act_ratio
            else:
                active_flops_sum += 1.0

            h_next = h + torch.tanh(self.ln(pre_act))
            h_acc = h_acc + weight * h_next
            accum_p = accum_p + weight
            step_counts = step_counts + still_running.squeeze(-1)

            h = h_next

            # If all samples in batch have halted, break early
            if (accum_p >= 0.99).all():
                break

        # Handle skipped tokens (if skip_mask is active)
        if skip_mask is not None and skip_mask.any():
            # For skipped samples, h_next is a lightweight momentum update
            h_acc = torch.where(skip_mask.unsqueeze(-1), h_prev + 0.1 * delta_x, h_acc)
            step_counts = torch.where(skip_mask, torch.zeros_like(step_counts), step_counts)

        ponder_cost = step_counts.mean()
        stats = {
            "step_counts": step_counts,
            "avg_steps": step_counts.mean().item(),
            "max_step_reached": step_counts.max().item(),
            "min_step_reached": step_counts.min().item(),
            "median_steps": step_counts.median().item(),
            "active_flops_ratio": (active_flops_sum / max(1, step))
        }

        return h_acc, ponder_cost, stats


class CausalConv1d(nn.Module):
    """
    Depthwise 1D Causal Convolution with left-padding.
    Binds neighboring tokens (e.g. key at t and value at t+1) locally
    before passing representations into recurrent memory.
    """
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x_trans = x.transpose(1, 2) # (B, d_model, T)
        padded = F.pad(x_trans, (self.kernel_size - 1, 0))
        out = self.act(self.conv(padded))
        return out.transpose(1, 2) # (B, T, d_model)


class KernelizedBoundSynapticMemory(nn.Module):
    """
    Kernelized & Bound Synaptic Memory (KBSM):
    Combines:
    1. Causal 1D depthwise convolution for local Key-Value binding.
    2. Rectified power-kernel mapping phi(k) = normalize(ReLU(k)^2 + eps)
       to exponentially suppress small noise dot products.
    3. Thalamic salience gating to freeze memory updates on distractor tokens.
    State size: strictly O(1) across any sequence length!
    """
    def __init__(self, d_model: int, num_heads: int = 4, head_dim: int = 16, conv_kernel: int = 4, use_power_kernel: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.use_power_kernel = use_power_kernel

        # 1. Local Causal Conv Binding
        self.local_conv = CausalConv1d(d_model, kernel_size=conv_kernel) if conv_kernel > 1 else None

        # 2. Linear Projections
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        # 3. Thalamic Sensory Gate
        self.salience_gate = nn.Linear(d_model, num_heads)
        nn.init.constant_(self.salience_gate.bias, -1.0)
        self.write_gate = nn.Linear(d_model, num_heads)
        self.decay_gate = nn.Linear(d_model, num_heads)

        self.head_norm = nn.LayerNorm(head_dim)

    def power_kernel(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_power_kernel:
            rect = F.relu(z)
            sq = rect * rect + 1e-5
            return F.normalize(sq, dim=-1)
        else:
            return F.normalize(z, dim=-1)

    def forward(
        self,
        x_seq: torch.Tensor,
        M_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x_seq.size()
        H = self.num_heads
        D = self.head_dim

        if self.local_conv is not None:
            x_bound = self.local_conv(x_seq)
        else:
            x_bound = x_seq

        if M_prev is None:
            M_prev = torch.zeros(B, H, D, D, device=x_seq.device)

        M = M_prev
        outputs = []

        for t in range(T):
            xt = x_bound[:, t]

            q = self.power_kernel(self.q_proj(xt).view(B, H, D))
            k = self.power_kernel(self.k_proj(xt).view(B, H, D))
            v = self.v_proj(xt).view(B, H, D)

            # Readout
            readout = torch.matmul(M, q.unsqueeze(-1)).squeeze(-1)

            # Thalamic gate
            s = torch.sigmoid(self.salience_gate(xt)).view(B, H, 1, 1)
            raw_alpha = torch.sigmoid(self.write_gate(xt)).view(B, H, 1, 1)
            raw_beta = torch.sigmoid(self.decay_gate(xt)).view(B, H, 1, 1)

            alpha = s * raw_alpha
            gamma = 1.0 - (s * raw_beta * 0.5)

            # Delta-rule write
            pred_v = torch.matmul(M, k.unsqueeze(-1)).squeeze(-1)
            error = v - pred_v
            delta_write = torch.matmul(error.unsqueeze(-1), k.unsqueeze(-2))
            M = gamma * M + alpha * delta_write

            readout_norm = self.head_norm(readout)
            out_t = self.out_proj(readout_norm.view(B, self.inner_dim))
            outputs.append(out_t)

        out_seq = torch.stack(outputs, dim=1)
        return out_seq, M

