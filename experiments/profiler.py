"""
Profiling suite for measuring:
- Empirical Wall-Clock Latency (ms/token)
- State / Cache Memory scaling with sequence length
- Parameter count and theoretical FLOPs
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, List

def measure_token_latency(model: nn.Module, vocab_size: int = 64, num_tokens: int = 100, device: str = 'cpu') -> float:
    """
    Measures empirical wall-clock generation latency in milliseconds per token.
    Uses sequential step-by-step inference.
    """
    model.eval()
    times = []
    
    # Warmup
    with torch.no_grad():
        x = torch.randint(0, vocab_size, (1, 1), device=device)
        state = None
        for _ in range(10):
            if hasattr(model, 'blocks'): # Transformer
                out, state, _ = model(x, kv_caches=state)
            else: # PSAN
                out, state, _ = model(x, state=state)

    # Measurement
    with torch.no_grad():
        state = None
        curr_token = torch.randint(0, vocab_size, (1, 1), device=device)
        for _ in range(num_tokens):
            t0 = time.perf_counter()
            if hasattr(model, 'blocks'):
                out, state, _ = model(curr_token, kv_caches=state)
            else:
                out, state, _ = model(curr_token, state=state)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0) # convert to ms
            curr_token = torch.argmax(out[:, -1:, :], dim=-1)

    # Return median ms/token to exclude system noise
    times.sort()
    return times[len(times) // 2]

def measure_state_memory_bytes(model: nn.Module, seq_len: int, device: str = 'cpu') -> int:
    """
    Measures the exact byte size of the internal recurrent state or KV-cache
    after processing a sequence of length seq_len.
    """
    model.eval()
    vocab_size = getattr(model, 'vocab_size', 64)
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    
    with torch.no_grad():
        if hasattr(model, 'blocks'):
            # Transformer KV-cache
            _, caches, _ = model(x)
            total_bytes = 0
            for k, v in caches:
                total_bytes += k.element_size() * k.nelement()
                total_bytes += v.element_size() * v.nelement()
            return total_bytes
        else:
            # Recurrent state (tuple or single tensor)
            _, state, _ = model(x)
            if isinstance(state, (tuple, list)):
                total_bytes = 0
                for item in state:
                    if item is not None and isinstance(item, torch.Tensor):
                        total_bytes += item.element_size() * item.nelement()
                return total_bytes
            elif isinstance(state, torch.Tensor):
                return state.element_size() * state.nelement()
            return 0

def profile_model_summary(model: nn.Module, model_name: str, test_lengths: List[int] = [64, 128, 256, 512]) -> Dict[str, Any]:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    latency_ms = measure_token_latency(model)
    
    memory_scaling = {}
    for L in test_lengths:
        mem_bytes = measure_state_memory_bytes(model, L)
        memory_scaling[L] = mem_bytes

    return {
        "name": model_name,
        "params": params,
        "latency_ms_per_token": latency_ms,
        "memory_scaling_bytes": memory_scaling
    }
