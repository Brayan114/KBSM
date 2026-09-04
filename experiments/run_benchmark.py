"""
Unified Experiment Runner & Systematic Ablation Suite.
Executes apples-to-apples training and evaluation across:
- M0: Standard Causal Transformer
- M1: Dense Recurrent Baseline
- M2: + Sparsity (Top-k)
- M3: + Predictive Delta Skip
- M4: + Dynamic Halting (Attractor Core)
- M5: + Fast Synaptic Memory (Gated Delta-Rule)
- M6: Full PSAN
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List

# Ensure current workspace is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.mqar import MQARDataset
from benchmarks.state_tracking import StateTrackingDataset
from models.transformer import CausalTransformer
from models.psan import PSAN
from experiments.profiler import profile_model_summary

def get_model_factory(vocab_size: int, d_model: int = 64):
    """Returns factory dictionary for all 7 ablation models."""
    return {
        "M0_Transformer": lambda: CausalTransformer(
            vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=128
        ),
        "M1_DenseRecurrent": lambda: PSAN(
            vocab_size=vocab_size, d_model=d_model,
            use_sparsity=False, use_predictive=False,
            use_dynamic_halting=False, use_synaptic_memory=False
        ),
        "M2_PlusSparsity": lambda: PSAN(
            vocab_size=vocab_size, d_model=d_model,
            use_sparsity=True, use_predictive=False,
            use_dynamic_halting=False, use_synaptic_memory=False
        ),
        "M3_PlusPredictive": lambda: PSAN(
            vocab_size=vocab_size, d_model=d_model,
            use_sparsity=False, use_predictive=True,
            use_dynamic_halting=False, use_synaptic_memory=False
        ),
        "M4_PlusDynamicHalting": lambda: PSAN(
            vocab_size=vocab_size, d_model=d_model,
            use_sparsity=False, use_predictive=False,
            use_dynamic_halting=True, use_synaptic_memory=False
        ),
        "M5_PlusSynapticMemory": lambda: PSAN(
            vocab_size=vocab_size, d_model=d_model,
            use_sparsity=False, use_predictive=False,
            use_dynamic_halting=False, use_synaptic_memory=True
        ),
        "M6_FullPSAN": lambda: PSAN(
            vocab_size=vocab_size, d_model=d_model,
            use_sparsity=True, use_predictive=True,
            use_dynamic_halting=True, use_synaptic_memory=True
        )
    }

def train_and_eval_task(
    model: nn.Module,
    train_dataset: Any,
    eval_dataset: Any,
    num_steps: int = 400,
    batch_size: int = 32,
    lr: float = 2e-3,
    ponder_weight: float = 0.005
) -> Dict[str, float]:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    model.train()
    for step in range(num_steps):
        inputs, targets, masks = train_dataset.get_batch(batch_size)
        optimizer.zero_grad()

        if hasattr(model, 'blocks'):
            logits, _, metrics = model(inputs)
        else:
            logits, _, metrics = model(inputs)

        # Masked loss calculation
        B, T, V = logits.shape
        loss_matrix = loss_fn(logits.view(-1, V), targets.view(-1)).view(B, T)
        masked_loss = (loss_matrix * masks.float()).sum() / masks.float().sum().clamp(min=1.0)

        # Add ponder penalty if applicable
        if "ponder_cost" in metrics and isinstance(metrics["ponder_cost"], torch.Tensor):
            total_loss = masked_loss + ponder_weight * metrics["ponder_cost"]
        else:
            total_loss = masked_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        inputs, targets, masks = eval_dataset.get_batch(min(500, eval_dataset.num_examples))
        if hasattr(model, 'blocks'):
            logits, _, metrics = model(inputs)
        else:
            logits, _, metrics = model(inputs)

        preds = torch.argmax(logits, dim=-1)
        correct = ((preds == targets) & masks).sum().float()
        total_eval_tokens = masks.sum().float().clamp(min=1.0)
        accuracy = (correct / total_eval_tokens).item() * 100.0

        loss_matrix = loss_fn(logits.view(-1, V), targets.view(-1)).view(inputs.size(0), inputs.size(1))
        eval_loss = ((loss_matrix * masks.float()).sum() / total_eval_tokens).item()

    return {
        "accuracy": accuracy,
        "eval_loss": eval_loss,
        "avg_steps": metrics.get("avg_steps", 1.0),
        "median_steps": metrics.get("median_steps", 1.0),
        "active_flops_ratio": metrics.get("active_flops_ratio", 1.0)
    }

def run_all_experiments():
    print("=================================================================")
    print("STARTING PSAN SYSTEMATIC ABLATION & BENCHMARK SUITE")
    print("=================================================================\n")

    vocab_size = 64
    d_model = 64

    # 1. Datasets
    print("Generating Datasets...")
    # MQAR Task: 4 key-value pairs, sequence length 128
    mqar_train = MQARDataset(num_examples=1500, seq_len=128, num_pairs=4, vocab_size=vocab_size, seed=42)
    mqar_eval = MQARDataset(num_examples=500, seq_len=128, num_pairs=4, vocab_size=vocab_size, seed=123)

    # State Tracking Task: Train on L=64, Extrapolate to L=128
    state_train = StateTrackingDataset(num_examples=1500, seq_len=64, vocab_size=vocab_size, seed=42)
    state_eval_in_domain = StateTrackingDataset(num_examples=500, seq_len=64, vocab_size=vocab_size, seed=123)
    state_eval_extrapolate = StateTrackingDataset(num_examples=500, seq_len=128, vocab_size=vocab_size, seed=456)

    factory = get_model_factory(vocab_size=vocab_size, d_model=d_model)
    results = {}

    for model_name, model_fn in factory.items():
        print(f"--- Running: {model_name} ---")
        
        # Fresh model instance
        torch.manual_seed(42)
        model = model_fn()
        
        # Profile hardware metrics (Params, Latency, Memory scaling)
        prof = profile_model_summary(model, model_name, test_lengths=[64, 128, 256, 512])
        print(f"  Params: {prof['params']:,} | Latency: {prof['latency_ms_per_token']:.3f} ms/tok | State RAM @512: {prof['memory_scaling_bytes'][512]} bytes")

        # Train & Eval on MQAR
        t0 = time.time()
        mqar_res = train_and_eval_task(model, mqar_train, mqar_eval, num_steps=350, batch_size=32)
        train_time_mqar = time.time() - t0
        print(f"  [MQAR] Acc: {mqar_res['accuracy']:.1f}% | Loss: {mqar_res['eval_loss']:.3f} (Train: {train_time_mqar:.1f}s)")

        # Train & Eval on State Tracking (with length extrapolation)
        torch.manual_seed(42)
        model_state = model_fn()
        t0 = time.time()
        state_in = train_and_eval_task(model_state, state_train, state_eval_in_domain, num_steps=350, batch_size=32)
        
        # Test extrapolation on L=128 without retraining
        model_state.eval()
        with torch.no_grad():
            extrap_inputs, extrap_targets, extrap_masks = state_eval_extrapolate.get_batch(500)
            if hasattr(model_state, 'blocks'):
                extrap_logits, _, _ = model_state(extrap_inputs)
            else:
                extrap_logits, _, _ = model_state(extrap_inputs)
            preds = torch.argmax(extrap_logits, dim=-1)
            extrap_acc = (((preds == extrap_targets) & extrap_masks).sum().float() / extrap_masks.sum().float() * 100.0).item()
        
        print(f"  [State Tracking] In-Domain (L=64): {state_in['accuracy']:.1f}% | Extrapolation (L=128): {extrap_acc:.1f}%")

        results[model_name] = {
            "params": prof["params"],
            "latency_ms_per_token": prof["latency_ms_per_token"],
            "state_memory_at_512_bytes": prof["memory_scaling_bytes"][512],
            "state_memory_scaling": prof["memory_scaling_bytes"],
            "mqar_accuracy": mqar_res["accuracy"],
            "mqar_eval_loss": mqar_res["eval_loss"],
            "mqar_avg_steps": mqar_res["avg_steps"],
            "mqar_active_flops_ratio": mqar_res["active_flops_ratio"],
            "state_acc_in_domain": state_in["accuracy"],
            "state_acc_extrapolate": extrap_acc,
            "extrapolation_ratio": (extrap_acc / max(1e-4, state_in["accuracy"]))
        }
        print()

    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Summary
    print("\n=================================================================")
    print("FINAL EXPERIMENTAL RESULTS SUMMARY TABLE")
    print("=================================================================\n")
    
    header = "| Model | Params | Latency (ms/tok) | State RAM @512 | MQAR Acc (%) | State L=64 Acc | State L=128 Extrap |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(sep)
    for name, r in results.items():
        row = f"| **{name}** | {r['params']:,} | {r['latency_ms_per_token']:.2f} | {r['state_memory_at_512_bytes']} B | {r['mqar_accuracy']:.1f}% | {r['state_acc_in_domain']:.1f}% | {r['state_acc_extrapolate']:.1f}% |"
        print(row)

if __name__ == "__main__":
    run_all_experiments()
