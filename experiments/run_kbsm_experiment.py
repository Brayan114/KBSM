"""
Experiment Runner: Testing Weapon 1 (1D Causal Conv) and Weapon 2 (Rectified Power Kernel)
on breaking the Multi-Query Associative Recall (MQAR) barrier.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.mqar import MQARDataset
from models.transformer import CausalTransformer
from models.kbsm_model import KBSMModel
from experiments.profiler import profile_model_summary

def train_eval_model(
    model: nn.Module,
    train_data: MQARDataset,
    eval_data: MQARDataset,
    num_steps: int = 500,
    batch_size: int = 32,
    lr: float = 3e-3
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    model.train()
    for step in range(num_steps):
        inputs, targets, masks = train_data.get_batch(batch_size)
        optimizer.zero_grad()

        if hasattr(model, 'blocks'):
            logits, _, _ = model(inputs)
        else:
            logits, _, _ = model(inputs)

        B, T, V = logits.shape
        loss_matrix = loss_fn(logits.view(-1, V), targets.view(-1)).view(B, T)
        loss = (loss_matrix * masks.float()).sum() / masks.float().sum().clamp(min=1.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    # Eval
    model.eval()
    with torch.no_grad():
        inputs, targets, masks = eval_data.get_batch(min(500, eval_data.num_examples))
        if hasattr(model, 'blocks'):
            logits, _, _ = model(inputs)
        else:
            logits, _, _ = model(inputs)

        preds = torch.argmax(logits, dim=-1)
        correct = ((preds == targets) & masks).sum().float()
        total_eval = masks.sum().float().clamp(min=1.0)
        acc = (correct / total_eval).item() * 100.0

        loss_matrix = loss_fn(logits.view(-1, V), targets.view(-1)).view(inputs.size(0), inputs.size(1))
        eval_loss = ((loss_matrix * masks.float()).sum() / total_eval).item()

    return acc, eval_loss

def run_kbsm_experiments():
    print("=================================================================")
    print("BREAKING THE MQAR RECALL BARRIER: WEAPONS 1 & 2 EVALUATION")
    print("=================================================================\n")

    vocab_size = 64
    d_model = 64
    seq_len = 128
    num_pairs = 4

    print(f"Generating MQAR Benchmark (K={num_pairs} pairs, L={seq_len})...")
    train_data = MQARDataset(num_examples=2500, seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size, seed=42)
    eval_data = MQARDataset(num_examples=500, seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size, seed=123)

    models_to_test = {
        "M0_Transformer (O(N) KV-Cache)": lambda: CausalTransformer(
            vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=128
        ),
        "Baseline_Thalamic (No Conv, Linear Kernel)": lambda: KBSMModel(
            vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16,
            conv_kernel=1, use_power_kernel=False
        ),
        "Weapon1_ConvBinding (K=4, Linear Kernel)": lambda: KBSMModel(
            vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16,
            conv_kernel=4, use_power_kernel=False
        ),
        "Weapon2_PowerKernel (No Conv, Rectified Kernel)": lambda: KBSMModel(
            vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16,
            conv_kernel=1, use_power_kernel=True
        ),
        "Full_KBSM (Weapon 1 + 2: Conv K=4 + Power Kernel)": lambda: KBSMModel(
            vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16,
            conv_kernel=4, use_power_kernel=True
        ),
    }

    results = {}
    for name, m_fn in models_to_test.items():
        print(f"--- Running: {name} ---")
        torch.manual_seed(42)
        model = m_fn()

        prof = profile_model_summary(model, name, test_lengths=[64, 128, 256, 512])
        t0 = time.time()
        acc, eval_loss = train_eval_model(model, train_data, eval_data, num_steps=500, batch_size=32, lr=3e-3)
        duration = time.time() - t0

        print(f"  Params: {prof['params']:,} | State RAM @512: {prof['memory_scaling_bytes'][512]} B | Latency: {prof['latency_ms_per_token']:.3f} ms/tok")
        print(f"  --> MQAR Accuracy: {acc:.1f}% | Eval Loss: {eval_loss:.3f} (Time: {duration:.1f}s)\n")

        results[name] = {
            "params": prof["params"],
            "state_ram_bytes_512": prof["memory_scaling_bytes"][512],
            "latency_ms_per_token": prof["latency_ms_per_token"],
            "mqar_accuracy": acc,
            "eval_loss": eval_loss,
            "train_duration_s": duration
        }

    os.makedirs("results", exist_ok=True)
    with open("results/kbsm_mqar_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=================================================================")
    print("FINAL MQAR BARRIER BREAKTHROUGH TABLE")
    print("=================================================================\n")
    header = "| Model | Params | State RAM @512 | Latency (ms/tok) | MQAR Acc (%) | Eval Loss |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(sep)
    for name, r in results.items():
        print(f"| **{name}** | {r['params']:,} | {r['state_ram_bytes_512']} B | {r['latency_ms_per_token']:.2f} | **{r['mqar_accuracy']:.1f}%** | {r['eval_loss']:.3f} |")

if __name__ == "__main__":
    run_kbsm_experiments()
