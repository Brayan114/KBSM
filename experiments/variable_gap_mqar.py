"""
Variable-Gap MQAR Benchmark.
Systematically evaluates associative recall accuracy as the separation distance
between Key and Value tokens is varied: g in {1, 2, 4, 8}.

Specifically tests the hypothesis:
Does KBSM's synaptic memory retain associations when the key-value gap exceeds
the local receptive field of the 1D causal convolution (K=4)?
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.mqar import MQARDataset
from models.transformer import CausalTransformer
from models.kbsm_model import KBSMModel

def train_eval_mqar(
    model: nn.Module,
    train_data: MQARDataset,
    eval_data: MQARDataset,
    device: torch.device,
    num_steps: int = 400,
    batch_size: int = 32,
    lr: float = 3e-3
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    model.train()
    for step in range(num_steps):
        inputs, targets, masks = train_data.get_batch(batch_size)
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(inputs)
        B, T, V = logits.shape
        loss_matrix = loss_fn(logits.view(-1, V), targets.view(-1)).view(B, T)
        loss = (loss_matrix * masks.float()).sum() / masks.float().sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        inputs, targets, masks = eval_data.get_batch(min(500, eval_data.num_examples))
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        logits, _, _ = model(inputs)
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & masks).sum().item()
        total = masks.sum().item()
        acc = (correct / total * 100.0) if total > 0 else 0.0
        B, T, V = logits.shape
        loss_matrix = loss_fn(logits.view(-1, V), targets.view(-1)).view(B, T)
        eval_loss = ((loss_matrix * masks.float()).sum() / masks.sum().clamp(min=1.0)).item()

    return acc, eval_loss

def run_variable_gap_benchmark(
    gaps=(1, 2, 4, 8),
    seq_len=128,
    num_pairs=4,
    num_steps=400,
    batch_size=32,
    lr=3e-3,
    device_name='auto',
    output_json='results/variable_gap_mqar.json',
    output_plot='results/variable_gap_mqar.png'
):
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)

    print("=================================================================")
    print("VARIABLE-GAP MQAR BENCHMARK: DECOUPLING CONVOLUTION & MEMORY")
    print(f"Testing Key-Value separation gaps: {list(gaps)} | Device: {device}")
    print("=================================================================\n")

    vocab_size = 64
    d_model = 64

    results = {
        "gaps": list(gaps),
        "seq_len": seq_len,
        "num_pairs": num_pairs,
        "num_steps": num_steps,
        "device": str(device),
        "KBSM_Full": [],
        "KBSM_NoConv": [],
        "Linear_Recurrent": [],
        "Transformer": []
    }

    os.makedirs('results', exist_ok=True)

    for g in gaps:
        print(f"\n#################################################################")
        print(f">>> BENCHMARK FOR SEPARATION GAP g = {g} tokens (Receptive Field K=4) <<<")
        if g <= 3:
            print(f"    Status: Gap ({g}) <= K (4) -> In-Receptive Field (Local binding active)")
        else:
            print(f"    Status: Gap ({g}) > K (4) -> Out-of-Receptive Field (Requires pure recurrent memory)")
        print(f"#################################################################")

        train_data = MQARDataset(num_examples=2000, seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size, gap=g, seed=42)
        eval_data = MQARDataset(num_examples=500, seq_len=seq_len, num_pairs=num_pairs, vocab_size=vocab_size, gap=g, seed=123)

        # 1. Full KBSM (K=4, Power Kernel, Thalamic Gating)
        print("  [1/4] Training Full KBSM (Conv K=4 + Power Kernel + Gating)...")
        torch.manual_seed(42)
        kbsm_model = KBSMModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, conv_kernel=4, use_power_kernel=True)
        t0 = time.time()
        kbsm_acc, kbsm_loss = train_eval_mqar(kbsm_model, train_data, eval_data, device=device, num_steps=num_steps, batch_size=batch_size, lr=lr)
        print(f"        --> Full KBSM: Acc = {kbsm_acc:.1f}%, Loss = {kbsm_loss:.3f} ({time.time()-t0:.1f}s)")
        results["KBSM_Full"].append(kbsm_acc)

        # 2. KBSM No-Conv (K=1, Power Kernel, Thalamic Gating)
        print("  [2/4] Training KBSM No-Conv (Pure Synaptic Memory, K=1)...")
        torch.manual_seed(42)
        noconv_model = KBSMModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, conv_kernel=1, use_power_kernel=True)
        t0 = time.time()
        noconv_acc, noconv_loss = train_eval_mqar(noconv_model, train_data, eval_data, device=device, num_steps=num_steps, batch_size=batch_size, lr=lr)
        print(f"        --> KBSM No-Conv: Acc = {noconv_acc:.1f}%, Loss = {noconv_loss:.3f} ({time.time()-t0:.1f}s)")
        results["KBSM_NoConv"].append(noconv_acc)

        # 3. Linear Recurrent Baseline (K=1, Linear Inner Product, Ungated)
        print("  [3/4] Training Linear Recurrent Baseline...")
        torch.manual_seed(42)
        lin_model = KBSMModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, conv_kernel=1, use_power_kernel=False)
        t0 = time.time()
        lin_acc, lin_loss = train_eval_mqar(lin_model, train_data, eval_data, device=device, num_steps=num_steps, batch_size=batch_size, lr=lr)
        print(f"        --> Linear Recurrent: Acc = {lin_acc:.1f}%, Loss = {lin_loss:.3f} ({time.time()-t0:.1f}s)")
        results["Linear_Recurrent"].append(lin_acc)

        # 4. Causal Transformer Baseline
        print("  [4/4] Training Causal Transformer Baseline...")
        torch.manual_seed(42)
        tf_model = CausalTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=128)
        t0 = time.time()
        tf_acc, tf_loss = train_eval_mqar(tf_model, train_data, eval_data, device=device, num_steps=num_steps, batch_size=batch_size, lr=lr)
        print(f"        --> Transformer: Acc = {tf_acc:.1f}%, Loss = {tf_loss:.3f} ({time.time()-t0:.1f}s)")
        results["Transformer"].append(tf_acc)

    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Successfully saved benchmark results to: {output_json}")

    # Generate Publication-Quality Plot
    try:
        plt.figure(figsize=(9, 6), dpi=300)
        x_indices = list(range(len(gaps)))
        gap_labels = [f"g={g}" for g in gaps]

        plt.plot(x_indices, results["KBSM_Full"], marker='o', linewidth=2.5, markersize=8, color='#2ca02c', label='KBSM (Full, Conv K=4)')
        plt.plot(x_indices, results["KBSM_NoConv"], marker='s', linewidth=2.0, markersize=7, color='#1f77b4', linestyle='--', label='KBSM (No Conv, K=1)')
        plt.plot(x_indices, results["Transformer"], marker='^', linewidth=1.8, markersize=7, color='#ff7f0e', linestyle='-.', label='Causal Transformer')
        plt.plot(x_indices, results["Linear_Recurrent"], marker='x', linewidth=1.5, markersize=7, color='#d62728', linestyle=':', label='Linear Recurrence')

        # Add vertical divider showing Conv Receptive Field limit
        plt.axvline(x=1.5, color='gray', linestyle='--', alpha=0.7, label='Conv K=4 Receptive Limit')

        plt.xticks(x_indices, gap_labels, fontsize=12)
        plt.yticks(range(0, 105, 10), fontsize=11)
        plt.xlabel("Key-Value Separation Distance (Gap g)", fontsize=13, fontweight='bold')
        plt.ylabel("Multi-Query Recall Accuracy (%)", fontsize=13, fontweight='bold')
        plt.title("Variable-Gap Associative Recall: Isolating Memory from Convolution", fontsize=14, fontweight='bold')
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.savefig(output_plot, dpi=300)
        plt.close()
        print(f"[+] Successfully saved benchmark figure to: {output_plot}")
    except Exception as e:
        print(f"[!] Warning: Plot generation failed with: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variable-Gap MQAR Benchmark")
    parser.add_argument("--gaps", nargs="+", type=int, default=[1, 2, 4, 8], help="Separation gaps to test")
    parser.add_argument("--seq_len", type=int, default=128, help="Total sequence length")
    parser.add_argument("--num_pairs", type=int, default=4, help="Number of key-value pairs")
    parser.add_argument("--num_steps", type=int, default=400, help="Training steps per model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--output_json", type=str, default="results/variable_gap_mqar.json")
    parser.add_argument("--output_plot", type=str, default="results/variable_gap_mqar.png")
    args = parser.parse_args()

    run_variable_gap_benchmark(
        gaps=tuple(args.gaps),
        seq_len=args.seq_len,
        num_pairs=args.num_pairs,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        output_json=args.output_json,
        output_plot=args.output_plot
    )
