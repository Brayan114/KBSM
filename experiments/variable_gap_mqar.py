"""
Comprehensive Variable-Gap MQAR Benchmark with Modern Baselines.
Evaluates associative recall accuracy as Key-Value separation distance g in {1, 2, 4, 8} varies.

Architectures Evaluated:
1. KBSM (Full, Conv K=4 + Rectified Power Kernel + Thalamic Gating)
2. KBSM No-Conv (K=1, Pure Recurrent Synaptic Memory)
3. Mamba-SSM (Selective State Space Model with K=4 Causal Conv)
4. Gated Linear Attention (GLA, Outer-Product Associative State with Data-Dependent Decay)
5. Causal Transformer (with Rotary Position Embeddings - RoPE)
6. Linear Recurrent Baseline (Ungated, Inner Product)
"""

import os
import sys
import json
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.mqar import MQARDataset
from models.transformer import CausalTransformer
from models.kbsm_model import KBSMModel
from models.mamba_ssm import MambaModel
from models.gla import GLAModel

def get_lr_scheduler(optimizer, num_steps, warmup_steps=35):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_eval_mqar(
    model: nn.Module,
    train_data: MQARDataset,
    eval_data: MQARDataset,
    device: torch.device,
    num_steps: int = 350,
    batch_size: int = 32,
    lr: float = 3e-3
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, num_steps, warmup_steps=int(num_steps * 0.1))
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
    num_steps=350,
    batch_size=32,
    device_name='auto',
    output_json='results/variable_gap_mqar.json',
    output_plot='results/variable_gap_mqar.png'
):
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)

    print("=================================================================")
    print("MASTER VARIABLE-GAP MQAR BENCHMARK WITH MODERN SOTA BASELINES")
    print(f"Testing Key-Value separation gaps: {list(gaps)} | Device: {device}")
    print("=================================================================\n")

    vocab_size = 64
    d_model = 64

    # Parameter-matched baseline definitions
    model_factories = {
        "KBSM_Full": (
            lambda: KBSMModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, conv_kernel=4, use_power_kernel=True),
            3e-3, "KBSM (Full, Conv K=4)"
        ),
        "KBSM_NoConv": (
            lambda: KBSMModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, conv_kernel=1, use_power_kernel=True),
            3e-3, "KBSM (No Conv, K=1)"
        ),
        "Mamba_SSM": (
            lambda: MambaModel(vocab_size=vocab_size, d_model=d_model, d_state=16, d_conv=4, expand=2, n_layers=2),
            2e-3, "Mamba-SSM (Gu & Dao 2023)"
        ),
        "GLA": (
            lambda: GLAModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, n_layers=2),
            2e-3, "Gated Linear Attention (GLA)"
        ),
        "Transformer_RoPE": (
            lambda: CausalTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=128, use_rope=True),
            1e-3, "Causal Transformer (with RoPE)"
        ),
        "Linear_Recurrent": (
            lambda: KBSMModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, conv_kernel=1, use_power_kernel=False),
            3e-3, "Linear Recurrence Baseline"
        ),
    }

    results = {
        "gaps": list(gaps),
        "seq_len": seq_len,
        "num_pairs": num_pairs,
        "num_steps": num_steps,
        "device": str(device)
    }
    for m_key in model_factories:
        results[m_key] = []

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

        for m_idx, (m_key, (m_fn, m_lr, m_label)) in enumerate(model_factories.items(), 1):
            print(f"  [{m_idx}/{len(model_factories)}] Training {m_label} (lr={m_lr})...")
            torch.manual_seed(42)
            model = m_fn()
            t0 = time.time()
            acc, loss = train_eval_mqar(model, train_data, eval_data, device=device, num_steps=num_steps, batch_size=batch_size, lr=m_lr)
            print(f"        --> {m_label}: Acc = {acc:.1f}%, Loss = {loss:.3f} ({time.time()-t0:.1f}s)")
            results[m_key].append(acc)

    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Successfully saved benchmark results to: {output_json}")

    # Generate Publication-Quality Plot
    try:
        plt.figure(figsize=(10, 6.2), dpi=300)
        x_indices = list(range(len(gaps)))
        gap_labels = [f"g={g}\n({'In-Window' if g <= 2 else 'Out-of-Window'})" for g in gaps]

        styles = {
            "KBSM_Full": {'color': '#2ca02c', 'marker': 'o', 'lw': 2.8, 'ls': '-', 'label': 'KBSM (Full, O(1) Memory)'},
            "KBSM_NoConv": {'color': '#1f77b4', 'marker': 's', 'lw': 2.0, 'ls': '--', 'label': 'KBSM (No Conv, K=1)'},
            "Mamba_SSM": {'color': '#9467bd', 'marker': 'D', 'lw': 2.2, 'ls': '-.', 'label': 'Mamba-SSM (Selective SSM)'},
            "GLA": {'color': '#8c564b', 'marker': 'v', 'lw': 2.0, 'ls': ':', 'label': 'Gated Linear Attention (GLA)'},
            "Transformer_RoPE": {'color': '#ff7f0e', 'marker': '^', 'lw': 2.2, 'ls': '-', 'label': 'Transformer (RoPE, O(N) Cache)'},
            "Linear_Recurrent": {'color': '#d62728', 'marker': 'x', 'lw': 1.6, 'ls': ':', 'label': 'Linear Recurrence Baseline'}
        }

        for m_key, st in styles.items():
            if m_key in results:
                plt.plot(x_indices, results[m_key], marker=st['marker'], linewidth=st['lw'], markersize=7, color=st['color'], linestyle=st['ls'], label=st['label'])

        # Add vertical divider showing Conv Receptive Field limit
        plt.axvline(x=1.5, color='gray', linestyle='--', alpha=0.7, label='Conv K=4 Receptive Limit')

        plt.xticks(x_indices, gap_labels, fontsize=11)
        plt.yticks(range(0, 105, 10), fontsize=11)
        plt.xlabel("Key-Value Separation Distance (Gap g)", fontsize=13, fontweight='bold')
        plt.ylabel("Multi-Query Recall Accuracy (%)", fontsize=13, fontweight='bold')
        plt.title("Variable-Gap MQAR: Decoupling Local Convolution from Recurrent State", fontsize=14, fontweight='bold')
        plt.legend(frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5, loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.savefig(output_plot, dpi=300)
        plt.close()
        print(f"[+] Successfully saved benchmark figure to: {output_plot}")
    except Exception as e:
        print(f"[!] Warning: Plot generation failed with: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variable-Gap MQAR Benchmark with Modern Baselines")
    parser.add_argument("--gaps", nargs="+", type=int, default=[1, 2, 4, 8], help="Separation gaps to test")
    parser.add_argument("--seq_len", type=int, default=128, help="Total sequence length")
    parser.add_argument("--num_pairs", type=int, default=4, help="Number of key-value pairs")
    parser.add_argument("--num_steps", type=int, default=350, help="Training steps per model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
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
        device_name=args.device,
        output_json=args.output_json,
        output_plot=args.output_plot
    )
