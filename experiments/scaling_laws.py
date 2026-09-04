"""
Scaling Laws Experiment: Loss vs. Compute (FLOPs) on Wikitext-2.
Trains KBSM-LM and Transformer-LM side-by-side and logs:
- Step-by-step Validation Loss and Perplexity
- Cumulative Training Compute (FLOPs)
- Convergence trajectory and Compute-Efficiency ratio
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.wikitext import WikitextDataset
from models.kbsm_lm import KBSMLanguageModel
from models.transformer_lm import TransformerLanguageModel

def evaluate_loss(model: nn.Module, val_dataset: WikitextDataset, num_batches: int = 15, batch_size: int = 16, device: str = 'cpu') -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = val_dataset.get_batch(batch_size, device=device)
            logits, _, _ = model(x)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(-1, V), y.view(-1))
            total_loss += loss.item()
    return total_loss / max(1, num_batches)

def train_scaling_run(
    model: nn.Module,
    model_name: str,
    train_dataset: WikitextDataset,
    val_dataset: WikitextDataset,
    num_steps: int = 350,
    batch_size: int = 16,
    lr: float = 2e-3,
    eval_every: int = 50,
    device: str = 'cpu'
) -> Dict[str, Any]:
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    params = model.count_parameters()
    flops_per_token = model.estimate_flops_per_token()
    tokens_per_step = batch_size * train_dataset.seq_len

    history = {
        "model_name": model_name,
        "params": params,
        "steps": [],
        "tokens_processed": [],
        "cumulative_flops": [],
        "train_losses": [],
        "val_losses": [],
        "val_perplexities": []
    }

    print(f"\n--- Training {model_name} (Params: {params:,} | FLOPs/tok: {flops_per_token:,.0f}) ---")
    model.train()
    running_train_loss = 0.0
    t0 = time.time()

    for step in range(1, num_steps + 1):
        x, y = train_dataset.get_batch(batch_size, device=device)
        optimizer.zero_grad()

        logits, _, _ = model(x)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(-1, V), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_train_loss += loss.item()

        if step % eval_every == 0 or step == num_steps:
            val_loss = evaluate_loss(model, val_dataset, batch_size=batch_size, device=device)
            val_ppl = math.exp(min(val_loss, 20.0))
            cum_tokens = step * tokens_per_step
            cum_flops = cum_tokens * flops_per_token
            avg_train_loss = running_train_loss / eval_every

            history["steps"].append(step)
            history["tokens_processed"].append(cum_tokens)
            history["cumulative_flops"].append(cum_flops)
            history["train_losses"].append(avg_train_loss)
            history["val_losses"].append(val_loss)
            history["val_perplexities"].append(val_ppl)

            print(f"  Step {step:3d}/{num_steps} | Tokens: {cum_tokens:,.0f} | Compute: {cum_flops:.2e} FLOPs | Train Loss: {avg_train_loss:.3f} | Val Loss: {val_loss:.3f} (PPL: {val_ppl:.1f})")
            running_train_loss = 0.0
            model.train()

    total_time = time.time() - t0
    history["total_time_s"] = total_time
    print(f"Finished {model_name} in {total_time:.1f}s.")
    return history

def run_scaling_benchmark():
    print("=================================================================")
    print("WIKITEXT-2 SCALING BENCHMARK: LOSS VS. COMPUTE")
    print("=================================================================\n")

    seq_len = 128
    batch_size = 16
    num_steps = 350
    device = 'cpu'

    train_data = WikitextDataset(split="train", seq_len=seq_len, max_tokens=200_000)
    val_data = WikitextDataset(split="valid", seq_len=seq_len, max_tokens=50_000)

    # 1. KBSM Language Model (~433k params)
    torch.manual_seed(42)
    kbsm_model = KBSMLanguageModel(
        vocab_size=train_data.vocab_size,
        d_model=128,
        n_layers=2,
        num_heads=4,
        head_dim=32,
        conv_kernel=4
    )
    kbsm_history = train_scaling_run(
        kbsm_model, "KBSM-LM", train_data, val_data,
        num_steps=num_steps, batch_size=batch_size, lr=2e-3, device=device
    )

    # 2. Causal Transformer Language Model (~560k params)
    torch.manual_seed(42)
    tf_model = TransformerLanguageModel(
        vocab_size=train_data.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4
    )
    tf_history = train_scaling_run(
        tf_model, "Transformer-LM", train_data, val_data,
        num_steps=num_steps, batch_size=batch_size, lr=2e-3, device=device
    )

    results = {
        "KBSM-LM": kbsm_history,
        "Transformer-LM": tf_history
    }

    os.makedirs("results", exist_ok=True)
    with open("results/scaling_laws_wikitext.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print Final Summary Comparison
    print("\n=================================================================")
    print("FINAL LOSS VS. COMPUTE COMPARISON")
    print("=================================================================\n")
    print("| Model | Parameters | Tokens Trained | Final Compute (FLOPs) | Final Val Loss | Final Val PPL |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, h in results.items():
        print(f"| **{name}** | {h['params']:,} | {h['tokens_processed'][-1]:,} | {h['cumulative_flops'][-1]:.2e} | **{h['val_losses'][-1]:.3f}** | **{h['val_perplexities'][-1]:.1f}** |")

if __name__ == "__main__":
    run_scaling_benchmark()
