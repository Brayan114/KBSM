"""
GPU Turnkey Scaling Script for KBSM-LM and Transformer-LM.
Supports 10M and 100M parameter scales on Wikitext-2 and TinyStories
with BF16/FP16 Mixed Precision, gradient accumulation, and FLOP tracking.

Usage Examples:
  python train_scale_gpu.py --model kbsm --scale 10M --data wikitext
  python train_scale_gpu.py --model transformer --scale 10M --data wikitext
  python train_scale_gpu.py --model kbsm --scale 100M --data tinystories
"""

import os
import sys
import math
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.wikitext import WikitextDataset
from models.kbsm_lm import KBSMLanguageModel
from models.transformer_lm import TransformerLanguageModel

SCALE_CONFIGS = {
    "1M": {
        "d_model": 256,
        "n_layers": 4,
        "num_heads": 8,
        "head_dim": 32,
        "d_ff": 1024,
    },
    "10M": {
        "d_model": 512,
        "n_layers": 8,
        "num_heads": 8,
        "head_dim": 64,
        "d_ff": 2048,
    },
    "100M": {
        "d_model": 1024,
        "n_layers": 16,
        "num_heads": 16,
        "head_dim": 64,
        "d_ff": 4096,
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Scale up KBSM or Transformer to 10M/100M params")
    parser.add_argument("--model", type=str, default="kbsm", choices=["kbsm", "transformer"])
    parser.add_argument("--scale", type=str, default="10M", choices=["1M", "10M", "100M"])
    parser.add_argument("--data", type=str, default="wikitext", choices=["wikitext", "tinystories"])
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32")
    return parser.parse_args()

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps, eta_min=1e-5):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    args = parse_args()
    print("=================================================================")
    print(f"FRONTIER SCALING RUN: {args.model.upper()} AT {args.scale} SCALE")
    print(f"Device: {args.device} | Precision: {args.dtype}")
    print("=================================================================\n")

    cfg = SCALE_CONFIGS[args.scale]

    # 1. Dataset
    print(f"Loading {args.data.upper()} Dataset...")
    train_data = WikitextDataset(split="train", seq_len=args.seq_len)
    val_data = WikitextDataset(split="valid", seq_len=args.seq_len)

    # 2. Model Initialization
    if args.model == "kbsm":
        model = KBSMLanguageModel(
            vocab_size=train_data.vocab_size,
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            d_ff=cfg["d_ff"],
            conv_kernel=4
        )
    else:
        model = TransformerLanguageModel(
            vocab_size=train_data.vocab_size,
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            max_seq_len=args.seq_len * 2
        )

    model = model.to(args.device)
    params = model.count_parameters()
    flops_per_token = model.estimate_flops_per_token()
    print(f"Model: {args.model.upper()} | Parameters: {params:,} | FLOPs/Token: {flops_per_token:,.0f}\n")

    # 3. Optimizer & Training Setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    loss_fn = nn.CrossEntropyLoss()

    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else (torch.float16 if args.dtype == "float16" else torch.float32)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    history = {
        "model": args.model,
        "scale": args.scale,
        "params": params,
        "history": []
    }

    print("Beginning Training Loop...")
    model.train()
    running_loss = 0.0
    t0 = time.time()
    effective_tokens_per_step = args.batch_size * args.grad_accum * args.seq_len

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(args.grad_accum):
            x, y = train_data.get_batch(args.batch_size, device=args.device)
            with torch.cuda.amp.autocast(enabled=(args.device == "cuda"), dtype=amp_dtype):
                logits, _, _ = model(x)
                B, T, V = logits.shape
                loss = loss_fn(logits.view(-1, V), y.view(-1)) / args.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * args.grad_accum

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        running_loss += accum_loss

        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            val_loss = 0.0
            eval_batches = 20
            with torch.no_grad():
                for _ in range(eval_batches):
                    vx, vy = val_data.get_batch(args.batch_size, device=args.device)
                    with torch.cuda.amp.autocast(enabled=(args.device == "cuda"), dtype=amp_dtype):
                        vlogits, _, _ = model(vx)
                        B, T, V = vlogits.shape
                        vloss = loss_fn(vlogits.view(-1, V), vy.view(-1))
                        val_loss += vloss.item()
            val_loss /= eval_batches
            val_ppl = math.exp(min(val_loss, 20.0))

            cum_tokens = step * effective_tokens_per_step
            cum_flops = cum_tokens * flops_per_token
            train_avg = running_loss / args.eval_every
            elapsed = time.time() - t0

            print(f"Step {step:4d}/{args.max_steps} | Tokens: {cum_tokens:,.0f} | Compute: {cum_flops:.2e} FLOPs | Train Loss: {train_avg:.3f} | Val Loss: {val_loss:.3f} (PPL: {val_ppl:.1f}) | {elapsed:.1f}s")
            history["history"].append({
                "step": step,
                "tokens": cum_tokens,
                "flops": cum_flops,
                "train_loss": train_avg,
                "val_loss": val_loss,
                "val_ppl": val_ppl
            })
            running_loss = 0.0
            model.train()

    # Save Checkpoint & Results
    os.makedirs("results", exist_ok=True)
    out_file = f"results/scaling_{args.model}_{args.scale}.json"
    with open(out_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nScaling history saved to {out_file}.")

if __name__ == "__main__":
    main()
