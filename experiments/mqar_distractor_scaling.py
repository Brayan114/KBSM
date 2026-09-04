import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.mqar import MQARDataset
from models.transformer import CausalTransformer
from models.kbsm_model import KBSMModel

def train_eval_mqar(model, train_data, eval_data, device='cpu', num_steps=350, batch_size=32, lr=3e-3):
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

def run_distractor_scaling():
    print('=================================================================')
    print('MQAR DISTRACTOR NOISE SCALING BENCHMARK')
    print('Testing exact recall as distractor noise scales (L=64 to L=512)')
    print('=================================================================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on compute device: {device}')

    seq_lens = [64, 128, 256, 512]
    num_pairs = 4
    vocab_size = 64
    d_model = 64

    results = {
        'seq_lens': seq_lens,
        'Transformer': [],
        'Linear_Recurrent': [],
        'KBSM': []
    }

    for L in seq_lens:
        print(f'\n>>> Benchmark for Sequence Length L={L} (Num Pairs K={num_pairs}) <<<')
        train_data = MQARDataset(num_examples=2000, seq_len=L, num_pairs=num_pairs, vocab_size=vocab_size, seed=42)
        eval_data = MQARDataset(num_examples=500, seq_len=L, num_pairs=num_pairs, vocab_size=vocab_size, seed=123)

        # 1. Causal Transformer
        print('  Training Causal Transformer...')
        torch.manual_seed(42)
        tf_model = CausalTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=128)
        t0 = time.time()
        tf_acc, tf_loss = train_eval_mqar(tf_model, train_data, eval_data, device=device)
        print(f'    Transformer: Acc = {tf_acc:.1f}%, Loss = {tf_loss:.3f} ({time.time()-t0:.1f}s)')
        results['Transformer'].append(tf_acc)

        # 2. Linear Recurrent
        print('  Training Linear Recurrent Baseline...')
        torch.manual_seed(42)
        lin_model = KBSMModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, conv_kernel=1, use_power_kernel=False)
        t0 = time.time()
        lin_acc, lin_loss = train_eval_mqar(lin_model, train_data, eval_data, device=device)
        print(f'    Linear Recurrent: Acc = {lin_acc:.1f}%, Loss = {lin_loss:.3f} ({time.time()-t0:.1f}s)')
        results['Linear_Recurrent'].append(lin_acc)

        # 3. Full KBSM
        print('  Training Full KBSM...')
        torch.manual_seed(42)
        kbsm_model = KBSMModel(vocab_size=vocab_size, d_model=d_model, num_heads=4, head_dim=16, conv_kernel=4, use_power_kernel=True)
        t0 = time.time()
        kbsm_acc, kbsm_loss = train_eval_mqar(kbsm_model, train_data, eval_data, device=device)
        print(f'    Full KBSM: Acc = {kbsm_acc:.1f}%, Loss = {kbsm_loss:.3f} ({time.time()-t0:.1f}s)')
        results['KBSM'].append(kbsm_acc)

    # Save JSON
    os.makedirs('results', exist_ok=True)
    with open('results/mqar_distractor_scaling.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved results to results/mqar_distractor_scaling.json')

    # Plot figure
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(seq_lens, results['KBSM'], 'o-', color='#1f77b4', linewidth=2.8, markersize=8, label='KBSM (O(1) Memory - Ours)')
    plt.plot(seq_lens, results['Transformer'], 's--', color='#ff7f0e', linewidth=2.2, markersize=7, label='Causal Transformer (O(N) KV-Cache)')
    plt.plot(seq_lens, results['Linear_Recurrent'], '^-.', color='#d62728', linewidth=2.0, markersize=7, label='Linear Recurrent Baseline (O(1) Memory)')

    plt.title('Multi-Query Associative Recall vs. Sequence Horizon', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Sequence Length L (Distractor Noise Horizon)', fontsize=11, fontweight='bold')
    plt.ylabel('Exact Recall Accuracy (%)', fontsize=11, fontweight='bold')
    plt.xticks(seq_lens, [f'L={l}\n(~{int(l*0.75)} dist.)' for l in seq_lens], fontsize=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='center right')
    plt.tight_layout()

    out_png = 'results/mqar_distractor_scaling.png'
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f'Saved plot to {out_png}')

    # Copy to paper/
    import shutil
    shutil.copy(out_png, 'paper/mqar_distractor_scaling.png')
    print('Copied plot to paper/mqar_distractor_scaling.png')

if __name__ == '__main__':
    run_distractor_scaling()
