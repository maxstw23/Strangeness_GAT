"""
Train a RealNVP normalizing flow on Anti-Omega events only.

Each event (variable-length kaon set) is aggregated into a 21-dimensional
summary vector:
  - For each of 5 features: mean, std, min, max across kaons → 20 dims
  - log(K) where K = number of kaons → 1 dim
Total: 21 dims

Flow models p(x | Anti). Omega events are never used during training.
At inference, low log p(x) → Omega event looks unlike Anti → BN-transport candidate.
"""
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import config
from flow_model import RealNVP

# ── Flow-specific hyperparameters ─────────────────────────────────────────────
FLOW_DIM        = 21        # 5 features × 4 moments + log(K)
N_LAYERS        = 6
HIDDEN          = 128
BATCH_SIZE      = 256
LR              = 1e-3
EPOCHS          = 200
PATIENCE        = 30        # early stopping on val NLL
LR_PATIENCE     = 10
LR_FACTOR       = 0.5
VAL_FRAC        = 0.2
FLOW_SAVE_PATH  = "models/omega_flow.pth"


def get_next_run_number():
    os.makedirs("logs", exist_ok=True)
    existing = glob.glob("logs/flow_run*.log")
    nums = []
    for f in existing:
        stem = os.path.basename(f).replace("flow_run", "").replace(".log", "")
        if stem.isdigit():
            nums.append(int(stem))
    return max(nums) + 1 if nums else 1


def aggregate_event(x: torch.Tensor) -> torch.Tensor:
    """Aggregate (K, 5) kaon feature matrix into 21-dim summary vector.

    Moments: mean, std, min, max for each of 5 features → 20 dims
    Plus log(K) → 1 dim. Total = 21 dims.
    """
    K = x.shape[0]
    mean = x.mean(dim=0)                          # (5,)
    std  = x.std(dim=0, unbiased=False)           # (5,) — 0 if K==1
    mn   = x.min(dim=0).values                    # (5,)
    mx   = x.max(dim=0).values                    # (5,)
    log_k = torch.tensor([math.log(K)], dtype=x.dtype)
    return torch.cat([mean, std, mn, mx, log_k])  # (21,)


def build_summary_dataset(raw_data, feature_means, feature_stds):
    """Convert raw data list to (summaries, labels) tensors."""
    summaries, labels = [], []
    for entry in tqdm(raw_data, desc="Aggregating events"):
        x, y = entry['x'], entry['y']
        label = y.squeeze().long().item()

        # Select and normalise features (same as train.py)
        x = x[:, config.FEATURE_IDX].float()
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds

        summaries.append(aggregate_event(x))
        labels.append(label)

    return torch.stack(summaries), torch.tensor(labels)


def run_training():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    run_number = get_next_run_number()
    log_path = f"logs/flow_run{run_number}.log"
    os.makedirs("models", exist_ok=True)

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f"Flow Run {run_number}")
    log(f"features={config.FEATURE_NAMES} | FLOW_DIM={FLOW_DIM} | N_LAYERS={N_LAYERS} | HIDDEN={HIDDEN}")
    log(f"BATCH_SIZE={BATCH_SIZE} | LR={LR} | EPOCHS={EPOCHS} | PATIENCE={PATIENCE}")

    # Load data and stats
    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    summaries, labels = build_summary_dataset(raw_data, feature_means, feature_stds)

    # Filter to Anti events only for training the flow
    is_anti = (labels == 1)
    anti_summaries = summaries[is_anti]
    omega_summaries = summaries[~is_anti]  # kept for reference, not used in training

    n_anti = anti_summaries.shape[0]
    n_omega = omega_summaries.shape[0]
    log(f"Dataset: {n_omega} Omega, {n_anti} Anti")
    log(f"Flow training on Anti events only: {n_anti} events")

    # Normalise summary vectors using Anti train stats (z-score)
    n_train_anti = int((1 - VAL_FRAC) * n_anti)
    # Shuffle Anti events
    perm = torch.randperm(n_anti, generator=torch.Generator().manual_seed(42))
    anti_summaries = anti_summaries[perm]

    train_anti = anti_summaries[:n_train_anti]
    val_anti   = anti_summaries[n_train_anti:]

    # Compute normalisation stats from Anti train set
    flow_mean = train_anti.mean(dim=0)
    flow_std  = train_anti.std(dim=0, unbiased=True).clamp(min=1e-6)

    log(f"Anti train: {train_anti.shape[0]} | Anti val: {val_anti.shape[0]}")

    # Normalise
    train_anti_norm = (train_anti - flow_mean) / flow_std
    val_anti_norm   = (val_anti   - flow_mean) / flow_std

    train_loader = DataLoader(
        TensorDataset(train_anti_norm),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_anti_norm),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # Build model
    flow = RealNVP(dim=FLOW_DIM, n_layers=N_LAYERS, hidden=HIDDEN).to(config.DEVICE)
    n_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    log(f"Flow parameters: {n_params:,}")

    optimizer = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE
    )

    best_val_nll = float('inf')
    epochs_no_improvement = 0

    log("\nStarting flow training (NLL loss on Anti events)...\n")
    for epoch in range(1, EPOCHS + 1):
        # Train
        flow.train()
        train_nll = 0.0
        n_train = 0
        for (x,) in train_loader:
            x = x.to(config.DEVICE)
            optimizer.zero_grad()
            loss = -flow.log_prob(x).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            optimizer.step()
            train_nll += loss.item() * x.shape[0]
            n_train += x.shape[0]
        train_nll /= n_train

        # Validate
        flow.eval()
        val_nll = 0.0
        n_val = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(config.DEVICE)
                nll = -flow.log_prob(x).mean()
                val_nll += nll.item() * x.shape[0]
                n_val += x.shape[0]
        val_nll /= n_val

        scheduler.step(val_nll)
        lr_now = optimizer.param_groups[0]['lr']

        improved = val_nll < best_val_nll
        if improved:
            best_val_nll = val_nll
            epochs_no_improvement = 0
            # Save checkpoint with normalisation stats
            torch.save({
                'model_state': flow.state_dict(),
                'flow_mean': flow_mean,
                'flow_std': flow_std,
                'flow_dim': FLOW_DIM,
                'n_layers': N_LAYERS,
                'hidden': HIDDEN,
                'feature_names': config.FEATURE_NAMES,
                'feature_idx': config.FEATURE_IDX,
            }, FLOW_SAVE_PATH)
        else:
            epochs_no_improvement += 1

        marker = " *" if improved else ""
        log(f"Epoch {epoch:3d} | train_nll={train_nll:.4f} | val_nll={val_nll:.4f} "
            f"| lr={lr_now:.2e} | best={best_val_nll:.4f}{marker}")

        if epochs_no_improvement >= PATIENCE:
            log(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    log(f"\nBest val NLL: {best_val_nll:.4f}")
    log(f"Checkpoint saved to {FLOW_SAVE_PATH}")


if __name__ == "__main__":
    run_training()
