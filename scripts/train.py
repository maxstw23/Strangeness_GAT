import sys
import os
import copy
import glob
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import config
from transformer_model import OmegaTransformer, OmegaTransformerEdge
from tqdm import tqdm

EMA_DECAY = 0.999  # EMA shadow model decay for stable checkpointing


def omega_rec_at_anti_target(p_scores, is_anti, is_omega, target=0.90):
    """Omega recall at the threshold where Anti recall is closest to target.

    Returns (omega_rec, actual_anti_rec, threshold).
    """
    thresholds = torch.linspace(0.01, 0.99, 980)
    best_t, best_diff = thresholds[0], float('inf')
    for t in thresholds:
        diff = abs((p_scores[is_anti] >= t).float().mean().item() - target)
        if diff < best_diff:
            best_diff, best_t = diff, t
    t = best_t.item()
    a_rec = (p_scores[is_anti] >= t).float().mean().item()
    o_rec = (p_scores[is_omega] < t).float().mean().item()
    return o_rec, a_rec, t

# ── Optional CLI feature override ─────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", choices=["padded", "unpadded"], default="padded")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature names to override config.FEATURE_NAMES")
    parser.add_argument("--target-anti-rec", type=float, default=0.90,
                        help="Anti recall target for operating-point metric (default: 0.90)")
    parser.add_argument("--subsample", action="store_true",
                        help="Subsample Omega K+ to Anti-Omega n_kaons distribution (unpadded data)")
    parser.add_argument("--edge-bias", action="store_true",
                        help="Use OmegaTransformerEdge: add pairwise (ΔR, Δy, cosΔφ) attention bias")
    args, _ = parser.parse_known_args()
    if args.data == "unpadded":
        config.DATA_PATH  = config.DATA_PATH_UNPADDED
        config.STATS_PATH = config.STATS_PATH_UNPADDED
    if args.features:
        names = [f.strip() for f in args.features.split(",")]
        config.FEATURE_NAMES = names
        config.FEATURE_IDX = [config.FEATURE_REGISTRY.index(f) for f in names]
        config.IN_CHANNELS = len(names)
        config.KSTAR_IDX = names.index("k_star") if "k_star" in names else None
    return args


def get_next_run_number():
    os.makedirs("logs", exist_ok=True)
    existing = glob.glob("logs/train_run*.log")
    nums = []
    for f in existing:
        stem = os.path.basename(f).replace("train_run", "").replace(".log", "")
        if stem.isdigit():
            nums.append(int(stem))
    return max(nums) + 1 if nums else 1


class KaonDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]  # (x, y)


def make_collate_fn(subsample_dist=None):
    """subsample_dist: list of Anti-Omega n_kaons values to sample from, or None."""
    def collate_fn(batch):
        xs, ys = zip(*batch)
        processed = []
        for x, y in zip(xs, ys):
            if subsample_dist is not None and y.item() == 0:  # Omega event
                k = min(random.choice(subsample_dist), x.shape[0])
                idx = torch.randperm(x.shape[0])[:k]
                x = x[idx]
            processed.append(x)
        max_len = max(x.shape[0] for x in processed)
        n_feat  = processed[0].shape[1]
        padded = torch.zeros(len(processed), max_len, n_feat)
        # mask: True = padding position (ignored by attention)
        mask = torch.ones(len(processed), max_len, dtype=torch.bool)
        for i, x in enumerate(processed):
            n = x.shape[0]
            padded[i, :n] = x
            mask[i, :n] = False
        return padded, torch.stack(ys), mask
    return collate_fn


def run_training(args, target_anti_rec=0.90):
    # Fix seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    run_number = get_next_run_number()
    log_path = f"logs/train_run{run_number}.log"

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    labels = []

    print("Preparing Transformer dataset...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        target = y.squeeze().long()
        labels.append(target.item())

        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds

        dataset.append((x, target))

    labels_t = torch.tensor(labels)
    n_o = (labels_t == 0).sum().item()
    n_a = (labels_t == 1).sum().item()

    # Anti-Omega n_kaons empirical distribution (for subsampling Omega events)
    anti_nk_dist = [x.shape[0] for x, lbl in dataset if lbl.item() == 1]

    log(f"Run {run_number} | features={config.FEATURE_NAMES} | IN_CHANNELS={config.IN_CHANNELS} | D_MODEL={config.D_MODEL}")
    log(f"Dataset: {n_o} Omega, {n_a} Anti-Omega")
    log(f"Loss: per-class-mean BCE | target Anti recall: {target_anti_rec:.2f}")
    log(f"EMA decay: {EMA_DECAY} | subsample={args.subsample}")

    # 80/20 split (seed already fixed above)
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    collate = make_collate_fn(anti_nk_dist if args.subsample else None)
    train_loader = DataLoader(KaonDataset(dataset[:split]), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(KaonDataset(dataset[split:]), batch_size=config.BATCH_SIZE,
                            collate_fn=collate,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    common = dict(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    )
    if args.edge_bias:
        dy_idx   = config.FEATURE_NAMES.index("d_y")
        dphi_idx = config.FEATURE_NAMES.index("d_phi")
        model = OmegaTransformerEdge(**common, dy_idx=dy_idx, dphi_idx=dphi_idx).to(config.DEVICE)
        log(f"Architecture: OmegaTransformerEdge (dy_idx={dy_idx}, dphi_idx={dphi_idx})")
    else:
        model = OmegaTransformer(**common).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    # EMA shadow model — evaluated each epoch for stable checkpointing
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_o_rec_at_target = -1.0
    epochs_no_improvement = 0

    log("Starting training...\n")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for x, y, mask in train_loader:
            x, y, mask = x.to(config.DEVICE), y.to(config.DEVICE), mask.to(config.DEVICE)
            optimizer.zero_grad()
            out = model(x, mask)
            p = torch.softmax(out, dim=1)[:, 1].clamp(1e-7, 1 - 1e-7)
            is_a = (y == 1)
            is_o = (y == 0)

            # Per-class-mean BCE
            loss = (-torch.log(p[is_a]).mean() + -torch.log(1 - p[is_o]).mean())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA shadow model
            with torch.no_grad():
                for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(live_p.data, alpha=1.0 - EMA_DECAY)

        # Evaluate using EMA model for stable metrics
        ema_model.eval()
        all_p_anti, all_y = [], []
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(config.DEVICE), y.to(config.DEVICE), mask.to(config.DEVICE)
                out = ema_model(x, mask)
                p_anti = torch.softmax(out, dim=1)[:, 1]
                all_p_anti.append(p_anti.cpu())
                all_y.append(y.cpu())

        p_anti = torch.cat(all_p_anti)
        labels_val = torch.cat(all_y)
        is_omega = (labels_val == 0)
        is_anti  = (labels_val == 1)

        # Operating-point metric: Omega recall at threshold where Anti recall ≈ target
        o_rec_at_t, a_rec_at_t, t_target = omega_rec_at_anti_target(
            p_anti, is_anti, is_omega, target=target_anti_rec
        )

        # Also log argmax score for reference
        preds = (p_anti >= 0.5).long()
        o_rec_argmax = ((preds == 0) & is_omega).sum().item() / is_omega.sum().item()
        a_rec_argmax = ((preds == 1) & is_anti).sum().item()  / is_anti.sum().item()
        score_argmax  = o_rec_argmax + a_rec_argmax - 1.0

        # Scheduler, checkpoint, and early stopping driven by operating-point metric
        scheduler.step(o_rec_at_t)

        line = (
            f"Epoch {epoch:03d} | "
            f"O@A={target_anti_rec:.2f}: {o_rec_at_t:.3f} (t={t_target:.2f}, A={a_rec_at_t:.3f}) | "
            f"Argmax: O={o_rec_argmax:.3f} A={a_rec_argmax:.3f} Score={score_argmax:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"No-improve: {epochs_no_improvement}/{config.EARLY_STOP_PATIENCE}"
        )
        log(line)

        if o_rec_at_t > best_o_rec_at_target:
            best_o_rec_at_target = o_rec_at_t
            epochs_no_improvement = 0
            torch.save(ema_model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= config.EARLY_STOP_PATIENCE:
            log(f"\nEarly stopping: no improvement for {config.EARLY_STOP_PATIENCE} epochs.")
            break

    log(f"\nBest Omega recall @ Anti={target_anti_rec:.2f}: {best_o_rec_at_target:.4f}  →  saved to {config.MODEL_SAVE_PATH}")
    log(f"Log saved to {log_path}")


if __name__ == "__main__":
    args = _parse_args()
    run_training(args, target_anti_rec=args.target_anti_rec)
