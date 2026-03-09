import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import config
from transformer_model import OmegaTransformer

OUT_PATH = "plots/recall_tradeoff.png"


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    return padded, torch.stack(ys), mask


def main():
    os.makedirs("plots", exist_ok=True)
    device = config.DEVICE

    model = OmegaTransformer(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    stats = torch.load(config.STATS_PATH)
    feat_means = stats['means'][config.FEATURE_IDX]
    feat_stds  = stats['stds'][config.FEATURE_IDX]
    raw_data = torch.load(config.DATA_PATH)

    dataset = []
    for entry in raw_data:
        x, y = entry['x'], entry['y']
        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feat_means) / feat_stds
        dataset.append((x, y.squeeze().long()))

    random.seed(42)
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    val_loader = DataLoader(dataset[split:], batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

    all_p, all_y = [], []
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, mask = x.to(device), mask.to(device)
            p = torch.softmax(model(x, mask), dim=1)[:, 1]
            all_p.append(p.cpu())
            all_y.append(y)

    p_anti = torch.cat(all_p).numpy()
    labels = torch.cat(all_y).numpy()
    is_omega = labels == 0
    is_anti = labels == 1
    n_omega = is_omega.sum()
    n_anti = is_anti.sum()
    print(f"Val set: {n_omega} Omega, {n_anti} Anti")

    # Dense threshold sweep
    thresholds = np.linspace(0.0, 1.0, 1000)
    a_recs, o_recs, scores = [], [], []
    for t in thresholds:
        preds = (p_anti >= t).astype(int)
        a_r = ((preds == 1) & is_anti).sum() / n_anti
        o_r = ((preds == 0) & is_omega).sum() / n_omega
        a_recs.append(a_r)
        o_recs.append(o_r)
        scores.append(a_r + o_r - 1.0)

    a_recs = np.array(a_recs)
    o_recs = np.array(o_recs)
    scores = np.array(scores)

    best_idx = np.argmax(scores)
    print(f"Best score {scores[best_idx]:.4f} at t={thresholds[best_idx]:.3f}: "
          f"Anti={a_recs[best_idx]:.4f}, Omega={o_recs[best_idx]:.4f}")

    # Physics operating point: threshold closest to Anti recall = 0.90
    TARGET_ANTI_REC = 0.90
    op_idx = np.argmin(np.abs(a_recs - TARGET_ANTI_REC))
    print(f"Operating point (Anti≈{TARGET_ANTI_REC}) at t={thresholds[op_idx]:.3f}: "
          f"Anti={a_recs[op_idx]:.4f}, Omega={o_recs[op_idx]:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Tradeoff curve: Anti recall vs Omega recall
    ax = axes[0]
    sc = ax.scatter(a_recs, o_recs, c=thresholds, cmap='viridis', s=4, alpha=0.8)
    ax.scatter(a_recs[op_idx], o_recs[op_idx], c='red', s=80, zorder=5,
               label=f'Operating point (Anti≈{TARGET_ANTI_REC}, t={thresholds[op_idx]:.2f})')
    ax.plot([0, 1], [1, 0], 'k--', lw=0.8, alpha=0.3, label='Random classifier')
    plt.colorbar(sc, ax=ax, label='Threshold')
    ax.set_xlabel('Anti Recall')
    ax.set_ylabel('Omega Recall')
    ax.set_title('Recall Tradeoff Curve')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Score vs threshold
    ax = axes[1]
    ax.plot(thresholds, scores, 'b-', lw=1.5)
    ax.axvline(thresholds[op_idx], color='red', ls='--', lw=1,
               label=f'Operating point t={thresholds[op_idx]:.2f}')
    ax.axhline(scores[op_idx], color='red', ls=':', lw=0.8)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score (Anti_rec + Omega_rec - 1)')
    ax.set_title('Score vs Threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. P(Anti) score distributions
    ax = axes[2]
    bins = np.linspace(0, 1, 60)
    ax.hist(p_anti[is_omega], bins=bins, alpha=0.6, density=True, label=f'Omega (n={n_omega})', color='steelblue')
    ax.hist(p_anti[is_anti], bins=bins, alpha=0.6, density=True, label=f'Anti (n={n_anti})', color='tomato')
    ax.axvline(thresholds[op_idx], color='red', ls='--', lw=1,
               label=f'Operating point t={thresholds[op_idx]:.2f}')
    ax.set_xlabel('P(Anti | event)')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
