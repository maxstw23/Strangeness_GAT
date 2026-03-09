"""analyze_subpopulations.py — Compare junction-like vs pair-produced Omega subpopulations.

Uses the current model checkpoint to score all Omega events, then splits them
into subpopulations by score percentile and compares k*, |y_K−y_Ω|, and
|y_K|−|y_Ω| distributions.

Subpopulations:
  - Junction-like Ω⁻: bottom 30% of p_anti (low P(pair-produced))
  - Pair-produced Ω⁻: top 50% of p_anti (high P(pair-produced))
  - Ω̄⁺ (Anti): all events — pure pair-produced reference
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import config
from transformer_model import OmegaTransformer
from tqdm import tqdm

JUNCTION_PERCENTILE    = 30   # bottom X% of Omega scores → junction-like
PAIREDPROD_PERCENTILE  = 50   # top (100-X)% of Omega scores → pair-produced-like


class KaonDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch):
    xs, ys, raw_xs = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask   = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    return padded, torch.stack(ys), mask, list(raw_xs)


def main():
    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    model = OmegaTransformer(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    stats = torch.load(config.STATS_PATH)
    means = stats['means'][config.FEATURE_IDX]
    stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    print("Preparing dataset...")
    for entry in tqdm(raw_data):
        raw_x = entry['x']          # unnormalized, all stored features
        y = entry['y'].squeeze().long()
        x = raw_x[:, config.FEATURE_IDX].clone()
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - means) / stds
        dataset.append((x, y, raw_x))

    loader = DataLoader(KaonDataset(dataset), batch_size=256,
                        collate_fn=collate_fn, num_workers=0)

    all_p, all_y, all_raw = [], [], []
    print("Scoring events...")
    with torch.no_grad():
        for x, y, mask, raw_xs in loader:
            x, mask = x.to(config.DEVICE), mask.to(config.DEVICE)
            p = torch.softmax(model(x, mask), dim=1)[:, 1].cpu()
            all_p.append(p); all_y.append(y); all_raw.extend(raw_xs)

    p_all = torch.cat(all_p)
    y_all = torch.cat(all_y)

    is_omega = (y_all == 0)
    is_anti  = (y_all == 1)

    p_omega = p_all[is_omega]
    lo = torch.quantile(p_omega, JUNCTION_PERCENTILE / 100.0).item()
    hi = torch.quantile(p_omega, PAIREDPROD_PERCENTILE / 100.0).item()

    # Collect raw feature values per group (per kaon)
    def gather(mask_events, score_mask=None):
        kstar, dy_abs, dy_signed = [], [], []
        idxs = torch.where(mask_events)[0]
        for idx in idxs:
            if score_mask is not None and not score_mask[idx]:
                continue
            raw = all_raw[idx.item()]       # shape (n_kaons, n_stored_features)
            kstar.append(raw[:, 1])         # k*
            dy_abs.append(raw[:, 2])        # |y_K − y_Ω|
            if raw.shape[1] > 10:
                dy_signed.append(raw[:, 10])  # |y_K| − |y_Ω|
        return (torch.cat(kstar).numpy(),
                torch.cat(dy_abs).numpy(),
                torch.cat(dy_signed).numpy() if dy_signed else None)

    junction_mask    = is_omega & (p_all < lo)
    pairedprod_mask  = is_omega & (p_all > hi)

    print(f"Junction-like (bottom {JUNCTION_PERCENTILE}%): {junction_mask.sum()} events")
    print(f"Pair-produced Ω⁻ (top {100-PAIREDPROD_PERCENTILE}%): {pairedprod_mask.sum()} events")
    print(f"Ω̄⁺ (Anti): {is_anti.sum()} events")

    junc  = gather(is_omega, junction_mask)
    pprod = gather(is_omega, pairedprod_mask)
    anti  = gather(is_anti)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels  = [f'Junction-like Ω⁻ (bottom {JUNCTION_PERCENTILE}%)',
                f'Pair-produced Ω⁻ (top {100-PAIREDPROD_PERCENTILE}%)',
                'Ω̄⁺ (reference)']
    colors  = ['#d62728', '#1f77b4', '#2ca02c']
    groups  = [junc, pprod, anti]

    configs = [
        (0, 'k*',          np.linspace(0, 4, 60),   'k* [GeV/c]'),
        (1, '|y_K − y_Ω|', np.linspace(0, 3, 60),   '|y_K − y_Ω|'),
        (2, '|y_K| − |y_Ω|', np.linspace(-2, 2, 60),'|y_K| − |y_Ω|'),
    ]

    for ax, (feat_i, title, bins, xlabel) in zip(axes, configs):
        for (grp, label, color) in zip(groups, labels, colors):
            vals = grp[feat_i]
            if vals is None:
                continue
            vals = np.clip(vals, bins[0], bins[-1])
            counts, edges = np.histogram(vals, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            norm = counts.sum() * (edges[1] - edges[0])
            ax.step(centers, counts / norm, where='mid', label=label, color=color, linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('dN/d(feature) [a.u.]', fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Subpopulation comparison: junction-like vs pair-produced Ω⁻', fontsize=14)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out = 'plots/subpopulation_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
