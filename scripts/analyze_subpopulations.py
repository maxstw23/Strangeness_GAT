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

    # ── Optimal cutoff analysis ────────────────────────────────────────────────
    pi = is_anti.float().sum().item() / is_omega.float().sum().item()
    print(f"\nπ = n_Anti/n_Omega = {pi:.4f}")
    opt_bn, opt_pp = find_optimal_cutoffs(
        p_omega.numpy(), p_all[is_anti].numpy(), pi)
    # Override percentiles with optimal values
    opt_junc_pct  = opt_bn[0] * 100
    opt_pprod_pct = opt_pp[0] * 100
    lo = float(opt_bn[1])
    hi = float(opt_pp[1])
    junction_mask   = is_omega & (p_all < lo)
    pairedprod_mask = is_omega & (p_all >= hi)

    print(f"\nJunction-like (bottom {opt_junc_pct:.1f}%, BN purity={opt_bn[2]:.3f}): {junction_mask.sum()} events")
    print(f"Pair-produced Ω⁻ (top {opt_pprod_pct:.1f}%, PP purity={min(opt_pp[2],1.0):.3f}): {pairedprod_mask.sum()} events")
    print(f"Ω̄⁺ (Anti): {is_anti.sum()} events")

    junc  = gather(is_omega, junction_mask)
    pprod = gather(is_omega, pairedprod_mask)
    anti  = gather(is_anti)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels  = [f'Junction-like Ω⁻ (bottom {opt_junc_pct:.0f}%, BN pur={opt_bn[2]:.2f})',
                f'Pair-produced Ω⁻ (top {opt_pprod_pct:.0f}%, PP pur={min(opt_pp[2],1.0):.2f})',
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


def find_optimal_cutoffs(p_omega, p_anti, pi, min_frac=0.05, max_frac=0.65):
    """Sweep percentile cutoffs and find the point that maximises purity * sqrt(N).

    For the BN-enriched (junction) sample: bottom X% of Omega scores.
    For the PP-enriched sample: top X% of Omega scores.

    Purity formulas (PU mixture model):
        PP purity(S) = π * f_A(S) / f_O(S)
        BN purity(S) = [f_O(S) − π * f_A(S)] / f_O(S)
    where f_X(S) = fraction of class X whose score falls in S.

    Figure of merit: purity * sqrt(N_events) — balances purity gain against
    statistical loss as the cut becomes stricter.
    """
    n_O = len(p_omega)
    n_A = len(p_anti)
    fracs = np.linspace(min_frac, max_frac, 200)

    results = {'bn': [], 'pp': []}
    for frac in fracs:
        # BN-enriched: bottom frac of Omega scores
        t_bn = np.quantile(p_omega, frac)
        f_O_bn = (p_omega < t_bn).mean()
        f_A_bn = (p_anti  < t_bn).mean()
        bn_pur = (f_O_bn - pi * f_A_bn) / f_O_bn if f_O_bn > 0 else 0.0
        n_bn   = int(f_O_bn * n_O)
        fom_bn = bn_pur * np.sqrt(n_bn) if bn_pur > 0 else 0.0
        results['bn'].append((frac, t_bn, bn_pur, n_bn, fom_bn))

        # PP-enriched: top frac of Omega scores
        t_pp = np.quantile(p_omega, 1.0 - frac)
        f_O_pp = (p_omega >= t_pp).mean()
        f_A_pp = (p_anti  >= t_pp).mean()
        pp_pur = pi * f_A_pp / f_O_pp if f_O_pp > 0 else 0.0
        pp_pur = min(pp_pur, 1.0)          # cap at 1 (saturation regime)
        n_pp   = int(f_O_pp * n_O)
        fom_pp = pp_pur * np.sqrt(n_pp) if pp_pur > 0 else 0.0
        results['pp'].append((frac, t_pp, pp_pur, n_pp, fom_pp))

    # Find optima
    bn_arr = np.array(results['bn'])
    pp_arr = np.array(results['pp'])
    opt_bn = bn_arr[np.argmax(bn_arr[:, 4])]
    opt_pp = pp_arr[np.argmax(pp_arr[:, 4])]

    print("\n── Optimal cutoffs (max purity × √N) ──────────────────")
    print(f"BN-enriched  : bottom {opt_bn[0]*100:.1f}%  "
          f"(t={opt_bn[1]:.3f})  purity={opt_bn[2]:.3f}  N={int(opt_bn[3])}  FOM={opt_bn[4]:.1f}")
    print(f"PP-enriched  : top    {opt_pp[0]*100:.1f}%  "
          f"(t={opt_pp[1]:.3f})  purity={opt_pp[2]:.3f}  N={int(opt_pp[3])}  FOM={opt_pp[4]:.1f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, color, label in [
        (axes[0], 'bn', '#d62728', 'BN-enriched (junction)'),
        (axes[0], 'pp', '#1f77b4', 'PP-enriched'),
    ]:
        arr = np.array(results[key])
        ax.plot(arr[:, 0] * 100, arr[:, 2], color=color, label=label)
    axes[0].set_xlabel("Percentile cut (%)")
    axes[0].set_ylabel("Mechanism purity")
    axes[0].set_title("Purity vs cut")
    axes[0].axvline(opt_bn[0]*100, color='#d62728', ls='--', alpha=0.6)
    axes[0].axvline(opt_pp[0]*100, color='#1f77b4', ls='--', alpha=0.6)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    for ax, key, color in [(axes[1], 'bn', '#d62728'), (axes[1], 'pp', '#1f77b4')]:
        arr = np.array(results[key])
        ax.plot(arr[:, 0] * 100, arr[:, 3], color=color)
    axes[1].set_xlabel("Percentile cut (%)")
    axes[1].set_ylabel("N events")
    axes[1].set_title("Statistics vs cut")
    axes[1].axvline(opt_bn[0]*100, color='#d62728', ls='--', alpha=0.6)
    axes[1].axvline(opt_pp[0]*100, color='#1f77b4', ls='--', alpha=0.6)
    axes[1].grid(True, alpha=0.3)

    for ax, key, color in [(axes[2], 'bn', '#d62728'), (axes[2], 'pp', '#1f77b4')]:
        arr = np.array(results[key])
        norm = arr[:, 4].max()
        ax.plot(arr[:, 0] * 100, arr[:, 4] / norm, color=color)
    axes[2].set_xlabel("Percentile cut (%)")
    axes[2].set_ylabel("FOM = purity × √N  (normalised)")
    axes[2].set_title("Figure of merit vs cut")
    axes[2].axvline(opt_bn[0]*100, color='#d62728', ls='--', alpha=0.6,
                    label=f"BN opt={opt_bn[0]*100:.0f}%")
    axes[2].axvline(opt_pp[0]*100, color='#1f77b4', ls='--', alpha=0.6,
                    label=f"PP opt={opt_pp[0]*100:.0f}%")
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Cutoff optimisation: purity × √N figure of merit", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/cutoff_optimisation.png", dpi=150, bbox_inches='tight')
    print("Saved → plots/cutoff_optimisation.png")

    return opt_bn, opt_pp


if __name__ == "__main__":
    main()
