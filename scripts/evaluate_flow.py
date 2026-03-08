"""
Evaluate the trained RealNVP flow on all events (Omega + Anti).

Scoring: log p(x | Anti-flow). Higher = more Anti-like = likely pair-produced.

Threshold sweep (over log p):
  - Anti recall  = fraction of Anti events with log p > t
  - Omega recall = fraction of Omega events with log p < t  (BN-transport candidates)
  - Score = Anti_recall + Omega_recall − 1

Outputs:
  - Threshold sweep table printed to stdout
  - plots/flow_score_dist.png — log p histogram for Omega vs Anti
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
from tqdm import tqdm
import config
from flow_model import RealNVP

FLOW_SAVE_PATH = "models/omega_flow.pth"
PLOT_PATH      = "plots/flow_score_dist.png"
N_THRESHOLDS   = 200


def aggregate_event(x: torch.Tensor) -> torch.Tensor:
    K = x.shape[0]
    mean = x.mean(dim=0)
    std  = x.std(dim=0, unbiased=False)
    mn   = x.min(dim=0).values
    mx   = x.max(dim=0).values
    log_k = torch.tensor([math.log(K)], dtype=x.dtype)
    return torch.cat([mean, std, mn, mx, log_k])


def load_summaries():
    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    summaries, labels = [], []
    for entry in tqdm(raw_data, desc="Aggregating events"):
        x, y = entry['x'], entry['y']
        label = y.squeeze().long().item()
        x = x[:, config.FEATURE_IDX].float()
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds
        summaries.append(aggregate_event(x))
        labels.append(label)
    return torch.stack(summaries), torch.tensor(labels)


def threshold_sweep(log_p, labels, n_thresholds=N_THRESHOLDS):
    """Sweep over log p thresholds and compute recall metrics."""
    is_anti  = (labels == 1)
    is_omega = (labels == 0)
    thresholds = torch.linspace(log_p.min().item(), log_p.max().item(), n_thresholds)

    best_score, best_t, best_a_rec, best_o_rec = -1.0, 0.0, 0.0, 0.0
    rows = []
    for t in thresholds:
        a_rec = ((log_p > t) & is_anti).sum().item()  / is_anti.sum().item()
        o_rec = ((log_p < t) & is_omega).sum().item() / is_omega.sum().item()
        score = a_rec + o_rec - 1.0
        rows.append((t.item(), a_rec, o_rec, score))
        if score > best_score:
            best_score, best_t, best_a_rec, best_o_rec = score, t.item(), a_rec, o_rec

    return rows, best_score, best_t, best_a_rec, best_o_rec


def plot_histogram(log_p_omega, log_p_anti):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(log_p_omega.min().item(), log_p_anti.min().item()),
        max(log_p_omega.max().item(), log_p_anti.max().item()),
        80
    )
    ax.hist(log_p_omega.numpy(), bins=bins, density=True, alpha=0.6,
            color='steelblue', label='Ω⁻ (Omega)')
    ax.hist(log_p_anti.numpy(),  bins=bins, density=True, alpha=0.6,
            color='tomato', label='Ω̄⁺ (Anti)')

    ax.set_xlabel('log p(x | Anti-flow)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Flow log-likelihood: Omega vs Anti', fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Histogram saved to {PLOT_PATH}")


def main():
    # Load checkpoint
    ckpt = torch.load(FLOW_SAVE_PATH, map_location='cpu')
    flow = RealNVP(
        dim=ckpt['flow_dim'],
        n_layers=ckpt['n_layers'],
        hidden=ckpt['hidden'],
    )
    flow.load_state_dict(ckpt['model_state'])
    flow.eval()
    flow_mean = ckpt['flow_mean']
    flow_std  = ckpt['flow_std']

    print(f"Flow checkpoint loaded: dim={ckpt['flow_dim']}, layers={ckpt['n_layers']}")
    print(f"Trained on features: {ckpt['feature_names']}\n")

    # Build summaries for all events
    summaries, labels = load_summaries()

    # Normalise with flow's own stats (computed on Anti train set)
    summaries_norm = (summaries - flow_mean) / flow_std

    # Compute log p in batches
    log_probs = []
    BATCH = 1024
    with torch.no_grad():
        for i in range(0, summaries_norm.shape[0], BATCH):
            batch = summaries_norm[i:i+BATCH]
            log_probs.append(flow.log_prob(batch))
    log_p = torch.cat(log_probs)

    is_anti  = (labels == 1)
    is_omega = (labels == 0)
    log_p_anti  = log_p[is_anti]
    log_p_omega = log_p[is_omega]

    print(f"Omega events: {is_omega.sum().item()} | log p: mean={log_p_omega.mean():.3f}, "
          f"std={log_p_omega.std():.3f}")
    print(f"Anti  events: {is_anti.sum().item()}  | log p: mean={log_p_anti.mean():.3f}, "
          f"std={log_p_anti.std():.3f}")

    # AUROC-style check: fraction of Anti > Omega in log p (should be > 0.5 if flow separates)
    n_check = min(5000, is_anti.sum().item(), is_omega.sum().item())
    rng = torch.Generator().manual_seed(0)
    a_sample = log_p_anti[torch.randperm(log_p_anti.shape[0], generator=rng)[:n_check]]
    o_sample = log_p_omega[torch.randperm(log_p_omega.shape[0], generator=rng)[:n_check]]
    auc_approx = (a_sample.unsqueeze(1) > o_sample.unsqueeze(0)).float().mean().item()
    print(f"Approx AUC (log p Anti > Omega): {auc_approx:.4f}\n")

    # Threshold sweep
    rows, best_score, best_t, best_a_rec, best_o_rec = threshold_sweep(log_p, labels)

    print("Threshold sweep (subset, every 20th):")
    print(f"{'threshold':>12} {'Anti_recall':>12} {'Omega_recall':>13} {'Score':>8}")
    for row in rows[::20]:
        t, a, o, s = row
        print(f"{t:12.3f} {a:12.4f} {o:13.4f} {s:8.4f}")

    print(f"\n{'='*55}")
    print(f"SWEEP OPTIMUM: t={best_t:.3f} | Anti={best_a_rec:.4f} | "
          f"Omega={best_o_rec:.4f} | Score={best_score:.4f}")
    print(f"BCE baseline score: 0.3161")
    delta = best_score - 0.3161
    sign = '+' if delta >= 0 else ''
    print(f"Delta vs BCE: {sign}{delta:.4f}")
    print(f"{'='*55}\n")

    # Anti recall at several fixed Omega recall levels
    print("Anti recall at fixed Omega recall targets:")
    for target_o in [0.50, 0.60, 0.70, 0.80, 0.90]:
        candidates = [(a, o, s) for _, a, o, s in rows if o >= target_o]
        if candidates:
            a_best, o_best, s_best = max(candidates, key=lambda r: r[0])
            print(f"  Omega_recall >= {target_o:.2f}: Anti_recall={a_best:.4f}, Score={s_best:.4f}")

    # Plot
    plot_histogram(log_p_omega, log_p_anti)


if __name__ == "__main__":
    main()
