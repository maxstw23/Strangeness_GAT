"""
mixture_decompose.py — AlphaMax mixture decomposition of Ω⁻ score distribution.

Physics framing:
  f_Omega(p) = (1 - f_BN) * f_pair(p) + f_BN * f_junction(p)
  where f_pair ≈ f_Anti (pair-produced mechanism is same for Ω̄⁺ and pair-produced Ω⁻).

AlphaMax estimate (Blanchard & Lee 2010):
  f_BN* = 1 - min_p [ f_Omega(p) / f_Anti(p) ]

Per-event posterior:
  P(BN-carrying | p(x)) = 1 - (1 - f_BN*) * f_Anti(p(x)) / f_Omega(p(x))

Outputs:
  - Estimated f_BN from AlphaMax
  - Per-event posterior saved to plots/posteriors.pt  (tensor of shape [N_omega_val])
  - Figure: plots/mixture_decompose.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
import random
import config
from transformer_model import OmegaTransformer
from tqdm import tqdm


KDE_BW = 0.05   # KDE bandwidth (Scott's rule tends to over-smooth here; fixed is safer)
GRID_N = 2000   # evaluation grid points
GUARD  = 0.01   # trim edges where KDE estimates are unreliable


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


def score_events():
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
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    print("Scoring events...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds
        dataset.append((x, y.squeeze().long()))

    # Reproduce exact 80/20 val split from training
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

    p = torch.cat(all_p).numpy()
    y = torch.cat(all_y).numpy()
    return p, y


def alphamax(p_omega, p_anti, bw=KDE_BW, n_grid=GRID_N, guard=GUARD):
    """
    AlphaMax estimator for the BN-carrying fraction f_BN.

    Given:  f_Omega = (1 - f_BN) * f_Anti + f_BN * f_junction
    Returns: f_BN = 1 - min_{p in (guard, 1-guard)} [ f_Omega(p) / f_Anti(p) ]

    The minimum is taken on the interior of [0,1] to avoid KDE edge artifacts.
    """
    grid = np.linspace(guard, 1 - guard, n_grid)

    kde_omega = gaussian_kde(p_omega, bw_method=bw)
    kde_anti  = gaussian_kde(p_anti,  bw_method=bw)

    f_omega = kde_omega(grid)
    f_anti  = kde_anti(grid)

    # Avoid division by near-zero Anti density (very low Anti scores are rare)
    safe = f_anti > 1e-6 * f_anti.max()
    ratio = np.where(safe, f_omega / f_anti, np.inf)

    f_bn = float(np.clip(1.0 - ratio.min(), 0.0, 1.0))
    return f_bn, grid, f_omega, f_anti


def posterior(p_scores, f_bn, kde_omega, kde_anti):
    """Per-event P(BN-carrying | p) for a set of Omega event scores."""
    f_o = kde_omega(p_scores)
    f_a = kde_anti(p_scores)
    # P(BN) = 1 - (1 - f_BN) * f_Anti(p) / f_Omega(p), clipped to [0, 1]
    post = 1.0 - (1.0 - f_bn) * f_a / np.maximum(f_o, 1e-9)
    return np.clip(post, 0.0, 1.0)


def plot(grid, f_omega, f_anti, f_bn, p_omega, p_anti, posteriors_omega):
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"AlphaMax mixture decomposition  |  f_BN = {f_bn:.3f}", fontsize=12)

    # --- Panel 1: Score distributions + decomposition ---
    ax = axes[0]
    ax.fill_between(grid, f_anti, alpha=0.35, color="royalblue", label="Anti (pair-produced)")
    ax.plot(grid, f_omega, color="tomato", lw=1.5, label="Omega (mixture)")

    f_pair_component = (1 - f_bn) * f_anti
    f_junc_component = np.clip(f_omega - f_pair_component, 0, None)
    ax.fill_between(grid, f_pair_component, alpha=0.3, color="royalblue",
                    linestyle="--", label=f"Pair-produced in Ω⁻  (×{1-f_bn:.2f})")
    ax.fill_between(grid, f_junc_component, alpha=0.3, color="darkorange",
                    label=f"BN-carrying in Ω⁻  (×{f_bn:.2f})")
    ax.set_xlabel("p(x) = P(pair-produced | kaons)")
    ax.set_ylabel("Density")
    ax.set_title("Score distributions")
    ax.legend(fontsize=7)

    # --- Panel 2: Posterior P(BN-carrying | p) ---
    ax = axes[1]
    kde_omega = gaussian_kde(p_omega, bw_method=KDE_BW)
    kde_anti  = gaussian_kde(p_anti,  bw_method=KDE_BW)
    post_grid = posterior(grid, f_bn, kde_omega, kde_anti)
    ax.plot(grid, post_grid, color="darkorange", lw=2)
    ax.axhline(f_bn, color="gray", ls="--", lw=1, label=f"f_BN = {f_bn:.3f}")
    ax.set_xlabel("p(x) = P(pair-produced | kaons)")
    ax.set_ylabel("P(BN-carrying | p(x))")
    ax.set_title("Per-event posterior")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # --- Panel 3: Posterior distribution over Omega events ---
    ax = axes[2]
    ax.hist(posteriors_omega, bins=50, color="darkorange", alpha=0.7, density=True)
    ax.set_xlabel("P(BN-carrying | p(x))")
    ax.set_ylabel("Density")
    ax.set_title("Posterior over Ω⁻ events")
    ax.axvline(f_bn, color="gray", ls="--", lw=1, label=f"mean ≈ f_BN = {f_bn:.3f}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = "plots/mixture_decompose.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    p, y = score_events()
    p_omega = p[y == 0]
    p_anti  = p[y == 1]
    print(f"Val set: {len(p_omega)} Omega, {len(p_anti)} Anti")

    f_bn, grid, f_omega_grid, f_anti_grid = alphamax(p_omega, p_anti)
    print(f"\nAlphaMax estimate: f_BN = {f_bn:.4f}")

    kde_omega = gaussian_kde(p_omega, bw_method=KDE_BW)
    kde_anti  = gaussian_kde(p_anti,  bw_method=KDE_BW)
    posteriors = posterior(p_omega, f_bn, kde_omega, kde_anti)

    os.makedirs("plots", exist_ok=True)
    torch.save({
        "p_omega": torch.tensor(p_omega, dtype=torch.float32),
        "p_anti":  torch.tensor(p_anti,  dtype=torch.float32),
        "posteriors_omega": torch.tensor(posteriors, dtype=torch.float32),
        "f_bn_alphamax": f_bn,
    }, "plots/posteriors.pt")
    print(f"Saved: plots/posteriors.pt")

    print(f"\nPosterior stats over Ω⁻ events:")
    print(f"  mean  = {posteriors.mean():.4f}  (should ≈ f_BN = {f_bn:.4f})")
    print(f"  median= {np.median(posteriors):.4f}")
    print(f"  P(BN-carrying) > 0.5: {(posteriors > 0.5).mean():.3f} of Ω⁻ events")
    print(f"  P(BN-carrying) > 0.7: {(posteriors > 0.7).mean():.3f} of Ω⁻ events")

    plot(grid, f_omega_grid, f_anti_grid, f_bn, p_omega, p_anti, posteriors)


if __name__ == "__main__":
    main()
