import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import random
import config
from adversarial_model import DensityRatioNet
from tqdm import tqdm

ADV_SAVE_PATH = "models/omega_adversarial.pth"
BCE_BASELINE  = 0.3161


class KaonDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


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


def evaluate():
    device = config.DEVICE
    print(f"Loading DensityRatioNet: {ADV_SAVE_PATH}")

    model = DensityRatioNet(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(device)
    model.load_state_dict(torch.load(ADV_SAVE_PATH, map_location=device))
    model.eval()

    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    print("Preparing evaluation dataset...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds
        dataset.append((x, y.squeeze().long()))

    # Exact same 80/20 split as training
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    val_loader = DataLoader(
        KaonDataset(dataset[split_idx:]), batch_size=config.BATCH_SIZE, collate_fn=collate_fn
    )

    all_f, all_labels = [], []
    with torch.no_grad():
        for x, y, mask in val_loader:
            f = model(x.to(device), mask.to(device))
            all_f.append(f.cpu())
            all_labels.append(y)

    f_all  = torch.cat(all_f)       # log-density-ratio scores, unconstrained
    labels = torch.cat(all_labels)
    is_omega = (labels == 0)
    is_anti  = (labels == 1)
    n_omega = is_omega.sum().item()
    n_anti  = is_anti.sum().item()
    r_anti_omega = n_anti / n_omega

    print(f"\nScore stats: f_Anti  mean={f_all[is_anti].mean():.3f} std={f_all[is_anti].std():.3f}")
    print(f"             f_Omega mean={f_all[is_omega].mean():.3f} std={f_all[is_omega].std():.3f}")
    print(f"             Δf = {f_all[is_anti].mean() - f_all[is_omega].mean():.3f}")

    # --- Physics operating point: Anti_recall = 0.90 ---
    print("\n" + "=" * 55)
    print("  Physics operating points (fixed Anti recall):")
    print(f"  {'Anti_rec':>8} | {'Threshold':>9} | {'Omega_rec':>9} | {'f_BN':>6}")
    print("  " + "-" * 42)
    for ar_target in [0.80, 0.85, 0.90, 0.95]:
        t_ar = torch.quantile(f_all[is_anti], 1.0 - ar_target).item()
        ar   = (f_all[is_anti]  > t_ar).float().mean().item()
        or_  = (f_all[is_omega] <= t_ar).float().mean().item()
        fbn  = or_ - (1.0 - ar) * r_anti_omega
        print(f"  {ar:>8.3f} | {t_ar:>9.3f} | {or_:>9.4f} | {fbn:>6.4f}")
    print("=" * 55)
    print("  BCE baseline @ Anti≈0.90: Omega_recall ≈ 0.318")

    # --- Threshold scan on f(x) ---
    # Use percentiles of the full distribution as thresholds (f is unconstrained)
    print("\nThreshold scan (percentiles of f distribution):")
    print(f"{'Threshold':>10} | {'Anti Recall':>11} | {'Omega Recall':>12} | {'Score':>7} | {'f_BN':>6}")
    print("-" * 60)
    thresholds_q = torch.linspace(0.05, 0.95, 19).tolist()
    thresholds = [torch.quantile(f_all, q).item() for q in thresholds_q]
    best_score_t, best_row = -1.0, None
    for t, q in zip(thresholds, thresholds_q):
        a_r = (f_all[is_anti]  > t).float().mean().item()
        o_r = (f_all[is_omega] <= t).float().mean().item()
        s   = a_r + o_r - 1.0
        f   = o_r - (1.0 - a_r) * r_anti_omega
        marker = " ← best" if s > best_score_t else ""
        if s > best_score_t:
            best_score_t = s
            best_row = (t, q, a_r, o_r, s, f)
        print(f"{t:>10.3f} | {a_r:>11.4f} | {o_r:>12.4f} | {s:>7.4f} | {f:>6.4f}{marker}")

    # Argmax at median threshold
    median_t = f_all.median().item()
    a_r_med  = (f_all[is_anti]  > median_t).float().mean().item()
    o_r_med  = (f_all[is_omega] <= median_t).float().mean().item()
    argmax_s = a_r_med + o_r_med - 1.0

    print(f"\nMedian threshold: t={median_t:.3f} | Anti={a_r_med:.4f} | Omega={o_r_med:.4f} | Score={argmax_s:.4f}")

    if best_row:
        t, q, a_r, o_r, s, f_bn = best_row
        delta = s - BCE_BASELINE
        sign  = "+" if delta >= 0 else ""
        print(
            f"Physics optimum: t={t:.3f} (q={q:.2f}) | Anti={a_r:.4f} | "
            f"Omega={o_r:.4f} | Score={s:.4f} | f_BN={f_bn:.4f}"
        )
        print(f"vs BCE baseline: {BCE_BASELINE:.4f} | Δ = {sign}{delta:.4f}")

    # --- Score distribution histogram ---
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 60
    ax.hist(f_all[is_omega].numpy(), bins=bins, alpha=0.6, label='Omega (Ω⁻)', density=True, color='steelblue')
    ax.hist(f_all[is_anti].numpy(),  bins=bins, alpha=0.6, label='Anti (Ω̄⁺)',  density=True, color='tomato')
    ax.set_xlabel('NCE score f(x) = log-density-ratio')
    ax.set_ylabel('Density')
    ax.set_title('NCE density-ratio score distribution')
    ax.legend()
    fig.tight_layout()
    out_path = "plots/adv_score_dist.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nHistogram saved to {out_path}")


if __name__ == "__main__":
    evaluate()
