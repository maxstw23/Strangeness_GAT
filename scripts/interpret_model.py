"""
Model interpretation: Analysis 1 (attention) and Analysis 2 (feature permutation importance).
See docs/interpretation.md for full description of each analysis.
"""
import sys, os
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

N_PERM_REPEATS = 15

# Plot axis limits per feature (used in attention analysis histograms)
_FEAT_LIMS = {
    "f_pt":           (0.0, 2.5),
    "k_star":         (0.0, 3.5),
    "d_y":            (0.0, 2.5),   # now |d_y|
    "d_phi":          (0.0, 3.2),   # now |d_phi|
    "o_pt":           (0.0, 3.5),
    "cos_theta_star": (-1.0, 1.0),
    "d_y_signed":     (-2.0, 2.0),
    "o_y_abs":        (0.0, 1.5),
}


# ── Shared utilities ──────────────────────────────────────────────────────────

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


def load_val_set():
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
    return dataset[split:], {'means': feat_means, 'stds': feat_stds}


def load_model():
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
    return model


def forward_with_attn(model, x, padding_mask):
    """Forward pass that also returns per-layer CLS attention weights (averaged over heads)."""
    batch_size = x.shape[0]
    h = model.input_proj(x)
    cls = model.cls_token.expand(batch_size, -1, -1)
    h = torch.cat([cls, h], dim=1)

    if padding_mask is not None:
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)
    else:
        full_mask = None

    attn_per_layer = []
    for layer in model.transformer.layers:
        # Pre-LN self-attention with weight extraction
        h_ln = layer.norm1(h)
        attn_out, attn_w = layer.self_attn(
            h_ln, h_ln, h_ln,
            key_padding_mask=full_mask,
            need_weights=True,
            average_attn_weights=True,   # (batch, seq+1, seq+1)
        )
        h = h + layer.dropout1(attn_out)
        # Pre-LN feedforward
        h_ln2 = layer.norm2(h)
        h = h + layer.dropout2(
            layer.linear2(layer.dropout(layer.activation(layer.linear1(h_ln2))))
        )
        attn_per_layer.append(attn_w.detach().cpu())

    logits = model.classifier(h[:, 0])
    return logits, attn_per_layer


def get_predictions(model, val_set):
    loader = DataLoader(val_set, batch_size=256, collate_fn=collate_fn)
    all_p, all_y = [], []
    with torch.no_grad():
        for x, y, mask in loader:
            x, mask = x.to(config.DEVICE), mask.to(config.DEVICE)
            p = torch.softmax(model(x, mask), dim=1)[:, 1]
            all_p.append(p.cpu())
            all_y.append(y)
    return torch.cat(all_p), torch.cat(all_y)


def best_threshold_score(p_anti, labels):
    """Score at the threshold that maximises Anti+Omega recall - 1."""
    is_omega = (labels == 0)
    is_anti  = (labels == 1)
    best_s, best_t = -1.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        preds = (p_anti >= t).long()
        o_r = ((preds == 0) & is_omega).sum().item() / is_omega.sum().item()
        a_r = ((preds == 1) & is_anti).sum().item() / is_anti.sum().item()
        s = o_r + a_r - 1.0
        if s > best_s:
            best_s, best_t = s, t
    return best_s, best_t


# ── Analysis 1: Attention ─────────────────────────────────────────────────────

def run_attention_analysis(model, val_set, stats):
    print("\n" + "="*60)
    print("Analysis 1: Attention")
    print("="*60)

    means = stats['means'].numpy()
    stds  = stats['stds'].numpy()

    loader = DataLoader(val_set, batch_size=64, collate_fn=collate_fn)

    # Per-event records: (attention_weights, unnormalized_kaon_features, label)
    omega_records, anti_records = [], []

    for x_pad, y, mask in loader:
        x_pad_dev = x_pad.to(config.DEVICE)
        mask_dev  = mask.to(config.DEVICE)

        with torch.no_grad():
            _, attn_layers = forward_with_attn(model, x_pad_dev, mask_dev)

        # Final layer CLS→kaon weights: row 0, columns 1: (skip CLS→CLS)
        cls_attn = attn_layers[-1][:, 0, 1:]  # (batch, max_seq)
        mask_cpu = mask.cpu()
        x_cpu    = x_pad.cpu()

        for i in range(len(y)):
            n = (~mask_cpu[i]).sum().item()
            if n == 0:
                continue
            attn_w = cls_attn[i, :n].numpy()                    # (n,)
            feats  = x_cpu[i, :n].numpy() * stds + means        # (n, 6) unnormalized

            record = (attn_w, feats)
            if y[i].item() == 0:
                omega_records.append(record)
            else:
                anti_records.append(record)

    print(f"  Events — Omega: {len(omega_records)}, Anti: {len(anti_records)}")

    def attn_weighted_mean(records):
        """Per-event attention-weighted mean of each feature."""
        out = []
        for attn, feats in records:
            w = attn / (attn.sum() + 1e-10)
            out.append((w[:, None] * feats).sum(axis=0))
        return np.array(out)   # (n_events, 6)

    def top_kaon_feats(records):
        """Features of the single most-attended kaon per event."""
        return np.array([feats[np.argmax(attn)] for attn, feats in records])

    o_aw  = attn_weighted_mean(omega_records)
    a_aw  = attn_weighted_mean(anti_records)
    o_top = top_kaon_feats(omega_records)
    a_top = top_kaon_feats(anti_records)

    # ── Numerical summary ────────────────────────────────────────────────────
    print(f"\n  {'Feature':<16}  {'Attn-wtd Omega':>15}  {'Attn-wtd Anti':>14}  {'Diff':>8}")
    print("  " + "-"*60)
    for fi, name in enumerate(config.FEATURE_NAMES):
        om, am = o_aw[:, fi].mean(), a_aw[:, fi].mean()
        print(f"  {name:<16}  {om:>15.4f}  {am:>14.4f}  {am-om:>+8.4f}")

    print(f"\n  {'Feature':<16}  {'Top-kaon Omega':>15}  {'Top-kaon Anti':>14}  {'Diff':>8}")
    print("  " + "-"*60)
    for fi, name in enumerate(config.FEATURE_NAMES):
        om, am = o_top[:, fi].mean(), a_top[:, fi].mean()
        print(f"  {name:<16}  {om:>15.4f}  {am:>14.4f}  {am-om:>+8.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    feat_lims = [_FEAT_LIMS.get(f, (-3.0, 3.0)) for f in config.FEATURE_NAMES]
    n_feat = len(config.FEATURE_NAMES)
    fig, axes = plt.subplots(2, n_feat, figsize=(4 * n_feat + 2, 7))
    fig.suptitle("Attention Analysis  |  top row: attn-weighted feature / event  |  "
                 "bottom row: most-attended kaon feature / event", fontsize=10)

    for fi, (name, (lo, hi)) in enumerate(zip(config.FEATURE_NAMES, feat_lims)):
        bins = np.linspace(lo, hi, 50)
        for row, (o_arr, a_arr, subtitle) in enumerate([
            (o_aw[:, fi],  a_aw[:, fi],  f"Attn-wtd {name}"),
            (o_top[:, fi], a_top[:, fi], f"Top-kaon {name}"),
        ]):
            ax = axes[row, fi]
            ax.hist(o_arr, bins=bins, density=True, alpha=0.55, color='steelblue', label='Omega')
            ax.hist(a_arr, bins=bins, density=True, alpha=0.55, color='tomato',   label='Anti')
            ax.set_title(subtitle, fontsize=8)
            ax.grid(True, alpha=0.3)
            if fi == 0 and row == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    path = "plots/attention_analysis.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Saved {path}")


# ── Analysis 2: Feature Permutation Importance ────────────────────────────────

def permute_feature_globally(val_set, fi):
    """Return a copy of val_set with feature fi permuted across all events and kaons."""
    # Collect all values for this feature globally
    all_vals = np.concatenate([x[:, fi].numpy() for x, y in val_set])
    np.random.shuffle(all_vals)

    perm_set = []
    idx = 0
    for (x, y) in val_set:
        n = x.shape[0]
        x_perm = x.clone()
        x_perm[:, fi] = torch.tensor(all_vals[idx:idx + n], dtype=torch.float)
        idx += n
        perm_set.append((x_perm, y))
    return perm_set


def run_permutation_importance(model, val_set, n_repeats=N_PERM_REPEATS):
    print("\n" + "="*60)
    print("Analysis 2: Feature Permutation Importance")
    print("="*60)

    p_base, labels = get_predictions(model, val_set)
    base_score, best_t = best_threshold_score(p_base, labels)
    print(f"  Baseline score: {base_score:.4f}  (optimal threshold t={best_t:.3f})")
    print(f"\n  {'Feature':<8}  {'Mean drop':>10}  {'Std':>8}  {'Relative %':>11}")
    print("  " + "-"*45)

    importance = {}
    for fi, fname in enumerate(config.FEATURE_NAMES):
        drops = []
        for _ in range(n_repeats):
            perm_set   = permute_feature_globally(val_set, fi)
            p_perm, _  = get_predictions(model, perm_set)
            perm_score = best_threshold_score(p_perm, labels)[0]
            drops.append(base_score - perm_score)
        mu, sd = np.mean(drops), np.std(drops)
        importance[fname] = (mu, sd)
        rel = 100 * mu / base_score if base_score > 0 else 0
        print(f"  {fname:<8}  {mu:>+10.4f}  {sd:>8.4f}  {rel:>10.1f}%")

    # ── Plot ─────────────────────────────────────────────────────────────────
    order  = sorted(importance, key=lambda k: importance[k][0], reverse=True)
    means  = [importance[k][0] for k in order]
    sds    = [importance[k][1] for k in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(order))
    bars = ax.barh(y_pos, means, xerr=sds, color='steelblue', alpha=0.8, capsize=5)
    # colour negative bars differently
    for bar, m in zip(bars, means):
        bar.set_color('tomato' if m < 0 else 'steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Score drop  (baseline − permuted)', fontsize=9)
    ax.set_title(f'Feature Permutation Importance  (n={n_repeats} repeats per feature)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    path = "plots/feature_importance.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Saved {path}")

    return importance


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("plots", exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    model           = load_model()
    val_set, stats  = load_val_set()

    run_attention_analysis(model, val_set, stats)
    run_permutation_importance(model, val_set)


if __name__ == "__main__":
    main()
