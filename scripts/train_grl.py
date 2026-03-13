"""train_grl.py — Gradient Reversal Layer (GRL) adversarial training.

Approach: DANN-style adversarial debiasing.
  Main task : classify Anti (pair-produced) vs Omega (unlabeled mixture)
              using per-class-mean BCE — identical to train.py
  Adversary : predict log(n_kaons) from gradient-reversed CLS embedding
              forces encoder to be invariant to kaon multiplicity

Why multiplicity?  The charge-balancing procedure pads K⁻ in Ω̄⁺ events to
match K⁺ count.  Any residual (n_kaons, label) correlation is a bias artifact
rather than physics.  The GRL removes this potential shortcut.

Training schedule:
  Phase 1 (GRL_PRETRAIN epochs): classifier trains alone (alpha=0, lambda=0).
    Gives encoder a stable initialisation before gradient reversal starts.
  Phase 2: DANN sigmoidal alpha ramp  alpha = 2/(1+exp(-10p)) − 1  where p
    is progress within phase 2.  Static ADV_LAMBDA throughout phase 2.

Loss:
    L = L_cls + λ * L_adv
    L_cls = BCE per-class-mean (as in train.py)
    L_adv = MSE(adv_pred, log(n_kaons))   [adversary maximises; encoder minimises]
    λ = ADV_LAMBDA (static; 0 during pre-train phase)

Checkpoint: same as train.py — best O@A=0.90 (Omega recall at Anti recall ≥ 0.90).
"""
import sys
import os
import copy
import glob
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import config
from adversarial_model import OmegaTransformerGRL
from tqdm import tqdm

# ── GRL hyperparameters ──────────────────────────────────────────────────────
GRL_ALPHA_MAX    = 1.0    # max GRL reversal strength (sigmoidal schedule)
GRL_PRETRAIN     = 15     # epochs to train classifier alone before enabling GRL
GRL_N_BINS       = 5      # number of within-class n_kaons quantile bins for adversary
# Checkpoint gate: only save when adv_loss exceeds this fraction of chance
# (log(n_bins) = chance).  Ensures saved model satisfies the multiplicity
# constraint.  E.g. 0.95 → adv_loss > 0.95 * log(5) ≈ 1.529 before saving.
GRL_ADV_GATE     = 0.95
EMA_DECAY        = 0.999
GRL_SAVE_PATH    = "models/omega_grl.pth"



def omega_rec_at_anti_target(p_scores, is_anti, is_omega, target=0.90):
    """Omega recall at the threshold where Anti recall is closest to target."""
    thresholds = torch.linspace(0.01, 0.99, 980)
    best_t, best_diff = thresholds[0], float('inf')
    for t in thresholds:
        diff = abs((p_scores[is_anti] >= t).float().mean().item() - target)
        if diff < best_diff:
            best_diff, best_t = diff, t
    t = best_t.item()
    return (p_scores[is_omega] < t).float().mean().item(), \
           (p_scores[is_anti] >= t).float().mean().item(), t


def eval_multiplicity_debiasing(encoder_model, val_loader, device, log, n_bins, n_epochs=40):
    """Freeze the encoder; train a fresh MLP to classify the global n_kaons bin
    (stat index 0) from the CLS embedding.  Report accuracy vs chance.

    acc ≈ 1/n_bins  →  embedding encodes no multiplicity info (fully debiased)
    acc >> 1/n_bins →  multiplicity readable from embedding (insufficient debiasing)
    """
    encoder_model.eval()
    all_cls, all_bins = [], []
    with torch.no_grad():
        for x, y, mask, lengths, bin_labels in val_loader:
            x, mask = x.to(device), mask.to(device)
            cls_emb = encoder_model.encode(x, mask)
            all_cls.append(cls_emb.cpu())
            all_bins.append(bin_labels[:, 0])   # stat 0 = n_kaons global bin

    cls_all  = torch.cat(all_cls)    # [N, d_model]
    bins_all = torch.cat(all_bins)   # [N], int64
    probe_in = cls_all

    d = probe_in.shape[1]
    fresh_adv = nn.Sequential(
        nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n_bins)
    ).to(device)
    opt_adv = torch.optim.Adam(fresh_adv.parameters(), lr=1e-3)

    n = len(probe_in)
    split = int(0.8 * n)
    idx = torch.randperm(n)
    tr_idx, va_idx = idx[:split], idx[split:]
    inp_tr = probe_in[tr_idx].to(device);  bins_tr = bins_all[tr_idx].to(device)
    inp_va = probe_in[va_idx].to(device);  bins_va = bins_all[va_idx].to(device)

    for _ in range(n_epochs):
        fresh_adv.train()
        opt_adv.zero_grad()
        nn.functional.cross_entropy(fresh_adv(inp_tr), bins_tr).backward()
        opt_adv.step()

    fresh_adv.eval()
    with torch.no_grad():
        preds = fresh_adv(inp_va).argmax(dim=1)
        acc = (preds == bins_va).float().mean().item()

    chance = 1.0 / n_bins
    log(f"Multiplicity debiasing probe: acc={acc:.4f} (chance={chance:.4f}) "
        f"→ {'DEBIASED' if acc < chance * 1.5 else 'NOT debiased'}")
    return acc


def get_next_run_number(prefix="grl_run"):
    os.makedirs("logs", exist_ok=True)
    existing = glob.glob(f"logs/{prefix}*.log")
    nums = [
        int(os.path.basename(f).replace(prefix, "").replace(".log", ""))
        for f in existing
        if os.path.basename(f).replace(prefix, "").replace(".log", "").isdigit()
    ]
    return max(nums) + 1 if nums else 1


class KaonDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(batch):
    xs, ys, bin_labels = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat  = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask   = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.float)
    # bin_labels is (n_stats,) per event; stack to (B, n_stats)
    return padded, torch.stack(ys), mask, lengths, torch.stack(bin_labels)


def run_training(args):
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Apply data-source overrides
    if args.data == "unpadded":
        config.DATA_PATH  = config.DATA_PATH_UNPADDED
        config.STATS_PATH = config.STATS_PATH_UNPADDED

    # Apply feature override (comma-separated list of names from FEATURE_REGISTRY)
    if args.features is not None:
        names = [f.strip() for f in args.features.split(",")]
        config.FEATURE_NAMES = names
        config.FEATURE_IDX   = [config.FEATURE_REGISTRY.index(f) for f in names]
        config.IN_CHANNELS   = len(names)
        config.KSTAR_IDX     = names.index("k_star") if "k_star" in names else None

    target_anti_rec = args.target_anti_rec

    global GRL_SAVE_PATH
    if args.data == "unpadded":
        GRL_SAVE_PATH = "models/omega_grl_unpadded.pth"

    log_prefix = "grl_unpadded_run" if args.data == "unpadded" else "grl_run"
    run_number = get_next_run_number(log_prefix)
    log_path = f"logs/{log_prefix}{run_number}.log"

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset, labels, n_kaons_list = [], [], []

    print("Preparing GRL dataset...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        target = y.squeeze().long()
        labels.append(target.item())
        n_kaons_list.append(x.shape[0])
        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds
        dataset.append((x, target))

    labels_t   = torch.tensor(labels)
    n_kaons_t  = torch.tensor(n_kaons_list, dtype=torch.float)
    n_o = (labels_t == 0).sum().item()
    n_a = (labels_t == 1).sum().item()

    # Global quantile bins for every adversary statistic:
    #   stat 0 : n_kaons
    #   stat 1..IN_CHANNELS : std of each feature fed to the classifier
    # Using global bins (not within-class) so the adversary targets the full
    # between-class multiplicity distribution — the artifact on unpadded data.
    import numpy as np

    # Broadcast features are per-event constants (same value for all kaons in an event),
    # so their std across kaons is always 0 — degenerate adversary targets.  Exclude them.
    BROADCAST_FEATURES = {"o_pt", "o_cos_psi1", "o_cos2_psi2", "o_y_abs", "net_kaon"}
    non_broadcast_idx = [i for i, f in enumerate(config.FEATURE_NAMES)
                         if f not in BROADCAST_FEATURES]  # local indices into x columns

    n_stats = 1 + len(non_broadcast_idx)  # n_kaons + std(f_i) for non-broadcast features

    # Build raw statistics matrix: (N, n_stats)
    ns_arr  = n_kaons_t.numpy()                                        # (N,)
    sig_arr = torch.stack([x[:, non_broadcast_idx].std(dim=0).nan_to_num()
                           for x, _ in dataset]).numpy()               # (N, len(non_broadcast_idx))
    stats_arr = np.concatenate([ns_arr[:, None], sig_arr], axis=1)     # (N, n_stats)

    # Compute global quantile boundaries for each statistic
    adv_boundaries = []
    for s in range(n_stats):
        b = np.quantile(stats_arr[:, s], np.linspace(0, 1, GRL_N_BINS + 1)[1:-1])
        adv_boundaries.append(b)

    # Assign global bin labels for each event: shape (N, n_stats)
    all_bin_labels = torch.zeros(len(dataset), n_stats, dtype=torch.long)
    for s in range(n_stats):
        bins = np.digitize(stats_arr[:, s], adv_boundaries[s]).astype(np.int64)
        all_bin_labels[:, s] = torch.from_numpy(bins)

    # Attach all bin labels to dataset tuples (replace single bin_label)
    dataset = [(x, y, all_bin_labels[i]) for i, (x, y) in enumerate(dataset)]

    log(f"GRL Run {run_number} | features={config.FEATURE_NAMES} | IN_CHANNELS={config.IN_CHANNELS} | target Anti recall={target_anti_rec:.2f}")
    log(f"Dataset: {n_o} Omega, {n_a} Anti-Omega")
    log(f"ADV_LAMBDA={args.adv_lambda} | GRL_N_BINS={GRL_N_BINS} (global) | n_stats={n_stats} | non_broadcast={[config.FEATURE_NAMES[i] for i in non_broadcast_idx]} | GRL_ALPHA_MAX={GRL_ALPHA_MAX} | pretrain={GRL_PRETRAIN} | sigmoidal schedule")

    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_loader = DataLoader(
        KaonDataset(dataset[:split]), batch_size=config.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        KaonDataset(dataset[split:]), batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    model = OmegaTransformerGRL(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
        n_bins=GRL_N_BINS,
        n_stats=n_stats,
    ).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"OmegaTransformerGRL: {n_params:,} parameters")

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_o_rec_at_target = -1.0
    epochs_no_improvement = 0
    adv_lambda = args.adv_lambda

    log(f"Pre-train phase: {GRL_PRETRAIN} epochs (alpha=0, classifier only)\n")
    log("Starting GRL training...\n")
    grl_phase_started = False
    for epoch in range(1, config.EPOCHS + 1):
        # Phase 1: pre-train classifier alone (alpha=0, no adversary)
        # Phase 2: DANN sigmoidal alpha from training progress after pre-train
        if epoch <= GRL_PRETRAIN:
            alpha = 0.0
            lambda_eff = 0.0
        else:
            if not grl_phase_started:
                # Reset LR and scheduler state so GRL phase gets a fresh start
                for pg in optimizer.param_groups:
                    pg['lr'] = config.LEARNING_RATE
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=5)
                best_o_rec_at_target = -1.0
                epochs_no_improvement = 0
                grl_phase_started = True
                log(f"\n[Epoch {epoch}] GRL phase start — LR reset to {config.LEARNING_RATE}\n")
            p_prog = (epoch - GRL_PRETRAIN) / (config.EPOCHS - GRL_PRETRAIN)
            alpha = GRL_ALPHA_MAX * (2.0 / (1.0 + math.exp(-10.0 * p_prog)) - 1.0)
            lambda_eff = adv_lambda

        model.train()
        epoch_cls_loss, epoch_adv_loss, n_steps = 0., 0., 0
        for x, y, mask, lengths, bin_labels in train_loader:
            x, y, mask  = x.to(config.DEVICE), y.to(config.DEVICE), mask.to(config.DEVICE)
            bin_labels   = bin_labels.to(config.DEVICE)   # (B, n_stats)

            optimizer.zero_grad()
            logits, adv = model(x, mask, alpha=alpha)     # adv: (B, n_stats, n_bins)

            p = torch.softmax(logits, dim=1)[:, 1].clamp(1e-7, 1 - 1e-7)
            is_a = (y == 1)
            is_o = (y == 0)

            # Main classification loss (per-class-mean BCE)
            loss_cls = -torch.log(p[is_a]).mean() + -torch.log(1 - p[is_o]).mean()

            # Adversary loss: mean CE over all n_stats heads
            # adv[:, s, :] are logits for statistic s; bin_labels[:, s] are targets.
            loss_adv = sum(
                nn.functional.cross_entropy(adv[:, s, :], bin_labels[:, s])
                for s in range(adv.shape[1])
            ) / adv.shape[1]

            loss = loss_cls + lambda_eff * loss_adv
            if torch.isnan(loss):
                continue  # skip NaN batch; EMA model is unaffected
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(live_p.data, alpha=1.0 - EMA_DECAY)

            epoch_cls_loss += loss_cls.item()
            epoch_adv_loss += loss_adv.item()
            n_steps += 1

        # Evaluate using EMA model
        ema_model.eval()
        all_p_anti, all_y = [], []
        with torch.no_grad():
            for x, y, mask, _len, _bins in val_loader:
                x, mask = x.to(config.DEVICE), mask.to(config.DEVICE)
                logits, _ = ema_model(x, mask, alpha=0.0)
                p_anti = torch.softmax(logits, dim=1)[:, 1]
                all_p_anti.append(p_anti.cpu())
                all_y.append(y)

        p_anti     = torch.cat(all_p_anti)
        labels_val = torch.cat(all_y)
        is_omega   = (labels_val == 0)
        is_anti    = (labels_val == 1)

        o_rec_at_t, a_rec_at_t, t_target = omega_rec_at_anti_target(
            p_anti, is_anti, is_omega, target=target_anti_rec
        )

        # Argmax score for reference
        preds = (p_anti >= 0.5).long()
        o_rec = ((preds == 0) & is_omega).sum().item() / is_omega.sum().item()
        a_rec = ((preds == 1) & is_anti).sum().item()  / is_anti.sum().item()
        score = o_rec + a_rec - 1.0

        adv_chance   = math.log(GRL_N_BINS)   # CE chance level for each head
        adv_gate_val = adv_chance * GRL_ADV_GATE
        adv_mean     = epoch_adv_loss / n_steps
        # Constrained when mean CE across all heads is close to chance
        constrained  = grl_phase_started and (adv_mean >= adv_gate_val)

        # Only step scheduler in constrained regime
        if constrained:
            scheduler.step(o_rec_at_t)

        line = (
            f"Epoch {epoch:03d} | α={alpha:.3f} λ={lambda_eff:.4f} | "
            f"cls={epoch_cls_loss/n_steps:.4f} adv={adv_mean:.4f}/{adv_chance:.4f} | "
            f"O@A={target_anti_rec:.2f}: {o_rec_at_t:.3f} (t={t_target:.2f}, A={a_rec_at_t:.3f}) | "
            f"Argmax: O={o_rec:.3f} A={a_rec:.3f} Score={score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"{'[CONSTRAINED] ' if constrained else ''}"
            f"No-improve: {epochs_no_improvement}/{config.EARLY_STOP_PATIENCE}"
        )
        log(line)

        # Save checkpoint only when adversary constraint is satisfied
        # (adv_loss close to chance → encoder not leaking multiplicity info)
        if constrained and o_rec_at_t > best_o_rec_at_target:
            best_o_rec_at_target = o_rec_at_t
            epochs_no_improvement = 0
            torch.save(ema_model.state_dict(), GRL_SAVE_PATH)
        elif not constrained:
            pass  # don't count pre-constraint epochs against no-improve
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= config.EARLY_STOP_PATIENCE:
            log(f"\nEarly stopping: no improvement for {config.EARLY_STOP_PATIENCE} epochs.")
            break

    # Always save the final EMA state (most-debiased for GRL runs)
    final_path = GRL_SAVE_PATH.replace(".pth", "_final.pth")
    torch.save(ema_model.state_dict(), final_path)
    log(f"\nBest Omega recall @ Anti={target_anti_rec:.2f}: {best_o_rec_at_target:.4f}  →  saved to {GRL_SAVE_PATH}")
    log(f"Final (most-debiased) model  →  saved to {final_path}")

    # Debiasing check: how much multiplicity info remains in the final embedding?
    log("\n── Multiplicity debiasing check (fresh adversary probe) ──")
    eval_multiplicity_debiasing(ema_model, val_loader, config.DEVICE, log, GRL_N_BINS)

    log(f"Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["padded", "unpadded"], default="padded",
                        help="padded: balanced dataset (default); unpadded: no K⁻ padding, directed rapidity features")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature names from FEATURE_REGISTRY (overrides config.FEATURE_NAMES)")
    parser.add_argument("--target-anti-rec", type=float, default=0.90,
                        help="Anti recall target for operating-point metric (default: 0.90)")
    parser.add_argument("--adv-lambda", type=float, default=0.1,
                        help="Weight of adversarial loss (default: 0.1). Applied statically "
                             "after the pre-train phase.")
    run_training(parser.parse_args())
