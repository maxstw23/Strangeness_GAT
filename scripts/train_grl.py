"""train_grl.py — Gradient Reversal Layer (GRL) adversarial training.

Approach: DANN-style adversarial debiasing.
  Main task : classify Anti (pair-produced) vs Omega (unlabeled mixture)
              using per-class-mean BCE — identical to train.py
  Adversary : predict log(n_kaons) from gradient-reversed CLS embedding
              forces encoder to be invariant to kaon multiplicity

Why multiplicity?  The charge-balancing procedure pads K⁻ in Ω̄⁺ events to
match K⁺ count.  Any residual (n_kaons, label) correlation is a bias artifact
rather than physics.  The GRL removes this potential shortcut.

Alpha schedule: linearly ramp 0 → GRL_ALPHA_MAX over GRL_WARMUP_EPOCHS, then
hold.  This lets the classifier converge before the adversary destabilises the
encoder.

Loss:
    L = L_cls + λ_adv * L_adv
    L_cls = BCE per-class-mean (as in train.py)
    L_adv = MSE(adv_pred, log(n_kaons))   [adversary maximises; encoder minimises]

Checkpoint: same as train.py — best argmax score (Anti_recall + Omega_recall − 1).
"""
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
from adversarial_model import OmegaTransformerGRL
from tqdm import tqdm

# ── GRL hyperparameters ──────────────────────────────────────────────────────
GRL_ALPHA_MAX    = 1.0    # max GRL reversal strength
GRL_WARMUP_EPOCHS = 30    # epochs to ramp alpha from 0 → GRL_ALPHA_MAX
ADV_LAMBDA       = 0.1    # weight of adversary loss
CHECKPOINT_THRESHOLD = 0.55
EMA_DECAY        = 0.999
GRL_SAVE_PATH    = "models/omega_grl.pth"


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
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat  = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask   = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    # also return actual sequence lengths for adversary target
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.float)
    return padded, torch.stack(ys), mask, lengths


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
    dataset, labels = [], []

    print("Preparing GRL dataset...")
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

    log(f"GRL Run {run_number} | features={config.FEATURE_NAMES} | IN_CHANNELS={config.IN_CHANNELS}")
    log(f"Dataset: {n_o} Omega, {n_a} Anti-Omega")
    log(f"ADV_LAMBDA={ADV_LAMBDA} | GRL_ALPHA_MAX={GRL_ALPHA_MAX} | warmup={GRL_WARMUP_EPOCHS} epochs")

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
    ).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"OmegaTransformerGRL: {n_params:,} parameters")

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_score = -1.0
    best_anti_rec = -1.0
    best_a_rec_055 = -1.0
    epochs_no_improvement = 0

    log("Starting GRL training...\n")
    for epoch in range(1, config.EPOCHS + 1):
        # Linear alpha ramp: 0 → GRL_ALPHA_MAX over first GRL_WARMUP_EPOCHS
        alpha = min(GRL_ALPHA_MAX, GRL_ALPHA_MAX * epoch / GRL_WARMUP_EPOCHS)

        model.train()
        epoch_cls_loss, epoch_adv_loss, n_steps = 0., 0., 0
        for x, y, mask, lengths in train_loader:
            x, y, mask = x.to(config.DEVICE), y.to(config.DEVICE), mask.to(config.DEVICE)
            log_n = torch.log(lengths.to(config.DEVICE).clamp(min=1))

            optimizer.zero_grad()
            logits, adv_pred = model(x, mask, alpha=alpha)

            p = torch.softmax(logits, dim=1)[:, 1].clamp(1e-7, 1 - 1e-7)
            is_a = (y == 1)
            is_o = (y == 0)

            # Main classification loss (per-class-mean BCE)
            loss_cls = -torch.log(p[is_a]).mean() + -torch.log(1 - p[is_o]).mean()

            # Adversary loss: MSE for predicting log(n_kaons)
            # Gradient reversal already handled in model.forward()
            loss_adv = nn.functional.mse_loss(adv_pred, log_n)

            loss = loss_cls + ADV_LAMBDA * loss_adv
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
            for x, y, mask, _ in val_loader:
                x, mask = x.to(config.DEVICE), mask.to(config.DEVICE)
                logits, _ = ema_model(x, mask, alpha=0.0)
                p_anti = torch.softmax(logits, dim=1)[:, 1]
                all_p_anti.append(p_anti.cpu())
                all_y.append(y)

        p_anti     = torch.cat(all_p_anti)
        labels_val = torch.cat(all_y)
        is_omega   = (labels_val == 0)
        is_anti    = (labels_val == 1)

        preds_argmax = (p_anti >= 0.5).long()
        o_rec = ((preds_argmax == 0) & is_omega).sum().item() / is_omega.sum().item()
        a_rec = ((preds_argmax == 1) & is_anti).sum().item()  / is_anti.sum().item()
        score = o_rec + a_rec - 1.0

        preds_055  = (p_anti >= CHECKPOINT_THRESHOLD).long()
        a_rec_055  = ((preds_055 == 1) & is_anti).sum().item() / is_anti.sum().item()

        scheduler.step(score)

        line = (
            f"Epoch {epoch:03d} | α={alpha:.3f} | "
            f"cls={epoch_cls_loss/n_steps:.4f} adv={epoch_adv_loss/n_steps:.4f} | "
            f"Omega Rec: {o_rec:.3f} | Anti Rec: {a_rec:.3f} | "
            f"A@{CHECKPOINT_THRESHOLD:.2f}: {a_rec_055:.3f} | Score: {score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"No-improve: {epochs_no_improvement}/{config.EARLY_STOP_PATIENCE}"
        )
        log(line)

        if score > best_score:
            best_score    = score
            best_anti_rec = a_rec
            best_a_rec_055 = a_rec_055
            epochs_no_improvement = 0
            torch.save(ema_model.state_dict(), GRL_SAVE_PATH)
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= config.EARLY_STOP_PATIENCE:
            log(f"\nEarly stopping: no improvement for {config.EARLY_STOP_PATIENCE} epochs.")
            break

    summary = (
        f"\nBest Score: {best_score:.4f} | Anti (argmax): {best_anti_rec:.4f} | "
        f"A@{CHECKPOINT_THRESHOLD:.2f}: {best_a_rec_055:.4f}  →  saved to {GRL_SAVE_PATH}"
    )
    log(summary)
    log(f"Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["padded", "unpadded"], default="padded",
                        help="padded: balanced dataset (default); unpadded: no K⁻ padding, directed rapidity features")
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated feature names from FEATURE_REGISTRY (overrides config.FEATURE_NAMES)")
    run_training(parser.parse_args())
