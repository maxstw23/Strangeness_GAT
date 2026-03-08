"""
NCE v2: Noise Contrastive Estimation with three targeted fixes.

Problem diagnosed from adv_run4:
  1. Scale explosion: f_Anti std = 15.6 >> f_Omega std = 2.8.
     The 90% Anti threshold fell to -3.024, below the Omega mean (-2.485),
     so most Omega events scored ABOVE the threshold → Omega_recall = 0.253 only.
  2. Unstable log_Z: logsumexp over 128 Omega events dominated by 1-2 outliers.
  3. Wrong checkpoint: optimising symmetric sweep score ≠ physics operating point.

Fixes:
  1. L2 regularisation  λ·(f_anti² + f_omega²).mean()  on f values.
     Compresses the Anti distribution → raises the 90% threshold →
     more Omega events fall below it → higher Omega_recall at Anti_recall=0.9.
  2. Accumulate OMEGA_ACCUM=4 Omega batches per step (512 events for log_Z).
     4× lower logsumexp variance; the 'max(f_omega)' term no longer dominates.
  3. Checkpoint by Omega_recall at Anti_recall=0.9 — the physics operating point.

NCE objective (minimise):
    L = -E_Anti[f] + log E_Omega[exp f]  +  λ·E[(f_anti² + f_omega²)]

At the optimum: f*(x) = log(p_Anti/p_Omega) + C.
Anti & pair-produced Ω⁻ score HIGH; BN-transport Ω⁻ score LOW.
"""
import sys
import os
import copy
import glob
import itertools
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
from torch.utils.data import DataLoader, Dataset
import random
import config
from adversarial_model import DensityRatioNet
from tqdm import tqdm

# --- Hyperparameters ---
BATCH_SIZE    = 128
OMEGA_ACCUM   = 4       # accumulate this many Omega batches per step → 4× larger log_Z denominator
REG_LAMBDA    = 0.01    # L2 penalty on f values — prevents scale explosion
ANTI_RECALL_TARGET = 0.90   # operating point for checkpoint metric
LR            = 5e-4
EPOCHS        = 200
PATIENCE      = 30
EMA_DECAY     = 0.999
ADV_SAVE_PATH = "models/omega_adversarial.pth"


def get_next_run_number():
    os.makedirs("logs", exist_ok=True)
    existing = glob.glob("logs/adv_run*.log")
    nums = [
        int(os.path.basename(f).replace("adv_run", "").replace(".log", ""))
        for f in existing
        if os.path.basename(f).replace("adv_run", "").replace(".log", "").isdigit()
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
    """Collate (x, y) pairs with padding — used for the combined val loader."""
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


def collate_x_only(batch):
    """Collate x tensors only — used for single-class Anti/Omega train loaders."""
    max_len = max(x.shape[0] for x in batch)
    n_feat = batch[0].shape[1]
    padded = torch.zeros(len(batch), max_len, n_feat)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, x in enumerate(batch):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    return padded, mask


def nce_loss_with_reg(f_anti, f_omega, reg_lambda):
    """NCE loss + L2 regularisation.

    L = -E_Anti[f] + log E_Omega[exp f] + λ·E[f²]

    The L2 term prevents scale explosion without changing the ordering of events.
    """
    n_omega = len(f_omega)
    anti_term = -f_anti.mean()
    log_Z     = torch.logsumexp(f_omega, dim=0) - math.log(n_omega)
    l2_reg    = reg_lambda * (f_anti.pow(2).mean() + f_omega.pow(2).mean())
    return anti_term + log_Z + l2_reg


def omega_recall_at_anti_target(f_anti, f_omega, anti_target=ANTI_RECALL_TARGET):
    """Return Omega_recall at the threshold where Anti_recall ≈ anti_target.

    Uses the empirical quantile of the Anti score distribution.
    """
    t = torch.quantile(f_anti, 1.0 - anti_target).item()
    actual_anti_recall  = (f_anti  > t).float().mean().item()
    omega_recall        = (f_omega <= t).float().mean().item()
    return omega_recall, actual_anti_recall, t


def compute_val_score(model, val_loader, device):
    """Evaluate on val set.

    Returns:
        omega_recall_at_90   — primary physics metric (checkpoint by this)
        sweep_score          — symmetric score for comparison
        anti_recall_actual   — actual Anti recall achieved at the threshold
    """
    model.eval()
    all_f, all_y = [], []
    with torch.no_grad():
        for x, y, mask in val_loader:
            f = model(x.to(device), mask.to(device))
            all_f.append(f.cpu())
            all_y.append(y)
    f_all = torch.cat(all_f)
    y_all = torch.cat(all_y)
    is_anti  = (y_all == 1)
    is_omega = (y_all == 0)

    f_anti_val  = f_all[is_anti]
    f_omega_val = f_all[is_omega]

    # Primary metric: Omega_recall at Anti_recall ≈ ANTI_RECALL_TARGET
    omega_rec, anti_rec, t = omega_recall_at_anti_target(f_anti_val, f_omega_val)

    # Symmetric sweep score (for comparison / logging)
    best_sweep = max(
        (f_anti_val > thr).float().mean().item() + (f_omega_val <= thr).float().mean().item() - 1.0
        for thr in torch.quantile(f_all, torch.linspace(0.05, 0.95, 19)).tolist()
    )

    return omega_rec, best_sweep, anti_rec


def run_training():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = config.DEVICE

    run_number = get_next_run_number()
    log_path = f"logs/adv_run{run_number}.log"

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    # Load and normalise features
    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    print("Preparing dataset...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        target = y.squeeze().long()
        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds
        dataset.append((x, target))

    # 80/20 split — identical seed to train.py
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    val_data   = dataset[split:]

    omega_train = [x for x, y in train_data if y.item() == 0]
    anti_train  = [x for x, y in train_data if y.item() == 1]

    log(f"NCE v2 Run {run_number} | features={config.FEATURE_NAMES}")
    log(f"Train: {len(omega_train)} Omega, {len(anti_train)} Anti | Val: {len(val_data)}")
    log(f"OMEGA_ACCUM={OMEGA_ACCUM} | REG_LAMBDA={REG_LAMBDA} | LR={LR} | EMA={EMA_DECAY}")
    log(f"Checkpoint metric: Omega_recall at Anti_recall≥{ANTI_RECALL_TARGET}")

    # Anti is the primary iteration dimension (391 batches); Omega is cycled
    anti_loader = DataLoader(
        anti_train, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_x_only, num_workers=2,
        pin_memory=True, persistent_workers=True, drop_last=True,
    )
    omega_loader = DataLoader(
        omega_train, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_x_only, num_workers=2,
        pin_memory=True, persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        KaonDataset(val_data), batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )

    model = DensityRatioNet(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"DensityRatioNet: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_omega_rec = -1.0
    epochs_no_improve = 0

    log("Starting NCE v2 training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Cycle Omega so it always covers anti_loader without running out
        omega_iter = itertools.cycle(omega_loader)
        anti_iter  = iter(anti_loader)

        epoch_loss, epoch_f_anti, epoch_f_omega, epoch_f_std_anti, n_steps = 0., 0., 0., 0., 0

        for x_anti, mask_anti in anti_iter:
            x_anti    = x_anti.to(device)
            mask_anti = mask_anti.to(device)

            # Accumulate OMEGA_ACCUM Omega batches for a stable log_Z denominator
            f_omega_parts = []
            for _ in range(OMEGA_ACCUM):
                x_o, m_o = next(omega_iter)
                f_omega_parts.append(model(x_o.to(device), m_o.to(device)))
            f_omega_all = torch.cat(f_omega_parts)  # (OMEGA_ACCUM × BATCH_SIZE,)

            f_anti_batch = model(x_anti, mask_anti)

            loss = nce_loss_with_reg(f_anti_batch, f_omega_all, REG_LAMBDA)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(live_p.data, alpha=1.0 - EMA_DECAY)

            epoch_loss       += loss.item()
            epoch_f_anti     += f_anti_batch.mean().item()
            epoch_f_omega    += f_omega_all.mean().item()
            epoch_f_std_anti += f_anti_batch.std().item()
            n_steps          += 1

        epoch_loss       /= n_steps
        epoch_f_anti     /= n_steps
        epoch_f_omega    /= n_steps
        epoch_f_std_anti /= n_steps

        omega_rec, sweep_score, anti_rec_actual = compute_val_score(ema_model, val_loader, device)
        scheduler.step(omega_rec)

        line = (
            f"Epoch {epoch:03d} | NCE: {epoch_loss:.4f} | "
            f"f_anti: {epoch_f_anti:.3f}±{epoch_f_std_anti:.2f} | f_omega: {epoch_f_omega:.3f} | "
            f"Δf: {epoch_f_anti - epoch_f_omega:.3f} | "
            f"Ω-rec@Anti0.9: {omega_rec:.4f} (Anti={anti_rec_actual:.3f}) | "
            f"Sweep: {sweep_score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"No-improve: {epochs_no_improve}/{PATIENCE}"
        )
        log(line)

        if omega_rec > best_omega_rec:
            best_omega_rec = omega_rec
            epochs_no_improve = 0
            torch.save(ema_model.state_dict(), ADV_SAVE_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            log(f"\nEarly stopping at epoch {epoch}.")
            break

    log(f"\nBest Omega_recall@Anti=0.9: {best_omega_rec:.4f} → saved to {ADV_SAVE_PATH}")
    log(f"Log saved to {log_path}")


if __name__ == "__main__":
    run_training()
