"""train_pseudolabel.py — Iterative EM-style pseudo-labeling for PU learning.

Motivation: the standard BCE loss treats ALL Omega events as BN-transport (label=0),
but ~half are pair-produced.  This label noise inflates the gradients for pair-produced
Omega events in the wrong direction.

The EM fix: use the current model's scores as soft labels.  Pair-produced Omega events
score high (p_anti ≈ 1) and should be down-weighted; BN-transport events score low
and should be kept at full weight.

Algorithm:
    E-step: score all events with current model → per-Omega weights w_i = 1 − p_anti(x_i)
    M-step: train one full pass with weighted Omega loss
    Repeat for EM_ITER iterations, resetting LR and patience each time

Loss at each M-step:
    L = mean(−log p    | Anti events)
      + sum(w_i · −log(1−p) | Omega events) / sum(w_i)

The first iteration uses equal weights (w_i = 1), so it is identical to standard BCE.
Subsequent iterations reweight Omega events by their inferred BN-transport probability.

Temperature parameter γ controls the sharpness of the reweighting:
    w_i = (1 − p_anti)^γ
γ = 1.0 → linear reweighting (gentle)
γ > 1.0 → sharper: high-scoring Omega events are more aggressively down-weighted
"""
import sys
import os
import copy
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import config
from transformer_model import OmegaTransformer
from tqdm import tqdm

# ── Pseudo-labeling hyperparameters ──────────────────────────────────────────
EM_ITER          = 4      # number of E/M iterations (iter 1 = standard BCE)
GAMMA            = 1.5    # reweighting temperature: w_i = (1 − p)^γ
WARMUP_LR_RESET  = True   # reset LR and patience each EM iteration
CHECKPOINT_THRESHOLD = 0.55
EMA_DECAY        = 0.999
PL_SAVE_PATH     = "models/omega_pseudolabel.pth"


def get_next_run_number():
    os.makedirs("logs", exist_ok=True)
    existing = glob.glob("logs/pl_run*.log")
    nums = [
        int(os.path.basename(f).replace("pl_run", "").replace(".log", ""))
        for f in existing
        if os.path.basename(f).replace("pl_run", "").replace(".log", "").isdigit()
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
    xs, ys, ws = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat  = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask   = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    return padded, torch.stack(ys), mask, torch.stack(ws)


def collate_val(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat  = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask   = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    return padded, torch.stack(ys), mask


@torch.no_grad()
def compute_weights(model, train_data_xy, device):
    """E-step: score all training Omega events and return per-event weights.

    Anti events always get weight 1.0 (they are pure labeled positives).
    Omega events get weight (1 − p_anti)^γ.
    """
    model.eval()
    weights = []
    # Process in small batches to avoid OOM
    batch_size = 512
    n = len(train_data_xy)
    all_p = []

    for start in range(0, n, batch_size):
        batch = train_data_xy[start:start + batch_size]
        xs = [x for x, _ in batch]
        max_len = max(x.shape[0] for x in xs)
        n_feat  = xs[0].shape[1]
        padded  = torch.zeros(len(xs), max_len, n_feat, device=device)
        mask    = torch.ones(len(xs), max_len, dtype=torch.bool, device=device)
        for i, x in enumerate(xs):
            padded[i, :x.shape[0]] = x.to(device)
            mask[i, :x.shape[0]]   = False

        out    = model(padded, mask)
        p_anti = torch.softmax(out, dim=1)[:, 1].cpu()
        all_p.append(p_anti)

    all_p = torch.cat(all_p)

    for idx, (_, y) in enumerate(train_data_xy):
        if y.item() == 1:                         # Anti: full weight
            weights.append(torch.tensor(1.0))
        else:                                      # Omega: (1 − p_anti)^γ
            w = (1.0 - all_p[idx].clamp(1e-7, 1 - 1e-7)) ** GAMMA
            weights.append(w)

    return weights


def run_training(start_iter=1, resume_log=None):
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if resume_log:
        log_path = resume_log
    else:
        run_number = get_next_run_number()
        log_path = f"logs/pl_run{run_number}.log"

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset_xy, labels = [], []

    print("Preparing pseudo-label dataset...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        target = y.squeeze().long()
        labels.append(target.item())
        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds
        dataset_xy.append((x, target))

    labels_t = torch.tensor(labels)
    n_o = (labels_t == 0).sum().item()
    n_a = (labels_t == 1).sum().item()

    run_label = f"PL Run {run_number}" if not resume_log else f"PL Run (resumed, log={resume_log})"
    log(f"{run_label} | features={config.FEATURE_NAMES} | IN_CHANNELS={config.IN_CHANNELS}")
    log(f"Dataset: {n_o} Omega, {n_a} Anti-Omega")
    log(f"EM_ITER={EM_ITER} | GAMMA={GAMMA}")

    random.shuffle(dataset_xy)
    split = int(0.8 * len(dataset_xy))
    train_xy = dataset_xy[:split]
    val_xy   = dataset_xy[split:]

    val_loader = DataLoader(
        KaonDataset([(x, y) for x, y in val_xy]),
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_val,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    model = OmegaTransformer(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"OmegaTransformer: {n_params:,} parameters")

    iter_save_fmt = PL_SAVE_PATH.replace(".pth", "_iter{}.pth")

    # Resume: load checkpoint from the last completed iteration
    if start_iter > 1:
        resume_ckpt = iter_save_fmt.format(start_iter - 1)
        model.load_state_dict(torch.load(resume_ckpt, map_location=config.DEVICE))
        log(f"Resumed from {resume_ckpt} (starting at EM iteration {start_iter})")

    global_best_score = -1.0

    for em_iter in range(start_iter, EM_ITER + 1):
        log(f"\n{'='*60}")
        log(f"EM Iteration {em_iter}/{EM_ITER}")

        # E-step: compute weights from current model (iter 1 = uniform weights)
        if em_iter == 1:
            log("Iter 1: uniform weights (standard BCE)")
            weights = [torch.tensor(1.0)] * len(train_xy)
        else:
            log("E-step: computing per-event weights from current model...")
            weights = compute_weights(model, train_xy, config.DEVICE)
            w_omega = [weights[i].item() for i, (_, y) in enumerate(train_xy) if y.item() == 0]
            log(f"  Omega weight stats: mean={sum(w_omega)/len(w_omega):.4f} "
                f"min={min(w_omega):.4f} max={max(w_omega):.4f}")

        # Build weighted dataset
        train_data = [(x, y, weights[i]) for i, (x, y) in enumerate(train_xy)]
        train_loader = DataLoader(
            KaonDataset(train_data),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

        # M-step: train to convergence
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_score   = -1.0
        best_anti    = -1.0
        best_055     = -1.0
        no_improve   = 0

        for epoch in range(1, config.EPOCHS + 1):
            model.train()
            for x, y, mask, w in train_loader:
                x, y, w, mask = (x.to(config.DEVICE), y.to(config.DEVICE),
                                  w.to(config.DEVICE), mask.to(config.DEVICE))
                optimizer.zero_grad()

                out = model(x, mask)
                p = torch.softmax(out, dim=1)[:, 1].clamp(1e-7, 1 - 1e-7)
                is_a = (y == 1)
                is_o = (y == 0)

                # Weighted Omega loss; Anti loss unweighted
                loss_anti  = -torch.log(p[is_a]).mean()
                w_o = w[is_o]
                loss_omega = (w_o * -torch.log(1 - p[is_o])).sum() / w_o.sum().clamp(min=1e-7)
                loss = loss_anti + loss_omega

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    for ep, lp in zip(ema_model.parameters(), model.parameters()):
                        ep.data.mul_(EMA_DECAY).add_(lp.data, alpha=1.0 - EMA_DECAY)

            # Eval
            ema_model.eval()
            all_p_anti, all_y = [], []
            with torch.no_grad():
                for xv, yv, mv in val_loader:
                    xv, mv = xv.to(config.DEVICE), mv.to(config.DEVICE)
                    pv = torch.softmax(ema_model(xv, mv), dim=1)[:, 1]
                    all_p_anti.append(pv.cpu())
                    all_y.append(yv)

            p_v      = torch.cat(all_p_anti)
            y_v      = torch.cat(all_y)
            is_omega = (y_v == 0)
            is_anti  = (y_v == 1)

            pred_am  = (p_v >= 0.5).long()
            o_rec    = ((pred_am == 0) & is_omega).sum().item() / is_omega.sum().item()
            a_rec    = ((pred_am == 1) & is_anti).sum().item()  / is_anti.sum().item()
            score    = o_rec + a_rec - 1.0

            pred_55  = (p_v >= CHECKPOINT_THRESHOLD).long()
            a_055    = ((pred_55 == 1) & is_anti).sum().item() / is_anti.sum().item()

            scheduler.step(score)
            line = (
                f"  [EM{em_iter}] Ep {epoch:03d} | "
                f"Ω:{o_rec:.3f} A:{a_rec:.3f} A@.55:{a_055:.3f} Score:{score:.4f} | "
                f"LR:{optimizer.param_groups[0]['lr']:.6f} | ni:{no_improve}/{config.EARLY_STOP_PATIENCE}"
            )
            log(line)

            if score > best_score:
                best_score  = score
                best_anti   = a_rec
                best_055    = a_055
                no_improve  = 0
                # Save if global best
                if score > global_best_score:
                    global_best_score = score
                    torch.save(ema_model.state_dict(), PL_SAVE_PATH)
            # Always save this iteration's best (overwrite each epoch within iter)
            if score >= best_score:
                torch.save(ema_model.state_dict(), iter_save_fmt.format(em_iter))
            else:
                no_improve += 1

            if no_improve >= config.EARLY_STOP_PATIENCE:
                log(f"  Early stopping at epoch {epoch}")
                break

        log(f"EM iter {em_iter} best: Score={best_score:.4f} Anti={best_anti:.4f} A@.55={best_055:.4f}")

    log(f"\nGlobal best score: {global_best_score:.4f} → saved to {PL_SAVE_PATH}")
    log(f"Log: {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-iter", type=int, default=1,
                        help="EM iteration to start from (loads iter{N-1}.pth checkpoint)")
    parser.add_argument("--resume-log", type=str, default=None,
                        help="Append to this existing log file instead of creating a new one")
    args = parser.parse_args()
    run_training(start_iter=args.start_iter, resume_log=args.resume_log)
