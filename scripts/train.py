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

CHECKPOINT_THRESHOLD = 0.55  # Logged for reference; checkpoint is now by argmax score
EMA_DECAY = 0.999             # EMA shadow model decay for stable checkpointing


def get_next_run_number():
    os.makedirs("logs", exist_ok=True)
    existing = glob.glob("logs/train_run*.log")
    nums = []
    for f in existing:
        stem = os.path.basename(f).replace("train_run", "").replace(".log", "")
        if stem.isdigit():
            nums.append(int(stem))
    return max(nums) + 1 if nums else 1


class KaonDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]  # (x, y)


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    # mask: True = padding position (ignored by attention)
    mask = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    return padded, torch.stack(ys), mask


def run_training():
    # Fix seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    run_number = get_next_run_number()
    log_path = f"logs/train_run{run_number}.log"

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    labels = []

    print("Preparing Transformer dataset...")
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

    log(f"Run {run_number} | features={config.FEATURE_NAMES} | IN_CHANNELS={config.IN_CHANNELS} | D_MODEL={config.D_MODEL}")
    log(f"Dataset: {n_o} Omega, {n_a} Anti-Omega")
    log(f"Loss: per-class-mean BCE")
    log(f"EMA decay: {EMA_DECAY}")

    # 80/20 split (seed already fixed above)
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_loader = DataLoader(KaonDataset(dataset[:split]), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(KaonDataset(dataset[split:]), batch_size=config.BATCH_SIZE,
                            collate_fn=collate_fn,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    model = OmegaTransformer(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    # EMA shadow model — evaluated each epoch for stable checkpointing
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_a_rec_055 = -1.0
    best_anti_rec = -1.0
    best_score = -1.0
    epochs_no_improvement = 0

    log("Starting training...\n")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for x, y, mask in train_loader:
            x, y, mask = x.to(config.DEVICE), y.to(config.DEVICE), mask.to(config.DEVICE)
            optimizer.zero_grad()
            out = model(x, mask)
            p = torch.softmax(out, dim=1)[:, 1].clamp(1e-7, 1 - 1e-7)
            is_a = (y == 1)
            is_o = (y == 0)

            # Per-class-mean BCE
            loss = (-torch.log(p[is_a]).mean() + -torch.log(1 - p[is_o]).mean())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA shadow model
            with torch.no_grad():
                for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(live_p.data, alpha=1.0 - EMA_DECAY)

        # Evaluate using EMA model for stable metrics
        ema_model.eval()
        all_p_anti, all_y = [], []
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(config.DEVICE), y.to(config.DEVICE), mask.to(config.DEVICE)
                out = ema_model(x, mask)
                p_anti = torch.softmax(out, dim=1)[:, 1]
                all_p_anti.append(p_anti.cpu())
                all_y.append(y.cpu())

        p_anti = torch.cat(all_p_anti)
        labels_val = torch.cat(all_y)
        is_omega = (labels_val == 0)
        is_anti  = (labels_val == 1)

        # Argmax metrics (for logging)
        preds_argmax = (p_anti >= 0.5).long()
        o_rec = ((preds_argmax == 0) & is_omega).sum().item() / is_omega.sum().item()
        a_rec = ((preds_argmax == 1) & is_anti).sum().item()  / is_anti.sum().item()
        score = o_rec + a_rec - 1.0

        # Threshold-aware checkpoint metric: Anti recall at t=0.55
        preds_055 = (p_anti >= CHECKPOINT_THRESHOLD).long()
        a_rec_055 = ((preds_055 == 1) & is_anti).sum().item() / is_anti.sum().item()

        # Scheduler and checkpoint both driven by argmax score
        scheduler.step(score)

        line = (
            f"Epoch {epoch:03d} | Omega Rec: {o_rec:.3f} | Anti Rec: {a_rec:.3f} | "
            f"A@{CHECKPOINT_THRESHOLD:.2f}: {a_rec_055:.3f} | Score: {score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"No-improve: {epochs_no_improvement}/{config.EARLY_STOP_PATIENCE}"
        )
        log(line)

        # Checkpoint EMA model by best argmax score — threshold sweep done post-hoc
        if score > best_score:
            best_score = score
            best_anti_rec = a_rec
            best_a_rec_055 = a_rec_055
            epochs_no_improvement = 0
            torch.save(ema_model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= config.EARLY_STOP_PATIENCE:
            log(f"\nEarly stopping: no improvement for {config.EARLY_STOP_PATIENCE} epochs.")
            break

    summary = (
        f"\nBest Score: {best_score:.4f} | Anti (argmax): {best_anti_rec:.4f} | "
        f"A@{CHECKPOINT_THRESHOLD:.2f}: {best_a_rec_055:.4f}  →  saved to {config.MODEL_SAVE_PATH}"
    )
    log(summary)
    log(f"Log saved to {log_path}")


if __name__ == "__main__":
    run_training()
