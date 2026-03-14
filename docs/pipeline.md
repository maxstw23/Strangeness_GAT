# Pipeline

## Overview

```
data/omega_kaon_all.root
        │
        ▼
scripts/preprocess_data.py  →  data/balanced_omega_anti.pt  +  _stats.pt
        │
        ├──▶  scripts/inspect_data.py      (sanity check)
        ├──▶  scripts/explore_data.py      (data exploration plots)
        │
        ▼
scripts/train.py  →  models/omega_transformer.pth
        │
        ├──▶  scripts/evaluate_physics.py        (threshold scan + physics metrics)
        ├──▶  scripts/plot_recall_tradeoff.py     (recall tradeoff curve)
        └──▶  scripts/interpret_model.py          (attention + permutation importance)
```

---

## Step 1: Preprocessing

```bash
source venv/bin/activate
python scripts/preprocess_data.py
```

Reads `data/omega_kaon_all.root` (TTree: `ml_tree`) and outputs:
- `data/balanced_omega_anti.pt` — list of per-event dicts with node features and binary label (0 = Ω⁻, 1 = Ω̄⁺)
- `data/balanced_omega_anti_stats.pt` — per-feature means and stds for normalisation

Processing per event:
1. Kaon multiplicities are balanced so N(same-sign) = N(opposite-sign) relative to the Omega, by sampling from a global kaon pool built across all events. Since K⁺ > K⁻ globally, embedded kaons are always K⁻ (symmetric for both Ω⁻ and Ω̄⁺ events).
2. Raw momenta converted to cylindrical coordinates.
3. Per-kaon node features computed:
   `[f_pt, k_star, d_eta, d_phi, rel_q, o_pt]`
   — see CLAUDE.md Feature table for definitions.
4. Feature statistics (mean, std) computed from all kaons across all events and saved.

---

## Step 2: Inspection (optional)

```bash
source venv/bin/activate
python scripts/inspect_data.py
```

Prints feature statistics and checks for NaNs and out-of-range values.

```bash
source venv/bin/activate
python scripts/explore_data.py
```

Produces `plots/data_exploration.png` comparing per-kaon feature distributions and per-event
aggregates between Omega and Anti events. See `data_exploration.md` for findings.

---

## Step 3: Training

```bash
source venv/bin/activate
python scripts/train.py
```

Trains OmegaTransformer on the preprocessed dataset. Per-epoch output:
```
Epoch 001 | Omega Rec: 0.562 | Anti Rec: 0.800 | A@0.55: 0.738 | Score: 0.3613 | LR: 0.000500 | No-improve: 0/25
```

- Checkpoint saved to `models/omega_transformer.pth` when argmax score improves.
- Early stops after `EARLY_STOP_PATIENCE = 25` epochs without improvement.
- Training logs saved to `logs/train_run<N>.log`.

Loss function: per-class-mean asymmetric BCE with `ANTI_WEIGHT = 2.0`. See CLAUDE.md for details.

### Dry-run option

To test the full pipeline quickly without waiting for convergence:

```bash
source venv/bin/activate
python scripts/train.py --dry-run
```

Dry-run mode:
- Uses first 1000 samples (instead of full dataset)
- Runs 2 epochs (instead of 200)
- Still saves checkpoint, logs, and produces all metrics
- Useful for verifying preprocessing, dataloaders, and model forward/backward passes

You can combine `--dry-run` with other options:
```bash
source venv/bin/activate
python scripts/train.py --dry-run --edge-bias --data unpadded
```

---

## Step 4: Evaluation

```bash
source venv/bin/activate
python scripts/evaluate_physics.py
```

Reports argmax metrics and a threshold scan on the validation set:
- **Omega Recall**: fraction of Ω⁻ events correctly classified
- **Anti Recall**: fraction of Ω̄⁺ events correctly classified
- **Physics Score**: Omega_recall + Anti_recall − 1
- **f_BN** (corrected): BN-carrying fraction estimate accounting for class imbalance ratio r

```bash
source venv/bin/activate
python scripts/plot_recall_tradeoff.py
```

Dense threshold sweep (1000 points) producing `plots/recall_tradeoff.png`:
- Left: recall tradeoff curve (Anti recall vs Omega recall as threshold varies)
- Centre: score vs threshold
- Right: P(Anti | event) score distributions for Omega and Anti events

---

## Step 5: Interpretation

```bash
source venv/bin/activate
python scripts/interpret_model.py
```

Runs two interpretability analyses on the saved checkpoint (see `interpretation.md`):
1. **Attention analysis** — CLS→kaon attention weights from the final transformer layer; attention-weighted feature means and most-attended kaon features per event, split by class → `plots/attention_analysis.png`
2. **Feature permutation importance** — global cross-event permutation of each feature, score drop measured over 15 repeats → `plots/feature_importance.png`
