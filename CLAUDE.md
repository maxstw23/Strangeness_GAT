# CLAUDE.md — OmegaNet

**OmegaNet** uses ML to study Omega hyperon production in heavy-ion collisions at RHIC BES-II energies.

## References

- **`docs/physics.md`** — Physics motivation: baryon number transport, gluon junction picture.
- **`docs/method.md`** — ML strategy: PU classification, charge blinding.
- **`docs/pipeline.md`** — End-to-end: preprocessing, training, evaluation.
- **`docs/experiments.md`** — All training runs and outcomes.
- **`docs/lessons.md`** — Hard-won rules. **Read before reporting any result.**

## Principles

- **Be objective**: Evaluate all ideas with full scrutiny. Flag flaws immediately. Wasted experiments are wasted physics time.
- **Centralize config**: All tunable parameters live in `config.py`. Edit only `FEATURE_NAMES` to change features; `IN_CHANNELS` is derived automatically.
- **Keep docs current**: After any significant change, update the relevant doc. Docs are the single source of truth.
- **Commit at checkpoints**: Small, focused commits after each meaningful unit of work.
- **Be frugal**: Batch reads/edits, use targeted searches. Token efficiency is a real constraint.

## Environment

- Python venv at `venv/`. Always use `venv/bin/python` from the project root.
- CUDA: RTX 3070, CUDA 12.8, PyTorch 2.6.0+cu124.

## Model

**OmegaTransformer** (`models/transformer_model.py`) — 2-layer Pre-LN Transformer, CLS token, no positional encoding.
Checkpoint: `models/omega_transformer.pth`

Config (`config.py`): `D_MODEL=128`, `NHEAD=4`, `NUM_LAYERS=2`, `DIM_FEEDFORWARD=256`, `KSTAR_CLIP=8.0`

## Features

Canonical order in `FEATURE_REGISTRY` (`config.py`). Active set controlled by `FEATURE_NAMES`.

**Current active features (6):** `f_pt`, `k_star`, `d_y`, `d_phi`, `cos_theta_star`, `d_y_signed`

| Name | Description |
|---|---|
| `f_pt` | Kaon pT [GeV/c] |
| `k_star` | Lorentz-invariant relative momentum in kaon-Ω pair rest frame [GeV/c] |
| `d_y` | \|y_K − y_Ω\| — absolute rapidity separation |
| `d_phi` | \|φ_K − φ_Ω\| in [0, π] |
| `cos_theta_star` | cos(θ*) = k*_z / \|k*\| — beam-axis angle in pair rest frame |
| `d_y_signed` | \|y_K\| − \|y_Ω\| — midrapidity proximity gap |

Excluded: `o_pt` (p̄ absorption biases Ω̄⁺ efficiency), `o_y_abs` (introduces biased rapidity shift), EP cosines (confirmed inert, run18).

The balanced dataset (`data/balanced_omega_anti.pt`) stores 12 features per kaon. Preprocessing uses event-mixed K⁻ pool (binned on |y_Ω|, pT_Ω quartiles) to eliminate padding kinematic artifacts. k* uses PDG masses: m_K = 0.493677, m_Ω = 1.67245 GeV/c².

## Scripts

| Script | Purpose |
|---|---|
| `scripts/preprocess_data.py` | ROOT → `data/balanced_omega_anti.pt`; `--mode unpadded` for unpadded variant |
| `scripts/train.py` | Train OmegaTransformer, save best checkpoint |
| `scripts/evaluate_physics.py` | Threshold scan + physics metrics on val set |
| `scripts/plot_recall_tradeoff.py` | Recall tradeoff curve → `plots/recall_tradeoff.png` |
| `scripts/interpret_model.py` | Attention + permutation importance → `plots/` |

## Training

Loss: **per-class-mean BCE** — `mean(−log p | Anti) + mean(−log(1−p) | Omega)`.
Checkpoint criterion: best **O@A=0.90** on validation set.
Scheduler: ReduceLROnPlateau on O@A=0.90, patience=5, factor=0.5.

Best result to date: **O@A=0.90 = 0.3345** (run25). See `docs/experiments.md` for full history.

## Physics Goal

Classifier assigns p(x) ≈ P(pair-produced | kaon kinematics) to each Ω⁻ event.
- **Positive**: Ω̄⁺ — all pair-produced (labeled)
- **Unlabeled**: Ω⁻ — mixture of pair-produced (fraction π) and BN-transport via junction

Goal: **separate the two mechanisms** and profile their kinematic differences (k*, Δy, cos θ*, flow).
Not f_BN estimation — thermal models already constrain that. The goal is mechanism separation.
