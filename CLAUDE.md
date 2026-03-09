# CLAUDE.md — OmegaNet

**OmegaNet** uses ML to study Omega hyperon production in heavy-ion collisions at RHIC BES-II energies.

## Key References

- **`docs/physics.md`** — Physics motivation: baryon number transport, gluon junction picture, and why Omega is the probe of choice.
- **`docs/method.md`** — ML strategy: how the Omega/anti-Omega classification task encodes the physics goal, and how charge blinding is implemented.
- **`docs/pipeline.md`** — How to run the project end-to-end: preprocessing, inspection, training, and evaluation.
- **`docs/data_exploration.md`** — Findings from exploratory analysis of the preprocessed dataset: feature distributions, ΔR, k*, and signal-to-noise discussion.
- **`docs/interpretation.md`** — Plan for four model interpretability analyses (attention, permutation importance, gradient attribution, CLS embedding correlation).

## Development Principles

- **Centralize configuration**: All tunable parameters (features, model hyperparams, paths) belong in `config.py`. Scripts must import from `config.py` rather than hardcoding values. This ensures a single-point change propagates everywhere (e.g., adding/removing a feature only requires editing `FEATURE_NAMES`).
- **Write elegant code**: Prefer clean, readable solutions over verbose ones. Favour list/dict comprehensions, well-named variables, and small focused functions. When regenerating or refactoring code, actively look for simpler expressions of the same logic — not just correctness, but clarity.
- **Future-proof by design**: When writing or refactoring code, prefer patterns that will remain correct as features, scripts, or hyperparameters evolve. Avoid magic numbers, hardcoded feature counts, or duplicate constants across files.
- **Keep docs current**: After any significant change (new feature, architecture update, training result, bug fix, new script), update the relevant markdown file (`CLAUDE.md`, `pipeline.md`, `method.md`, `physics.md`, `interpretation.md`, or `data_exploration.md`). The docs are the single source of truth for the project state.
- **Commit at natural checkpoints**: After completing a meaningful unit of work (new feature, bug fix, doc update, successful training run), create a git commit. Don't wait until the end of a long session — small, focused commits make history readable and rollbacks safe.
- **Be frugal with tokens**: Before running any expensive operation (training, large script execution, extensive searches), pause to plan ways to reduce token usage. Batch related reads/edits, avoid re-reading files, use targeted searches instead of broad exploration. Token efficiency is a real constraint — optimize before executing.
- **Do not use f_BN as a training metric**: f_BN is a derived scalar that depends on the threshold and counting ratio assumptions — it is not the physics goal and must not be used to compare training runs. The correct diagnostics for model quality are the sweep-optimum score (Omega_recall + Anti_recall − 1) and, ultimately, the kinematic profiles of the separated subpopulations. f_BN is well-constrained by thermal models; the goal is mechanism separation, not f_BN estimation.

## Environment

- Python venv at `venv/`. Always use `venv/bin/python` to run scripts.
- Scripts must be run from the project root so that relative paths in `config.py` resolve correctly.
- CUDA: RTX 3070, CUDA 12.8, PyTorch 2.6.0+cu124.

## Model

Current model: **OmegaTransformer** (`models/transformer_model.py`)
- 2-layer Pre-LN Transformer, CLS token, no positional encoding, 275K parameters
- Checkpoint: `models/omega_transformer.pth`

Config (`config.py`):
- `IN_CHANNELS = 9` — derived from `len(FEATURE_NAMES)`; edit only `FEATURE_NAMES` to change features
- `D_MODEL = 128`, `NHEAD = 4`, `NUM_LAYERS = 2`, `DIM_FEEDFORWARD = 256`
- `KSTAR_CLIP = 8.0` — applied to k* (feature index 1) before normalisation

## Features

Ten features per kaon stored in `data/balanced_omega_anti.pt` (canonical order = `FEATURE_REGISTRY`).
Active set controlled by `FEATURE_NAMES` in `config.py` (currently 9 features; `o_pt` excluded).

| Index | Name | Description |
|---|---|---|
| 0 | `f_pt` | Kaon transverse momentum [GeV/c], capped at 2 GeV/c |
| 1 | `k_star` | Lorentz-invariant relative momentum in kaon-Ω pair rest frame [GeV/c] |
| 2 | `d_y` | \|y_kaon − y_Omega\| — absolute rapidity separation (padded); **d_y_signed = sign(y_Ω)×(y_K−y_Ω)** in unpadded |
| 3 | `d_phi` | \|φ_kaon − φ_Omega\| in [0, π] — absolute azimuthal separation (no preferred direction) |
| 4 | `o_pt` | Omega pT [GeV/c], broadcast — **excluded** (p̄ absorption biases Ω̄⁺ reconstruction) |
| 5 | `cos_theta_star` | cos(θ*) = k*_z / \|k*\| — beam-axis angle in pair rest frame (source elongation probe) |
| 6 | `o_cos_psi1` | cos(φ_Ω − Ψ₁) — Omega v₁ alignment with EPD 1st-order event plane, broadcast |
| 7 | `o_cos2_psi2` | cos(2(φ_Ω − Ψ₂)) — Omega v₂ alignment with EPD 2nd-order event plane, broadcast |
| 8 | `f_cos_psi1` | cos(φ_K − Ψ₁) — kaon v₁ alignment with EPD 1st-order event plane, per-kaon |
| 9 | `f_cos2_psi2` | cos(2(φ_K − Ψ₂)) — kaon v₂ alignment with EPD 2nd-order event plane, per-kaon |
| 10 | `o_y_abs` | \|y_Omega\| — Omega's displacement from midrapidity [rapidity units], broadcast; **unpadded dataset only** |

**Unpadded dataset** (`data/unpadded_omega_anti.pt`): 11 features per kaon; no K⁻ padding, preserving the natural n_kaons(Ω⁻) > n_kaons(Ω̄⁺) asymmetry. Feature 2 stores the directed rapidity gap d_y_signed: negative means kaon is on the midrapidity side of the Omega (expected for junction-transport Ω⁻); near zero for pair-produced. Feature 10 is \|y_Omega\|, larger for junction Ω⁻ (beam-remnant origin).

k* uses PDG masses: m_K = 0.493677 GeV/c², m_Ω = 1.67245 GeV/c².
Feature means/stds are auto-computed by preprocessing and saved to `data/balanced_omega_anti_stats.pt`.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/preprocess_data.py` | ROOT → `data/balanced_omega_anti.pt` + stats; `--mode unpadded` → `data/unpadded_omega_anti.pt` |
| `scripts/inspect_data.py` | Print feature statistics and sanity checks |
| `scripts/explore_data.py` | Full data exploration plots → `plots/data_exploration.png` |
| `scripts/train.py` | Train OmegaTransformer, save best checkpoint |
| `scripts/evaluate_physics.py` | Threshold scan + physics metrics on val set |
| `scripts/plot_recall_tradeoff.py` | Dense threshold sweep + recall tradeoff curve → `plots/recall_tradeoff.png` |
| `scripts/interpret_model.py` | Attention analysis + feature permutation importance → `plots/` |

## Training

Loss: **per-class-mean BCE**:
```
loss = mean(−log p(x)    | Anti events)    # P(pair-produced) → 1
     + mean(−log(1−p(x)) | Omega events)   # P(pair-produced) → 0
```
- Biased approximation of the true PU objective (pair-produced Ω⁻ are incorrectly penalized), but robust in the weak-signal regime where PU gradient subtraction introduces too much variance
- Checkpoint saved by best argmax score (Anti recall + Omega recall − 1 at t = 0.5)
- Scheduler: ReduceLROnPlateau on argmax score, patience = 5, factor = 0.5

Key training findings:
- Score ceiling ~0.31–0.32 (argmax t=0.5) confirmed across 21 BCE runs, 5 NCE/DensityRatio runs, GRL adversarial, consistency regularization, and EM pseudo-labeling — see `docs/experiments.md`
- Per-kaon feature distributions are nearly identical between Omega and Anti events
- Signal lives in multi-kaon correlations — visible as clear separation in P(Anti) score distributions
- EP cosine features (event-plane alignment) add no signal (run18 vs run16)
- Ultra-minimal features [k*, cos_θ*] collapse to 0.157 — f_pt, d_y, d_phi are essential
- GRL adversarial (multiplicity debiasing) and consistency regularization both give ≈0.308, below BCE baseline
- EM pseudo-labeling correctly down-weights pair-produced Ω⁻ (mean weight 0.45 ≈ π), shifting Anti recall 0.635→0.711, but sweep-optimal score unchanged — genuine physics ceiling, not label-noise artifact
- Threshold is a post-hoc choice; operating point for physics analysis set by val-set threshold sweep

## Physics Goal

Use the Ω̄⁺ sample as a **pure labeled proxy for the pair-produced mechanism** to train a classifier
that assigns each Ω⁻ event a score p(x) ≈ P(pair-produced | kaon kinematics).

The PU mixture is about **production mechanism**, not particle identity:
- **Positive (P)**: Ω̄⁺ events — all pair-produced, used as the labeled positive class
- **Unlabeled (U)**: Ω⁻ events — mixture of pair-produced (fraction π) and BN-transport via junction

**The measurement goal is not f_BN** — that is well-constrained by thermal models. The goal is to
**separate the two production mechanisms** and study how BN-carrying Ω⁻ differ kinematically and
dynamically from pair-produced Ω⁻:
- Events with high p(x) → likely pair-produced → profile their kinematics
- Events with low p(x) → likely BN-carrying → profile their kinematics
- Differences in k*, Δy, cos(θ*), flow observables, etc. characterise the junction mechanism
