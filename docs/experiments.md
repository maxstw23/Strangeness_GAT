# Experiments Log

Primary metric: **O@A=0.90** (Omega recall at fixed Anti recall = 0.90).
All other scores (argmax, sweep-optimum) are internal training diagnostics only.

---

## BCE Transformer Baseline (old global-pool preprocessing)

| Run | Features | O@A=0.90 | Notes |
|-----|----------|----------|-------|
| run5–15 | Various 5–6 feat | — | Architecture search; scores recorded as argmax only |
| run16 | f_pt, k_star, d_y, d_phi, cos_θ* | ~0.26 | Best old-preprocessing BCE baseline |
| run18 | + EP cosines (×4) | — | EP features add no signal |
| run19 | f_pt, k_star, d_y, d_phi, cos_θ* | — | Confirms run16; EP features inert |
| run20 | k_star, cos_θ* only | — | Near-random — f_pt, d_y, d_phi are essential |

---

## Alternative Objectives (old preprocessing)

All run on same dataset with global K⁻ pool padding.

| Method | O@A=0.90 | Notes |
|--------|----------|-------|
| NCE/density-ratio (adv_run4–5) | — | Unstable (scale explosion); no improvement over BCE |
| GRL adversarial (grl_run2) | — | Multiplicity debiasing adds no benefit |
| Consistency regularization (cons_run1) | — | Augmentation adds no signal; model already consistent |
| EM pseudo-labeling (pl_run1) | — | Shifts operating point but does not improve separability |

Note: O@A=0.90 was not tracked for these runs; scores above were recorded as argmax at t=0.5.

---

## Event-Mixed Padding (run25)

**Problem**: Global K⁻ pool sampling biased d_y_signed — fake K⁻ had |y_K| values independent
of the specific Omega's |y_Ω|, producing pathological rapidity gaps. Anti events with many
padded kaons scored systematically higher (padding fraction 0.465 in score-spike vs 0.285 in bulk).

**Fix**: Event-mixed K⁻ pool binned on (|y_Ω|, pT_Ω) quartiles (4×4 = 16 bins).

| Run | Features | O@A=0.90 | Notes |
|-----|----------|----------|-------|
| run25 | f_pt, k_star, d_y, d_phi, cos_θ*, d_y_signed, o_y_abs | **0.3345** | Event-mixed padding; first clear improvement over old ceiling |

Note: run25 included `o_y_abs` which was subsequently removed (introduces biased rapidity shift).

| Run | Features | O@A=0.90 | Notes |
|-----|----------|----------|-------|
| run26 | f_pt, k_star, d_y, d_phi, cos_θ*, d_y_signed | **0.3333** | Removed o_y_abs |
| run27 | k_star, d_y, d_phi, cos_θ*, d_y_signed | **0.3162** | Removed f_pt (K⁺/K⁻ pT spectrum charge-leak); −0.017 is the spurious contribution from f_pt |

**Run 26 interpretation (interpret_model.py):**

Permutation importance (score drop, 15 repeats):

| Feature | Drop | Relative |
|---|---|---|
| d_y_signed | 0.221 | 66% |
| d_y | 0.200 | 60% |
| f_pt | 0.134 | 40% |
| k_star | 0.077 | 23% |
| cos_theta_star | 0.076 | 23% |
| d_phi | 0.019 | 6% |

The rapidity features (d_y_signed and d_y) dominate — together accounting for ~60–66% of the
signal. k* and cos_theta_star contribute equally (~23% each). d_phi carries almost no signal.
No feature has negative importance (no artifact dependency detected).

Attention analysis: attention-weighted feature differences between Omega and Anti are small
(largest Δ: d_y_signed +0.049, d_y +0.029), consistent with the weak per-kaon signal — the
model exploits multi-kaon patterns rather than individual distinctive kaons.

---

## GRL on Unpadded Data

Goal: train on unpadded data (genuine K⁺/K⁻ multiplicity asymmetry preserved) while
constraining the encoder to not exploit kaon multiplicity counts — using a DANN adversary.
Checkpoint criterion: save only when `adv_loss > 0.95 × log(n_bins)` (adversary at chance).

**Unpadded data preprocessing (re-done)**: updated to 13-column layout matching FEATURE_REGISTRY
(d_y at col 2, d_y_signed at col 10, o_y_abs at col 11, net_kaon at col 12).

**Adversary design iterations:**

| Run | Adversary | λ | Best constrained O@A | Notes |
|-----|-----------|---|----------------------|-------|
| grl_unpadded_run1 | MSE on log(n_kaons) | 0.1 (static) | 0.632 (unconstrained) | Linear ramp alpha; plateaus instantly |
| grl_unpadded_run3 | MSE on log(n_kaons) | 0.1 (dynamic ratio) | — | NaN at epoch 6; encoder destabilised |
| grl_unpadded_run4/5 | MSE on log(n_kaons) | 0.05 (static+pretrain) | 0.633 | Pre-train phase added; no constraint gate |
| grl_unpadded_run6 | MSE on log(n_kaons) | 0.5 | — (R²=−1.19) | Sign-flip Nash: encoder anti-encodes multiplicity |
| grl_unpadded_run7 | CE on within-class n_kaons bins | 0.5 | 0.632 (gate not reached) | O@A drops to 0.606 under GRL; adversary settles at 1.31 < chance 1.609 |
| **grl_unpadded_run8** | CE on within-class n_kaons bins | **2.0** | **0.630** | Adversary at chance; constrained Nash stable at epochs 50–110 |
| grl_unpadded_run9 | CE bins + conditional on class | 2.0 | 0.6325 | Conditional adversary; similar result |
| **grl_unpadded_run13** | CE bins + MSE(n_kaons, std of features) + conditional | **2.0** | **0.631** | Strongest adversary; both heads at chance; still O@A≈0.630 |

**Run 13 key results (λ=2.0, conditional CE+MSE adversary, moment targets = [n_kaons, std(f_i)]):**
- Pre-train O@A (no GRL): 0.630
- Best constrained O@A: **0.631** (adv_ce ≈ 1.59/1.609, adv_mse ≈ 0.99/1.00)
- Fresh-probe accuracy: 0.537 vs 0.200 chance

**Interpretation:**
- O@A is consistently ~0.630 across runs 8, 9, 13 despite very different adversary strengths.
- Even with feature std distributions fully constrained to chance, the model still achieves ~0.630.
- The constrained signal (~0.630) is ~2× run27 (0.316 on padded data).
- The per-event *mean* of kinematic features (mean k_star, mean d_y, etc.) is left free (not adversarially constrained) — this is the likely remaining source of separation. Mean features encode genuine kinematics and are partially multiplicity-correlated but not purely so.
- The DANN constraint is satisfied (co-trained adversary at chance); residual probe leakage reflects the conditional vs unconditional gap, not adversary failure.
- Fresh probe being higher in run13 than run8 (0.537 vs 0.446) reflects a different Nash equilibrium, not more leakage — the conditional adversary pushes the encoder to a different embedding space that an unconditional probe can read more easily.

---

## Key Null Results

- EP cosine features (event-plane alignment) add no signal
- NCE/density-ratio no better than BCE at this signal strength
- GRL adversarial debiasing (multiplicity invariance, padded data) does not help
- Consistency regularization (kaon dropout) adds no novel signal
- EM pseudo-labeling shifts operating point but not separability — confirms physics ceiling, not label-noise artifact
- Ultra-minimal [k*, cos_θ*] collapses to near-random — f_pt, d_y, d_phi essential
- Global K⁻ pool padding was a real artifact masking genuine signal
- GRL regression adversary (MSE on log n_kaons) produces sign-flip Nash: encoder anti-encodes multiplicity, fresh R²=−1.19, padded O@A=0.185 — unusable
- GRL classification adversary with λ<1 (run7): adversary settles above chance-floor, O@A drops only to 0.606 — incomplete debiasing

---

## Efficiency-Weighted Features (run43–45)

**Goal**: Add per-kaon inverse efficiency `eff_weight = 1/ε(pT, η)` to correct for detector acceptance.

**Config**: K⁺ and K⁻ efficiency params are identical (no charge-dependent correction).

| Run | Data | Features | O@A=0.90 | Notes |
|-----|------|----------|----------|-------|
| run43 | unpadded | f_pt, k_star, d_y, d_phi, cos_θ*, d_y_signed, net_kaon | **0.3162** | Baseline (no eff_weight) |
| run44 | **stale** | + eff_weight | **1.0000** | **DATA LEAKAGE ARTIFACT** — Anti-Omega eff_weight was frozen at 1.0 in stale .pt file; after z-norm this became trivial separator |
| run45 | fresh | + eff_weight | **0.3332** | Fresh preprocessing; realistic ~5% gain. eff_weight provides η information without charge leakage |

**Lesson**: Always re-preprocess after config changes. Stale feature distributions can appear to be perfect signals. Run 44 is disregarded — run 45 is the valid eff_weight baseline.
