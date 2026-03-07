# Data Exploration Findings

**Date**: 2026-03-04
**Script**: `scripts/explore_data.py`
**Plot**: `plots/data_exploration.png`

---

## Dataset Overview

| | Omega (Ω⁻) | Anti (Ω̄⁺) | Ratio |
|---|---|---|---|
| Events | 103,751 | 50,081 | 2.072 |
| Total kaons | 3,663,776 | 1,748,730 | — |
| Kaons / event | 35.31 ± 8.69 | 34.92 ± 8.58 | — |
| same-sign kaons | 50.0% | 50.0% | — |

The 2.072 ratio is consistent with N(Ω)/N(Ω̄) at 14.6 GeV. Kaon multiplicity and same/opp-sign balance are symmetric between classes by construction (multiplicity matching worked correctly).

---

## Feature Distributions: Per-Kaon Level

| Feature | Omega mean | Anti mean | Omega std | Anti std |
|---|---|---|---|---|
| f_pt [GeV/c] | 0.6040 | 0.6043 | 0.339 | 0.339 |
| d_pt [GeV/c] | 1.860 | 1.910 | 0.727 | 0.717 |
| d_eta | 0.0094 | 0.0093 | 1.093 | 1.070 |
| d_phi | 0.0008 | 0.0005 | 1.815 | 1.816 |
| rel_q | 0.000 | 0.000 | 1.000 | 1.000 |

**All five per-kaon distributions are nearly indistinguishable between Omega and Anti events.** The means agree to 3+ significant figures. No 1D feature shows visible separation.

---

## ΔR = √(Δη² + Δφ²)

| | Omega | Anti |
|---|---|---|
| Mean ΔR (per kaon) | 1.945 | 1.935 |
| Per-event mean ΔR | 1.945 | 1.934 |
| Per-event min ΔR | 0.378 | 0.376 |

ΔR distributions are nearly identical. Even the **nearest kaon** per event sits at ΔR ~ 0.38 on average — not particularly close to the Omega — and is indistinguishable between classes.

---

## Key Structural Observations

### 1. d_phi is approximately uniform over [−π, π]
The kaons have no azimuthal clustering around the Omega. This is the expected signature of a high-multiplicity heavy-ion event where uncorrelated background kaons dominate. If there were jet-like production or a clear spatial partner kaon, we would see a peak near Δφ = 0.

### 2. d_eta is broadly Gaussian (std ~ 1.1), centered at 0
No rapidity clustering. Kaons are spread across the detector acceptance with no preference to be near the Omega.

### 3. f_pt is capped at 2 GeV/c with a soft exponential spectrum
The kaon pT spectrum peaks at low pT (~0.3 GeV/c) and is hard-cut at 2 GeV/c by the detector acceptance. Identical for both classes.

### 4. rel_q is 50/50 by construction
The multiplicity-matching procedure has fully symmetrized the charge balance. There is no residual charge asymmetry.

### 5. Same-sign vs opposite-sign kaons are kinematically identical within each class
Within both Omega and Anti events, the d_eta and d_phi distributions of same-sign and opposite-sign kaons overlap completely. There is no hint that one charge group is more spatially correlated with the Omega than the other.

---

## Implications for the Model

### The signal is not visible in 1D marginals
With ~35 kaons per event, the vast majority are uncorrelated background kaons. Any kinematic difference between BN-carrying and pair-produced Omegas — if it exists — is buried under this background. The model cannot find signal by looking at single-kaon features; it must learn a **collective, multi-kaon pattern**.

### The Transformer architecture is in principle well-suited
The CLS-token Transformer can attend over all kaons and form a global event representation. It is not restricted to single-kaon statistics. The fact that the trained model achieves a non-trivial score (~0.40) despite the flat marginals confirms there is some multi-kaon signal present.

### Signal is very subtle — the dominant challenge is the signal-to-noise ratio
With ~35 kaons, even if 1–2 are genuinely associated with the Omega, the input is ~95–97% noise. The model must learn to identify and weight the few informative kaons — an attention-based "filtering" task rather than a bulk-statistics task.

### The physically motivated features (ΔR, k*) show no gross separation
This may mean:
- The BN-carrying signal manifests as a correlation pattern across multiple kaons rather than a single-kaon anomaly.
- The signal in ΔR or d_pt may exist but only for a small subset of kaons (the "partner" kaons) that are diluted in the per-kaon statistics.
- Alternatively, k* (relative momentum in the pair rest frame) is not computed here and may carry more discriminating power than d_pt.

---

## k* as a Feature (updated 2026-03-04)

`d_pt` was replaced by k* (Lorentz-invariant relative momentum in the kaon-omega pair rest frame):

```
k* = sqrt[(s - (m_K + m_Ω)²)(s - (m_K - m_Ω)²)] / (2√s)
s  = (E_kaon + E_omega)² - |p_kaon + p_omega|²
```

using PDG masses m_K = 0.493677 GeV/c², m_Ω = 1.67245 GeV/c².

| | Omega | Anti |
|---|---|---|
| k* mean [GeV/c] | 0.8044 | 0.8083 |
| k* std [GeV/c] | 0.388 | 0.386 |
| k* range [GeV/c] | [0.004, 3.57] | [0.006, 3.59] |

**k* shows no meaningful 1D separation** — the difference in means (0.004 GeV/c) is negligible relative to the std (0.39 GeV/c). The distribution shapes are indistinguishable between classes.

Femtoscopically, the region of interest is k* < 0.2 GeV/c (quantum correlation / source-size regime). This region is present in the data but does not produce visible class separation in the marginal distribution.

**Overall conclusion from all data exploration**: No individual kaon feature — f_pt, k*, d_eta, d_phi, rel_q, ΔR — shows class separation in 1D marginal distributions. The signal, if present, lives entirely in multi-kaon correlations across the ~35 kaons per event. This makes the Transformer's attention mechanism the essential component: it must learn to identify the small subset of correlated kaons and suppress the background.

---

## Open Questions / Next Steps

1. **Look at the closest-kaon features**: Extract d_eta, d_phi, d_pt for only the kaon with minimum ΔR per event. Does this show any Omega/Anti separation?
2. **k* as a feature**: Consider computing the Lorentz-invariant k* = |p_kaon - p_omega|/2 in the pair rest frame as an additional or replacement feature. It is more physically motivated than d_pt.
3. **Event-level correlation features**: Does the variance of ΔR within an event, or the number of kaons inside ΔR < 0.5, differ between classes?
4. **Training strategy**: Given that the signal is subtle and multi-kaon, the model architecture is sound. The training instability is the real obstacle to reaching the physics target operating point (Anti recall → 1, Omega recall → 0.5).
