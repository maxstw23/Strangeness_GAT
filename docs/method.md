# Method

## Core Idea

The dataset consists of event-by-event (EbyE) records, each containing one Omega or anti-Omega
and the kaons reconstructed in the same event. The two Ω⁻ populations — BN-carrying and
pair-produced — are not labeled in data and cannot be separated by direct observation.

The key asymmetry that makes indirect separation possible:

- **All Ω̄⁺** are pair-produced (they cannot carry net baryon number via the junction).
- **Ω⁻** is a mixture: roughly half pair-produced, half BN-carrying (at BES-II energies where
  N(Ω) ≈ 2·N(Ω̄)).

This means Ω̄⁺ events serve as a **pure labeled sample of the pair-produced production mechanism**.
In the Positive-Unlabeled (PU) learning sense:
- **Positive (P)**: Ω̄⁺ events — known to be pair-produced.
- **Unlabeled (U)**: Ω⁻ events — mixture of pair-produced and BN-carrying.

The model is trained to learn p(x) ≈ P(pair-produced | kaon kinematics), using Ω̄⁺ as the
labeled proxy. **The goal is not to estimate f_BN** — that is well-constrained by thermal
models. The goal is to use p(x) as a **soft event-by-event label for production mechanism**,
enabling kinematic profiling of the two subpopulations.

## Blinding the Model to Baryon Identity

The model must not have access — direct or indirect — to the sign of the Omega candidate.
Two potential sources of leakage are addressed:

### 1. Kaon Charge Encoding

At BES-II energies, K+ are more abundant than K- globally. A model given absolute kaon charges
could trivially infer the Omega sign by counting same-sign vs opposite-sign kaons, or by
exploiting the spatial correlation between an Ω⁻ and its physically associated K+ partners.

To remove this, all kaon charges are encoded as **relative sign with respect to the Omega**:
each kaon is labeled *same-sign* or *opposite-sign* relative to the Omega candidate in that
event. Under this encoding, Ω⁻ and Ω̄⁺ events are symmetric by construction — the model sees
an identical feature space regardless of which baryon is present.

### 2. Kaon Multiplicity Matching

Even with relative-sign encoding, a residual shortcut exists: at BES-II energies, the global
K+/K- asymmetry means that Ω⁻ events tend to have more opposite-sign kaons than same-sign
kaons, while Ω̄⁺ events are closer to balance. A model could exploit this multiplicity
difference to infer the baryon sign without using any spatial or momentum information.

To resolve this, **kaon multiplicity is force-matched**: for each Ω̄⁺ event, additional K⁻ are
sampled until the K⁻ count equals the K⁺ count. Fake K⁻ are drawn from an **event-mixed pool**
built from other Ω̄⁺ events with similar Omega kinematics, binned on (|y_Ω|, pT_Ω) quartiles
(4×4 = 16 bins). This ensures fake kaons have realistic kinematic relationships to the Omega
they are paired with — a global pool was found to produce pathological d_y_signed values that
the model exploited as a padding artifact.

## Model Architecture

The classifier is an **OmegaTransformer** (`models/transformer_model.py`): a 2-layer Pre-LN
Transformer encoder with a CLS token, no positional encoding, and a 2-class output head.

```
Input: (n_kaons × 6) feature matrix per event
  ↓ Linear projection → d_model = 128
  ↓ Prepend CLS token
  ↓ TransformerEncoder (2 layers, 4 heads, FFN dim 256, Pre-LayerNorm)
  ↓ CLS token output → MLP classifier → 2-class softmax
Output: p(x) = P(pair-produced)
```

Key design choices:
- **No positional encoding**: the model is permutation-invariant over kaons — consistent with the
  physics, where there is no meaningful ordering among associated kaons.
- **CLS token aggregation**: the CLS token attends to all kaons and aggregates their evidence;
  this allows the model to learn multi-kaon correlations rather than single-kaon features.
- **274,626 parameters** — deliberately small to avoid overfitting on weak signal.

Input features (6, after ablation studies):

| Feature | Description |
|---|---|
| `f_pt` | Kaon pT [GeV/c] |
| `k*` | Lorentz-invariant relative momentum in K-Ω pair rest frame |
| `\|y_K − y_Ω\|` | Absolute rapidity separation |
| `\|φ_K − φ_Ω\|` | Absolute azimuthal separation |
| `cos θ*` | Beam-axis angle in pair rest frame |
| `\|y_K\| − \|y_Ω\|` | Midrapidity proximity gap (signed) |

Excluded features: `o_pt` (p̄ absorption biases Ω̄⁺ efficiency), `o_y_abs` (introduces biased
rapidity shift), EP cosines (confirmed inert, run18).

## Loss Function

The current implementation uses **per-class-mean BCE**:

```
loss = mean(−log p(x)     | Anti events)    # P(pair-produced) → 1 for labeled positives
     + mean(−log(1−p(x))  | Omega events)   # P(pair-produced) → 0 for unlabeled
```

This is a biased approximation of the true PU objective: it incorrectly penalizes
pair-produced Ω⁻ for "looking like Anti," since the Omega target of 0 is wrong for the
pair-produced sub-population. However, in practice the threshold sweep (see below) compensates
post-hoc, and BCE produces more stable training than PU losses in the weak-signal regime.

**Why PU losses fail here**: Standard nnPU/uPU losses are designed for problems where positive
and negative classes are reasonably separable. In our case, pair-produced Ω⁻ and Ω̄⁺ have
*identical* kaon kinematics by hypothesis — the PU correction subtracts a quantity of the same
order as the signal, introducing variance that overwhelms the gradient. BCE with threshold sweep
is a lower-variance estimator for the same underlying quantity.

## Expected Outcome and Downstream Use

If BN-carrying and pair-produced Omegas differ in their associated kaon distributions, the
model should produce a discriminating score p(x):

| Population | Score distribution |
|---|---|
| Ω̄⁺ (all pair-produced) | p(x) concentrated near 1 |
| Pair-produced Ω⁻ | p(x) concentrated near 1 (same mechanism) |
| BN-carrying Ω⁻ | p(x) concentrated near 0 (different mechanism) |

The Omega score distribution will be a **mixture** of the two components. The operating point
is fixed at **Anti recall = 0.90** (O@A=0.90); Omega recall at this threshold is the primary
training metric.

**Primary downstream use**: Apply the trained model to the Ω⁻ sample and use p(x) as a
soft mechanism label. Profile kinematic observables (k*, Δy, cos θ*, flow angles, p_T) as a
function of p(x) to characterise how BN-carrying Omegas differ from pair-produced ones.
This provides the first data-driven, event-by-event handle on the junction production mechanism.

**Recall metrics are diagnostics, not the measurement**: The physics output is the
differential kinematic profile, not a single number like f_BN.

## Subpopulation Purity Estimation

Given the PU mixture model p_Ω(x) = π · p_PP(x) + (1−π) · p_BN(x), the mechanism purity
of any score-selected subset S of Ω⁻ events is:

```
PP purity(S) = π · f_Anti(S) / f_Omega(S)
BN purity(S) = [f_Omega(S) − π · f_Anti(S)] / f_Omega(S)
```

where f_X(S) = fraction of class X whose score falls in S, and π = N(Ω̄⁺)/N(Ω⁻) ≈ 0.478
(estimated from dataset proportions, consistent with thermal model expectation at BES-II).

The optimal operating points (maximising purity × √N figure of merit) are:

| Sample | Score cut | Mechanism purity | N events |
|---|---|---|---|
| PP-enriched  | Top 32% of Ω⁻ by score | 99% pair-produced | ~35,600 |
| BN-enriched  | Bottom 65% of Ω⁻ by score | 77% BN-carrying | ~72,000 |
| BN-enriched (tight) | Bottom 30% of Ω⁻ by score | 86% BN-carrying | ~33,000 |

The PP-enriched sample is essentially saturated at 100% purity for all cuts stricter than ~32%,
because the model places nearly all Anti events above that threshold. The BN sample purity
rises slowly as the cut tightens, so the FOM optimum is at a relatively loose cut (65%).
