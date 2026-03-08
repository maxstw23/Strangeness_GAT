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

To resolve this, **kaon multiplicity is force-matched**: for each event, additional kaons of
the minority relative-sign are embedded by sampling from the global kaon momentum spectrum
until the number of same-sign and opposite-sign kaons are equal. The embedded kaons carry no
kinematic correlation with the Omega, ensuring that any signal the model learns must come from
the genuine spatial or momentum structure of real associated kaons.

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

The Omega score distribution will be a **mixture** of the two components. A threshold sweep
identifies the operating point that maximises separation quality.

**Primary downstream use**: Apply the trained model to the Ω⁻ sample and use p(x) as a
soft mechanism label. Profile kinematic observables (k*, Δy, cos θ*, flow angles, p_T) as a
function of p(x) to characterise how BN-carrying Omegas differ from pair-produced ones.
This provides the first data-driven, event-by-event handle on the junction production mechanism.

**Recall metrics are diagnostics, not the measurement**: Anti recall and Omega recall at a
chosen threshold are used to assess separation quality, but the physics output is the
differential kinematic profile, not a single number like f_BN.
