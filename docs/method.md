# Method

## Core Idea

The dataset consists of event-by-event (EbyE) records, each containing one Omega or anti-Omega
and the kaons reconstructed in the same event. The two populations of Ω⁻ — BN-carrying and
pair-produced — are not labeled in data and cannot be separated by direct observation. However,
the following asymmetry motivates an indirect approach:

- **All Ω̄⁺** are pair-produced (they cannot carry net baryon number).
- **Ω⁻** is a mixture: roughly half pair-produced, half BN-carrying (at BES-II energies where
  N(Ω) ≈ 2·N(Ω̄)).

This means Ω̄⁺ events serve as a **pure sample** of the pair-produced production mechanism.
If a pair-produced Ω⁻ is kinematically indistinguishable from a Ω̄⁺, then a model trained to
classify Omega vs anti-Omega events purely from kinematics should:

- Correctly identify **all** Ω̄⁺ events (high anti-Omega recall).
- Correctly identify only the **BN-carrying** Ω⁻ events (≈ 50% Omega recall), while
  misclassifying the pair-produced Ω⁻ as anti-Omega.

The model's recall asymmetry is therefore a direct, data-driven estimator of the BN-carrying
fraction of Ω⁻.

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

## Expected Outcome and Interpretation

If the kinematic hypothesis holds — that BN-carrying and pair-produced Omegas differ in their
associated kaon distributions — the trained model should exhibit a characteristic recall
asymmetry:

| Population        | Expected Model Behavior                         |
|-------------------|-------------------------------------------------|
| Ω̄⁺ (all pair-produced) | Correctly classified → high anti-Omega recall   |
| Pair-produced Ω⁻  | Misclassified as anti-Omega (looks the same)    |
| BN-carrying Ω⁻   | Correctly classified → contributes to Omega recall |

The **BN-carrying fraction** of Ω⁻ is then estimated as:

```
f_BN = Omega recall = TP_Omega / (TP_Omega + FN_Omega)
```

At 14.6 GeV, where N(Ω) ≈ 2·N(Ω̄), we expect f_BN ≈ 0.5 if the model works as intended.
A deviation from this expectation would indicate either incomplete separation or a violation
of the kinematic hypothesis.
