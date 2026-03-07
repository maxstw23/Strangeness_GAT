# Model Interpretation Plan

## Overview

The trained OmegaTransformer classifies events as Ω⁻ or Ω̄⁺ from the kinematics of ~35 kaons per
event. The threshold used for the binary decision is a post-hoc choice and does not affect the
quality of the learned representation. The central physics question is:

> **Which features, or combinations of features across kaons in an event, drive the decision boundary?**

If the model has learned the BN-carrying signal, we expect it to rely on kaons that are
kinematically correlated with the Omega — i.e., low k*, small ΔR — rather than on bulk event
properties. The four analyses below probe this from different angles.

---

## Analysis 1 — Attention Analysis

### What it is
For each event, the CLS token aggregates information from all kaon tokens via attention. The
attention weights from CLS→kaon in the **final transformer layer** encode which kaons the model
considers most informative for the classification decision.

For every validation event we extract:
- The CLS attention weight assigned to each kaon (after softmax over the sequence)
- The **attention-weighted mean** of each feature per event (a soft summary of "what the model
  looks at")
- The features of the **single most-attended kaon** per event

These are then compared between Omega and Anti events.

### Expected findings if physics signal is present
- The model should attend more strongly to kaons with **small k*** (femtoscopically correlated,
  likely produced in association with the Omega via strangeness conservation)
- Attended kaons should have **smaller ΔR** (geometrically close to the Omega)
- For Omega events, the most-attended kaon may preferentially be **opposite-sign** (K⁺, the
  strangeness-balancing partner of Ω⁻)
- The attention-weighted k* distribution should differ between Omega and Anti events, while the
  raw (uniform-weight) distribution does not — this would confirm the model is exploiting
  within-event structure rather than bulk statistics

### Caveats
- Attention weights are not guaranteed to correspond to causal importance (attention ≠ attribution)
- With 2 layers and 4 heads, attention in layer 2 already incorporates information mixed by layer 1
- The broadcast feature `o_pt` is identical for all kaons in an event; its attention-weighted mean
  simply equals the Omega pT and is not a meaningful per-kaon attention signal

---

## Analysis 2 — Feature Permutation Importance

### What it is
For each of the 6 input features ([f_pt, k_star, d_eta, d_phi, rel_q, o_pt]), permute its values
**globally across all events and kaons** in the validation set, breaking the correlation between
that feature and the event label. Measure the drop in model score (Anti recall + Omega recall − 1
at the optimal threshold).

A large score drop means the model relies heavily on that feature. A small or negative drop
(permuted sometimes better than baseline) means the feature contributes little.

Repeated N times per feature to get mean ± std of the importance score.

### Expected findings
- **k_star** should be important if the femtoscopic signal is real
- **o_pt** should be important if the BN-carrying fraction is pT-dependent (junction stopping
  prefers lower Omega pT)
- **d_eta, d_phi** (and thus ΔR) may be important if spatial proximity encodes the signal
- **rel_q** may or may not matter — after multiplicity matching the charge distribution is
  symmetric, but the model may still exploit the fact that strangeness-balancing K⁺ are
  opposite-sign to Ω⁻
- **f_pt** (kaon pT) is the least physically motivated and likely least important

### Caveats
- Permuting `o_pt` permutes a broadcast feature — all kaons in an event get a different (randomly
  drawn) Omega pT, effectively destroying the event-level context. This is a valid test.
- Cross-event permutation preserves the marginal distribution of each feature but destroys its
  correlation with all other features and with the label. Single-feature drops can underestimate
  importance when features are correlated (e.g., k_star and ΔR are related).

---

## Analysis 3 — Gradient × Input Attribution

### What it is
For each validation event, compute the gradient of the model's Anti-class logit with respect to
each input feature of each kaon: ∂logit_Anti / ∂x_{i,f}. Multiply by the input value (integrated
gradients approximation) to get a per-kaon, per-feature attribution score.

Average the magnitude of this attribution over all events, separated by Omega vs Anti class, to
get a global feature sensitivity map.

### Expected findings
- High gradient magnitude on k_star at small k* values would confirm femtoscopic sensitivity
- If o_pt has high gradient magnitude, the model is using Omega pT as a discriminating context
- Comparing gradient maps between Omega and Anti events may reveal asymmetric feature usage

### Caveats
- Gradient × input can be noisy for saturated predictions (gradient vanishes near p=0 or p=1)
- Unlike permutation importance, it is a local, per-sample measure; the global average may wash
  out event-specific patterns
- Should be used to cross-check Analysis 2, not as a standalone result

---

## Analysis 4 — CLS Embedding Correlation with Event-Level Summaries

### What it is
Extract the CLS token embedding (d_model-dimensional vector) after the final transformer layer for
every validation event. Project it to 2D via PCA or UMAP and visualize, coloring by:
- Event label (Omega vs Anti)
- Per-event minimum k* (the closest kaon in momentum space)
- Per-event Omega pT
- Model score P(Anti | event)

Additionally, compute the Pearson correlation between the scalar model score P(Anti | event) and
per-event summary statistics (mean ΔR, min k*, o_pt, kaon multiplicity).

### Expected findings
- If the CLS embedding separates by class in PCA/UMAP space, the model has learned a meaningful
  latent representation of event type
- Strong correlation between P(Anti) and min k* per event would confirm the femtoscopic hypothesis
- Strong correlation with o_pt would confirm junction pT dependence
- Weak correlation with kaon multiplicity confirms the model is not exploiting a multiplicity
  artifact (which would be a red flag)

### Caveats
- PCA/UMAP are dimensionality reduction tools and may not reveal all structure
- Correlation between P(Anti) and event summaries is observational, not causal
