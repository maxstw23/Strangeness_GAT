# Feature Ideas

## Current features: [f_pt, k_star, d_eta, d_phi]
- o_pt removed: biased by pT-dependent detector efficiency and p̄→Λ̄→Ω̄⁺ feed-down
- rel_q removed: only opposite-sign kaons kept (strangeness-balancing partners)
- f_pt kept for now but also acceptance-biased; may revisit

---

## Candidate additions

### 1. cos(θ*) — pair orientation in the rest frame

**Definition:** cosine of the polar angle of the kaon momentum vector **k*** in the
K-Omega pair rest frame (PRF), with respect to the beam axis (z).

    cos(θ*) = k*_z / |k*|

**Key property:** In the PRF the kaon has momentum **k*** and the Omega has **−k***.
They are exactly antiparallel, so cos(θ*) is a property of the *pair*, not of either
individual particle. No ambiguity.

**Complement to k*:** k* is the magnitude of the relative momentum; cos(θ*) is its
direction. Together they are the two independent Lorentz-invariant quantities fully
characterising the pair kinematics (the azimuthal angle φ* averages out by symmetry).

**Physics motivation:** The femtoscopic source in heavy-ion collisions is elongated
along the beam axis (R_long > R_out ~ R_side), so the correlation function
CF(**k***) is not spherically symmetric — it depends on both |k*| and θ*.
Using only scalar k* integrates over all directions and loses this directional
information. If BN-carrying Ω⁻ (junction fragmentation, longitudinal boost) produce
a more anisotropic k* distribution than pair-produced Ω̄⁺, cos(θ*) could be
discriminating.

**Bias concern:** None in principle — K⁺/Ω⁻ and K⁻/Ω̄⁺ pairs are symmetric under
charge conjugation in a symmetric detector. cos(θ*) is purely kinematic.

**Implementation:** Already compute the full Lorentz boost for k* in preprocess_data.py;
extracting k*_z costs nothing extra.

**Caveat:** If the source is approximately spherical at the scale of the signal,
cos(θ*) will be uniform and uninformative. Effectiveness is an open question.

---

### 2. Δy — rapidity difference (instead of / alongside Δη)

**Definition:**
    Δy = y_K − y_Ω
    y = 0.5 × ln((E + p_z) / (E − p_z))

**Why better than Δη:** For massive particles (m_K = 494 MeV, m_Ω = 1672 MeV),
rapidity and pseudorapidity differ non-trivially, especially at low pT. Δy is the
physically correct Lorentz-invariant longitudinal separation. Δη is only an
approximation valid in the massless (ultrarelativistic) limit.

**Bias concern:** None — Δy is symmetric between Ω⁻/K⁺ and Ω̄⁺/K⁻ pairs.

**Implementation:** Requires computing particle energies (already done for k*).
