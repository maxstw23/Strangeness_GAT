# Physics Background

## Baryon Number Transport and the Gluon Junction

The origin of baryon number (BN) in high-energy nuclear collisions remains an open question.
In the conventional picture, baryon number is carried by valence quarks, each contributing 1/3.
An alternative picture proposes that baryon number is instead carried by a topological gluonic
structure called the **gluon junction** — a Y-shaped configuration where three gluon strings meet.
Such structures are notoriously difficult to probe experimentally.

## The Omega Hyperon as a Probe

The Omega hyperon (Ω⁻, quark content *sss*) is a uniquely sensitive probe of the junction
mechanism. Because it contains no up or down quarks, it shares no valence quark content with the
incoming nucleons. Any Ω⁻ that carries baryon number from the initial state must do so via the
junction, which "captures" three sea strange quarks. In contrast, the conventional string
fragmentation picture requires multiple string breaks to produce *ss̄* pairs, which is suppressed
by the strange quark mass.

## The Omega Surplus at BES-II Energies

At lower RHIC Beam Energy Scan II (BES-II) energies (√s_NN = 7.7–27 GeV), a significant surplus
of Ω⁻ over Ω̄⁺ is observed. For example, at √s_NN = 14.6 GeV, N(Ω) ≈ 2·N(Ω̄). Since strangeness
must be produced in pairs, this surplus implies that a substantial fraction of Ω⁻ carry net
baryon number transported from the initial-state nucleons via the junction mechanism.

This leads to the identification of **two distinct populations** of Ω⁻:

1. **BN-carrying Omegas**: produced via the junction mechanism, transporting baryon number from
   the incoming nuclei into the final state.
2. **Pair-produced Omegas**: produced via standard strangeness pair creation, kinematically
   analogous to Ω̄⁺ production.

## The Open Question and Working Hypothesis

There are currently no theoretical guidelines for distinguishing these two populations on an
event-by-event basis using kinematic or dynamic observables. As a working hypothesis, this project
assumes that the kaons produced in association with a BN-carrying Omega — through strangeness
conservation — will exhibit a characteristically different "distance" from the Omega compared to
kaons associated with a pair-produced Omega. Candidate observables for this separation include:

- **ΔR** = √(Δφ² + Δy²): geometric separation in azimuthal angle and rapidity
- **k\***: relative momentum in the pair rest frame (femtoscopic observable)
- **p_T**: transverse momentum of the Omega or associated kaon

## Project Goal

The goal of this machine learning project is to exploit these kinematic differences to find a
data-driven method for distinguishing between BN-carrying and pair-produced Omega hyperons,
providing a novel experimental handle on the gluon junction picture of baryon number transport.
