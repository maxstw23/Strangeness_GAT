import torch
import os

# Paths and Naming
DATA_PATH = "data/balanced_omega_anti.pt"
STATS_PATH = "data/balanced_omega_anti_stats.pt"
DATA_PATH_UNPADDED  = "data/unpadded_omega_anti.pt"
STATS_PATH_UNPADDED = "data/unpadded_omega_anti_stats.pt"
MODEL_SAVE_PATH = "models/omega_transformer.pth"
os.makedirs("models", exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
EPOCHS = 200
EARLY_STOP_PATIENCE = 25  # stop if no new best Anti recall for this many epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PFN hyperparameters (kept for evaluate_physics.py backward compat)
HIDDEN_CHANNELS = 64

KSTAR_CLIP = 8.0

# ── Kaon efficiency correction (2D: pT, eta) ──────────────────────────────────
# GetEfficiency2D formula: (P0 + P3*pT + P4*pT²) * exp(-(P1/pT)^P2) * exp(-((eta/P5)²)^P6)
# Parametrized separately for K+ and K- across 9 centrality bins (indices cent-1)
EFFICIENCY_PARAMS = {
    "Kp": {
        "P0": [0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441],
        "P1": [0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589],
        "P2": [2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077],
        "P3": [0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208],
        "P4": [-0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983],
        "P5": [1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937],
        "P6": [3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326],
    },
    "Km": {
        "P0": [0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441, 0.77991441],
        "P1": [0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589, 0.16110589],
        "P2": [2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077, 2.1266077],
        "P3": [0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208, 0.038097208],
        "P4": [-0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983, -0.0048513983],
        "P5": [1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937, 1.5617937],
        "P6": [3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326, 3.4819326],
    },
}

# ── Feature configuration ──────────────────────────────────────────────────────
# All features produced by preprocess_data.py in canonical order.
# Indices 0–9: balanced (padded) dataset; indices 0–10: unpadded dataset.
# In the unpadded dataset index 2 ("d_y") stores the directed rapidity gap
# d_y_signed = sign(y_Omega) × (y_K − y_Omega); index 10 is o_y_abs.
FEATURE_REGISTRY = [
    "f_pt", "k_star", "d_y", "d_phi", "o_pt", "cos_theta_star",   # 0–5
    "o_cos_psi1", "o_cos2_psi2", "f_cos_psi1", "f_cos2_psi2",     # 6–9
    "d_y_signed",                                                    # 10: |y_K|−|y_Ω| (padded+unpadded)
    "o_y_abs",                                                       # 11: |y_Ω| broadcast
    "net_kaon",                                                      # 12: n_K+(real)−n_K−(real), broadcast
    "eff_weight",                                                    # 13: inverse efficiency 1/ε(pT,η) — correction weight (low ε → high weight)
]

# Active features for this run — edit ONLY this list to change the feature set.
# o_pt excluded: p̄ absorption asymmetry biases Ω̄⁺ reconstruction efficiency.
FEATURE_NAMES = [
    "k_star", "d_y", "d_phi", "cos_theta_star",
    "d_y_signed", "net_kaon", "eff_weight"
]

# Derived — do not edit manually.
FEATURE_IDX = [FEATURE_REGISTRY.index(f) for f in FEATURE_NAMES]
IN_CHANNELS = len(FEATURE_NAMES)
KSTAR_IDX   = FEATURE_NAMES.index("k_star") if "k_star" in FEATURE_NAMES else None

# ── Efficiency helper function ─────────────────────────────────────────────────
def get_efficiency_2d(pT, eta, cent, particle):
    """
    Compute kaon detection efficiency from 2D (pT, eta) parametrization.

    Args:
        pT: kaon transverse momentum (GeV/c)
        eta: kaon pseudorapidity
        cent: centrality bin (1–9)
        particle: "Kp" or "Km"

    Returns:
        efficiency (float, 0–1)
    """
    import numpy as np
    params = EFFICIENCY_PARAMS[particle]
    idx = cent - 1  # 0-indexed array
    p0, p1, p2, p3, p4, p5, p6 = (
        params["P0"][idx], params["P1"][idx], params["P2"][idx],
        params["P3"][idx], params["P4"][idx], params["P5"][idx], params["P6"][idx],
    )
    pt_term = (p0 + p3*pT + p4*pT**2) * np.exp(-np.power(p1/pT, p2))
    eta_term = np.exp(-np.power(np.power(eta/p5, 2), p6))
    return pt_term * eta_term

def get_inv_eff(pT, eta, cent, charge):
    """Inverse efficiency 1/ε(pT, η) for the kaon's actual charge.
    Low efficiency → high weight: each detected kaon represents more true kaons.
    charge: +1 for K+, -1 for K-.
    """
    particle = "Kp" if charge > 0 else "Km"
    eps = get_efficiency_2d(pT, eta, cent, particle)
    return 1.0 / max(eps, 1e-4)

# Transformer Hyperparameters
D_MODEL = 128
NHEAD = 4              # head_dim = 32
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT_RATE = 0.1