import torch
import os

# Paths and Naming
DATA_PATH = "data/balanced_omega_anti.pt"
STATS_PATH = "data/balanced_omega_anti_stats.pt"
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

# ── Feature configuration ──────────────────────────────────────────────────────
# All features produced by preprocess_data.py in canonical order.
# TODO: add "phi_omega_minus_psi_ep" (event plane angle) once ROOT file is updated.
FEATURE_REGISTRY = ["f_pt", "k_star", "d_y", "d_phi", "o_pt", "cos_theta_star"]

# Active features for this run — edit ONLY this list to change the feature set.
# o_pt excluded: p̄ absorption asymmetry biases Ω̄⁺ reconstruction efficiency.
FEATURE_NAMES = ["f_pt", "k_star", "d_y", "d_phi", "cos_theta_star"]

# Derived — do not edit manually.
FEATURE_IDX = [FEATURE_REGISTRY.index(f) for f in FEATURE_NAMES]
IN_CHANNELS = len(FEATURE_NAMES)
KSTAR_IDX   = FEATURE_NAMES.index("k_star") if "k_star" in FEATURE_NAMES else None

# Transformer Hyperparameters
D_MODEL = 128
NHEAD = 4              # head_dim = 32
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT_RATE = 0.1