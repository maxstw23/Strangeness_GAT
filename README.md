# OmegaNet

Machine learning for Omega hyperon production in heavy-ion collisions at RHIC BES-II energies.

## Overview

OmegaNet studies baryon number transport via the gluon junction picture using a 2-layer Pre-LN Transformer model with pileup classification and charge blinding.

**Key Features:**
- Preprocesses ROOT data into unpadded, unbalanced datasets with subsampling
- Trains OmegaTransformer with per-class-mean BCE loss
- Evaluates physics metrics (recall at Purity=0.90)
- Permutation importance and attention analysis

## Setup

```bash
bash setup.sh
source venv/bin/activate
```

This installs PyTorch (CUDA-enabled), tqdm, uproot, and awkward.

**Requirements:**
- Python 3.8+
- CUDA 12.8+
- RTX 3070 (or CUDA-capable GPU)

## Quick Start

```bash
# Preprocess ROOT data (unpadded, unbalanced with subsampling)
python scripts/preprocess_data.py

# Train model
python scripts/train.py

# Evaluate on validation set
python scripts/evaluate_physics.py

# Plot recall tradeoff
python scripts/plot_recall_tradeoff.py

# Interpret model
python scripts/interpret_model.py
```

## Project Structure

```
OmegaNet/
├── config.py                      # Centralized configuration
├── models/
│   ├── transformer_model.py       # OmegaTransformer architecture
│   └── omega_transformer.pth      # Best checkpoint
├── scripts/
│   ├── preprocess_data.py         # ROOT → balanced .pt
│   ├── train.py                   # Train OmegaTransformer
│   ├── evaluate_physics.py        # Physics metrics
│   ├── plot_recall_tradeoff.py    # Recall vs Purity curve
│   └── interpret_model.py         # Attention + importance
├── data/
│   └── omega_anti.pt              # Preprocessed dataset (unpadded, unbalanced)
└── plots/                         # Evaluation outputs
```

## Training

**Loss:** Per-class-mean BCE
- `mean(−log p | Anti) + mean(−log(1−p) | Omega)`

**Checkpoint:** Best O@A=0.90 on validation set

**Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)

## Documentation

- **`docs/physics.md`** — Physics motivation and baryon number transport
- **`docs/method.md`** — ML strategy and feature engineering
- **`docs/pipeline.md`** — End-to-end workflow
- **`docs/experiments.md`** — All training runs and outcomes
- **`docs/lessons.md`** — Hard-won rules and debugging tips

## License

Internal research project.
