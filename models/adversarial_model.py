"""adversarial_model.py — adversarial and density-ratio models for OmegaNet.

Contents:
  _TransformerBase       — shared Pre-LN Transformer backbone
  DensityRatioNet        — NCE log-density-ratio estimator (alias: WeighterNet)
  OmegaTransformerGRL    — classifier + GRL-based multiplicity adversary
"""
import torch
import torch.nn as nn


# ── Gradient Reversal ─────────────────────────────────────────────────────────

class _GradRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return _GradRevFn.apply(x, alpha)


# ── Shared backbone ───────────────────────────────────────────────────────────

class _TransformerBase(nn.Module):
    """Shared Pre-LN Transformer backbone: input projection + CLS token + encoder."""

    def __init__(self, in_channels, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def encode(self, x, padding_mask=None):
        """Return CLS token embedding, shape (B, d_model)."""
        B = x.shape[0]
        h = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        return h[:, 0]  # CLS token


class DensityRatioNet(_TransformerBase):
    """Estimates the log-density-ratio f(x) = log[p_Anti(x) / p_Omega(x)] + C.

    Trained with the NCE objective:
        L = -E_Anti[f(x)] + log E_Omega[exp f(x)]

    At the optimum, high f(x) ↔ event looks like Anti (pair-produced).
    Low f(x)  ↔ event looks like BN-transport Omega.

    Output is unconstrained (no sigmoid) — a log-density-ratio.
    Use exp(f(x)) ∝ density ratio weight for importance weighting;
    use f(x) directly for ranking/threshold sweep.
    """

    def __init__(self, in_channels, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__(in_channels, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, padding_mask=None):
        return self.head(self.encode(x, padding_mask)).squeeze(-1)  # (B,), unconstrained


# Alias kept for backward compatibility with any code importing WeighterNet
WeighterNet = DensityRatioNet


# ── GRL adversarial classifier ────────────────────────────────────────────────

class OmegaTransformerGRL(_TransformerBase):
    """OmegaTransformer + gradient-reversal adversary for multiplicity debiasing.

    Three heads share the same CLS embedding:
      1. classifier  — predicts Anti/Omega logits (main task)
      2. adv_cls     — classifies GLOBAL n_kaons quantile bin (n_bins classes)
                       Unconditional: no class label. CE loss. Prevents sign-flip Nash.
                       On unpadded data n_kaons ∝ class, so global bins directly target
                       the between-class multiplicity artifact.
      3. adv_reg     — regresses [n_kaons, std(f_i)] for each feature. Unconditional. MSE.

    Both adversary heads are UNCONDITIONAL (no class label). This forces the encoder
    to be blind to the raw kaon count and its distributional spread — the primary
    artifact on unpadded data where Ω⁻ events have many K⁺ and Ω̄⁺ events have few K⁻.

    Per-event MEANS of features are deliberately not targeted: they carry the genuine
    kinematic physics signal (junction fragmentation kinematics) we are trying to measure.
    """

    def __init__(self, in_channels, d_model, nhead, num_layers, dim_feedforward,
                 dropout=0.1, n_bins=5, n_stats=None):
        super().__init__(in_channels, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.n_bins  = n_bins
        # n_stats: n_kaons + std(f_i) for each non-broadcast classifier feature.
        # Defaults to 1 + in_channels for backwards compatibility.
        self.n_stats = n_stats if n_stats is not None else 1 + in_channels

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # Adversary: shared trunk → n_stats independent CE heads (one per statistic).
        # trunk width mirrors classifier; output reshaped to (B, n_stats, n_bins).
        self.adv_trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )
        self.adv_head = nn.Linear(d_model // 2, self.n_stats * n_bins)

    def forward(self, x, padding_mask=None, alpha=1.0):
        cls    = self.encode(x, padding_mask)                              # (B, d_model)
        logits = self.classifier(cls)                                      # (B, 2)
        h      = self.adv_trunk(grad_reverse(cls, alpha))                  # (B, d_model//2)
        adv    = self.adv_head(h).view(-1, self.n_stats, self.n_bins)      # (B, n_stats, n_bins)
        return logits, adv
