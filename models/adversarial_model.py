import torch
import torch.nn as nn


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
