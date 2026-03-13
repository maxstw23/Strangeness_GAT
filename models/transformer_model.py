import torch
import torch.nn as nn


class OmegaTransformerEdge(nn.Module):
    """OmegaTransformer with pairwise kaon-kaon edge bias in attention.

    For each pair (i, j) of kaons, computes geometric edge features
    (ΔR, Δy, cos Δφ) and maps them to per-head attention biases via a small
    MLP.  The bias is added to the attention logits before softmax, allowing
    the model to explicitly see inter-kaon spatial structure that CLS pooling
    alone cannot capture.

    Args:
        dy_idx:   index of d_y   in the (post-normalisation) feature tensor
        dphi_idx: index of d_phi in the (post-normalisation) feature tensor
    """

    def __init__(self, in_channels, d_model, nhead, num_layers,
                 dim_feedforward, dropout=0.1, dy_idx=1, dphi_idx=2):
        super().__init__()
        self.nhead    = nhead
        self.dy_idx   = dy_idx
        self.dphi_idx = dphi_idx

        self.input_proj = nn.Linear(in_channels, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # 3 edge features → one additive bias per attention head
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, nhead),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x, padding_mask=None):
        # x: (B, N, C) — normalised kaon features
        B, N, _ = x.shape

        # ── Pairwise edge features ────────────────────────────────────────────
        dy   = x[:, :, self.dy_idx  ].unsqueeze(2) - x[:, :, self.dy_idx  ].unsqueeze(1)  # [B,N,N]
        dphi = x[:, :, self.dphi_idx].unsqueeze(2) - x[:, :, self.dphi_idx].unsqueeze(1)  # [B,N,N]
        dR   = torch.sqrt(dy**2 + dphi**2 + 1e-8)                                          # [B,N,N]

        edge_feat = torch.stack([dR, dy, torch.cos(dphi)], dim=-1)   # [B,N,N,3]
        edge_bias = self.edge_mlp(edge_feat)                           # [B,N,N,nhead]

        # Reshape to [B*nhead, N, N]; CLS gets zero bias → pad to [B*nhead, N+1, N+1]
        edge_bias = edge_bias.permute(0, 3, 1, 2).reshape(B * self.nhead, N, N)
        attn_bias = torch.zeros(B * self.nhead, N + 1, N + 1,
                                device=x.device, dtype=x.dtype)
        attn_bias[:, 1:, 1:] = edge_bias   # kaon-kaon block; CLS rows/cols = 0

        # ── Standard forward ──────────────────────────────────────────────────
        h   = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        h   = torch.cat([cls, h], dim=1)   # [B, 1+N, d_model]

        if padding_mask is not None:
            cls_mask     = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        h = self.transformer(h, mask=attn_bias, src_key_padding_mask=padding_mask)
        return self.classifier(h[:, 0])


class OmegaTransformer(nn.Module):
    def __init__(self, in_channels, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, d_model)

        # Learned CLS token — its output is used for classification
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

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x, padding_mask=None):
        # x: (batch, seq_len, in_channels)
        batch_size = x.shape[0]

        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (batch, 1+seq_len, d_model)

        # Extend padding mask — CLS token is never masked
        if padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        h = self.transformer(h, src_key_padding_mask=padding_mask)

        # Classify from CLS token
        return self.classifier(h[:, 0])
