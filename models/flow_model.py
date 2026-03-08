"""
RealNVP normalizing flow for Anti-Omega density estimation.

Architecture: 6 affine coupling layers with alternating split direction.
Each coupling layer uses an MLP to compute scale and shift from the
first half of the input, transforming the second half.

Input: 21-dimensional per-event summary vector (see train_flow.py for
aggregation details).
"""
import torch
import torch.nn as nn
import math


class CouplingLayer(nn.Module):
    """Affine coupling layer (Dinh et al. 2017, RealNVP).

    Splits input into (x1, x2) at index `split_idx`.
    Transforms x2 using scale/shift computed from x1; x1 passes through.
    Alternates which half is transformed via the `reverse` flag.
    """

    def __init__(self, dim: int, split_idx: int, hidden: int = 128, reverse: bool = False):
        super().__init__()
        self.split_idx = split_idx
        self.reverse = reverse
        d1 = split_idx if not reverse else (dim - split_idx)
        d2 = dim - d1

        self.net = nn.Sequential(
            nn.Linear(d1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d2 * 2),  # outputs [s, t] concatenated
        )

    def _split(self, x):
        if not self.reverse:
            return x[:, :self.split_idx], x[:, self.split_idx:]
        else:
            return x[:, self.split_idx:], x[:, :self.split_idx]

    def _merge(self, x1, x2):
        if not self.reverse:
            return torch.cat([x1, x2], dim=1)
        else:
            return torch.cat([x2, x1], dim=1)

    def forward(self, x):
        """x → (y, log_det_jacobian)"""
        x1, x2 = self._split(x)
        st = self.net(x1)
        s_raw, t = st.chunk(2, dim=1)
        # Clamp scale for stability: s ∈ [-2, 2]
        s = 2.0 * torch.tanh(s_raw / 2.0)
        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return self._merge(x1, y2), log_det

    def inverse(self, y):
        """y → x (for sampling)"""
        x1, y2 = self._split(y)
        st = self.net(x1)
        s_raw, t = st.chunk(2, dim=1)
        s = 2.0 * torch.tanh(s_raw / 2.0)
        x2 = (y2 - t) * torch.exp(-s)
        return self._merge(x1, x2)


class RealNVP(nn.Module):
    """Stack of affine coupling layers forming a normalizing flow.

    Base distribution: N(0, I_dim).
    """

    def __init__(self, dim: int = 21, n_layers: int = 6, hidden: int = 128):
        super().__init__()
        split_idx = dim // 2
        self.layers = nn.ModuleList([
            CouplingLayer(dim, split_idx, hidden=hidden, reverse=(i % 2 == 1))
            for i in range(n_layers)
        ])
        self.dim = dim
        self._log_2pi = math.log(2 * math.pi)

    def forward(self, x):
        """Map data → latent space. Returns (z, total_log_det)."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            log_det += ld
        return z, log_det

    def inverse(self, z):
        """Map latent → data space (for sampling)."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        """Log probability under the flow model: log p(x)."""
        z, log_det = self.forward(x)
        log_p_z = -0.5 * (self.dim * self._log_2pi + (z ** 2).sum(dim=1))
        return log_p_z + log_det

    @torch.no_grad()
    def sample(self, n: int, device='cpu'):
        """Sample n points from the learned distribution."""
        z = torch.randn(n, self.dim, device=device)
        return self.inverse(z)
