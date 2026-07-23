"""VAT FiLM conditioning (design: notes/vat-conditioning-design.md).

A per-token/per-frame [B, vat_dim, T] conditioning sequence is projected by a
shared trunk, then per-block zero-initialized heads emit channel-wise scale
and shift: x = x * (1 + gamma) + beta. Zero-init makes the conditioned
network exactly the unconditioned checkpoint at step 0 of a warm start.

All ops are pointwise Conv1d + mul/add — the op families already proven
through the litert-torch fixed-shape export path.
"""

import torch
import torch.nn as nn


class VATTrunk(nn.Module):
    """Shared projection of the raw VAT sequence to the conditioning space.

    Input [B, vat_dim, T] -> [B, cond_dim, T]. Kept in-graph so exported
    graphs take the raw, model-independent [1, vat_dim, T] tensor as input.
    """

    def __init__(self, vat_dim=3, cond_dim=256):
        super().__init__()
        self.vat_dim = vat_dim
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Conv1d(vat_dim, cond_dim, 1),
            nn.SiLU(),
            nn.Conv1d(cond_dim, cond_dim, 1),
            nn.SiLU(),
        )

    def forward(self, vat):
        return self.net(vat)


class FiLMLayer(nn.Module):
    """Zero-initialized per-channel scale+shift from the conditioning trunk."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.head = nn.Conv1d(cond_dim, 2 * channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, c, mask=None):
        """x [B, C, T]; c [B, cond_dim, T]; mask [B, 1, T] or None."""
        gamma, beta = torch.chunk(self.head(c), 2, dim=1)
        x = x * (1.0 + gamma) + beta
        if mask is not None:
            x = x * mask
        return x
