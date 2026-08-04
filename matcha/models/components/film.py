"""VAT FiLM conditioning (design: notes/vat-channels.md).

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
        # THE MODEL-CORE WIDTH SEAM.
        # Both conditioning paths (text_encoder and decoder) funnel through a trunk, so
        # this is the one place that sees every VAT tensor entering the network — which
        # is why the check lives here rather than at each caller.
        #
        # It matters for the delivery migration (vat_dim 3 -> 4, direction-contract-v2).
        # Without it a width mismatch surfaces as a Conv1d channel error from inside
        # nn.Sequential, naming neither the expected width nor where the wrong one came
        # from; and a filelist and a checkpoint that disagree by one channel is exactly
        # the mistake the migration invites. The failure must name both numbers.
        #
        # Shapes are static under torch.export, so this is evaluated at trace time and
        # leaves nothing in the exported graph.
        if vat.dim() != 3:
            raise ValueError(
                f"VATTrunk expects [B, vat_dim, T]; got {tuple(vat.shape)}. "
                "Per-utterance (B, vat_dim) tensors must be expanded by the caller."
            )
        if vat.shape[1] != self.vat_dim:
            raise ValueError(
                f"VAT width mismatch: this trunk was built for vat_dim={self.vat_dim} "
                f"but received {vat.shape[1]} channels (shape {tuple(vat.shape)}). "
                "The checkpoint, the model config and the filelist must agree — see "
                "notes/todo.md §1 (delivery-channel seams)."
            )
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
