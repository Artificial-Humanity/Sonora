"""Decoder v2: a DiT estimator for the flow-matching field (docs/model-decisions.md § Decoder v2).

A drop-in replacement for the U-Net `Decoder`. It takes the same arguments in `forward` and
returns the same shape, so `CFM`, CFG and the ODE solver do not change. `CFM` selects it
when the decoder config says `type: dit`.

The block design is StableTTS's `DiTConVBlock` (MIT, https://github.com/KdaiP/StableTTS):

  * adaLN-Zero. Each block takes shift, scale and gate for attention and for the FFN
    from the conditioning. The gates start at zero, so every block starts as the identity.
  * A CONVOLUTIONAL FFN inside every block. Tiny plain transformers lose the fine spectral
    texture that convolutions get for free, and texture is exactly what this spike is for.
  * U-Net-like long skips across the stack (first half pushes, second half pops).
  * RoPE on half of each head, and LayerNorm without affine (adaLN supplies the affine).
  * A convolutional prenet on the encoder output `mu` (masked between layers; see there).

⚠ DEPARTURES FROM STABLETTS, ALL DELIBERATE:

1. THE CONDITIONING IS PER FRAME, NOT ONE VECTOR PER UTTERANCE.
StableTTS modulates with one vector per utterance. Our V/A/T and delivery channels are per
TOKEN, expanded to frames through the alignment, and a vector per utterance would average
away every in-utterance change of direction. So the conditioning here is a [B, C, T]
sequence: time embedding + speaker projection (both constant over T) + the VAT trunk's
output (per frame). The adaLN heads are 1x1 convolutions over that sequence, which is the
same arithmetic as StableTTS's Linear, applied at each frame.

The speaker enters the conditioning, not the input. The U-Net concatenates the speaker
embedding to its input channels; `CFM` therefore passes the DiT the input width WITHOUT
the speaker channels.

2. TIME ENTERS THROUGH adaLN, AS IN THE DiT PAPER, NOT THROUGH A SEPARATE FiLM. StableTTS's
`DitWrapper` applies a per-block time FiLM to `x` before each block and sends only the
speaker through adaLN. Here time is summed into `c`. One consequence: at init the output
does not depend on `t` until the adaLN heads move off zero, which the first steps do.

3. THE ATTENTION MASK IS A BOOLEAN KEY MASK (see `MultiHeadAttention`), not a [T, T] float
mask filled with -finfo.max.

4. THE mu PRENET IS MASKED BETWEEN LAYERS (see there).

⚠ EXPORT: everything here is Conv1d, LayerNorm, matmul and pointwise ops, and RoPE is
computed from the sequence length in `forward` (no cache held between calls), so a
fixed-shape trace sees no Python state. The split-graph export itself is NOT validated
yet. It is one of the adoption gates, not a property of this file.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from matcha.models.components.film import VATTrunk


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
        self.dim = dim

    def forward(self, t, scale=1000):
        if t.ndim < 1:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device).float() * -(math.log(10000) / (half - 1)))
        arg = scale * t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat((arg.sin(), arg.cos()), dim=-1)


def rope(x, d):
    """Rotary position embedding on the first `d` features of each head.

    x [B, H, T, Dh]. The angles come from T at call time, so nothing is cached.
    """
    t = x.shape[2]
    theta = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = torch.arange(t, device=x.device).float().unsqueeze(1) * theta.unsqueeze(0)   # [T, d/2]
    ang = torch.cat((ang, ang), dim=-1)                                                  # [T, d]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x_rope, x_pass = x[..., :d], x[..., d:]
    half = d // 2
    rotated = torch.cat((-x_rope[..., half:], x_rope[..., :half]), dim=-1)
    return torch.cat((x_rope * cos + rotated * sin, x_pass), dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads, p_dropout=0.0):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.rope_dim = self.head_dim // 2
        self.p_dropout = p_dropout
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        for conv in (self.conv_q, self.conv_k, self.conv_v):
            nn.init.xavier_uniform_(conv.weight)

    def forward(self, x, key_mask):
        """x [B, C, T]; key_mask [B, 1, 1, T] bool, True where a frame is real."""
        b, c, t = x.shape

        def heads(y):
            return y.view(b, self.n_heads, self.head_dim, t).transpose(2, 3)   # [B, H, T, Dh]

        q = rope(heads(self.conv_q(x)), self.rope_dim)
        k = rope(heads(self.conv_k(x)), self.rope_dim)
        v = heads(self.conv_v(x))
        # A boolean KEY mask, broadcast over queries: padded frames are never attended to.
        # Padded queries still produce output, and the caller masks it. A [T, T] float mask
        # filled with -finfo.max (StableTTS) is avoided: under fp16 autocast it is one cast
        # away from -inf, and it allocates T*T per batch item for nothing.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask,
                                             dropout_p=self.p_dropout if self.training else 0.0)
        return self.conv_o(out.transpose(2, 3).reshape(b, c, t))


class ConvFFN(nn.Module):
    def __init__(self, channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, mask):
        x = self.drop(F.silu(self.conv_1(x * mask)))
        return self.conv_2(x * mask) * mask


class DiTConvBlock(nn.Module):
    """One adaLN-Zero transformer block with a convolutional FFN (StableTTS `DiTConVBlock`)."""

    def __init__(self, channels, filter_channels, n_heads, kernel_size=3, p_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.attn = MultiHeadAttention(channels, n_heads, p_dropout)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False)
        self.ffn = ConvFFN(channels, filter_channels, kernel_size, p_dropout)
        # adaLN-Zero: SiLU then a per-frame projection to six modulation tensors. The
        # projection starts at zero, so every gate is 0 and the block is the identity.
        self.ada_ln = nn.Conv1d(channels, 6 * channels, 1)
        nn.init.zeros_(self.ada_ln.weight)
        nn.init.zeros_(self.ada_ln.bias)

    @staticmethod
    def _norm(norm, x):
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x, c, mask, key_mask):
        """x [B, C, T]; c [B, C, T] conditioning; mask [B, 1, T]; key_mask [B, 1, 1, T]."""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada_ln(F.silu(c)).chunk(6, dim=1)
        h = self._norm(self.norm1, x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(h * mask, key_mask) * mask
        h = self._norm(self.norm2, x) * (1 + scale_f) + shift_f
        x = x + gate_f * self.ffn(h, mask)
        return x * mask


class DiTDecoder(nn.Module):
    """The flow-matching estimator as a DiT. Same `forward` contract as the U-Net `Decoder`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=256,
        filter_channels=1024,
        n_layers=6,
        n_heads=4,
        kernel_size=3,
        dropout=0.1,
        spk_emb_dim=0,
        vat_dim=3,
        vat_cond_dim=0,
        use_lsc=True,
    ):
        super().__init__()
        if use_lsc and n_layers % 2:
            raise ValueError(f"long skips pair the first and second halves: n_layers must be even, got {n_layers}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vat_dim = vat_dim
        self.vat_cond_dim = vat_cond_dim
        self.use_lsc = use_lsc

        # `in_channels` is noise + mu (2 * n_feats), without speaker channels: see module doc.
        n_feats = in_channels // 2
        # ⚠ MASKED BETWEEN LAYERS, UNLIKE STABLETTS. Its prenet is a bare nn.Sequential, so
        # after the first conv the padded frames hold SiLU(bias), and the next kernel-3 conv
        # reads that into the last real frame. The result depends on how much padding the
        # batch happened to have. `scripts/gates/test_dit_decoder.py` gate 3 found it
        # (max|diff| 4e-3 between a clip alone and the same clip padded in a batch).
        self.mu_prenet = nn.ModuleList([
            nn.Conv1d(n_feats, filter_channels, kernel_size, padding=kernel_size // 2),
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2),
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
        ])
        self.in_proj = nn.Conv1d(in_channels - n_feats + hidden_channels, hidden_channels, 1)

        self.time_emb = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_channels, filter_channels),
            nn.SiLU(),
            nn.Linear(filter_channels, hidden_channels),
        )
        self.spk_proj = nn.Linear(spk_emb_dim, hidden_channels) if spk_emb_dim > 0 else None

        # VAT: the same trunk as the U-Net (the width seam lives there), then a projection
        # into the conditioning. Not zero-initialised: the U-Net's FiLM heads are zero so a
        # WARM start stays exact, and this estimator is always fresh. The unconditional
        # state CFG needs is VAT = 0 at the INPUT, which `vat_cond_dropout` trains; it does
        # not need a zero projection. A zero here would only put a second zero factor in
        # series with the adaLN heads and slow the VAT path's start (review, 2026-09-26).
        use_vat = vat_cond_dim > 0
        self.vat_trunk = VATTrunk(vat_dim, vat_cond_dim) if use_vat else None
        self.vat_proj = nn.Conv1d(vat_cond_dim, hidden_channels, 1) if use_vat else None

        self.blocks = nn.ModuleList(
            [DiTConvBlock(hidden_channels, filter_channels, n_heads, kernel_size, dropout) for _ in range(n_layers)]
        )
        self.lsc = (
            nn.ModuleList(
                [
                    nn.Conv1d(2 * hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
                    for _ in range(n_layers // 2)
                ]
            )
            if use_lsc
            else None
        )
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Same contract as `Decoder.forward`.

        x, mu [B, n_feats, T]; mask [B, 1, T]; t [B] or scalar; spks [B, spk_emb_dim] or None;
        cond [B, vat_dim, T] or None (None is neutral: zeros). Returns [B, out_channels, T].
        """
        key_mask = mask.bool().unsqueeze(1)                                   # [B, 1, 1, T]

        c = self.time_mlp(self.time_emb(t).to(x.dtype))                       # [B or 1, C]
        if self.spk_proj is not None and spks is not None:
            c = c + self.spk_proj(spks)
        c = c.unsqueeze(-1).expand(x.shape[0], -1, x.shape[-1])              # [B, C, T]
        if self.vat_trunk is not None:
            if cond is None:
                cond = torch.zeros(x.shape[0], self.vat_dim, x.shape[-1], dtype=x.dtype, device=x.device)
            c = c + self.vat_proj(self.vat_trunk(cond * mask))

        m = mu * mask
        for i, conv in enumerate(self.mu_prenet):
            m = conv(m)
            m = (F.silu(m) if i < len(self.mu_prenet) - 1 else m) * mask
        h = self.in_proj(torch.cat((x, m), dim=1)) * mask

        skips = []
        half = len(self.blocks) // 2
        for i, block in enumerate(self.blocks):
            if self.lsc is not None:
                if i < half:
                    skips.append(h)
                else:
                    h = self.lsc[i - half](torch.cat((h, skips.pop()), dim=1)) * mask
            h = block(h, c, mask, key_mask)

        return self.final_proj(h * mask) * mask
