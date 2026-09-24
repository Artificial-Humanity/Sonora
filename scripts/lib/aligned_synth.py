"""Generate a mel from noise on the alignment of a REAL recording.

Two tools need this one operation, and it lives here so that they cannot drift apart:

  * `measure_synth_mel_error.py` scores the generated mel against the true one;
  * `render_aligned_mels.py` saves it as training input for the vocoder.

⚠⚠ THE DURATIONS COME FROM MAS AGAINST THE TRUE MEL, NOT FROM THE DURATION PREDICTOR. The
generated mel is then the same length as the recording, frame for frame. The measurement
needs that to compare frames, and the vocoder fine-tune needs it to pair each mel frame with
the 256 samples of real audio it must learn to produce. Everything else is inference: the
decoder solves the ODE from noise with the encoder output as its condition, so the mel
carries the texture the model really synthesises.
"""

import math

import torch
from omegaconf import OmegaConf

import matcha.utils.monotonic_align as monotonic_align
from matcha.data.text_mel_datamodule import TextMelDataset
from matcha.utils.model import fix_len_compatibility, sequence_mask


def build_dataset(cfg, filelist):
    d = cfg.data
    vat_dim = d.get("vat_dim", cfg.model.get("vat_dim", 3))
    return TextMelDataset(
        filelist_path=filelist, n_spks=d.n_spks, cleaners=d.cleaners,
        add_blank=d.add_blank, n_fft=d.n_fft, n_mels=d.n_feats,
        sample_rate=d.sample_rate, hop_length=d.hop_length, win_length=d.win_length,
        f_min=d.f_min, f_max=d.f_max,
        data_parameters=OmegaConf.to_container(d.data_statistics, resolve=True),
        seed=1234, load_durations=d.load_durations,
        load_vat=d.get("load_vat", vat_dim > 0), vat_dim=vat_dim,
    )


@torch.no_grad()
def aligned_generate(model, batch, n_timesteps, temperature, spk_override=None,
                     guidance=1.0):
    """Returns (generated mel, true mel, mask), all normalised and `y.shape[-1]` frames long.

    Both mels are in the corpus-normalised space the model works in. Denormalise with the
    model's own `mel_mean`/`mel_std` before a vocoder sees one.
    """
    x, x_lengths = batch["x"], batch["x_lengths"]
    y, y_lengths = batch["y"], batch["y_lengths"]
    spks = batch["spks"] if spk_override is None else torch.full_like(batch["spks"],
                                                                     spk_override)
    vat = batch.get("vat")

    if model.n_spks > 1:
        spks = model.spk_emb(spks)
    if model.use_vat:
        if vat is None:
            vat = torch.zeros(x.shape[0], model.vat_dim, dtype=torch.float32, device=x.device)
        if vat.dim() == 2:
            vat = vat.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    else:
        vat = None

    mu_x, _logw, x_mask = model.encoder(x, x_lengths, spks, vat=vat)
    y_max_length = y.shape[-1]
    y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
    attn_mask_squeezed = x_mask.transpose(1, 2) * y_mask

    const = -0.5 * math.log(2 * math.pi) * model.n_feats
    factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
    y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
    y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
    mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
    attn = monotonic_align.maximum_path(y_square - y_mu_double + mu_square + const,
                                        attn_mask_squeezed).detach()

    # ⚠ THE DECODER'S UNET DOWNSAMPLES, so the frame count it is given must be compatible
    # or the solve fails on a shape mismatch. `synthesise` pads to fix_len_compatibility
    # for the same reason. Pad here, solve, then CROP BACK to the true length — the
    # padding frames mean nothing to a score or to a vocoder.
    padded = fix_len_compatibility(y_max_length)
    mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
    vat_y = None
    if model.use_vat and vat is not None:
        vat_y = torch.matmul(attn.transpose(1, 2), vat.transpose(1, 2)).transpose(1, 2)

    def pad_to(t, n):
        return torch.nn.functional.pad(t, (0, n - t.shape[-1])) if t.shape[-1] < n else t

    mu_y_p = pad_to(mu_y, padded)
    vat_y_p = pad_to(vat_y, padded) if vat_y is not None else None
    y_mask_p = sequence_mask(y_lengths, padded).unsqueeze(1).to(x_mask)

    gen = model.decoder(mu_y_p, y_mask_p, n_timesteps, temperature, spks, cond=vat_y_p,
                        guidance=guidance)
    return gen[:, :, :y_max_length], y, y_mask[:, :, :y_max_length]
