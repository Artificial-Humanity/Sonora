"""Classifier-free-guidance amplification demo on the VAT channels.

The derisk model trained with vat_cond_dropout=0.15, which zeroes the whole
VAT vector for dropped rows — so the unconditional mode the model learned IS
vat=0 (the trained neutral). That makes CFG-style extrapolation available at
inference with no retraining: run the decoder's Euler loop with both the
conditional and unconditional vector fields and step along
    v = v_uncond + s * (v_cond - v_uncond),  s > 1.
Guidance is applied to the decoder field only; the encoder (durations, mu)
stays at the conditional VAT — standard practice, and it keeps all renders of
the same (text, vat) pair frame-aligned across s values.

Also renders raw input extrapolation (vat values beyond the [-1,1] training
range, s=1) for comparison — the cruder lever.

Fully CPU. Same throwaway-container recipe as the other render scripts.

Usage:
    python scripts/render_guidance_demo.py --out <dir> [--spk 245]
        [--text "..."] [--seed 1234]
"""

import argparse
import json
import math
import os
import sys

HIFIGAN = os.environ.get("SONORA_HIFIGAN_DIR", "/data/model-training/vocoder/hifi-gan")
sys.path.insert(0, HIFIGAN)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import soundfile as sf
import torch

from matcha.utils.model import (
    denormalize,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

CKPT_DEFAULT = ("/data/model-training/sonora/logs/train/derisk_energy/runs/"
                "2026-07-15_00-20-31/checkpoints/checkpoint_epoch=099.ckpt")

# (label, (v, a, t), guidance)
RENDERS = [
    ("e0_s1",   (0.0, 0.0, 0.0), 1.0),   # neutral reference
    ("e+1_s1",  (0.0, 1.0, 0.0), 1.0),   # plain conditional (today's ceiling)
    ("e+1_s2",  (0.0, 1.0, 0.0), 2.0),   # CFG x2
    ("e+1_s3",  (0.0, 1.0, 0.0), 3.0),   # CFG x3
    ("e+2_raw", (0.0, 2.0, 0.0), 1.0),   # input extrapolation
    ("e+3_raw", (0.0, 3.0, 0.0), 1.0),
    ("e-1_s2",  (0.0, -1.0, 0.0), 2.0),  # amplified hush, the flip side
]


def load_acoustic(path):
    from matcha.models.matcha_tts import MatchaTTS

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MatchaTTS(**dict(ckpt["hyper_parameters"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def load_vocoder(ckpt_path, config_path):
    from env import AttrDict
    from models import Generator

    h = AttrDict(json.load(open(config_path)))
    g = Generator(h)
    g.load_state_dict(torch.load(ckpt_path, map_location="cpu")["generator"])
    g.eval()
    g.remove_weight_norm()
    return g, h


def guided_synthesise(model, x, x_lengths, n_timesteps, temperature, spks,
                      length_scale, vat, guidance):
    """model.synthesise with a CFG Euler loop in place of decoder.forward."""
    if model.n_spks > 1:
        spks = model.spk_emb(spks.long())

    if model.use_vat and vat is not None and vat.dim() == 2:
        vat = vat.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    mu_x, logw, x_mask = model.encoder(x, x_lengths, spks, vat=vat)

    w = torch.exp(logw) * x_mask
    w_ceil = torch.ceil(w) * length_scale
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_max_length = y_lengths.max()
    y_max_length_ = fix_len_compatibility(y_max_length)

    y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
    attn_mask_squeezed = x_mask.transpose(1, 2) * y_mask
    attn = generate_path(w_ceil.squeeze(1), attn_mask_squeezed).unsqueeze(1)

    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
    mu_y = mu_y.transpose(1, 2)

    vat_y = torch.matmul(attn.squeeze(1).transpose(1, 2),
                         vat.transpose(1, 2)).transpose(1, 2)
    uncond = torch.zeros_like(vat_y)

    # CFG Euler loop (decoder.solve_euler with field extrapolation)
    dec = model.decoder
    z = torch.randn_like(mu_y) * temperature
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu_y.device)
    t, dt = t_span[0], t_span[1] - t_span[0]
    xt = z
    for step in range(1, len(t_span)):
        v_c = dec.estimator(xt, y_mask, mu_y, t, spks, vat_y)
        if guidance != 1.0:
            v_u = dec.estimator(xt, y_mask, mu_y, t, spks, uncond)
            v = v_u + guidance * (v_c - v_u)
        else:
            v = v_c
        xt = xt + dt * v
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    decoder_outputs = xt[:, :, :y_max_length]
    return denormalize(decoder_outputs, model.mel_mean, model.mel_std)


def rms_db(wav):
    return 20 * math.log10(max(float(np.sqrt((wav.astype(np.float64) ** 2).mean())), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocoder",
                    default="/data/model-training/vocoder/cp_hifigan_24k/g_02510000")
    ap.add_argument("--vocoder-config",
                    default=os.path.join(HIFIGAN, "config_24k_80band.json"))
    ap.add_argument("--text", default="We won! We actually won the championship!")
    ap.add_argument("--spk", type=int, default=245)
    ap.add_argument("--n-timesteps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.667)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from matcha.text import text_to_sequence
    from matcha.text.op_g2p import OpenPhonemizerG2P
    from matcha.utils.utils import intersperse

    g2p = OpenPhonemizerG2P()
    ipa = g2p.phonemize(args.text)
    bad = g2p.validate(ipa)
    if bad:
        raise SystemExit(f"out-of-vocab characters after G2P: {bad}")
    seq, _ = text_to_sequence(ipa, ["no_cleaners"])
    x = torch.tensor(intersperse(seq, 0), dtype=torch.long)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long)
    spks = torch.tensor([args.spk], dtype=torch.long)

    model = load_acoustic(args.checkpoint)
    vocoder, h = load_vocoder(args.vocoder, args.vocoder_config)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for label, (v, a, t), s in RENDERS:
        vat = torch.tensor([[v, a, t]])
        torch.manual_seed(args.seed)
        with torch.no_grad():
            mel = guided_synthesise(model, x, x_lengths, args.n_timesteps,
                                    args.temperature, spks, args.length_scale,
                                    vat, s)
            wav = vocoder(mel).squeeze().numpy()
        fname = f"guidance_{label}.wav"
        sf.write(os.path.join(args.out, fname), wav, h.sampling_rate)
        rows.append({"wav": fname, "vat": [v, a, t], "guidance": s,
                     "seconds": round(len(wav) / h.sampling_rate, 2),
                     "rms_db": round(rms_db(wav), 2)})
        print(f"  {label:8s} vat={v:+.0f},{a:+.0f},{t:+.0f} s={s:.0f}: "
              f"{rows[-1]['seconds']}s  {rows[-1]['rms_db']} dB")

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "vocoder": args.vocoder,
                   "text": args.text, "spk": args.spk, "seed": args.seed,
                   "n_timesteps": args.n_timesteps,
                   "temperature": args.temperature,
                   "length_scale": args.length_scale, "rows": rows}, f, indent=2)
    print(f"\n{len(rows)} renders + report -> {args.out}")


if __name__ == "__main__":
    main()
