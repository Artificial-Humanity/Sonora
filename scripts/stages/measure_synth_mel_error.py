#!/usr/bin/env python3
"""Generate a mel from noise on the model's OWN alignment and score it against the truth.

WHY THIS EXISTS AND WHAT THE LAST MEASUREMENT COULD NOT REACH
-------------------------------------------------------------
`score_holdout.py` scores the TRAINING OBJECTIVE: it hands the model the true mel and the
MAS alignment and asks how well one step of the flow-matching field is predicted. Run
across a pitch-stratified sample it came back flat — `diff` rho +0.132 against F0,
permutation p = 0.44 — so the model's single-step prediction carries no pitch gradient.

That is not the same claim as "the model synthesises a low voice correctly". The hum is
heard in FREE-RUNNING GENERATION, where the decoder integrates its own trajectory from
noise over many steps. A vector field that is accurate on average at random points along
the path can still integrate to a bad trajectory: the errors compound in a direction the
averaged single-step loss does not see.

So this tool solves the ODE. It reuses the encoder and the MAS alignment from the real
recording, then generates the mel the way inference does, and compares frame-for-frame
against the true mel the alignment came from.

⚠ DURATIONS COME FROM MAS, NOT FROM THE DURATION PREDICTOR, AND THAT IS THE POINT. Using
predicted durations would change the LENGTH of the output, so a frame-wise comparison
would measure timing drift rather than timbre — and the hum is a timbre complaint. Holding
the alignment to the truth isolates the decoder path, which is the part this measurement
is about. A pitch-dependent duration failure is a separate question and `dur` in the
previous run was flat (rho +0.127, p = 0.46).

⚠⚠ THE UNITS ARE MEL STANDARD DEVIATIONS, chosen to be readable beside the vocoder probe.
`y` arrives already normalised by the corpus `data_statistics`, so a mean absolute error of
0.109 in this space IS the 10.9% figure the single-pass vocoder round trip reported. That
round trip is the floor the ear could just detect at 4 of 8 pairs; anything far below it is
inaudible whatever its correlation with pitch.

⚠ THE POSITIVE CONTROL IS A DELIBERATELY WRONG SPEAKER and it runs on every invocation.
The metric has to respond to a degradation that is known to be real before a null from it
means anything — an instrument that returns the same number for the right voice and the
wrong one is measuring nothing, and would report "no pitch effect" just as confidently.
The run refuses if the wrong speaker does not score worse.

Usage (in the ROCm container — score_holdout.sh's wrapper shape):
    python scripts/stages/measure_synth_mel_error.py \
        --filelist /data/.../pitch_error/clips.txt \
        --model-config <run>/.hydra/config.yaml \
        --ckpt /path/to.ckpt \
        --out /data/.../pitch_error/synth_mel.csv
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import zlib

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.environ.get("SONORA_REPO", "/sonora"))

from hydra.utils import instantiate  # noqa: E402

import matcha.utils.monotonic_align as monotonic_align  # noqa: E402
from matcha.data.license_wall import enforce  # noqa: E402
from matcha.data.text_mel_datamodule import TextMelBatchCollate, TextMelDataset  # noqa: E402
from matcha.utils.model import fix_len_compatibility, sequence_mask  # noqa: E402


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
def synth_error(model, batch, spk_override, n_timesteps, temperature, n_bands=8):
    """Mean absolute error, in mel std units, between a generated mel and the true one.

    The alignment is MAS against the TRUE mel, exactly as `forward()` derives it, so the
    generated mel is the same length as the target and the comparison is frame-for-frame.
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
    # for the same reason. Pad here, solve, then CROP BOTH SIDES BACK to the true length —
    # scoring the padding would dilute the error with frames that mean nothing.
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

    gen = model.decoder(mu_y_p, y_mask_p, n_timesteps, temperature, spks, cond=vat_y_p)
    gen = gen[:, :, :y_max_length]

    m = y_mask[:, :, :y_max_length]
    denom = m.sum() * y.shape[1]
    l1 = float((torch.abs(gen - y) * m).sum() / denom)

    # ⚠⚠ THE TARGET'S OWN SPREAD IS RETURNED BESIDE THE ERROR, AND IT IS THE CONTROL THAT
    # DECIDES WHETHER A GRADIENT IS REAL. `l1` is an ABSOLUTE distance in corpus-normalised
    # mel units. If high-pitched voices simply have more variable mels — more energy moving
    # between bins — their absolute error rises with no help from the model, and a
    # correlation against F0 would be a property of the recordings rather than a finding
    # about synthesis. Dividing by the target's own mean absolute deviation asks the
    # scale-free question instead: what FRACTION of this voice's own variation did the
    # model fail to reproduce?
    y_mean = (y * m).sum() / denom
    y_mad = float((torch.abs(y - y_mean) * m).sum() / denom)

    # ⚠⚠ PER-BAND ERROR, BECAUSE A SCALAR OVER 80 BINS CANNOT SEE A HUM. The whole-mel
    # number rises with pitch (rho +0.865), which is the opposite of what five blind ear
    # tests reported, and the most likely reason is that the two are not measuring the same
    # thing. A buzz is a STRUCTURED error in a few low bins. Averaged against 80, it barely
    # moves the mean, while diffuse high-frequency detail that no listener objects to moves
    # it a great deal. Splitting the error by band lets the low bins answer separately.
    err = (torch.abs(gen - y) * m).sum(dim=(0, 2)).squeeze()      # per mel bin
    frames = m.sum()
    n_feats = y.shape[1]
    step = n_feats // n_bands
    bands = [float(err[b * step:(b + 1) * step].sum() / (frames * step))
             for b in range(n_bands)]
    return l1, y_mad, bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=2, help="ODE solves per clip")
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.667)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bands", type=int, default=8,
                    help="split the mel error into this many equal bands of bins")
    ap.add_argument("--control-every", type=int, default=8,
                    help="run the wrong-speaker positive control on every Nth clip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enforce([args.filelist])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.model_config)
    collate = TextMelBatchCollate(cfg.data.n_spks)
    ds = build_dataset(cfg, args.filelist)
    idx = list(range(len(ds)))[: args.limit or None]

    model = instantiate(cfg.model)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit("REFUSING: checkpoint does not match this config "
                         "(missing=%d unexpected=%d)." % (len(missing), len(unexpected)))
    model = model.to(device).eval()
    print("  %d clips x %d solves x %d steps on %s"
          % (len(idx), args.samples, args.timesteps, device.type), flush=True)

    rows, t0 = [], time.time()
    ctrl_true, ctrl_wrong = [], []
    for n, i in enumerate(idx):
        clip = os.path.basename(ds.filepaths_and_text[i][0])
        batch = collate([ds[i]])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        acc, mad, band_acc = 0.0, 0.0, []
        for s in range(args.samples):
            # Paired with score_holdout's scheme so the same clip sees the same noise
            # across runs and across checkpoints.
            torch.manual_seed(zlib.crc32(("%s|%d" % (clip, s)).encode()))
            e, mad, bands = synth_error(model, batch, None, args.timesteps,
                                        args.temperature, args.bands)
            acc += e
            band_acc = [a + b for a, b in zip(band_acc, bands)] if band_acc else list(bands)
        err = acc / args.samples
        row = {"clip": clip, "mel_l1": round(err, 6), "y_mad": round(mad, 6),
               "mel_l1_rel": round(err / mad, 6) if mad else ""}
        for b, v in enumerate(band_acc):
            row["band%d" % b] = round(v / args.samples, 6)
        rows.append(row)

        if args.control_every and n % args.control_every == 0:
            true_spk = int(batch["spks"][0])
            wrong = (true_spk + cfg.data.n_spks // 2) % cfg.data.n_spks
            torch.manual_seed(zlib.crc32(("%s|0" % clip).encode()))
            ctrl_wrong.append(synth_error(model, batch, wrong, args.timesteps,
                                          args.temperature, args.bands)[0])
            ctrl_true.append(err)
        if (n + 1) % 50 == 0:
            print("    %d/%d  %.2f clips/s" % (n + 1, len(idx),
                                               (n + 1) / (time.time() - t0)), flush=True)

    mean_true = sum(r["mel_l1"] for r in rows) / len(rows)
    print("\n  mean mel L1 = %.4f mel std  (the single-pass vocoder floor is 0.109)"
          % mean_true, flush=True)

    # ⚠⚠ THE CONTROL IS A GATE, NOT A PRINTOUT. A null result from an instrument that
    # cannot tell the right voice from a wrong one is not evidence of anything.
    if ctrl_true:
        ct, cw = sum(ctrl_true) / len(ctrl_true), sum(ctrl_wrong) / len(ctrl_wrong)
        print("  positive control: correct speaker %.4f  vs  WRONG speaker %.4f  (%.2fx)"
              % (ct, cw, cw / ct if ct else float("nan")), flush=True)
        if cw <= ct:
            raise SystemExit(
                "REFUSING to report: the wrong speaker scored %.4f, no worse than the "
                "correct one at %.4f. The metric does not respond to a degradation known "
                "to be real, so a pitch null from it would mean nothing." % (cw, ct))
    else:
        raise SystemExit("REFUSING: --control-every 0 disables the positive control.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fields = (["clip", "mel_l1", "y_mad", "mel_l1_rel"]
                  + ["band%d" % b for b in range(args.bands)])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # Band edges in Hz, so a column name maps to a frequency without re-deriving it.
    try:
        import librosa
        f = librosa.mel_frequencies(n_mels=cfg.data.n_feats, fmin=cfg.data.f_min,
                                    fmax=cfg.data.f_max)
        step = cfg.data.n_feats // args.bands
        band_hz = [[round(float(f[b * step]), 1),
                    round(float(f[min(len(f) - 1, (b + 1) * step - 1)]), 1)]
                   for b in range(args.bands)]
    except Exception:
        band_hz = None
    meta = {"filelist": args.filelist, "ckpt": args.ckpt, "clips": len(rows),
            "bands": args.bands, "band_hz": band_hz,
            "samples": args.samples, "timesteps": args.timesteps,
            "temperature": args.temperature, "mean_mel_l1": round(mean_true, 6),
            "control_correct": round(sum(ctrl_true) / len(ctrl_true), 6),
            "control_wrong_speaker": round(sum(ctrl_wrong) / len(ctrl_wrong), 6),
            "is_holdout": False}
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("  wrote %d rows -> %s" % (len(rows), args.out), flush=True)


if __name__ == "__main__":
    main()
