#!/usr/bin/env python3
"""Measure the HARMONIC-TO-NOISE RATIO of rendered audio against the real recording.

WHY A FOURTH MEASUREMENT, AND WHY IT IS NOT ANOTHER DISTANCE
------------------------------------------------------------
Three measurements now disagree with the ear. Teacher-forced diff loss is flat against
pitch. Whole-mel synthesis error RISES with pitch (rho +0.865). Split into 8 bands, every
band rises with pitch and the two strongest are 432-820 Hz and 863-1296 Hz. Five blind ear
tests say the opposite: the hum is worse at LOW pitch, most recently 12 LOW / 0 HIGH at
gaps >= 25 Hz, p = 0.0005.

⚠⚠ A MEL MAGNITUDE SPECTROGRAM IS A POOR WITNESS FOR A BUZZ, and that is the most likely
reason for the disagreement. Mel band 0 holds about 10 bins across 0-388 Hz, roughly 39 Hz
per bin, so a 93 Hz voice's harmonic spacing is barely resolved there. An error that leaves
the magnitudes close to right while destroying the ratio of harmonic energy to noise sounds
mechanical to a listener and moves a magnitude distance very little. So this tool measures
the quantity a buzz actually changes, on the waveform, after the vocoder — the signal the
owner heard.

HNR is computed from the normalised autocorrelation peak at the pitch period, the same
quantity `measure_speaker_f0.py` already thresholds on to decide a frame is voiced:
HNR_dB = 10 log10(r / (1 - r)). A perfectly periodic frame has r -> 1 and HNR -> +inf; a
noisy one has r -> 0 and HNR -> -inf. A robotic buzz is not low HNR, it is the WRONG HNR:
a synthetic voice that is too periodic sounds mechanical, and one that is too noisy sounds
breathy or rough. The reported quantity is therefore the SIGNED DIFFERENCE from the real
recording of the same utterance, not an absolute score.

⚠ THE VOCODER ROUND TRIP IS THE POSITIVE CONTROL AND IT IS NOT OPTIONAL. The real audio is
also passed mel -> vocoder -> audio and measured the same way. That path is known to be
audible — a signal-detection ear test caught it on 4 of 8 pairs with a zero false-alarm
rate — so an HNR measure that cannot see it cannot see anything, and a null from it about
pitch would be worthless. The run refuses if the round trip does not move HNR at all.

⚠ THREE SIGNALS, ONE UTTERANCE, ONE ALIGNMENT. The generated mel uses the MAS alignment
taken from the real recording, so all three signals carry the same words at the same
durations and the same length. Free-running duration prediction would change the length
and make a frame distribution comparison answer a timing question instead.

Usage (in the ROCm container):
    python scripts/stages/measure_harmonicity.py \
        --filelist /data/.../pitch_error/clips.txt \
        --model-config <run>/.hydra/config.yaml --ckpt /path/to.ckpt \
        --out /data/.../pitch_error/harmonicity.csv
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.environ.get("SONORA_REPO", "/sonora"))

from hydra.utils import instantiate  # noqa: E402

import matcha.utils.monotonic_align as monotonic_align  # noqa: E402
from matcha.cli import load_vocoder_24k, to_waveform  # noqa: E402
from matcha.data.license_wall import enforce  # noqa: E402
from matcha.data.text_mel_datamodule import TextMelBatchCollate, TextMelDataset  # noqa: E402
from matcha.utils.model import denormalize, fix_len_compatibility, sequence_mask  # noqa: E402


def hnr_frames(x, sr, fmin, fmax, rms_floor, periodicity):
    """Per-frame (HNR dB, F0 Hz) over voiced frames, from the normalised ACF peak.

    Deliberately the SAME framing, window and voicing rule as
    `scripts/tools/measure_speaker_f0.f0_frames` — 40 ms window, 10 ms hop, mean removed,
    RMS floor, ACF peak searched between the period bounds. Two estimators disagreeing
    about which frames are voiced would make the F0 axis and the HNR axis incomparable.
    """
    w, hop = int(0.040 * sr), int(0.010 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    hnr, f0 = [], []
    for s in range(0, len(x) - w, hop):
        fr = x[s:s + w].astype(np.float64)
        fr = fr - fr.mean()
        if np.sqrt(np.mean(fr ** 2)) < rms_floor:
            continue
        ac = np.correlate(fr, fr, "full")[w - 1:]
        ac = ac / (ac[0] or 1.0)
        seg = ac[lo:hi]
        if not len(seg):
            continue
        k = int(np.argmax(seg))
        r = float(seg[k])
        if r < periodicity:
            continue
        # Clamped before the log: r at or above 1.0 is a numerical artifact of a
        # near-constant frame, not an infinitely periodic one, and one inf poisons a mean.
        r = min(max(r, 1e-6), 1.0 - 1e-6)
        hnr.append(10.0 * math.log10(r / (1.0 - r)))
        f0.append(sr / (lo + k))
    return np.asarray(hnr), np.asarray(f0)


def summarize(x, sr, args):
    h, f = hnr_frames(x, sr, args.fmin, args.fmax, args.rms_floor, args.periodicity)
    if not len(h):
        return None
    return {"hnr": float(np.median(h)), "f0": float(np.median(f)),
            # Frame-to-frame F0 scatter. A buzz often shows here before it shows in HNR:
            # an over-regular synthetic voice has LESS jitter than a real one.
            "jitter": float(np.median(np.abs(np.diff(f)))) if len(f) > 1 else 0.0,
            "voiced": len(h)}


@torch.no_grad()
def gen_mel(model, batch, n_timesteps, temperature):
    """A mel generated from noise on the MAS alignment of the real recording."""
    x, x_lengths = batch["x"], batch["x_lengths"]
    y, y_lengths = batch["y"], batch["y_lengths"]
    spks, vat = batch["spks"], batch.get("vat")
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
    T = y.shape[-1]
    y_mask = sequence_mask(y_lengths, T).unsqueeze(1).to(x_mask)
    const = -0.5 * math.log(2 * math.pi) * model.n_feats
    factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
    log_prior = (torch.matmul(factor.transpose(1, 2), y ** 2)
                 - torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                 + torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1) + const)
    attn = monotonic_align.maximum_path(
        log_prior, x_mask.transpose(1, 2) * y_mask).detach()

    padded = fix_len_compatibility(T)
    mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
    vat_y = (torch.matmul(attn.transpose(1, 2), vat.transpose(1, 2)).transpose(1, 2)
             if (model.use_vat and vat is not None) else None)

    def pad_to(t, n):
        return torch.nn.functional.pad(t, (0, n - t.shape[-1])) if t.shape[-1] < n else t

    gen = model.decoder(pad_to(mu_y, padded),
                        sequence_mask(y_lengths, padded).unsqueeze(1).to(x_mask),
                        n_timesteps, temperature, spks,
                        cond=pad_to(vat_y, padded) if vat_y is not None else None)
    return gen[:, :, :T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.667)
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=350.0)
    ap.add_argument("--rms-floor", type=float, default=0.01)
    ap.add_argument("--periodicity", type=float, default=0.3)
    ap.add_argument("--min-control-db", type=float, default=0.05,
                    help="refuse if the vocoder round trip moves HNR less than this")
    args = ap.parse_args()

    enforce([args.filelist])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.model_config)
    d = cfg.data
    vat_dim = d.get("vat_dim", cfg.model.get("vat_dim", 3))
    ds = TextMelDataset(
        filelist_path=args.filelist, n_spks=d.n_spks, cleaners=d.cleaners,
        add_blank=d.add_blank, n_fft=d.n_fft, n_mels=d.n_feats,
        sample_rate=d.sample_rate, hop_length=d.hop_length, win_length=d.win_length,
        f_min=d.f_min, f_max=d.f_max,
        data_parameters=OmegaConf.to_container(d.data_statistics, resolve=True),
        seed=1234, load_durations=d.load_durations,
        load_vat=d.get("load_vat", vat_dim > 0), vat_dim=vat_dim)
    collate = TextMelBatchCollate(d.n_spks)

    model = instantiate(cfg.model)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit("REFUSING: checkpoint does not match this config "
                         "(missing=%d unexpected=%d)." % (len(missing), len(unexpected)))
    model = model.to(device).eval()
    vocoder, sr = load_vocoder_24k(device)
    if sr != d.sample_rate:
        raise SystemExit("REFUSING: the vocoder runs at %d Hz and the corpus at %d. The "
                         "F0 and HNR axes would be scaled against each other."
                         % (sr, d.sample_rate))

    idx = list(range(len(ds)))[: args.limit or None]
    rows, t0 = [], time.time()
    # ⚠ NO GRAD FOR THE WHOLE LOOP. `gen_mel` carries its own decorator, but `to_waveform`
    # runs the vocoder outside it and returns a tensor that requires grad, so `.numpy()`
    # refuses. Wrapping the loop is better than sprinkling `.detach()`: it also stops the
    # vocoder building a graph over 288 clips it will never backward through.
    torch.set_grad_enabled(False)
    for n, i in enumerate(idx):
        path = ds.filepaths_and_text[i][0]
        clip = os.path.basename(path)
        batch = collate([ds[i]])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        real, rsr = sf.read(path, dtype="float64")
        if real.ndim > 1:
            real = real.mean(axis=1)
        if rsr != sr:
            raise SystemExit("REFUSING: %s is %d Hz, the vocoder is %d." % (clip, rsr, sr))

        # ⚠⚠ DENORMALISED BEFORE VOCODING. `y` carries the corpus normalisation and
        # `to_waveform` wants un-normalised log-mel. Feeding it the normalised tensor
        # produces audio that is wrong in a way that still sounds like speech, which is
        # exactly the failure that would be mistaken for a model defect here.
        true_mel = denormalize(batch["y"], model.mel_mean, model.mel_std)
        rt = to_waveform(true_mel, vocoder, None).cpu().numpy()

        torch.manual_seed(zlib.crc32(("%s|0" % clip).encode()))
        syn_mel = denormalize(gen_mel(model, batch, args.timesteps, args.temperature),
                              model.mel_mean, model.mel_std)
        syn = to_waveform(syn_mel, vocoder, None).cpu().numpy()

        row = {"clip": clip}
        ok = True
        for tag, sig in (("real", real), ("rt", rt), ("syn", syn)):
            s = summarize(np.asarray(sig).squeeze(), sr, args)
            if s is None:
                ok = False
                break
            for k, v in s.items():
                row["%s_%s" % (tag, k)] = round(v, 4)
        if not ok:
            continue
        row["d_hnr_rt"] = round(row["rt_hnr"] - row["real_hnr"], 4)
        row["d_hnr_syn"] = round(row["syn_hnr"] - row["real_hnr"], 4)
        row["d_jitter_syn"] = round(row["syn_jitter"] - row["real_jitter"], 4)
        rows.append(row)
        if (n + 1) % 50 == 0:
            print("    %d/%d  %.2f clips/s" % (n + 1, len(idx),
                                               (n + 1) / (time.time() - t0)), flush=True)

    if not rows:
        raise SystemExit("REFUSING: no clip produced a voiced frame in all three signals.")

    def mean(k):
        return sum(r[k] for r in rows) / len(rows)

    print("\n  %d clips" % len(rows))
    print("  HNR dB   real %6.2f   round-trip %6.2f   synth %6.2f"
          % (mean("real_hnr"), mean("rt_hnr"), mean("syn_hnr")), flush=True)
    print("  delta    round-trip %+6.2f   synth %+6.2f"
          % (mean("d_hnr_rt"), mean("d_hnr_syn")), flush=True)

    # ⚠⚠ GATE, NOT A PRINTOUT. The round trip is known-audible; a measure blind to it
    # cannot support a null about pitch.
    if abs(mean("d_hnr_rt")) < args.min_control_db:
        raise SystemExit(
            "REFUSING to report: the vocoder round trip moved median HNR by %+.3f dB, "
            "below --min-control-db %.2f. That path is KNOWN audible (4 of 8 pairs, zero "
            "false alarms), so a measure that cannot see it cannot support any claim "
            "about pitch." % (mean("d_hnr_rt"), args.min_control_db))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"filelist": args.filelist, "ckpt": args.ckpt, "clips": len(rows),
                   "mean_d_hnr_rt": round(mean("d_hnr_rt"), 4),
                   "mean_d_hnr_syn": round(mean("d_hnr_syn"), 4),
                   "is_holdout": False}, f, indent=2)
    print("  wrote %d rows -> %s" % (len(rows), args.out), flush=True)


if __name__ == "__main__":
    main()
