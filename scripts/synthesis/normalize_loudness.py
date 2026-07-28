# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "soundfile", "pyloudnorm"]
# ///
"""normalize_loudness — bring every clip in a bank to one integrated loudness.

WHY THIS EXISTS (measured 2026-07-28). Across the 193 clips in v1 the per-engine
median RMS spread was **5.1 dB** — dia -17.0, moss85 -20.1, longcat -21.3,
qwen -22.1 dBFS. That is not a cosmetic problem:

  1. It biases the audition. Louder reads as more present and more confident, so
     the ear is being told which engine it is before it judges the performance.
  2. It contaminates the arousal label. The instrument's A channel keys partly on
     level, so `measured_z.A` was encoding *which engine rendered the clip* rather
     than how activated the speaker was. Valence/arousal labels are already the
     known bottleneck (vat3 failed on labels, not training) — an engine-shaped
     term in the arousal channel is exactly the sort of thing that causes it.

So this runs at BUILD time: after render, before audition and before qc_gate's
measurement pass. Both consumers then see one level.

Target is **-23 LUFS integrated (EBU R128)** with a **-1.0 dBFS peak ceiling**. If
the gain needed to hit -23 would push the peak above the ceiling, we take the
smaller gain and land quiet — preserving the waveform matters more than hitting a
number, and a clipped clip is unusable while a 2 dB-quiet one is merely imperfect.
Every such case is reported.

ORIGINALS ARE NEVER DESTROYED. Each source wav is copied to `_pre_loudnorm/`
beside it before being rewritten, and the applied gain is recorded in
`loudnorm.jsonl`. Idempotent: a clip already listed in the sidecar is skipped, so
re-running after adding one engine to a bank does not re-gain the rest.

Usage:
    uv run normalize_loudness.py --dir <bank_dir> [--target -23.0] [--dry-run]
"""
import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

SIDECAR = "loudnorm.jsonl"
BACKUP_DIR = "_pre_loudnorm"
# pyloudnorm's BS.1770 meter uses a 400 ms block; anything shorter has no
# integrated loudness at all and must be passed through untouched.
MIN_SECONDS = 0.45


def measure(data, rate):
    """Integrated loudness, or None when the clip is too short/silent to have one."""
    if len(data) < int(MIN_SECONDS * rate):
        return None
    lufs = pyln.Meter(rate).integrated_loudness(data)
    # Digital silence and near-silence return -inf; there is nothing to normalize to.
    return None if not np.isfinite(lufs) else lufs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="bank directory containing *.wav")
    ap.add_argument("--target", type=float, default=-23.0, help="target LUFS (default -23, EBU R128)")
    ap.add_argument("--peak-ceiling", type=float, default=-1.0, help="max peak dBFS after gain")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    wavs = sorted(p for p in root.rglob("*.wav") if BACKUP_DIR not in p.parts)
    if not wavs:
        sys.exit(f"no wavs under {root}")

    sidecar = root / SIDECAR
    done = set()
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except Exception:
                    pass

    backup = root / BACKUP_DIR
    ceiling_amp = 10 ** (args.peak_ceiling / 20.0)

    changed = skipped = untouched = 0
    ceiling_hits, records = [], []

    for wav in wavs:
        rel = str(wav.relative_to(root))
        if rel in done:
            skipped += 1
            continue

        data, rate = sf.read(str(wav), always_2d=False)
        if data.ndim > 1:                      # meter and gain both want mono
            data = data.mean(axis=1)

        lufs = measure(data, rate)
        if lufs is None:
            untouched += 1
            records.append({"file": rel, "skipped": "too-short-or-silent"})
            continue

        gain_db = args.target - lufs
        gain = 10 ** (gain_db / 20.0)
        peak = float(np.max(np.abs(data))) if len(data) else 0.0

        # Back the gain off rather than clip. Landing quiet is recoverable; a
        # squared-off waveform is not, and it would read as distortion in audit.
        capped = False
        if peak * gain > ceiling_amp and peak > 0:
            gain = ceiling_amp / peak
            gain_db = 20 * np.log10(gain)
            capped = True
            ceiling_hits.append((rel, lufs, args.target - lufs, gain_db))

        rec = {
            "file": rel,
            "lufs_in": round(float(lufs), 2),
            "lufs_target": args.target,
            "gain_db": round(float(gain_db), 2),
            "peak_capped": capped,
        }
        records.append(rec)
        changed += 1

        if args.dry_run:
            continue

        dest = backup / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():                  # never overwrite a pristine original
            shutil.copy2(wav, dest)
        sf.write(str(wav), data * gain, rate, subtype="PCM_16")

    if not args.dry_run and records:
        with sidecar.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    verb = "would normalize" if args.dry_run else "normalized"
    print(f"{verb} {changed}  already-done {skipped}  too-short/silent {untouched}  (target {args.target} LUFS)")
    if ceiling_hits:
        print(f"  {len(ceiling_hits)} clip(s) hit the {args.peak_ceiling} dBFS ceiling and landed short of target:")
        for rel, lufs, wanted, got in ceiling_hits[:10]:
            print(f"    {rel}  in {lufs:.1f} LUFS  wanted {wanted:+.1f} dB  applied {got:+.1f} dB")
        if len(ceiling_hits) > 10:
            print(f"    … and {len(ceiling_hits) - 10} more")


if __name__ == "__main__":
    main()
