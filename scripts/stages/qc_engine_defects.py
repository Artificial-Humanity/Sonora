# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "librosa", "soundfile"]
# ///
"""Engine-specific defect detectors — ADVISORY flags for the ear, never auto-drops.

qc_gate covers defects every engine can have (truncation, collapse, loudness). These
two are specific to the scrutinized pair and are what the tier system needs in order
to ever hand those engines a ride-along back: coverage traded for instrumentation,
the same bargain qwen made.

Both are calibrated against the owner's own audition notes, and both are reported
honestly below — including where the calibration is thin. Neither drops a clip. They
add ids to qc_flags.txt, which pick_audit_subset always routes to the ear.

--------------------------------------------------------------------------------
RADIO TIMBRE (moss_vg) — validated, 100% recall at ~12% of batch
--------------------------------------------------------------------------------
MOSS intermittently applies a mid-century radio/telephone character. That is a
BANDWIDTH defect: energy collapses into roughly 300-3400 Hz. Score = energy outside
that band over energy inside it; lower means more band-limited.

Measured on delivery-v1-narration (n=64 moss_vg clips, 4 ear-confirmed positives):
all four rank in the top 8 — #1, #4, #6, #8. Recall 4/4, precision 4/8. The three
keeps interleaved are mildly band-limited clips the owner accepted, which is the
right failure direction for a flag: it costs an audition, not a clip.

--------------------------------------------------------------------------------
LEADING GAP (zonos) — partially validated, and deliberately narrowly named
--------------------------------------------------------------------------------
Zonos sometimes emits a short burst, a long silence, and only then the read. Score =
gap / blip for any sub-0.4 s segment followed by a >=0.8 s gap before the main
segment.

It ranks the one ear-confirmed instance at #3 of 71 and reproduces the note's own
description ("puff of air followed by three seconds of silence" -> "blip 0.17s then
3.67s gap"). It scores ZERO on castle-of-otranto_nar_0036, which the first draft of
this file counted as the same defect. It is not: that clip's note is "the start
sounds more like 'num my poverty'" — a spurious phoneme at onset with no silence at
all. A detector for THAT needs forced alignment against the first reference word, not
an energy profile, and is not attempted here. This one measures leading gaps and is
named for it.

Usage:
  qc_engine_defects.py --campaign delivery-v1-narration [--append-flags]
"""
import argparse
import csv
import json
import os
import pathlib

import librosa
import numpy as np

RATINGS_DIR = pathlib.Path("/data/model-training/datasets/sonora-expressive-registers")
DATASETS = pathlib.Path("/data/model-training/datasets")
SR = 24000

# Thresholds are drawn at the observed separation boundary, not fitted: on the
# calibration batch every ear-confirmed radio clip sits below 0.10 while the bulk of
# the distribution sits above it, and every leading-gap clip the ear complained about
# sits above 10 while the highest accepted clip sits at 4.75.
RADIO_MAX_OUT_RATIO = 0.10
LEADING_GAP_MIN = 10.0
DETECTORS = {"moss_vg": "radio_timbre", "zonos": "leading_gap"}


def radio_score(y, sr=SR):
    """Out-of-band over in-band energy. Lower = more band-limited = more radio-like."""
    n = len(y)
    if n < sr // 4:
        return None
    S = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    inband = S[(f >= 300) & (f <= 3400)].sum()
    outband = S[(f > 3400) & (f <= 9000)].sum() + S[(f >= 50) & (f < 300)].sum()
    return float(outband / max(inband, 1e-12))


def leading_gap_score(y, sr=SR):
    """gap/blip for a short burst followed by a long silence before the main read."""
    iv = librosa.effects.split(y, top_db=35)
    if len(iv) < 2:
        return 0.0, None
    durs = [(e - s) / sr for s, e in iv]
    main = int(np.argmax(durs))
    best, detail = 0.0, None
    for k in range(main):
        blip = durs[k]
        gap = (iv[k + 1][0] - iv[k][1]) / sr
        if blip <= 0.4 and gap >= 0.8:
            sc = gap / max(blip, 0.05)
            if sc > best:
                best, detail = sc, f"blip {blip:.2f}s then {gap:.2f}s gap"
    return float(best), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--append-flags", action="store_true",
                    help="append flagged ids to <campaign>/qc_flags.txt (deduped) so "
                         "pick_audit_subset --flags routes them to the ear")
    args = ap.parse_args()

    with open(RATINGS_DIR / "ratings.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["campaign"] == args.campaign]
    if not rows:
        raise SystemExit(f"no ratings rows for campaign {args.campaign}")

    flagged, results = [], []
    for r in rows:
        det = DETECTORS.get(r["engine"])
        if not det or not r["link"]:
            continue
        p = (RATINGS_DIR / r["link"]).resolve()
        if not p.is_file():
            continue
        y, _ = librosa.load(str(p), sr=SR, mono=True)
        if det == "radio_timbre":
            v = radio_score(y)
            hit, detail = (v is not None and v < RADIO_MAX_OUT_RATIO), None
        else:
            v, detail = leading_gap_score(y)
            hit = v >= LEADING_GAP_MIN
        results.append({"id": r["id"], "engine": r["engine"], "detector": det,
                        "score": v, "flag": bool(hit), "detail": detail,
                        "status": r["status"]})
        if hit:
            flagged.append(r["id"])

    out = DATASETS / args.campaign / "qc_engine_defects.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    by_det = {}
    for r in results:
        by_det.setdefault(r["detector"], [0, 0])
        by_det[r["detector"]][0] += 1
        by_det[r["detector"]][1] += r["flag"]
    for d, (n, k) in sorted(by_det.items()):
        print(f"{d:14s} {k:3d} flagged of {n:3d} scored")
    print(f"wrote {out}")

    if args.append_flags and flagged:
        fp = DATASETS / args.campaign / "qc_flags.txt"
        have = set()
        if fp.exists():
            have = {ln.strip() for ln in fp.read_text().splitlines() if ln.strip()}
        new = [i for i in flagged if i not in have]
        with open(fp, "a", encoding="utf-8") as f:
            for i in new:
                f.write(i + "\n")
        print(f"appended {len(new)} new ids to {fp} ({len(flagged) - len(new)} already present)")


if __name__ == "__main__":
    main()
