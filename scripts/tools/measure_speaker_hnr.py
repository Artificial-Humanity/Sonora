"""Per-speaker HARMONIC-TO-NOISE RATIO across a corpus, beside pitch.

WHY THIS AND NOT `measure_speaker_f0.py`
-----------------------------------------
On 2026-09-20 the hum stopped being a pitch story. Renders of low voices come out TOO
PERIODIC — +0.73 dB above the real recording — and renders of high voices not periodic
enough, and the model compresses the whole periodicity range toward its middle at a slope
of 0.728. But a speaker's natural HNR correlates with their F0 at +0.922, so in that
36-speaker sample pitch and periodicity were very nearly one variable. Holding HNR fixed,
pitch explained nothing: the partial correlation was +0.004.

That result needs a sample where the two come apart, and this tool finds one. It measures
BOTH quantities per speaker from one autocorrelation pass, so the two axes cannot disagree
about which frames are voiced, and it reports the joint distribution rather than each
margin — the count that matters is how many speakers sit OFF the diagonal.

⚠ THE EXISTING 36-SPEAKER SAMPLE CANNOT ANSWER IT. Four of its speakers break the pattern
and all four sit within 20 Hz of the median pitch and 2 dB of the median HNR, so they are
off-diagonal only by a hair. A real test needs speakers far out on one axis and ordinary on
the other, which is a question about the whole corpus rather than about that draw.

⚠ HNR HERE IS A PROPERTY OF THE RECORDING, NOT OF THE VOICE ALONE. Room noise, microphone
distance and the recording chain all push it down, and LibriTTS-R's partitions differ in
exactly those respects — `train-other-500` is the noisier half and holds 55.8% of the rows.
The partition is therefore reported per speaker and a downstream sample must match on it,
or "low HNR" and "noisier partition" will be the same column again.

Usage:
    python scripts/tools/measure_speaker_hnr.py \
        --corpus data/libritts_r_full_vat_v7 \
        --speakers-sampled 400 --clips-per-speaker 5 \
        --json /tmp/speaker_hnr.json
"""

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib import periodicity                                     # noqa: E402
from lib.corpus_filelist import dataset_of, partition_of        # noqa: E402
from matcha.delivery import VAT_DIM                             # noqa: E402


def self_test(args):
    """A synthetic stack at known noise levels. HNR must fall as noise rises.

    ⚠ THIS RUNS ON EVERY INVOCATION and refuses, for the same reason the F0 tool's does.
    The measured effect is a difference of well under one decibel between groups, so an
    estimator that is merely monotonic-ish would produce a confident number that means
    nothing. The check is ORDERING plus a floor on the spread, not an absolute calibration:
    autocorrelation HNR is not the same estimator as Praat's and the two do not agree on an
    absolute scale.
    """
    sr, f0, dur = 24000, 120.0, 1.0
    t = np.arange(int(sr * dur)) / sr
    clean = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 21))
    clean = clean / np.abs(clean).max()
    rng = np.random.default_rng(0)
    print("SELF-TEST  synthetic stack at %.0f Hz, rising noise" % f0)
    got = []
    for snr_db in (40, 30, 20, 10, 5):
        noise = rng.normal(0, 1, len(clean))
        noise = noise / np.sqrt(np.mean(noise ** 2))
        amp = np.sqrt(np.mean(clean ** 2)) / (10 ** (snr_db / 20.0))
        _f, h = periodicity.frames(clean + amp * noise, sr, args.fmin, args.fmax,
                                   args.rms_floor, args.periodicity)
        if not len(h):
            raise SystemExit("REFUSING: the self-test signal at %d dB SNR produced no "
                             "voiced frame. The voicing rule cannot see a pure harmonic "
                             "stack, so it will not see speech either." % snr_db)
        m = float(np.median(h))
        got.append((snr_db, m))
        print("    SNR %2d dB -> HNR %6.2f dB" % (snr_db, m))
    if any(got[i][1] <= got[i + 1][1] for i in range(len(got) - 1)):
        raise SystemExit("REFUSING: measured HNR is not monotonic in SNR: %s. The "
                         "estimator cannot order two recordings, so it cannot support a "
                         "correlation." % ", ".join("%d->%.2f" % g for g in got))
    spread = got[0][1] - got[-1][1]
    if spread < args.min_selftest_spread:
        raise SystemExit("REFUSING: 40 dB of SNR moved HNR by only %.2f dB, below "
                         "--min-selftest-spread %.2f. A compressed estimator would report "
                         "a null from any sample." % (spread, args.min_selftest_spread))
    print("   monotonic, spread %.2f dB  ✓\n" % spread)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--speakers-sampled", type=int, default=400)
    ap.add_argument("--clips-per-speaker", type=int, default=5)
    ap.add_argument("--dataset", default="LibriTTS_R")
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=350.0)
    ap.add_argument("--rms-floor", type=float, default=0.01)
    ap.add_argument("--periodicity", type=float, default=0.3)
    ap.add_argument("--min-selftest-spread", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    self_test(args)

    fp = Path(args.corpus) / ("%s_op.txt" % args.split)
    if not fp.is_file():
        raise SystemExit("REFUSING: no %s." % fp)
    by_spk = defaultdict(list)
    for lineno, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 4:
            raise SystemExit("REFUSING: %s line %d has %d fields, wanted 4."
                             % (fp, lineno, len(parts)))
        if len(parts[3].split(",")) != VAT_DIM:
            raise SystemExit("REFUSING: %s line %d is %d-wide and this checkout states "
                             "VAT_DIM as %d." % (fp, lineno, len(parts[3].split(",")),
                                                 VAT_DIM))
        # ⚠⚠ THIS TESTED THE WRONG THING until 2026-09-21. It read
        # `partition_of(parts[0]) == "(unkeyed)"`, and `partition_of` returns the DATASET
        # NAME for a non-LibriTTS row — so an Emilia row came back "emilia_kept_24k",
        # never "(unkeyed)", and `--dataset LibriTTS_R` restricted nothing at all. The
        # 1526-speaker survey of 2026-09-21 came out all-LibriTTS-R anyway, because
        # Emilia's 10,653 rows spread too thinly for any of its speakers to clear
        # --min-rows — by luck of the corpus, not by this line.
        if args.dataset and dataset_of(parts[0]) != args.dataset:
            continue
        by_spk[int(parts[1])].append(parts[0])

    eligible = sorted(s for s, w in by_spk.items() if len(w) >= args.min_rows)
    if len(eligible) < args.speakers_sampled:
        print("  only %d speakers have >= %d rows; measuring all of them"
              % (len(eligible), args.min_rows))
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(eligible, min(args.speakers_sampled, len(eligible))))

    out, skipped = {}, []
    for n, s in enumerate(chosen):
        f0s, hnrs, parts = [], [], []
        for wav in rng.sample(by_spk[s], min(args.clips_per_speaker, len(by_spk[s]))):
            if not os.path.exists(wav):
                continue
            x, sr = sf.read(wav, dtype="float64")
            if x.ndim > 1:
                x = x.mean(axis=1)
            f, h = periodicity.frames(x, sr, args.fmin, args.fmax, args.rms_floor,
                                      args.periodicity)
            if len(f):
                f0s.append(float(np.median(f)))
                hnrs.append(float(np.median(h)))
                parts.append(partition_of(wav))
        if not f0s:
            skipped.append(s)
            continue
        out[s] = {"f0": round(statistics.median(f0s), 2),
                  "hnr": round(statistics.median(hnrs), 3),
                  "rows": len(by_spk[s]),
                  "partition": statistics.mode(parts), "clips": len(f0s)}
        if (n + 1) % 50 == 0:
            print("    %d/%d speakers" % (n + 1, len(chosen)), flush=True)

    if not out:
        raise SystemExit("REFUSING: no speaker produced a voiced frame.")
    f0v = [v["f0"] for v in out.values()]
    hnv = [v["hnr"] for v in out.values()]
    mf, mh = statistics.median(f0v), statistics.median(hnv)
    quad = defaultdict(int)
    for v in out.values():
        quad["%s F0 / %s HNR" % ("low" if v["f0"] < mf else "high",
                                 "low" if v["hnr"] < mh else "high")] += 1

    print("\n=== %d speakers measured, %d skipped ===" % (len(out), len(skipped)))
    print("  F0  median %.1f Hz   range %.1f - %.1f" % (mf, min(f0v), max(f0v)))
    print("  HNR median %.2f dB   range %.2f - %.2f" % (mh, min(hnv), max(hnv)))
    print("\n  JOINT DISTRIBUTION — the off-diagonal count is the point:")
    for k in sorted(quad):
        print("    %-22s %4d" % (k, quad[k]))
    off = quad["low F0 / high HNR"] + quad["high F0 / low HNR"]
    print("    %-22s %4d  (%.0f%%)" % ("OFF-DIAGONAL", off, 100 * off / len(out)))

    Path(args.json).write_text(json.dumps(
        {"corpus": args.corpus, "split": args.split, "fmin": args.fmin,
         "fmax": args.fmax, "min_rows": args.min_rows,
         "clips_per_speaker": args.clips_per_speaker, "skipped": skipped,
         "median_f0": mf, "median_hnr": mh,
         "speakers": {str(k): v for k, v in out.items()}}, indent=2), encoding="utf-8")
    print("\n  wrote %s" % args.json)


if __name__ == "__main__":
    main()
