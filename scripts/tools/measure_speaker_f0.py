#!/usr/bin/env python3
"""Measure each corpus speaker's median fundamental frequency, and the corpus's coverage of it.

WHY THIS EXISTS (2026-09-19)
----------------------------
The robotic hum is NOT in the training audio. A blind test over source recordings settled
it: both same-speaker controls came back all ties, and the speaker whose renders drew
"highly robotic" in 8 of 8 notes has source clips the owner called pleasant and clear
(`notes/STATE.md` § Source audio). So the model is producing the hum, not inheriting it.

The owner's hypothesis for why — offered from the ear, before any of this was measured —
is that the hum is a FAILED EMULATION of a deep voice. The deep readers carry a natural
low vibration, the model has barely seen it, and what comes out is a buzz that resembles
it. This measures the coverage half of that claim.

⚠ THE HEADLINE IS NOT A SPEAKER'S F0. It is `rows_below`: the share of training rows that
sit under a given pitch. A model asked to render 93 Hz has whatever evidence that number
describes and no more, and the relevant scarcity is NOT clips-per-speaker — the two voices
that split on the hum are ranks 1 and 2 of 5,332 by row count.

THE ESTIMATOR IS AUTOCORRELATION OVER VOICED FRAMES, and it is checked two ways because a
pitch number that is quietly wrong would move a corpus decision:

  * `--self-test` (ON BY DEFAULT, printed every run) measures synthetic harmonic stacks at
    six known pitches spanning the range and refuses if any is off by more than `--tolerance`
    Hz. Measured 2026-09-19: worst error 0.77 Hz over 85-230 Hz.
  * `--cross-check` runs librosa's pYIN over a subsample and reports the difference.
    Measured on the three probe speakers: within 2.3 Hz. pYIN is far slower, which is why
    it is a spot check and not the estimator.

⚠ VOICED-FRAME SELECTION IS THE WHOLE ESTIMATOR. Silence and unvoiced consonants have no
pitch, and averaging them in drags every speaker toward the middle of the search range —
which would compress exactly the differences this is looking for. Frames below `--rms-floor`
or whose autocorrelation peak is below `--periodicity` are dropped, and a clip with no
surviving frame is reported rather than counted as zero.

⚠ THE SEARCH RANGE IS A PRIOR. `--fmin 55` will not find a 45 Hz voice and `--fmax 350`
will read a child or a very high soprano as its subharmonic. The range is printed, and any
speaker whose median lands within 5% of either end is flagged as possibly clipped by it.

Run:  .venv/bin/python scripts/tools/measure_speaker_f0.py \
          --corpus data/libritts_r_full_vat_v7 --min-rows 100 --speakers-sampled 140 \
          --json /tmp/speaker_f0.json
Exit: 0 measured; 2 unreadable input or a failed self-test; 3 no speaker met --min-rows.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "soundfile", "librosa"]
# ///
import argparse
import collections
import json
import os
import random
import sys

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO,):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as np  # noqa: E402

# `scripts/` is on the path for `lib.periodicity`, the shared estimator. The repo root is
# already inserted above for `matcha`; this adds the sibling directory, not a second copy.
_SCRIPTS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SCRIPTS not in _sys.path:
    _sys.path.insert(0, _SCRIPTS)
from lib import periodicity as periodicity_lib  # noqa: E402
import soundfile as sf  # noqa: E402

from matcha.delivery import VAT_DIM  # noqa: E402


def _refuse(msg):
    print("REFUSED: %s" % msg, file=sys.stderr)
    raise SystemExit(2)


def _empty(msg):
    print("REFUSED: %s" % msg, file=sys.stderr)
    raise SystemExit(3)


def f0_frames(x, sr, fmin, fmax, rms_floor, periodicity):
    """Per-frame F0 over the voiced frames only. Empty when nothing is voiced.

    ⚠ THE ESTIMATOR MOVED TO `scripts/lib/periodicity.py` and this is now a delegation.
    `measure_harmonicity.py` needs the same voicing decision to build an HNR axis that is
    comparable with this F0 axis, and it had a byte-identical copy of the loop. The
    self-test below still runs on every invocation, so it now guards the shared code.
    """
    f0, _hnr = periodicity_lib.frames(x, sr, fmin, fmax, rms_floor, periodicity)
    return f0


def clip_f0(path, fmin, fmax, rms_floor, periodicity):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    f = f0_frames(x, sr, fmin, fmax, rms_floor, periodicity)
    return float(np.median(f)) if len(f) else float("nan")


def self_test(args):
    """POSITIVE CONTROL, run on every invocation. Synthetic stacks at a known pitch.

    A harmonic stack is not a voice, and passing this does not make the estimator right on
    speech — that is what --cross-check is for. What it does catch is the class of error
    that would silently move a corpus decision: an off-by-one in the lag-to-Hz conversion,
    a search range that excludes the pitches being looked for, an octave error.
    """
    sr = 24000
    t = np.arange(int(2.0 * sr)) / sr
    worst, rows = 0.0, []
    for true in (85.0, 95.0, 110.0, 140.0, 180.0, 230.0):
        x = sum((0.8 ** k) * np.sin(2 * np.pi * true * (k + 1) * t) for k in range(7))
        x = 0.3 * x / np.max(np.abs(x))
        f = f0_frames(x, sr, args.fmin, args.fmax, args.rms_floor, args.periodicity)
        got = float(np.median(f)) if len(f) else float("nan")
        err = abs(got - true) if np.isfinite(got) else float("inf")
        worst = max(worst, err)
        rows.append((true, got, err))
    print("SELF-TEST  synthetic harmonic stacks, %.0f-%.0f Hz search range"
          % (args.fmin, args.fmax))
    for true, got, err in rows:
        print("   %6.1f Hz -> %6.1f   error %+.2f" % (true, got, got - true))
    if worst > args.tolerance:
        _refuse("the estimator is off by %.2f Hz on a synthetic tone of known pitch, and "
                "--tolerance is %.2f. Every number this would print is suspect."
                % (worst, args.tolerance))
    print("   worst error %.2f Hz, within --tolerance %.2f  ✓\n" % (worst, args.tolerance))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--min-rows", type=int, default=100,
                    help="speakers with fewer rows than this are not measured")
    ap.add_argument("--speakers-sampled", type=int, default=0,
                    help="0 = every eligible speaker")
    ap.add_argument("--clips-per-speaker", type=int, default=4)
    ap.add_argument("--also", default="", help="comma-separated speakers to include regardless")
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=350.0)
    ap.add_argument("--rms-floor", type=float, default=0.01)
    ap.add_argument("--periodicity", type=float, default=0.35)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--no-self-test", action="store_true")
    ap.add_argument("--cross-check", type=int, default=0,
                    help="run librosa pYIN over this many speakers as an independent check")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.fmin >= args.fmax:
        _refuse("--fmin %.1f is not below --fmax %.1f" % (args.fmin, args.fmax))
    if not args.no_self_test:
        self_test(args)

    path = os.path.join(args.corpus, "%s_op.txt" % args.split)
    if not os.path.exists(path):
        _refuse("%s holds no %s_op.txt" % (args.corpus, args.split))
    rows = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            p = line.split("|")
            if len(p) != 4:
                _refuse("%s line %d states %d fields and a filelist row has 4" % (path, n, len(p)))
            if len(p[3].split(",")) != VAT_DIM:
                _refuse("%s line %d is not a %d-wide filelist" % (path, n, VAT_DIM))
            rows[int(p[1])].append(p[0])

    eligible = sorted(s for s, v in rows.items() if len(v) >= args.min_rows)
    if not eligible:
        _empty("no speaker in %s has %d rows — the most any has is %d"
               % (path, args.min_rows, max(len(v) for v in rows.values())))

    rng = random.Random(args.seed)
    # ⚠ COPY, NOT AN ALIAS. `--also` appends to `chosen`, and with a bare assignment
    # that append landed in `eligible` too — inflating the "of N speakers with >= M
    # rows" line by however many extras were named, i.e. reporting a population that
    # did not meet the threshold the same sentence claims for it.
    chosen = list(eligible)
    if args.speakers_sampled and args.speakers_sampled < len(eligible):
        chosen = rng.sample(eligible, args.speakers_sampled)
    for tok in args.also.split(","):
        tok = tok.strip()
        if not tok:
            continue
        s = int(tok)
        if s not in rows:
            _refuse("--also names speaker %d and the %s split holds no such speaker"
                    % (s, args.split))
        if s not in chosen:
            chosen.append(s)
    chosen = sorted(set(chosen))

    print("MEASURING %d speakers (of %d with >= %d rows) x %d clips"
          % (len(chosen), len(eligible), args.min_rows, args.clips_per_speaker))

    f0, weight, silent = {}, {}, []
    for s in chosen:
        vals = []
        for fp in rng.sample(rows[s], min(args.clips_per_speaker, len(rows[s]))):
            v = clip_f0(fp, args.fmin, args.fmax, args.rms_floor, args.periodicity)
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            silent.append(s)
            continue
        f0[s] = float(np.median(vals))
        weight[s] = len(rows[s])

    if not f0:
        _empty("no clip in any sampled speaker produced a voiced frame — check --rms-floor "
               "(%.3f) and --periodicity (%.2f)" % (args.rms_floor, args.periodicity))
    if silent:
        print("⚠ %d speakers produced no voiced frame and are EXCLUDED, not counted as "
              "zero: %s" % (len(silent), silent[:5]))

    vals = np.array([f0[s] for s in sorted(f0)])
    edge = [s for s in f0
            if f0[s] < args.fmin * 1.05 or f0[s] > args.fmax * 0.95]
    print("\n=== speaker-level F0 (Hz) ===")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print("  p%-3d %6.1f" % (q, np.percentile(vals, q)))
    print("  min  %6.1f    max %6.1f" % (vals.min(), vals.max()))
    if edge:
        print("  ⚠ %d speakers sit within 5%% of a search-range end and may be clipped by "
              "it: %s" % (len(edge), sorted(edge)[:5]))

    total = sum(weight.values())
    print("\n=== coverage: how much of the corpus sits BELOW a pitch ===")
    print("  (rows, not speakers — this is what the model has to learn a register from)")
    for thr in (90, 100, 110, 120, 140, 160):
        sel = [s for s in f0 if f0[s] < thr]
        r = sum(weight[s] for s in sel)
        print("  under %3d Hz   %3d of %3d speakers (%4.1f%%)   %6.1f%% of rows"
              % (thr, len(sel), len(f0), 100.0 * len(sel) / len(f0), 100.0 * r / total))

    if args.cross_check:
        import librosa                                            # noqa: PLC0415
        print("\n=== CROSS-CHECK: librosa pYIN, an independent algorithm ===")
        picks = rng.sample(sorted(f0), min(args.cross_check, len(f0)))
        diffs = []
        for s in picks:
            yin = []
            for fp in rng.sample(rows[s], min(3, len(rows[s]))):
                x, sr = sf.read(fp, dtype="float64")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                y, _, _ = librosa.pyin(x.astype("float32"), fmin=args.fmin,
                                       fmax=args.fmax, sr=sr)
                y = y[np.isfinite(y)]
                if len(y):
                    yin.append(float(np.median(y)))
            if yin:
                d = f0[s] - float(np.median(yin))
                diffs.append(d)
                print("  spk%-5d  autocorr %6.1f   pYIN %6.1f   %+.1f" % (s, f0[s], np.median(yin), d))
        if diffs:
            a = np.abs(diffs)
            print("  median |difference| %.1f Hz, worst %.1f Hz over %d speakers"
                  % (np.median(a), a.max(), len(diffs)))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"corpus": args.corpus, "split": args.split,
                       "fmin": args.fmin, "fmax": args.fmax,
                       "min_rows": args.min_rows,
                       "clips_per_speaker": args.clips_per_speaker,
                       "speakers": {str(s): {"f0": round(f0[s], 2), "rows": weight[s]}
                                    for s in sorted(f0)},
                       "no_voiced_frame": silent}, fh, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
