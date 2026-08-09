#!/usr/bin/env python3
"""stratify_holdout_sweep — read a checkpoint sweep by CONDITIONING LABEL, not in aggregate.

WHY THIS EXISTS (2026-08-09, PR-H3)
------------------------------------
`score_holdout` answers "did the run get better". It cannot answer "did it get better
*at the thing we changed*", because a channel-specific regression hides inside an
aggregate that improved: v5's mean total moved −0.0606 while any one channel could have
gone the other way underneath it.

That distinction is what a pre-registered failure signature is FOR. The T-saturation
prediction (quality-gap-plan.md § "PREDICTION, 2026-08-08") named a shape — *T worse on
the holdout while V and A are unchanged or better* — and the sweep held the data to read
it, per clip, for both checkpoints, under identical constants. Nothing read it. Rung 1 was
marked done and the front moved on, which is how a pre-registration decays into a ritual
exactly where it was supposed to pay.

So this is the read-out, as an instrument rather than a one-off: it recomputes from
artifacts already on disk, costs no GPU, and can be pointed at any future rung's sweep.

WHAT IT MEASURES
----------------
Per clip, `delta = loss(pick) - loss(baseline)`. Negative is better. Clips are then split
by the ABSOLUTE value of each conditioning label — the top and bottom quintile of |V|,
|A|, |T| independently — and the reported statistic is

    extreme-minus-central delta

i.e. how much less the channel's extremes improved than its centre. **Positive means the
extremes lagged: the regression shape.** A channel-specific defect shows up here and is
invisible in the mean.

⚠ Read the SIGN and the CI, not the magnitude. This is a teacher-forced loss on clean
audiobook narration; it is a relative instrument (see `score_holdout` on why absolute
numbers do not compare across normalization-constant changes). Quintiles rather than a
correlation because the prediction was written about extremes, and a correlation would
average the tails away — the same reason the mean hid the question in the first place.

⚠ It cannot see legs (b) and (c) of a perceptual prediction. A loss says nothing about
whether T=+1 *sounds like* Emilia-domain audio. When a prediction has ear legs, this
closes one leg and the ear still owes the others.

    .venv/bin/python scripts/stratify_holdout_sweep.py \
        --sweep /data/model-training/sonora/holdout_eval/v5_ckpt_sweep.csv \
        --filelist data/libritts_r_holdout_devclean/holdout_8w.txt \
        --baseline vat5_init --pick vat5_ep019
"""
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import argparse
import csv
import os
import random
import statistics as st

CHANNELS = (("V", 0), ("A", 1), ("T", 2))


def labels(filelist):
    """clip basename -> (V, A, T), from the filelist's own conditioning field.

    The labels are read from the SAME file the sweep scored, so a stratification can never
    be run against a different derivation's numbers than the losses came from.
    """
    out = {}
    for line in open(filelist, encoding="utf-8"):
        f = line.rstrip("\n").split("|")
        if len(f) < 4:
            continue
        vat = f[-1].split(",")
        if len(vat) < 3:
            continue
        out[os.path.basename(f[0])] = tuple(float(x) for x in vat[:3])
    return out


def sweep_rows(path, metric):
    by = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["ckpt"], {})[r["clip"]] = float(r[metric])
    return by


def bootstrap(extreme, central, n, seed):
    """95% CI on the difference of means. Resampled independently within each stratum,
    which is the right null here: the question is whether the two groups differ, not
    whether either differs from zero."""
    rng = random.Random(seed)
    gaps = []
    for _ in range(n):
        a = st.mean([extreme[rng.randrange(len(extreme))] for _ in extreme])
        b = st.mean([central[rng.randrange(len(central))] for _ in central])
        gaps.append(a - b)
    gaps.sort()
    return gaps[int(n * 0.025)], gaps[int(n * 0.975)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", required=True, help="score_holdout per-clip CSV")
    ap.add_argument("--filelist", required=True, help="the holdout filelist it scored")
    ap.add_argument("--baseline", required=True, help="ckpt name to compare FROM")
    ap.add_argument("--pick", required=True, help="ckpt name to compare TO")
    ap.add_argument("--metric", default="total", help="CSV column (default: total)")
    ap.add_argument("--quantile", type=float, default=0.20,
                    help="tail fraction per stratum (default 0.20)")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lab = labels(args.filelist)
    by = sweep_rows(args.sweep, args.metric)
    for name in (args.baseline, args.pick):
        if name not in by:
            raise SystemExit(f"{name!r} not in the sweep. Present: {sorted(by)}")

    clips = [c for c in by[args.baseline] if c in by[args.pick] and c in lab]
    if not clips:
        raise SystemExit(
            "no clip is present in both checkpoints AND the filelist — the sweep and the "
            "filelist are probably from different derivations.")
    missing = len(by[args.baseline]) - len(clips)
    delta = {c: by[args.pick][c] - by[args.baseline][c] for c in clips}
    overall = st.mean(delta.values())
    better = 100 * sum(1 for d in delta.values() if d < 0) / len(delta)

    print(f"sweep     : {args.sweep}")
    print(f"contrast  : {args.pick} - {args.baseline}   (metric: {args.metric})")
    print(f"clips     : {len(clips)} paired and labelled"
          + (f"  ⚠ {missing} unmatched and EXCLUDED" if missing else ""))
    print(f"\nOVERALL   : {overall:+.4f}   (improved on {better:.1f}% of clips)\n")
    print(f"per-channel, top/bottom {args.quantile:.0%} of |label|:")
    print(f"  {'channel':8s} {'extreme':>9} {'central':>9} {'gap':>9}   "
          f"{'95% CI':>20}   verdict")

    results = {}
    for name, idx in CHANNELS:
        vals = sorted(abs(lab[c][idx]) for c in clips)
        hi = vals[int(len(vals) * (1 - args.quantile))]
        lo = vals[int(len(vals) * args.quantile)]
        ex = [delta[c] for c in clips if abs(lab[c][idx]) >= hi]
        ce = [delta[c] for c in clips if abs(lab[c][idx]) <= lo]
        gap = st.mean(ex) - st.mean(ce)
        clo, chi = bootstrap(ex, ce, args.resamples, args.seed)
        # "Excludes zero" is the whole verdict. A gap whose CI straddles zero is not a
        # small effect, it is an absent one, and reporting it as small is how a null
        # becomes a hedge.
        sig = not (clo < 0 < chi)
        results[name] = (gap, clo, chi, sig)
        print(f"  {name:8s} {st.mean(ex):+9.4f} {st.mean(ce):+9.4f} {gap:+9.4f}   "
              f"[{clo:+.4f}, {chi:+.4f}]   "
              + ("EXCLUDES zero" if sig else "crosses zero — no effect"))

    print("\nreading it: a POSITIVE gap means that channel's extremes improved LESS than")
    print("its centre — the channel-specific regression shape. Negative means the extremes")
    print("improved MORE. A CI crossing zero means the channel shows nothing.")
    lagging = [n for n, (g, _, _, s) in results.items() if s and g > 0]
    print("\nchannels showing the regression shape: "
          + (", ".join(lagging) if lagging else "NONE"))


if __name__ == "__main__":
    main()
