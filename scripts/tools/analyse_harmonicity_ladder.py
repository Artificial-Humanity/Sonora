"""Compare checkpoints on HNR RANGE COMPRESSION, paired on the speakers they share.

⚠⚠ THE MISTAKE THIS TOOL EXISTS TO PREVENT. On 2026-09-21 the first checkpoint ladder was
read by printing each rung's slope with its own bootstrap CI, noticing that the CIs
overlapped, and reporting the ladder as FLAT — "maturity is not the lever". That was an
underpowered null presented as a finding. Every rung scores the SAME speakers, so the
comparison is paired and the right statistic is a CI on the DIFFERENCE, resampled jointly.
Run paired, the same data gave +0.073 [-0.057, +0.177] — the same direction, merely
unresolvable at 36 speakers — and a pitch-matched sample of 48 then put it at
+0.128 [+0.061, +0.201], monotone across all five rungs. Overlapping marginal CIs are not
a null, and two marginal CIs are the wrong test whenever the samples are the same.

WHAT THE SLOPE IS
-----------------
Least squares of the RENDERED per-speaker HNR on the REAL per-speaker HNR. 1.0 means the
model reproduces each voice's periodicity faithfully; below 1.0 it pulls every voice
toward one middling amount, which is what strips the breath and irregularity out of a
rough voice and leaves something closer to a pure oscillation. The vocoder round trip is
the floor — it is the same measurement with the acoustic model taken out — so a slope at
the round trip's value means the model adds no compression of its own.

⚠ THE SLOPE IS THE STATISTIC, NOT THE MEAN. Mean `d_hnr_syn` moves the LEVEL and says
nothing about the RANGE. On the pitch-matched ladder the means run -0.611, -0.011, -0.195,
-0.123, +0.147 — no order at all — while the slopes run 0.537, 0.792, 0.825, 0.880, 0.920.
Reading the means would have said the ladder was noise for a second time.

⚠ THE WITHIN-PAIR TEST NEEDS A PAIRED FILELIST. Passing --key from
`build_hnr_error_filelist.py` adds the decisive statistic: for two speakers at the same
pitch, is the rougher one rendered more over-periodic than their partner? Pitch is held
fixed by construction, so a pitch effect cannot produce it. Without --key only the slopes
are reported.

Usage:
    python scripts/tools/analyse_harmonicity_ladder.py \
        --csv /data/model-training/sonora/pitch_error/harmonicity_hnr_ladder.csv \
        --key /data/model-training/sonora/pitch_error/hnr_clips.key.json \
        --ref ep000_s007011
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def slope(pts):
    n = len(pts)
    mx = sum(a for a, _ in pts) / n
    my = sum(b for _, b in pts) / n
    sxx = sum((a - mx) ** 2 for a, _ in pts)
    if sxx <= 0:
        raise SystemExit("REFUSING: every speaker carries the same real HNR, so the "
                         "regression has no x-axis. A compression slope needs a RANGE to "
                         "compress.")
    return sum((a - mx) * (b - my) for a, b in pts) / sxx


def speaker_means(rows, spk_of, cols):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for c in cols:
            agg[spk_of(r["clip"])][c].append(float(r[c]))
    return {s: {c: sum(v) / len(v) for c, v in d.items()} for s, d in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="output of measure_harmonicity.py")
    ap.add_argument("--key", default=None,
                    help="hnr_clips.key.json; enables the within-pair test")
    ap.add_argument("--ref", default=None,
                    help="checkpoint every other is differenced against")
    ap.add_argument("--boots", type=int, default=5000)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rows = defaultdict(list)
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if "ckpt" not in r:
                raise SystemExit("REFUSING: %s has no `ckpt` column, so it holds ONE "
                                 "checkpoint and there is no ladder to compare."
                                 % args.csv)
            rows[r["ckpt"]].append(r)
    if len(rows) < 2:
        raise SystemExit("REFUSING: %d checkpoint(s) in %s. The paired comparison this "
                         "tool exists for needs at least two." % (len(rows), args.csv))

    key = None
    if args.key:
        key = json.loads(Path(args.key).read_text())["clips"]
        # ⚠ THE KEY MUST COVER THE CSV, and a partial overlap is the dangerous case. A key
        # from a DIFFERENT filelist raised a bare KeyError on the first missing clip, which
        # reads as a crash rather than as "you handed me the wrong pairing". Had the two
        # filelists shared some clips, the speaker map would have been right for those and
        # absent for the rest, and a lenient lookup would have silently analysed a subset.
        missing = {r["clip"] for rs in rows.values() for r in rs} - set(key)
        if missing:
            raise SystemExit(
                "REFUSING: --key %s does not name %d of the %d clips in %s (e.g. %s).\n"
                "  A key from a different filelist maps speakers WRONGLY, not partially."
                % (args.key, len(missing),
                   len({r["clip"] for rs in rows.values() for r in rs}), args.csv,
                   sorted(missing)[0]))
        spk_of = lambda c: key[c]["spk"]                                 # noqa: E731
    else:
        spk_of = lambda c: c.split("_")[0]                               # noqa: E731

    cols = ("real_hnr", "syn_hnr", "rt_hnr", "d_hnr_syn")
    M = {ck: speaker_means(rs, spk_of, cols) for ck, rs in rows.items()}
    cks = sorted(M)
    sp = sorted(M[cks[0]])
    for ck in cks:
        if sorted(M[ck]) != sp:
            raise SystemExit(
                "REFUSING: %s scored %d speakers and %s scored %d. The comparison is "
                "PAIRED on the speakers, and differencing two different populations "
                "reports their composition as a checkpoint effect."
                % (cks[0], len(sp), ck, len(M[ck])))

    ref = args.ref or cks[0]
    if ref not in M:
        raise SystemExit("REFUSING: --ref %r is not in %s. Present: %s"
                         % (ref, args.csv, ", ".join(cks)))

    rng = random.Random(args.seed)
    base = list(range(len(sp)))
    boots = [[rng.randrange(len(sp)) for _ in sp] for _ in range(args.boots)]
    sl = lambda ck, idx: slope([(M[ck][sp[i]]["real_hnr"],                # noqa: E731
                                 M[ck][sp[i]]["syn_hnr"]) for i in idx])
    floor = slope([(M[ref][s]["real_hnr"], M[ref][s]["rt_hnr"]) for s in sp])

    plist = []
    if key:
        pairs = defaultdict(dict)
        for c in key.values():
            if "pair" in c and "side" in c:
                pairs[c["pair"]][c["side"]] = c["spk"]
        plist = [(d["rough"], d["clean"]) for d in pairs.values() if len(d) == 2]
        if not plist:
            print("⚠ --key carries no rough/clean pairs; the within-pair test is skipped.",
                  file=sys.stderr)

    print("%d speakers, %d checkpoints. Vocoder round-trip floor: slope %.3f"
          % (len(sp), len(cks), floor))
    print("ref = %s\n" % ref)
    head = "checkpoint              slope   95% CI           delta vs ref     95% CI"
    if plist:
        head += "          within-pair    p       +/n"
    print(head)
    for ck in cks:
        bs = sorted(sl(ck, b) for b in boots)
        dd = sorted(sl(ck, b) - sl(ref, b) for b in boots)
        lo, hi = dd[int(.025 * args.boots)], dd[int(.975 * args.boots) - 1]
        line = ("%-22s  %.3f  [%.3f,%.3f]   %+.3f          [%+.3f,%+.3f]%s"
                % (ck, sl(ck, base), bs[int(.025 * args.boots)],
                   bs[int(.975 * args.boots) - 1], sl(ck, base) - sl(ref, base),
                   lo, hi, "" if lo <= 0 <= hi else " *"))
        if plist:
            d = [M[ck][R]["d_hnr_syn"] - M[ck][C]["d_hnr_syn"] for R, C in plist]
            n = len(d)
            mean = sum(d) / n
            # ⚠ SIGN-FLIP, NOT LABEL-SHUFFLE. The pair is the unit and "which of these two
            # is the rough one" is the label being tested, so the null flips each pair's
            # sign independently. Shuffling clips across pairs would test a different and
            # much easier hypothesis.
            hits = sum(1 for _ in range(args.perms)
                       if abs(sum(x if rng.random() < .5 else -x for x in d) / n)
                       >= abs(mean))
            line += ("   %+.3f dB  %.4f  %2d/%d"
                     % (mean, (hits + 1) / (args.perms + 1.0),
                        sum(1 for x in d if x > 0), n))
        print(line)
    print("\n* = the delta's CI excludes zero. Slopes are only comparable WITHIN one "
          "sample;\n  the floor above is this sample's, not a constant.")


if __name__ == "__main__":
    main()
