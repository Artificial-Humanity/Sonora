"""Measure the layering/comb signature on the ceiling bench's own clips, against the ear.

⚠⚠ THE VALIDATION IS THE OWNER'S 0-5 RATINGS, NOT A SYNTHETIC SELF-TEST ALONE. Eight
detectors for this defect class have failed, and several of them passed a synthetic check
first. The clips here are exactly the ones rated in the `ceiling` bench, so the question is
not "does this respond to an echo I inserted" but "does it order the clips the way the ear
did". The self-test still runs and still refuses, because a detector that cannot find a
delay it was handed cannot be trusted on one it was not.

⚠ A TAIL, NOT A MEAN. The defect is local to individual syllables; whole-clip averages
averaged it away in the previous attempts.

Usage:
    python scripts/tools/measure_comb.py \
        --key /data/model-training/sonora/eartest/_keys/ceiling.key.json \
        --test /data/model-training/sonora/eartest/ceiling
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import comb                                          # noqa: E402


def self_test(sr=24000, min_ratio=1.5):
    """A real-ish voiced signal, then the same signal plus a delayed copy.

    Refuses unless strength RISES with echo gain and the recovered lag is near the one
    inserted. ⚠ The clean baseline is reported too: a detector that scores a clean signal
    as highly as an echoed one has no dynamic range, and every previous failure in this
    family looked exactly like that.
    """
    # ⚠⚠ THE TEST SIGNAL MUST NOT HAVE A HARD SPECTRAL EDGE. A harmonic stack truncated
    # at k=24 stops dead at 3 kHz, and that cliff rings in the cepstrum: the first version
    # of this self-test produced a 4.6 ms artifact LARGER than the 19 ms echo it had
    # inserted, and the detector was blamed for a defect in its own test signal. An
    # exponential rolloff plus a little breath noise has no edge to ring.
    rng = np.random.default_rng(0)
    t = np.arange(int(1.5 * sr)) / sr
    f0 = 127.0
    nyq = sr / 2
    x = np.zeros_like(t)
    for k in range(1, int(nyq / f0)):
        x += np.exp(-0.18 * k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
    x += 0.02 * rng.standard_normal(len(t))                  # breath floor
    x *= (0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t))           # syllable-rate envelope
    x /= np.max(np.abs(x))
    base, _, _ = comb.echo_strength(x, sr)
    print("SELF-TEST  127 Hz voiced signal with breath floor, 1.5 s")
    print("    clean                 strength %6.2f" % base)
    got = []
    for lag, gain in ((0.0055, 0.6), (0.019, 0.6), (0.045, 0.6), (0.019, 0.3)):
        s, rl, _ = comb.echo_strength(comb.add_echo(x, sr, lag, gain), sr)
        print("    echo %5.1f ms gain %.1f   strength %6.2f   recovered lag %5.1f ms"
              % (lag * 1000, gain, s, rl * 1000))
        got.append((lag, gain, s, rl))
    for lag, gain, s, rl in got:
        if s < base * min_ratio:
            raise SystemExit(
                "REFUSING: a %.1f ms echo at gain %.1f scored %.2f against a clean "
                "baseline of %.2f, under the %.1fx this needs. A detector without "
                "dynamic range on a KNOWN echo cannot be believed on an unknown one."
                % (lag * 1000, gain, s, base, min_ratio))
        if abs(rl - lag) > 0.004:
            raise SystemExit(
                "REFUSING: an echo inserted at %.1f ms was recovered at %.1f ms. The "
                "strength number may still respond to something, but it is not "
                "measuring the delay it claims to." % (lag * 1000, rl * 1000))
    print("    responds to echo and recovers the lag  ✓\n")


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--min-selftest-ratio", type=float, default=1.5)
    args = ap.parse_args()

    self_test(min_ratio=args.min_selftest_ratio)

    key = json.loads(Path(args.key).read_text())
    truth = key["items"]
    clip_of = {}
    for name, meta in key["clips"].items():
        clip_of[(meta["pair"], meta["side"])] = (name, meta["label"])

    sev = {}
    vp = Path(args.test) / "verdicts" / "verdicts.csv"
    if vp.is_file():
        for r in csv.DictReader(vp.open(newline="", encoding="utf-8")):
            if r.get("sev_a", "") != "" and r.get("sev_b", "") != "":
                sev[r["item"]] = {truth[r["item"]]["A_label"]: int(r["sev_a"]),
                                  truth[r["item"]]["B_label"]: int(r["sev_b"])}

    clips = Path(args.test) / "clips"
    per = defaultdict(list)
    paired = []
    for item, t in sorted(truth.items()):
        row = {"item": item, "kind": t["kind"], "hnr": t["hnr"]}
        for side in ("A", "B"):
            name, label = clip_of[(item, side)]
            x, sr = sf.read(str(clips / ("%s.wav" % name)), dtype="float64")
            if x.ndim > 1:
                x = x.mean(axis=1)
            s, lag, nf = comb.echo_strength(x, sr)
            row[label] = s
            row[label + "_lag"] = lag
            per[label].append(s)
            if item in sev:
                paired.append((label, s, sev[item][label], t["hnr"]))
        paired_lags = [row.get(k + "_lag", 0) for k in ("real", "rt", "model") if k in row]
        print("  %-7s %-14s  %s" % (item, t["kind"],
              "  ".join("%s %.2f @%.0fms" % (k, row[k], 1000 * row[k + "_lag"])
                        for k in ("real", "rt", "model") if k in row)))
        del paired_lags

    print("\nSTRENGTH BY CONDITION (tail across frames; higher = more layered)")
    for lbl in ("real", "rt", "model"):
        v = per.get(lbl, [])
        if v:
            print("  %-6s n=%2d  mean %.2f  median %.2f  max %.2f"
                  % (lbl, len(v), float(np.mean(v)), float(np.median(v)), max(v)))

    if paired:
        s = [p[1] for p in paired]
        e = [p[2] for p in paired]
        rho = spearman(s, e)
        rng = random.Random(args.seed)
        hits = 0
        for _ in range(args.perms):
            sh = e[:]
            rng.shuffle(sh)
            if abs(spearman(s, sh)) >= abs(rho):
                hits += 1
        print("\nDOES IT PREDICT THE EAR?  n=%d clips with a rating" % len(paired))
        print("  rho(detector, owner severity) = %+.3f   permutation p = %.4f"
              % (rho, (hits + 1) / (args.perms + 1.0)))
        ms = [(p[1], p[2]) for p in paired if p[0] == "model"]
        if len(ms) > 3:
            print("  within MODEL clips only: rho = %+.3f  (n=%d)"
                  % (spearman([a for a, _ in ms], [b for _, b in ms]), len(ms)))


if __name__ == "__main__":
    main()
