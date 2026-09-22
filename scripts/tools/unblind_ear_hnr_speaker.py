"""Unblind the rough-versus-clean speaker bench.

BOTH STATISTICS ARE REPORTED, AND WHICH ONE LEADS WAS NAMED BEFORE THE DATA
---------------------------------------------------------------------------
DIRECTION — of the pairs where a side was named, how often was it the lower-HNR speaker?
Scored only on named items, because "no difference" carries no direction and folding it in
as half a point drags any result toward the prediction.

DETECTION — how often was a side named at all? This file first asserted that detection
"carries no information because both sides are different voices". That reasoning was
WRONG, and it is left here as the correction: the question asked is which side has more
HUM, so "no difference" means equal hum rather than identical audio, and a listener who
hears a hum difference more often at a 2.2 dB gap than at a 0.06 dB gap has reported
something real.

⚠⚠ BUT DIRECTION WAS NOMINATED FIRST AND DETECTION WAS NOT. On the 2026-09-22 run
direction came out 7/10 (p=0.34) and detection 0.83 vs 0.50 (p=0.14) — the larger-looking
number is the one chosen after seeing it. Treat detection as a hypothesis for the next
bench, not as this bench's result.

⚠⚠ THE CONTROL IS THE INSTRUMENT, AND IT IS REPORTED BESIDE THE CONTRAST ALWAYS. Control
pairs are matched on pitch AND on periodicity, so nothing predicts a side. A listener who
names one side as often there as on the contrast pairs is reporting voice identity, and
the contrast number means nothing on its own. The headline is the GAP.

Usage:
    python scripts/tools/unblind_ear_hnr_speaker.py \
        --key /data/model-training/sonora/eartest/_keys/hnr_speaker.key.json \
        --test /data/model-training/sonora/eartest/hnr_speaker
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def fisher_greater(a, b, c, d):
    """One-sided Fisher: P(contrast detections >= observed). Exact, because with a dozen
    items a side the normal approximation to a proportion is not usable."""
    n1, n2, k = a + b, c + d, a + c
    tot = num = 0.0
    for x in range(max(0, k - n2), min(n1, k) + 1):
        pr = math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
        tot += pr
        if x >= a:
            num += pr
    return num / tot


def binom_p(k, n, p=0.5):
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    if "items" not in key:
        raise SystemExit("REFUSING: %s has no `items` block, so it is not this bench's "
                         "key." % args.key)
    truth = key["items"]
    vpath = Path(args.test) / "verdicts" / "verdicts.csv"
    if not vpath.is_file():
        raise SystemExit("REFUSING: no %s — nothing was judged." % vpath)

    rows = defaultdict(list)
    with vpath.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("choice"):
                continue
            if r["item"] not in truth:
                raise SystemExit("REFUSING: verdict for %r has no entry in the key. The "
                                 "key and the served manifest have parted, and every "
                                 "side assignment below would be a guess." % r["item"])
            t = truth[r["item"]]
            rows[t["kind"]].append({
                "item": r["item"], "t": t, "v": r,
                "named": r["choice"] in ("A", "B"),
                "chose_rough": r["choice"] == t["rough_side"]})

    missing = set(truth) - {x["item"] for v in rows.values() for x in v}
    if missing:
        print("⚠ %d of %d items unjudged. The classes are interleaved, so an early stop "
              "removes them unevenly." % (len(missing), len(truth)))

    if "control" not in rows:
        raise SystemExit("REFUSING: no control pairs were judged. The contrast number "
                         "alone cannot be told from a listener hearing voice identity, "
                         "which is the whole reason the control exists.")

    print("%-9s %-14s %-28s" % ("class", "named a side", "direction (of those named)"))
    out = {}
    for kind in ("contrast", "control"):
        rs = rows.get(kind, [])
        if not rs:
            print("%-9s (none judged)" % kind)
            continue
        named = [x for x in rs if x["named"]]
        k = sum(1 for x in named if x["chose_rough"])
        out[kind] = (k, len(named), len(rs))
        d = ("%d/%d chose the ROUGHER speaker  p=%.3f"
             % (k, len(named), binom_p(k, len(named))) if named else "n/a")
        print("%-9s %2d/%-2d  %.2f    %s" % (kind, len(named), len(rs),
                                             len(named) / len(rs), d))

    if "contrast" in out and "control" in out:
        (ck, cn, cN), (lk, ln, lN) = out["contrast"], out["control"]
        print("\nDETECTION (secondary — see the header): %d/%d vs %d/%d, "
              "one-sided Fisher p = %.4f"
              % (cn, cN, ln, lN, fisher_greater(cn, cN - cn, ln, lN - ln)))
        if cn and ln:
            pc, pl = ck / cn, lk / ln
            # ⚠ PERMUTATION, NOT A z ON TWO PROPORTIONS. With a dozen items a side the
            # normal approximation is not usable, and the question is specifically whether
            # the CONTRAST rate exceeds the CONTROL rate — shuffling the class labels over
            # the pooled outcomes answers exactly that.
            pool = ([1] * ck + [0] * (cn - ck) + [1] * lk + [0] * (ln - lk))
            rng = random.Random(args.seed)
            obs = pc - pl
            hits = 0
            for _ in range(args.perms):
                rng.shuffle(pool)
                if (sum(pool[:cn]) / cn - sum(pool[cn:]) / ln) >= obs:
                    hits += 1
            print("\nTHE PREDICTION: contrast should exceed control.")
            print("  rougher-side rate  contrast %.2f  vs  control %.2f   gap %+.2f"
                  % (pc, pl, obs))
            print("  one-sided permutation p = %.4f" % ((hits + 1) / (args.perms + 1.0)))

    if args.verbose:
        print("\nitem   kind      rough  clean  HNR gap  dF0    rough_side  choice  note")
        for kind in ("contrast", "control"):
            for x in sorted(rows.get(kind, []), key=lambda r: r["item"]):
                t, v = x["t"], x["v"]
                print("%-6s %-9s %-6s %-6s %6.2f  %+5.1f  %-10s %-7s %s"
                      % (x["item"], kind, t["rough"], t["clean"], t["hnr_gap"],
                         t["d_f0"], t["rough_side"], v["choice"],
                         (v.get("note") or "")[:52]))


if __name__ == "__main__":
    main()
