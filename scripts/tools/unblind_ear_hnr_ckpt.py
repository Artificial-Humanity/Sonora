"""Unblind the differential checkpoint bench and report it as SIGNAL DETECTION.

⚠⚠ THE HEADLINE IS THE GAP BETWEEN THE HALVES, NOT EITHER HALF ALONE. The measurement
predicts the later checkpoint has less hum, and that the gain is concentrated on speakers
the model renders TOO PERIODICALLY — the rough, low-HNR ones — while their pitch-matched
clean partners barely move. A listener who splits the clean half as hard as the rough half
has reported a general preference for the later checkpoint, which this bench cannot
distinguish from a hundred other things training improved. The two halves differ in
periodicity and NOT in pitch, so the difference between them is the whole result.

⚠ `unblind_ear_ab.py` CANNOT READ THIS BENCH, which is why this file exists rather than a
flag on that one. It groups by `set` and expects exactly two labels inside each; here one
set holds three item kinds, and a catch trial carries the SAME label on both sides. That
bench would silently treat every catch trial as an ordinary pair, pool the classes, and
report a confident number for a test that did not happen.

DETECTION AND DIRECTION ARE TWO QUESTIONS AND ARE REPORTED SEPARATELY
---------------------------------------------------------------------
  DETECTION — did the listener hear any difference? Its null is the catch trials, which
  serve one checkpoint twice from two noise draws. The rate of non-"same" answers there is
  the FALSE-ALARM rate, and without it a hit rate cannot be told from a listener who
  simply dislikes choosing "same".

  DIRECTION — of the pairs where a difference was heard, how often was the SELECTED
  checkpoint named as the hummier one? That is the prediction's sign. Direction is scored
  only on detected pairs, because an undetected pair carries no direction at all, and
  folding "same" in as half a point would inflate a null toward the prediction.

d' uses the log-linear correction (x+0.5)/(n+1) so a clean 0 or a clean ceiling still
produces a finite number, and the correction is applied to EVERY cell rather than only to
the degenerate ones — applying it selectively biases the comparison between halves.

Usage:
    python scripts/tools/unblind_ear_hnr_ckpt.py \
        --key /data/model-training/sonora/eartest/_keys/hnr_ckpt.key.json \
        --test /data/model-training/sonora/eartest/hnr_ckpt
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def z(p):
    """Inverse normal CDF (Acklam), good to ~1e-9 — plenty for a 24-item bench."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def binom_p(k, n, p=0.5):
    """Two-sided exact binomial."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--selected", default="selected",
                    help="arm name the prediction says is HUMMIER")
    ap.add_argument("--verbose", action="store_true", help="print every item")
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    if "items" not in key:
        raise SystemExit("REFUSING: %s has no `items` block, so it is not this bench's "
                         "key. unblind_ear_ab.py's keys are shaped differently and would "
                         "be read here as an empty test." % args.key)
    truth = key["items"]
    arms = {a for v in truth.values() for a in (v["A_arm"], v["B_arm"])}
    if args.selected not in arms:
        raise SystemExit("REFUSING: --selected %r is not an arm in this key. Present: %s"
                         % (args.selected, ", ".join(sorted(arms))))

    vpath = Path(args.test) / "verdicts" / "verdicts.csv"
    if not vpath.is_file():
        raise SystemExit("REFUSING: no %s — nothing was judged." % vpath)
    verdicts = {}
    with vpath.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("choice"):
                verdicts[r["item"]] = r

    missing = set(truth) - set(verdicts)
    if missing:
        # ⚠ A PARTIAL BENCH IS NOT A SMALLER BENCH. The classes are interleaved, so
        # stopping early removes them unevenly and the gap between halves — the whole
        # result — is then partly a report of where the listener stopped.
        print("⚠ %d of %d items unjudged (%s). Reporting the %d that were."
              % (len(missing), len(truth), ", ".join(sorted(missing)[:4]),
                 len(truth) - len(missing)))

    by_kind = defaultdict(list)
    for item, v in verdicts.items():
        if item not in truth:
            raise SystemExit("REFUSING: verdict for %r has no entry in the key. The key "
                             "and the served manifest have parted, and every side "
                             "assignment below would be a guess." % item)
        t = truth[item]
        heard = v["choice"] in ("A", "B")
        chose_selected = heard and t["%s_arm" % v["choice"]] == args.selected
        by_kind[t["kind"]].append({"item": item, "heard": heard,
                                   "chose_selected": chose_selected, "t": t, "v": v})

    catch = by_kind.get("catch", [])
    if not catch:
        raise SystemExit("REFUSING: no catch trials were judged. Every number below is a "
                         "hit rate with no false-alarm rate to price it against, which is "
                         "indistinguishable from a listener who never says 'no "
                         "difference'.")
    fa_k, fa_n = sum(1 for c in catch if c["heard"]), len(catch)
    F = (fa_k + 0.5) / (fa_n + 1)
    print("FALSE ALARMS  %d/%d catch trials heard as different  (rate %.3f, corrected "
          "%.3f)\n" % (fa_k, fa_n, fa_k / fa_n, F))

    print("%-7s %-18s %-22s %s" % ("class", "detected", "direction (of detected)", "d'"))
    stats = {}
    for kind in ("rough", "clean"):
        rows = by_kind.get(kind, [])
        if not rows:
            print("%-7s (none judged)" % kind)
            continue
        hk, hn = sum(1 for r in rows if r["heard"]), len(rows)
        H = (hk + 0.5) / (hn + 1)
        det = z(H) - z(F)
        dk = sum(1 for r in rows if r["chose_selected"])
        stats[kind] = {"H": H, "d": det, "hk": hk, "hn": hn, "dk": dk, "dn": hk}
        dirn = ("%d/%d chose %s  p=%.3f" % (dk, hk, args.selected, binom_p(dk, hk))
                if hk else "n/a (nothing detected)")
        print("%-7s %2d/%-2d  rate %.3f   %-22s %+.2f" % (kind, hk, hn, hk / hn, dirn, det))

    if "rough" in stats and "clean" in stats:
        r, c = stats["rough"], stats["clean"]
        print("\nTHE PREDICTION: rough should exceed clean.")
        print("  detection  rough %.3f  vs  clean %.3f   gap %+.3f" %
              (r["hk"] / r["hn"], c["hk"] / c["hn"], r["hk"] / r["hn"] - c["hk"] / c["hn"]))
        print("  d'         rough %+.2f  vs  clean %+.2f   gap %+.2f" %
              (r["d"], c["d"], r["d"] - c["d"]))
        tot_k, tot_n = r["dk"] + c["dk"], r["dn"] + c["dn"]
        print("  direction  %d/%d of ALL detected pairs named %s as hummier  p=%.3f"
              % (tot_k, tot_n, args.selected, binom_p(tot_k, tot_n)))

    if args.verbose:
        print("\nitem   kind   spk   F0     HNR   A/B arms                choice  note")
        for kind in ("rough", "clean", "catch"):
            for r in sorted(by_kind.get(kind, []), key=lambda x: x["item"]):
                t, v = r["t"], r["v"]
                print("%-6s %-6s %-5s %5.1f %5.2f  %-10s/%-10s %-7s %s"
                      % (r["item"], kind, t["spk"], t["f0"], t["hnr"], t["A_arm"],
                         t["B_arm"], v["choice"], (v.get("note") or "")[:60]))


if __name__ == "__main__":
    main()
