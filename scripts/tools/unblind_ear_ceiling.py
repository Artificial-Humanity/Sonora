"""Unblind the ceiling bench: severity by CONDITION, and the two differences that matter.

⚠⚠ THE SEVERITY RATING IS THE DATUM AND THE PAIRING IS THE CONTROL. Each item rates two of
{real, rt, model} on the SAME source recording, minutes apart, so a within-item difference
is immune to the criterion drift that makes absolute levels wander between sittings. The
per-condition means are reported too, and they are the softer number — read the paired
differences first.

    real_vs_rt     the mel/vocoder round trip alone. If the hum is here, the ceiling is
                   the 80-bin mel and retraining the acoustic model cannot reach it.
    rt_vs_model    what the acoustic model adds ON TOP of that round trip.
    real_vs_model  the total, for scale.

⚠ THE THREE PAIRINGS ARE THE CONTROLS FOR EACH OTHER, which is why this bench has no catch
trials. Whichever pairing returns ~0 prices the listener's criterion for the others: a
listener who rates everything 2 would show it as a null difference everywhere.

Usage:
    python scripts/tools/unblind_ear_ceiling.py \
        --key /data/model-training/sonora/eartest/_keys/ceiling.key.json \
        --test /data/model-training/sonora/eartest/ceiling
"""

import argparse
import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    truth = json.loads(Path(args.key).read_text())["items"]
    vpath = Path(args.test) / "verdicts" / "verdicts.csv"
    if not vpath.is_file():
        raise SystemExit("REFUSING: no %s — nothing was judged." % vpath)

    rows = []
    with vpath.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["item"] not in truth:
                raise SystemExit("REFUSING: verdict for %r has no entry in the key. The "
                                 "key and the manifest have parted." % r["item"])
            if r.get("sev_a", "") == "" or r.get("sev_b", "") == "":
                continue
            t = truth[r["item"]]
            rows.append({"item": r["item"], "t": t, "note": r.get("note", ""),
                         t["A_label"]: int(r["sev_a"]),
                         t["B_label"]: int(r["sev_b"])})
    if not rows:
        raise SystemExit("REFUSING: no item carries BOTH severities. A one-sided item "
                         "has no within-item difference, which is this bench's "
                         "drift-immune statistic.")

    missing = len(truth) - len(rows)
    if missing:
        print("⚠ %d of %d items not fully rated; reporting the %d that were.\n"
              % (missing, len(truth), len(rows)))

    per = defaultdict(list)
    for r in rows:
        for lbl in ("real", "rt", "model"):
            if lbl in r:
                per[lbl].append(r[lbl])
    print("SEVERITY BY CONDITION (0 = cannot hear it, 5 = Freak-a-Zoid)")
    for lbl in ("real", "rt", "model"):
        v = per.get(lbl, [])
        if v:
            print("  %-6s n=%2d  mean %.2f  median %.1f  range %d-%d  %s"
                  % (lbl, len(v), st.mean(v), st.median(v), min(v), max(v),
                     "".join(str(x) for x in sorted(v))))

    rng = random.Random(args.seed)
    print("\nPAIRED WITHIN ITEM (the drift-immune statistic)")
    print("%-14s %2s  %-22s %-8s %s" % ("pairing", "n", "mean difference", "p", "split"))
    for kind in ("real_vs_rt", "rt_vs_model", "real_vs_model"):
        rs = [r for r in rows if r["t"]["kind"] == kind]
        if not rs:
            print("%-14s (none rated)" % kind)
            continue
        lo, hi = kind.split("_vs_")
        d = [r[hi] - r[lo] for r in rs]
        n = len(d)
        m = st.mean(d)
        # sign-flip permutation: the item is the unit and the label order is the null
        hits = sum(1 for _ in range(args.perms)
                   if abs(sum(x if rng.random() < .5 else -x for x in d) / n) >= abs(m))
        pos = sum(1 for x in d if x > 0)
        neg = sum(1 for x in d if x < 0)
        print("%-14s %2d  %-22s %-8.4f %d up / %d down / %d tied"
              % (kind, n, "%+.2f  (%s over %s)" % (m, hi, lo),
                 (hits + 1) / (args.perms + 1.0), pos, neg, n - pos - neg))

    if args.verbose:
        print("\nitem   kind           spk     HNR   real  rt  model   note")
        for r in sorted(rows, key=lambda x: x["item"]):
            t = r["t"]
            cell = lambda k: ("%d" % r[k]) if k in r else "·"       # noqa: E731
            print("%-6s %-14s %-7s %5.2f   %-4s %-3s %-6s %s"
                  % (r["item"], t["kind"], t["spk"], t["hnr"], cell("real"),
                     cell("rt"), cell("model"), (r["note"] or "")[:44]))


if __name__ == "__main__":
    main()
