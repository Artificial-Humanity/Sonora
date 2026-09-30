"""Unblind the baseline hum bench and apply its pre-registered rule.

Reads the key written by `render_ear_baseline.py render` and the app's verdicts. Reports:

  ours     our model over its round trip, per item (must reproduce the known effect)
  stock    stock Matcha over its round trip, per item
  matched  (ours gap - stock gap) per HNR-matched pair, both items rated

each as a mean with an EXACT two-sided sign-flip p, then the outcome from
`baseline_bench.outcome`. No randomness, so the numbers do not move between runs.

    .venv/bin/python scripts/tools/unblind_ear_baseline.py \\
        --key /data/model-training/sonora/eartest/_keys/baseline_hum.key.json \\
        --test /data/model-training/sonora/eartest/baseline_hum
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import baseline_bench as bb                          # noqa: E402

READING = {
    "matcha": "Stock Matcha adds hum like ours: the recipe is at fault, not our history.",
    "lineage": "Ours adds more than stock, and stock adds none: our history introduced it.",
    "both": "Stock adds hum AND ours adds more: part recipe, part history.",
    "inconclusive": "Neither contrast is significant: the bench cannot separate them.",
    "invalid": "Ours did not reproduce its known hum over the round trip: no reading.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    truth = json.loads(Path(args.key).read_text())["items"]
    vpath = Path(args.test) / "verdicts" / "verdicts.csv"
    if not vpath.is_file():
        raise SystemExit("REFUSING: no %s — nothing was judged." % vpath)

    gap = {}
    with vpath.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["item"] not in truth:
                raise SystemExit("REFUSING: verdict for %r has no entry in the key."
                                 % r["item"])
            if r.get("sev_a", "") == "" or r.get("sev_b", "") == "":
                continue
            t = truth[r["item"]]
            sev = {t["A_label"]: int(r["sev_a"]), t["B_label"]: int(r["sev_b"])}
            rt, model = t["kind"].split("_vs_")
            gap[r["item"]] = sev[model] - sev[rt]
    missing = len(truth) - len(gap)
    if missing:
        print("⚠ %d of %d items not fully rated; they are left out.\n"
              % (missing, len(truth)))

    fam = {"stock": {}, "ours": {}}
    for item, d in gap.items():
        fam[truth[item]["family"]][truth[item]["match"]] = (item, d)

    def line(name, d):
        if not d:
            print("%-8s (none rated)" % name)
            return (0.0, 1.0)
        m, p = st.mean(d), bb.sign_flip_p(d)
        up, dn = sum(x > 0 for x in d), sum(x < 0 for x in d)
        print("%-8s n=%2d  mean %+.2f  p = %.4f  %d worse / %d better / %d tied"
              % (name, len(d), m, p, up, dn, len(d) - up - dn))
        return (m, p)

    print("MODEL OVER ITS OWN ROUND TRIP (severity 0-5; + = the model hums more)")
    ours = line("ours", [d for _i, d in fam["ours"].values()])
    stock = line("stock", [d for _i, d in fam["stock"].values()])
    both = sorted(set(fam["ours"]) & set(fam["stock"]))
    print("\nHNR-MATCHED PAIRS (ours gap - stock gap)")
    matched = line("matched", [fam["ours"][m][1] - fam["stock"][m][1] for m in both])

    got = bb.outcome(ours, stock, matched)
    print("\nOUTCOME: %s — %s" % (got.upper(), READING[got]))

    print("\nmatch  HNR    ours gap  stock gap")
    for m in both:
        print("  %2d  %5.2f   %+d        %+d"
              % (m, truth[fam["ours"][m][0]]["hnr"], fam["ours"][m][1], fam["stock"][m][1]))


if __name__ == "__main__":
    main()
