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

# "Not shown" is not "absent": twelve items cannot show that a difference is zero.
READING = {
    "matcha": ("Stock Matcha adds hum over its own vocoder; ours adding MORE is not shown. "
               "The recipe produces it, so our history is not needed to explain it."),
    "lineage": ("Ours adds more hum than stock, and stock adding hum is not shown. Our "
                "training history is implicated: bisect the lineage."),
    "both": "Stock adds hum AND ours adds more: part recipe, part history.",
    "inconclusive": ("Neither contrast is shown (or the round-trip floors are not "
                     "comparable): the bench cannot separate recipe from history."),
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

    gap, sev_by = {}, {}
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
            for lbl, v in sev.items():
                sev_by.setdefault(lbl, []).append(v)
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

    print("SEVERITY BY CONDITION (0 = cannot hear it, 5 = Freak-a-Zoid)")
    for lbl in ("stockrt", "stock", "oursrt", "ours"):
        v = sev_by.get(lbl, [])
        if v:
            print("  %-8s n=%2d  mean %.2f  %s" % (lbl, len(v), st.mean(v),
                                                "".join(str(x) for x in sorted(v))))
    rt_means = None
    if sev_by.get("stockrt") and sev_by.get("oursrt"):
        rt_means = (st.mean(sev_by["stockrt"]), st.mean(sev_by["oursrt"]))
        ok = abs(rt_means[0] - rt_means[1]) < 1.0 and max(rt_means) < 4.0
        print("  round-trip floors %.2f / %.2f: %s" % (
            rt_means[0], rt_means[1], "comparable" if ok else
            "NOT comparable (a point apart, or one at 4+) — the matched contrast is not read"))

    print("\nMODEL OVER ITS OWN ROUND TRIP (+ = the model hums more)")
    ours = line("ours", [d for _i, d in fam["ours"].values()])
    stock = line("stock", [d for _i, d in fam["stock"].values()])
    both = sorted(set(fam["ours"]) & set(fam["stock"]))
    print("\nHNR-MATCHED PAIRS (ours gap - stock gap)")
    matched = line("matched", [fam["ours"][m][1] - fam["stock"][m][1] for m in both])

    got = bb.outcome(ours, stock, matched, rt_means)
    print("\nOUTCOME: %s — %s" % (got.upper(), READING[got]))

    print("\nmatch  HNR    ours gap  stock gap")
    for m in both:
        print("  %2d  %5.2f   %+d        %+d"
              % (m, truth[fam["ours"][m][0]]["hnr"], fam["ours"][m][1], fam["stock"][m][1]))


if __name__ == "__main__":
    main()
