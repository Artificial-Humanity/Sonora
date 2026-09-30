"""Unblind the lineage hum bench and apply its pre-registered reading.

Reads the key written by `render_ear_lineage.py render` and the app's verdicts, and prints
each checkpoint's gap (model minus round trip), each step's change voice by voice, the
first-to-last change, and the outcome from `lineage_bench.reading`. Exact tests; no
randomness.

For scale only (a different bench, different voices — not compared statistically): the
baseline bench put stock `matcha_vctk` at +0.75 and vat7 ep005 at +2.50.

    .venv/bin/python scripts/tools/unblind_ear_lineage.py \\
        --key /data/model-training/sonora/eartest/_keys/lineage_hum.key.json \\
        --test /data/model-training/sonora/eartest/lineage_hum
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import lineage_bench as lb                           # noqa: E402

ORDER = ["derisk", "vat3", "vat5", "vat6", "vat7"]
READING = {
    "step": "The excess enters at the named step(s); look at what that step changed.",
    "first": ("The hum is already there at derisk-energy, the first move off stock, and no "
              "later rise is shown: look at what that first step changed."),
    "gradual": "No single step is shown, but it rose from first to last: a creep.",
    "inconclusive": "No step and no first-to-last rise is shown.",
    "invalid": "vat7 did not reproduce its known hum over the round trip: no reading.",
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

    gaps = {c: {} for c in ORDER}
    sev = {c: [] for c in ORDER + ["rt"]}
    with vpath.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["item"] not in truth:
                raise SystemExit("REFUSING: verdict for %r has no entry in the key."
                                 % r["item"])
            if r.get("sev_a", "") == "" or r.get("sev_b", "") == "":
                continue
            t = truth[r["item"]]
            s = {t["A_label"]: int(r["sev_a"]), t["B_label"]: int(r["sev_b"])}
            gaps[t["ckpt"]][t["spk"]] = s[t["ckpt"]] - s["rt"]
            sev[t["ckpt"]].append(s[t["ckpt"]])
            sev["rt"].append(s["rt"])
    rated = sum(len(v) for v in gaps.values())
    if rated < len(truth):
        print("⚠ %d of %d items not fully rated; they are left out.\n"
              % (len(truth) - rated, len(truth)))

    print("SEVERITY (0 = cannot hear it, 5 = Freak-a-Zoid)")
    for c in ["rt"] + ORDER:
        if sev[c]:
            print("  %-7s n=%2d  mean %.2f  %s" % (c, len(sev[c]), st.mean(sev[c]),
                                               "".join(str(x) for x in sorted(sev[c]))))

    r = lb.reading(gaps, ORDER)
    print("\nGAP PER CHECKPOINT (model - round trip; for scale, baseline bench: stock +0.75, "
          "vat7 +2.50)")
    for c in ORDER:
        m, p = r["level"][c]
        print("  %-7s mean %+.2f  p = %.4f" % (c, m, p))
    print("\nSTEP CHANGES (voice by voice; + = more hum after the step)")
    for a, b, m, p, n in r["steps"] + [r["span"]]:
        print("  %-7s -> %-7s n=%d  mean %+.2f  p = %.4f" % (a, b, n, m, p))

    print("\nOUTCOME: %s%s — %s" % (
        r["outcome"].upper(),
        (" " + ", ".join("%s->%s" % s for s in r["rises"])) if r["rises"] else "",
        READING[r["outcome"]]))

    spks = sorted({t["spk"] for t in truth.values()})
    hnr = {t["spk"]: t["hnr"] for t in truth.values()}
    print("\nvoice  HNR   " + "  ".join("%-6s" % c for c in ORDER))
    for s in spks:
        print("%5d %5.2f  %s" % (s, hnr[s], "  ".join(
            "%+-6d" % gaps[c][s] if s in gaps[c] else "  .   " for c in ORDER)))


if __name__ == "__main__":
    main()
