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
# Four step tests, UNCORRECTED: a named step is where to look next, not a proof.
READING = {
    "step": ("Hum rose at the named step(s) — uncorrected, so where to look next. Exposure "
             "to these voices also falls along the chain (see the pre-registration)."),
    "present_at_first": ("derisk-energy already hums and no later rise is shown. The step "
                         "from stock is outside this bench, and stock itself hums (+0.75), "
                         "so this does not say the first step caused it."),
    "gradual": "No single step is shown, but it rose from first to last: a creep.",
    "inconclusive": "No step and no first-to-last rise is shown. Not shown is not absent.",
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

    if sorted({t["ckpt"] for t in truth.values()}) != sorted(ORDER):
        raise SystemExit("REFUSING: the key's checkpoints are not %s." % ORDER)
    gaps = {c: {} for c in ORDER}
    sev = {c: [] for c in ORDER + ["rt"]}
    floor = {c: [] for c in ORDER}
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
            floor[t["ckpt"]].append(s["rt"])
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
        fl = ("  (its round trips %.2f)" % st.mean(floor[c])) if floor[c] else ""
        print("  %-7s mean %+.2f  p = %.4f%s" % (c, m, p, fl))
    print("\nSTEP CHANGES (voice by voice; + = more hum after the step; uncorrected)")
    for a, b, m, p, n in r["steps"] + [r["span"]]:
        d = [gaps[b][s] - gaps[a][s] for s in sorted(set(gaps[a]) & set(gaps[b]))]
        print("  %-7s -> %-7s n=%d  mean %+.2f  p = %.4f  %d up / %d down / %d same"
              % (a, b, n, m, p, sum(x > 0 for x in d), sum(x < 0 for x in d),
                 sum(x == 0 for x in d)))

    print("\nOUTCOME: %s%s — %s" % (
        r["outcome"].upper(),
        (" " + ", ".join("%s->%s" % s for s in r["rises"])) if r["rises"] else "",
        READING[r["outcome"]]))
    if r["falls"]:
        print("  also: hum FELL at %s" % ", ".join("%s->%s" % s for s in r["falls"]))
    if r["present_at_first"] and r["outcome"] != "present_at_first":
        print("  also: derisk-energy already hums (gap > 0, p < 0.05)")

    spks = sorted({t["spk"] for t in truth.values()})
    hnr = {t["spk"]: t["hnr"] for t in truth.values()}
    print("\nvoice  HNR   " + "  ".join("%-6s" % c for c in ORDER))
    for s in spks:
        print("%5d %5.2f  %s" % (s, hnr[s], "  ".join(
            "%+-6d" % gaps[c][s] if s in gaps[c] else "  .   " for c in ORDER)))


if __name__ == "__main__":
    main()
