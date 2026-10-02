"""Unblind the first-step hum bench and apply its pre-registered reading.

Reads the key written by `render_ear_first_step.py render` and the app's verdicts, and
prints each family's gap (model minus round trip) with its round-trip floor, the three
matched contrasts, and the outcome from `first_step_bench.reading`. Exact tests; no
randomness.

    .venv/bin/python scripts/tools/unblind_ear_first_step.py \\
        --key /data/model-training/sonora/eartest/_keys/first_step_hum.key.json \\
        --test /data/model-training/sonora/eartest/first_step_hum
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import first_step_bench as fs                        # noqa: E402

READING = {
    "first_step": ("derisk-energy already hums more than stock, and vat7 adding more is not "
                   "shown: the excess came in at the first step (corpus, 24 kHz vocoder and "
                   "conditioning changed together). derisk saw these voices most, so if "
                   "anything this understates it."),
    "both_steps": "derisk-energy hums more than stock, and vat7 more again: both stages added.",
    "later": ("vat7 hums more than derisk-energy, and derisk-energy over stock is not shown. "
              "Either the excess came in after the first step, or derisk-energy is masked "
              "by having trained on these voices ~10x more than vat7 did."),
    "inconclusive": ("vat7's excess is reproduced, but neither step is shown to carry it. "
                     "Not shown is not absent."),
    "invalid": ("vat7 did not reproduce its excess over stock, or the round-trip floors are "
                "not comparable: no reading."),
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
    if {t["family"] for t in truth.values()} != set(fs.FAMILIES):
        raise SystemExit("REFUSING: the key's families are not %s." % (fs.FAMILIES,))

    gaps = {f: {} for f in fs.FAMILIES}
    sev = {f: [] for f in fs.FAMILIES}
    floor = {f: [] for f in fs.FAMILIES}
    with vpath.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["item"] not in truth:
                raise SystemExit("REFUSING: verdict for %r has no entry in the key."
                                 % r["item"])
            if r.get("sev_a", "") == "" or r.get("sev_b", "") == "":
                continue
            t = truth[r["item"]]
            fam = t["family"]
            s = {t["A_label"]: int(r["sev_a"]), t["B_label"]: int(r["sev_b"])}
            gaps[fam][t["match"]] = s[fam] - s[fam + "rt"]
            sev[fam].append(s[fam])
            floor[fam].append(s[fam + "rt"])
    rated = sum(len(v) for v in gaps.values())
    if rated < len(truth):
        print("⚠ %d of %d items not fully rated; they are left out.\n"
              % (len(truth) - rated, len(truth)))
    if not all(floor.values()):
        raise SystemExit("REFUSING: a family has no rated item.")
    floors = {f: st.mean(v) for f, v in floor.items()}

    r = fs.reading(gaps, floors)
    print("GAP PER FAMILY (model - round trip; baseline bench: stock +0.75, vat7 +2.50)")
    for f in fs.FAMILIES:
        m, p, n = r["level"][f]
        print("  %-6s n=%2d  model %.2f  round trip %.2f  gap %+.2f  p = %.4f"
              % (f, n, st.mean(sev[f]), floors[f], m, p))
    print("\nMATCHED CONTRASTS (per match; uncorrected)")
    for k, (m, p, n) in r["contrasts"].items():
        a, b = k.split("-")
        d = [gaps[a][x] - gaps[b][x] for x in sorted(set(gaps[a]) & set(gaps[b]))]
        print("  %-13s n=%2d  mean %+.2f  p = %.4f  %d up / %d down / %d same"
              % (k, n, m, p, sum(x > 0 for x in d), sum(x < 0 for x in d),
                 sum(x == 0 for x in d)))
    if not r["floors_ok"]:
        print("\n⚠ round-trip floors not comparable (within 1 point of stock's, all under 4)")

    print("\nOUTCOME: %s — %s" % (r["outcome"].upper(), READING[r["outcome"]]))
    if r["vat7_below_derisk"]:
        print("  also: vat7 hums LESS than derisk-energy (p < 0.05)")

    hnr = {}
    for t in truth.values():
        hnr.setdefault(t["match"], {})[t["family"]] = t["hnr"]
    print("\nmatch  HNR(vctk/libri)  " + "  ".join("%-6s" % f for f in fs.FAMILIES))
    for m in sorted(hnr):
        print("%5d  %5.2f / %5.2f    %s" % (m, hnr[m]["stock"], hnr[m]["derisk"], "  ".join(
            "%+-6d" % gaps[f][m] if m in gaps[f] else "  .   " for f in fs.FAMILIES)))


if __name__ == "__main__":
    main()
