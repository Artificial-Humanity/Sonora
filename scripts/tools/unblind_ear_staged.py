"""Unblind one staged-rebuild check and apply its pre-registered reading.

Reads the key written by `render_ear_staged.py render` and the app's verdicts, and prints
each arm's gap (model minus round trip) with its floor, the check's contrasts, the number
of tests (uncorrected), and the outcome from `staged_bench.reading`. Exact; no randomness.

    .venv/bin/python scripts/tools/unblind_ear_staged.py --check 1 \\
        --key /data/model-training/sonora/.keystage/staged_check1.key.json \\
        --test /data/model-training/sonora/eartest/staged_check1
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import staged_bench as sb                            # noqa: E402

OUTCOME = {
    (1, "go"): ("R hums more than stock after 10 epochs: the recipe's extra hum is reachable "
                "at this budget. GO: train C0, S1 and S2, then checks 2 and 3."),
    (1, "stop"): ("R over stock is not shown. STOP: the extra hum needs more than 10 epochs to "
                  "appear on the derisk recipe. Nothing else trains; the owner decides whether "
                  "to repeat at a longer budget. Not shown is not absent."),
    (1, "invalid"): ("derisk over stock is not shown, or the round-trip floors are not "
                     "comparable: no reading. The owner decides whether to re-draw voices."),
    (2, "shown"): "",
    (2, "not_shown"): ("Neither C0 nor S1 is shown to hum more than stock. Not shown is not "
                       "absent."),
    (2, "invalid"): "The round-trip floors are not comparable: no reading.",
    (3, "shown"): "",
    (3, "not_shown"): ("Neither the 24 kHz path nor the conditioning is shown to add the hum. "
                       "Not shown is not absent."),
    (3, "invalid"): ("Neither R over S1 nor S1's own excess is shown: the hum is not visible "
                     "on these voices, so no reading."),
}
SHOWN = {
    "fine_tuning": ("C0 hums more than stock: fine-tuning itself adds the hum. The cause is "
                    "the recipe, not the data."),
    "data": ("S1 hums more than stock and C0 is not shown to: the new data adds it (corpus, "
             "speakers or G2P, bundled here)."),
    "audio_path": "S2 hums more than S1: the 24 kHz path or our vocoder adds the hum.",
    "conditioning": "R hums more than S2: the VAT/FiLM conditioning adds it.",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args(argv)
    arms = sb.CHECKS[args.check]["arms"]

    key = json.loads(Path(args.key).read_text())
    if key.get("check") != args.check:
        raise SystemExit("REFUSING: the key is check %s's, not check %d's."
                         % (key.get("check"), args.check))
    truth = key["items"]
    if {t["arm"] for t in truth.values()} != set(arms):
        raise SystemExit("REFUSING: the key's arms are not %s." % (arms,))
    vpath = Path(args.test) / "verdicts" / "verdicts.csv"
    if not vpath.is_file():
        raise SystemExit("REFUSING: no %s — nothing was judged." % vpath)

    gaps = {a: {} for a in arms}
    sev = {a: [] for a in arms}
    floor = {a: [] for a in arms}
    with vpath.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["item"] not in truth:
                raise SystemExit("REFUSING: verdict for %r has no entry in the key." % r["item"])
            if r.get("sev_a", "") == "" or r.get("sev_b", "") == "":
                continue
            t = truth[r["item"]]
            a = t["arm"]
            s = {t["A_label"]: int(r["sev_a"]), t["B_label"]: int(r["sev_b"])}
            gaps[a][t["unit"]] = s[a] - s[a + "rt"]
            sev[a].append(s[a])
            floor[a].append(s[a + "rt"])
    rated = sum(len(v) for v in gaps.values())
    if rated < len(truth):
        print("⚠ %d of %d items not fully rated; they are left out.\n"
              % (len(truth) - rated, len(truth)))
    if not all(floor.values()):
        raise SystemExit("REFUSING: an arm has no rated item.")
    floors = {a: st.mean(v) for a, v in floor.items()}

    r = sb.reading(args.check, gaps, floors)
    print("CHECK %d — GAP PER ARM (model - round trip; one-sided p)" % args.check)
    for a in arms:
        m, p, n = r["level"][a]
        print("  %-6s n=%2d  model %.2f  round trip %.2f  gap %+.2f  p = %.4f"
              % (a, n, st.mean(sev[a]), floors[a], m, p))
    print("\nCONTRASTS (per unit; one-sided; uncorrected — %d tests in this check)"
          % r["n_tests"])
    for k, (m, p, n) in r["contrasts"].items():
        a, b = k.split("-")
        d = [gaps[a][u] - gaps[b][u] for u in sorted(set(gaps[a]) & set(gaps[b]))]
        print("  %-12s n=%2d  mean %+.2f  p = %.4f  %d up / %d down / %d same"
              % (k, n, m, p, sum(x > 0 for x in d), sum(x < 0 for x in d),
                 sum(x == 0 for x in d)))
    if r["floors_ok"] is False:
        print("\n⚠ round-trip floors not comparable (within 1 point of stock's, all under 4)")

    print("\nOUTCOME: %s — %s" % (r["outcome"].upper(), OUTCOME[(args.check, r["outcome"])]))
    for s in r["shown"]:
        print("  shown: %s" % SHOWN[s])

    hnr = {}
    for t in truth.values():
        hnr.setdefault(t["unit"], {})[t["arm"]] = t["hnr"]
    print("\nunit  HNR per arm          " + "  ".join("%-6s" % a for a in arms))
    for u in sorted(hnr):
        print("%4d  %-20s %s" % (u, "/".join("%.2f" % hnr[u][a] for a in arms if a in hnr[u]),
                                 "  ".join("%+-6d" % gaps[a][u] if u in gaps[a] else "  .   "
                                           for a in arms)))
    return r


if __name__ == "__main__":
    main()
