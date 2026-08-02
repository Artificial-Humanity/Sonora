"""gate_calibration — measure the instruments against the owner's ear.

WHY THIS EXISTS. The goal is dataset pipeline 1.0: a pipeline trusted enough that
the owner SPOT-CHECKS instead of auditing every clip (`audit_sampler.py` already
implements the sampling half). Spot-checking is not a workflow change, it is a
measurement claim — it is only safe once the instruments provably catch what the
ear catches. Nobody has ever measured that.

Every full-coverage round is a free opportunity to measure it, but only if the
instrument verdicts are frozen BEFORE the owner rates. Afterwards there is nothing
to compare against and the round is spent.

Reads a campaign's frozen qc_verdicts.jsonl and the owner's ratings.csv, and
prints the confusion between them:

    instrument keep=True/False  x  ear score >= --keep-score

The cell that decides pipeline 1.0 is FALSE NEGATIVES BY THE INSTRUMENT'S
STANDARD: clips the instrument would have certified unseen that the owner
rejected. Those are what a spot-check would have missed. The opposite cell
(instrument rejects, ear keeps) costs only wasted renders, not corpus quality —
worth knowing, but not disqualifying.

Read the numbers with the gate's semantics in mind. `keep` in qc_verdict is a
LABEL-CONFIRMATION gate: it asks whether the measured affect points the same way
as the INTENDED V/A/T, not whether the clip sounds good. A beautiful clip whose
director-assigned labels were wrong is a legitimate not-keep. So a large
instrument-rejects/ear-keeps cell may indict the DIRECTOR's labels rather than
the engine or the instrument — check which before concluding anything.

Usage:
    .venv/bin/python scripts/synthesis/gate_calibration.py --campaign-dir /data/.../revisit-v1 \
        [--verdicts qc_verdicts.per_engine.jsonl] [--keep-score 4]
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

RATINGS = "/data/model-training/datasets/sonora-expressive-registers/ratings.csv"


def parse_score(v):
    m = re.search(r"[1-5]", (v or "").strip())
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--verdicts", default="qc_verdicts.jsonl",
                    help="filename within --campaign-dir")
    ap.add_argument("--ratings", default=RATINGS)
    ap.add_argument("--keep-score", type=int, default=4,
                    help="ear keep threshold (>= this score)")
    args = ap.parse_args()

    vpath = os.path.join(args.campaign_dir, args.verdicts)
    verdicts = {r["id"]: r for r in
                (json.loads(l) for l in open(vpath, encoding="utf-8") if l.strip())}

    # Only genuine ear verdicts calibrate the gate. Two kinds of row used to
    # slip in: machine-folded rows carrying a fabricated score, and rows the
    # owner DROPPED that still hold the score they were given before the drop
    # (38 of them in delivery-v1-narration). Both were counted as ear keeps, so
    # the false-negative rate that gates pipeline 1.0 was computed over
    # polluted cells.
    ear, skipped = {}, {"folded": 0, "dropped": 0}
    with open(args.ratings, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = parse_score(row.get("score"))
            if s is None or row["id"] not in verdicts:
                continue
            if "folded: staged unheard" in (row.get("note") or ""):
                skipped["folded"] += 1
                continue
            if (row.get("status") or "") in ("dropped", "reroll"):
                skipped["dropped"] += 1
                continue
            ear[row["id"]] = s
    if any(skipped.values()):
        print(f"excluded from ear evidence: {skipped['folded']} machine-folded, "
              f"{skipped['dropped']} dropped/reroll rows still carrying a score")

    rated = [i for i in verdicts if i in ear]
    print(f"campaign : {args.campaign_dir}")
    print(f"verdicts : {args.verdicts} ({len(verdicts)} clips)")
    print(f"rated    : {len(rated)} of {len(verdicts)}"
          f"  ({len(verdicts) - len(rated)} awaiting the ear)\n")
    if not rated:
        print("Nothing to compare yet — the instrument verdicts are frozen and waiting.")
        print("Re-run this after the audit round to get the calibration numbers.")
        return

    cell = defaultdict(list)
    for i in rated:
        cell[(bool(verdicts[i].get("keep")), ear[i] >= args.keep_score)].append(i)

    tp, fn = len(cell[(True, True)]), len(cell[(True, False)])
    fp, tn = len(cell[(False, True)]), len(cell[(False, False)])
    print(f"                     ear keep (>={args.keep_score})   ear reject")
    print(f"  instrument keep    {tp:>14}   {fn:>11}")
    print(f"  instrument reject  {fp:>14}   {tn:>11}\n")

    certified = tp + fn
    if certified:
        print(f"FALSE NEGATIVES — certified by instrument, rejected by ear: "
              f"{fn}/{certified} = {fn/certified:.1%}")
        print("  ^ this is the spot-check risk: clips that would have shipped unheard")
        for i in cell[(True, False)][:10]:
            print(f"    {i}  ear={ear[i]}")
    if fp:
        print(f"\nover-rejection — instrument rejected, ear kept: {fp}"
              f" ({fp/len(rated):.0%} of rated). Costs renders, not quality;"
              f" check whether the DIRECTOR's labels were wrong before blaming the gate.")
    agree = (tp + tn) / len(rated)
    print(f"\noverall agreement: {agree:.1%}")


if __name__ == "__main__":
    main()
