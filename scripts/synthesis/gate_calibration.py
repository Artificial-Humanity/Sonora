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

--sweep: SETTING A THRESHOLD, RATHER THAN GUESSING ONE (C-M4)
------------------------------------------------------------
The confusion table above judges a gate that already exists. `--sweep` is for the step
before: a measure has been taken, no threshold has been chosen, and the choice has to come
from the ear rather than from what looks tidy. It is the procedure that produced
PAUSE_HARD_MAX — sweep candidate thresholds against clips the owner has already rated, and
read off two numbers at each one:

    CATCH      of the clips the ear dropped FOR THIS REASON, how many would be flagged
    FALSE FLAG of the clips the ear KEPT, how many would be flagged anyway

A threshold is a trade between those, and the trade is not symmetric: a false flag costs an
audition slot, a miss puts a broken clip in the corpus. Which is why the pause gate ended
up with two levels — hard where no good clip reaches, advisory where the ear decides.

"Dropped for this reason" comes from the owner's own note, matched by `--drop-note`. That
is the weak link and it is deliberate: the notes are free text, so the match is printed
along with the clips it selected, for the owner to sanity-check before believing the table.

Usage:
    .venv/bin/python scripts/synthesis/gate_calibration.py --campaign-dir /data/.../revisit-v1 \
        [--verdicts qc_verdicts.per_engine.jsonl] [--keep-score 4]

    # what threshold should head_lost_frac have?
    .venv/bin/python scripts/synthesis/gate_calibration.py --campaign-dir /data/.../librivox-v2 \
        --sweep head_lost_frac --drop-note 'cut off|truncat|starts? (late|mid)|missing'

    # ...and does the owner's 4 s floor survive the change of instrument?
    .venv/bin/python scripts/synthesis/gate_calibration.py --campaign-dir /data/.../librivox-v2 \
        --sweep speech_dur_vad --direction below --drop-note 'short|clipped|too brief'
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_common  # noqa: E402

RATINGS = "/data/model-training/datasets/sonora-expressive-registers/ratings.csv"

# Candidate thresholds per measure. Chosen to bracket the plausible range at a resolution
# the ear evidence can actually distinguish — a finer grid reads as precision the sample
# size does not support.
SWEEP_GRID = {
    "head_lost_frac": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    "tail_lost_frac": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    "head_words_lost": [1, 2, 3, 4, 5, 8, 12],
    "speech_dur_vad": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
    "speech_dur": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
    "worst_pause": [1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0],
    "asr_wer": [0.10, 0.20, 0.30, 0.35, 0.50],
}
# Which side of the threshold is the DEFECT. A fraction lost is bad when high; a duration
# is bad when low. Getting this backwards inverts the whole table, so it is named per
# measure rather than inferred.
SWEEP_DIRECTION = {"speech_dur_vad": "below", "speech_dur": "below"}


DROP_MARK = "x"          # the audition app's reject marker; `pick_audit_subset` reads it too


def parse_score(v):
    """Ear score 1-5, **0 for the drop marker `x`**, None when there is no verdict at all.

    `x` returned None until 2026-08-08, and every caller then did `if score is None:
    continue` — so **the ear's own rejections were discarded by the tool whose entire job
    is to calibrate gates against ear verdicts.** Measured when it was found: 253 of the
    286 dropped/reroll rows on record carry `x`, so **88% of every rejection was
    invisible**, and a sweep over a campaign full of drops reported "no rated clip matches
    the defect" — which reads as "this campaign is clean".

    It fails by doing nothing, and the number it produces looks like an answer. Both
    thresholds this tool has ever set (`PAUSE_HARD_MAX`, `TAIL_LOST_MAX`) were chosen
    against a defect set missing most of the drops; they are not thereby wrong, but the
    evidence behind them is thinner than their comments claim and both deserve a re-sweep
    now that the rejections are visible.

    0 rather than None because the downstream test is `score < keep_score`: a drop has to
    compare as worse than any keep, and None cannot.
    """
    s = (v or "").strip().lower()
    if s == DROP_MARK:
        return 0
    m = re.search(r"[1-5]", s)
    return int(m.group()) if m else None


def flags(value, threshold, direction):
    """Would this measure be flagged at this threshold? None when it was not measured."""
    if value is None:
        return None
    return value < threshold if direction == "below" else value > threshold


def load_measures(campaign_dir, filename="qc_measures.jsonl"):
    """clip id -> the measures row. Last record wins, as everywhere else.

    Edge-loss is BACKFILLED for rows written before qc_gate recorded it. Every one of
    those rows already carries `text` and `asr_hyp`, so the measure is exactly
    recoverable — which matters, because it means a threshold can be calibrated against
    eleven campaigns of existing evidence instead of waiting for the gate to be re-run
    over every wav on disk.
    """
    path = os.path.join(campaign_dir, filename)
    out, backfilled = {}, 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("id"):
            continue
        if r.get("head_lost_frac") is None and r.get("text") and r.get("asr_hyp"):
            e = synth_common.edge_loss(r["text"], r["asr_hyp"])
            r["head_lost_frac"], r["head_words_lost"] = e["head_frac"], e["head_words"]
            backfilled += 1
        out[r["id"]] = r
    if backfilled:
        print(f"note: edge-loss recomputed for {backfilled} row(s) predating the "
              f"measure, from their stored text + asr_hyp")
    return out


def load_ear(ratings_path, ids):
    """id -> (score, status, note) for clips a HUMAN actually rated.

    Same exclusions as the confusion table: a machine-folded row carries no ear verdict,
    and a dropped row's score is the one it held BEFORE the drop.
    """
    out = {}
    with open(ratings_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["id"] not in ids:
                continue
            note = row.get("note") or ""
            if "folded: staged unheard" in note:
                continue
            score = parse_score(row.get("score"))
            if score is None:
                continue
            out[row["id"]] = (score, (row.get("status") or "").strip(), note)
    return out


def cmd_sweep(args):
    """Print catch vs false-flag at each candidate threshold, from the ear's own labels."""
    measures = {}
    for d in args.campaign_dir:
        measures.update(load_measures(d, args.measures))
    ear = load_ear(args.ratings, set(measures))
    direction = args.direction or SWEEP_DIRECTION.get(args.sweep, "above")
    grid = ([float(x) for x in args.grid.split(",")] if args.grid
            else SWEEP_GRID.get(args.sweep))
    if not grid:
        raise SystemExit(f"no candidate grid for {args.sweep!r} — pass --grid a,b,c")

    note_re = re.compile(args.drop_note, re.I) if args.drop_note else None
    # The DEFECT set: clips the ear rejected AND whose note names this failure. Without
    # the note match every unrelated rejection counts as a miss and the sweep reads as
    # though no threshold works.
    defects, keeps, unmeasured = [], [], 0
    for cid, (score, status, note) in ear.items():
        value = measures[cid].get(args.sweep)
        if value is None:
            unmeasured += 1
            continue
        rejected = status in ("dropped", "reroll") or score < args.keep_score
        if rejected and (note_re.search(note) if note_re else True):
            defects.append((cid, value, note.strip()))
        elif not rejected:
            keeps.append((cid, value))

    print("campaign : " + (args.campaign_dir[0] if len(args.campaign_dir) == 1
                            else f"{len(args.campaign_dir)} pooled: "
                                 + ", ".join(os.path.basename(d.rstrip("/"))
                                             for d in args.campaign_dir)))
    print(f"measure  : {args.sweep}  (flagged when {direction} the threshold)")
    print(f"rated    : {len(ear)} clip(s) with an ear verdict"
          + (f"; {unmeasured} lack this measure" if unmeasured else ""))
    if note_re:
        print(f"defects  : {len(defects)} ear-rejected AND note matches "
              f"/{args.drop_note}/")
    else:
        print(f"defects  : {len(defects)} ear-rejected (NO --drop-note: every rejection "
              f"counts, so CATCH is diluted by unrelated ones)")
    print(f"keeps    : {len(keeps)} ear-kept\n")
    if not defects:
        print("Nothing to calibrate against: no rated clip matches the defect. Either")
        print("this campaign has none, or --drop-note does not match how they were")
        print("described. The notes are free text — read a few before trusting a zero.")
        return
    if len(defects) < 10:
        print(f"⚠ {len(defects)} labelled defect(s) is thin. PAUSE_HARD_MAX was set")
        print("  against 21, and called 95% its practical ceiling on that basis.\n")

    print(f"  {'threshold':>10}   {'catch':>14}   {'false flag':>16}")
    for t in grid:
        c = sum(1 for _, v, _ in defects if flags(v, t, direction))
        f = sum(1 for _, v in keeps if flags(v, t, direction))
        print(f"  {t:>10}   {c:>4}/{len(defects):<4} {c/len(defects):>6.0%}   "
              f"{f:>4}/{len(keeps):<4} {f/max(len(keeps),1):>6.0%}")

    print(f"\nthe {min(len(defects), 12)} labelled defect(s), so the note match can be "
          f"checked rather than trusted:")
    for cid, v, note in sorted(defects, key=lambda t: -t[1])[:12]:
        print(f"  {args.sweep}={v:<8.3f} {cid}")
        print(f"      note: {note[:110]}")
    print("\nNothing is written. Choose the threshold, then set it in qc_gate.py with the")
    print("numbers that chose it — a constant without its evidence is a guess by 2027.")


def main():
    ap = argparse.ArgumentParser()
    # REPEATABLE since 2026-08-08. One campaign at a time structurally guarantees a thin
    # defect set — a gate's evidence is however many clips one bank happened to break —
    # and thin is exactly the complaint the sweep prints about itself. PAUSE_HARD_MAX was
    # chosen this way. Pass it more than once to pool.
    ap.add_argument("--campaign-dir", required=True, action="append",
                    help="campaign directory; repeat to pool evidence across campaigns")
    ap.add_argument("--verdicts", default="qc_verdicts.jsonl",
                    help="filename within --campaign-dir")
    ap.add_argument("--ratings", default=RATINGS)
    ap.add_argument("--keep-score", type=int, default=4,
                    help="ear keep threshold (>= this score)")
    ap.add_argument("--sweep", metavar="MEASURE",
                    help="calibrate a threshold for this qc_measures field instead of "
                         f"scoring an existing gate. Known: {', '.join(sorted(SWEEP_GRID))}")
    ap.add_argument("--measures", default="qc_measures.jsonl",
                    help="filename within --campaign-dir, for --sweep")
    ap.add_argument("--grid", help="comma-separated candidate thresholds, overriding "
                                   "the built-in grid for this measure")
    ap.add_argument("--direction", choices=("above", "below"),
                    help="which side of the threshold is the DEFECT (default: per measure)")
    ap.add_argument("--drop-note", metavar="REGEX",
                    help="restrict the defect set to ear-rejected clips whose note "
                         "matches. Without it every rejection counts, including ones "
                         "this measure has no view of, and CATCH reads far too low.")
    args = ap.parse_args()

    if args.sweep:
        return cmd_sweep(args)

    vpath = os.path.join(args.campaign_dir[0], args.verdicts)
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
    print("campaign : " + (args.campaign_dir[0] if len(args.campaign_dir) == 1
                            else f"{len(args.campaign_dir)} pooled: "
                                 + ", ".join(os.path.basename(d.rstrip("/"))
                                             for d in args.campaign_dir)))
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
