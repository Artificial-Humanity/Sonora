#!/usr/bin/env python3
"""Regenerate every A-channel number quoted in `notes/direction-contract-v3-proposal.md` § 3b.

WHY THIS EXISTS (2026-08-11, issue #34)
---------------------------------------
§ 3b's resolution originally shipped ~15 load-bearing numbers with **no committed
derivation**. The `bb76085` diagnostic that produced the first half was stdout-only and was
lost when an OOM crash-loop killed the terminal; the ep010 delivery probe that produced the
second half existed only as 90 wav files and a `measures.csv` under `/data`. The note said
the numbers had been "put in the tree so neither exists solely as run output again", which
was true of the *numbers* and false of the *derivation* — so the next time one is questioned
the recovery cost is what it was the first time: re-derive from scratch and hope the method
matches.

This script is that method, written down. It is **strictly read-only** — it opens two
artifacts, prints, and writes nothing.

WHAT IT DELIBERATELY DOES NOT DO. It is not a gate. It asserts nothing and exits 0 whenever
it can read its inputs, because the artifacts live under `/data` and a clean checkout does
not have them; a gate that fails on a missing mount is a gate someone switches off.
`scripts/test_doc_claims.py` is the gate, it owns the corpus/checkpoint registry, and it is
deliberately **not** touched here.

TWO SEPARATE POOLINGS, AND THE REASON THEY ARE BOTH RIGHT
---------------------------------------------------------
The probe's design is 2 texts x 5 lanes x A in {-1, 0, +1} x 3 repeats. That admits two
different "cells", and § 3b needs both:

* **(text, lane, A)** — 30 cells of n = 3. This is the *noise floor*: the only variation
  inside such a cell is the sampler's, with text held fixed. Pooled sd 0.7616 dB, df = 60.
  ⚠ The note first quoted **0.554 dB** here and called it "pooled". It is not pooled — it is
  the **mean of the 30 per-cell population sds** (ddof = 0), which is biased low twice over:
  once by ddof = 0 on n = 3, and once because a mean of sds is not a pooled sd. The number is
  printed below so the arithmetic of the old claim stays checkable.

* **(lane, A)** — 15 cells of n = 6, texts pooled. This is the right denominator for any
  statement *comparing cell means*, because the quantities being compared (a lane's mean at
  A = 0, a lane's response between A = -1 and A = +1) are themselves averaged over both
  texts. Pooled sd 0.8457 dB, df = 75; se of a difference of two 6-render means 0.4883 dB.

The second is the larger of the two because it absorbs the text effect, which is real and
which any between-cell comparison has to carry.

SLOPE IS NOT INTERCEPT — THE DISTINCTION THE WHOLE SECTION TURNS ON
-------------------------------------------------------------------
The three-reference-frames concern (issue #14) is about **intercept**: whether A = 0 denotes
the same absolute loudness in two lanes whose training rows were z-scored against different
reference populations. The probe's headline measurement is a **slope**: dB of output movement
per unit A, measured *inside* each lane. A uniform slope says the dial is wired. It cannot
say where the dial's zero sits. Both are printed below, separately and labelled, because
conflating them is exactly what issue #33 was filed about.

Run:  .venv/bin/python scripts/derive_a_channel_stats.py
Exit: 0 always when the inputs are readable; 2 if an input is missing.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

LABELS = "/data/model-training/sonora/expressive_registers_measures/labels_v6.jsonl"
MANIFEST = "/data/model-training/sonora/expressive_registers_measures/labels_v6_manifest.json"
PROBE = "/data/model-training/sonora/probes/delivery_ep010/measures.csv"

# The two loudness values `label_expressive_registers.py:44` names as declared targets, and
# the range it gives for the third. The third is written as a RANGE in the source because it
# was never a target — see `loudness_targets()`.
T_23, T_20 = -23.0, -20.4
BAND_LO, BAND_HI = -27.1, -26.3
TOL = 0.05        # "at the target" — loudnorm lands within a few hundredths
BAND_PAD = 0.5    # generosity granted to the third band, since it is a range not a value


def pooled_sd(cells):
    """Pooled within-cell sd and its df. The textbook definition, ddof = 1 per cell."""
    num = sum((len(v) - 1) * float(np.var(v, ddof=1)) for v in cells.values() if len(v) > 1)
    den = sum(len(v) - 1 for v in cells.values() if len(v) > 1)
    return float(np.sqrt(num / den)), den


def h(title):
    print(f"\n{title}\n{'-' * len(title)}")


# --- part 1: the corpus side, from labels_v6.jsonl ----------------------------------


def corpus(rows, manifest):
    h("1. WHAT PER-CAMPAIGN CENTRING DID TO BETWEEN-LANE LOUDNESS STRUCTURE")
    print(f"rows {len(rows)}   campaigns {len({r['campaign'] for r in rows})}")

    def lane_mean_sd(field):
        m = collections.defaultdict(list)
        for r in rows:
            m[r["delivery"]].append(r[field])
        lm = {k: float(np.mean(v)) for k, v in m.items()}
        return lm, float(np.std(list(lm.values()), ddof=0))

    before, sd_b = lane_mean_sd("lufs_native")
    after, sd_a = lane_mean_sd("lufs_adjusted")
    print("\n  lane-mean LUFS (blank lane is the un-laned remainder):")
    for lane in sorted(before, key=lambda k: str(k)):
        n = sum(1 for r in rows if r["delivery"] == lane)
        print(f"    {str(lane) or '(blank)':12s} n={n:4d}  native {before[lane]:8.3f}"
              f"   adjusted {after[lane]:8.3f}")
    print(f"\n  sd of lane-mean LUFS, before centring   {sd_b:.4f} dB")
    print(f"  sd of lane-mean LUFS, after  centring   {sd_a:.4f} dB")
    if sd_b == 0.0:
        # A DENOMINATOR WITH NO GUARD KILLED EVERY SECTION, not just this one: `corpus()` is
        # the first thing `main()` calls, so a bare ZeroDivisionError here took §§ 2-8 with
        # it. sd_b is 0 whenever the artifact carries one lane, or several whose native
        # means coincide — an ordinary re-run on a single-lane bank, which is precisely the
        # case this script's docstring says it exists to survive.
        print("  between-lane structure surviving        UNDEFINED — the lane means do not "
              "vary\n    before centring (sd = 0), so there is no between-lane structure "
              "for centring to\n    remove. This is the expected reading on a single-lane "
              "bank, not a failure.")
    else:
        print(f"  between-lane structure surviving        {100 * sd_a / sd_b:.1f}%"
              f"   (removed {100 * (1 - sd_a / sd_b):.1f}%)")

    h("2. CAMPAIGN IS A NEAR-PERFECT PROXY FOR LANE — BUT ONLY ONE PAIR IS ONE-TO-ONE")
    by_camp = collections.defaultdict(collections.Counter)
    by_lane = collections.defaultdict(collections.Counter)
    for r in rows:
        by_camp[r["campaign"]][r["delivery"]] += 1
        by_lane[r["delivery"]][r["campaign"]] += 1
    n10 = {k: v for k, v in by_camp.items() if sum(v.values()) >= 10}
    # ⚠ THE UN-LANED REMAINDER IS EXCLUDED HERE TOO, and this is the number § 3b quotes
    # ("13 of the 20 campaigns with n >= 10 are >=90% one delivery lane"). The `both` count
    # further down was fixed first and this one was left, four lines above the comment
    # explaining why it is wrong — so a campaign that is 100% un-laned still counted as
    # ">= 90% one lane". `max()` over the named lanes only, and a campaign with no named
    # rows at all cannot qualify.
    def top_named(counter):
        named = [n for lane, n in counter.items() if lane]
        return max(named) if named else 0

    ge90 = [k for k, v in n10.items() if top_named(v) / sum(v.values()) >= 0.90]
    print(f"  campaigns with n >= 10: {len(n10)};  of those, >= 90% one NAMED lane: "
          f"{len(ge90)}   (un-laned rows never qualify)")

    print("\n  per lane: how concentrated is it in its top campaign?")
    for lane, c in sorted(by_lane.items(), key=lambda kv: str(kv[0])):
        tot = sum(c.values())
        camp, n = c.most_common(1)[0]
        camp_tot = sum(by_camp[camp].values())
        print(f"    {str(lane) or '(blank)':12s} n={tot:4d} from {len(c):2d} campaign(s); "
              f"top `{camp}` = {n}/{tot} ({100 * n / tot:.1f}% of the lane) and that "
              f"campaign is {n}/{camp_tot} ({100 * n / camp_tot:.1f}%) this lane")
    # COUNTED, not asserted. This line used to name Newscaster and the number 1 in prose,
    # in the file whose entire purpose is that § 3b's numbers regenerate. Same class of
    # defect as the § 8 conclusion below: a sentence that cannot be re-derived sits beside
    # numbers that can, and it is the sentence that gets quoted.
    # ⚠ THE UN-LANED REMAINDER IS EXCLUDED, DELIBERATELY, AND SAYS SO. The blank lane is a
    # key of `by_lane` like any other, so it was silently counted here and then rendered as
    # an empty string in the parenthetical — `(, Newscaster)`, which is the line that gets
    # pasted into § 3b. This section's claim is "campaign is a proxy for LANE", and a
    # campaign that is 100% un-laned says nothing about lane structure: it is the absence of
    # a lane, not a fifth one. The `(blank)` row two blocks up still shows its concentration,
    # so excluding it here hides nothing.
    both = [lane for lane, c in by_lane.items()
            if lane and c and c.most_common(1)[0][1] == sum(c.values())
            and c.most_common(1)[0][1] == sum(by_camp[c.most_common(1)[0][0]].values())]
    print(f"\n  ⚠ {len(both)} lane/campaign pair(s) are 100% in BOTH directions"
          + (f" ({', '.join(sorted(map(str, both)))})." if both else ".")
          + "  (the un-laned remainder is not a lane for this claim and is excluded)")
    sp = by_lane.get("Speech", collections.Counter())
    if sp:
        print("    Speech draws from " + ", ".join(f"`{k}` {v}" for k, v in sp.most_common())
              + f" = {sum(sp.values())}; its top campaign is "
              f"{100 * sp.most_common(1)[0][1] / sum(sp.values()):.1f}% of the lane.")

    h("3. NEWSCASTER'S A IS A CONSTANT — AND THE VARIANCE WAS NEVER THERE TO REMOVE")
    # ⚠ THIS SECTION NAMES A SPECIFIC LANE AND CAMPAIGN, so it is the one most likely to
    # find nothing on a re-run. `np.mean([])` is a RuntimeWarning and a nan, `np.std(ddof=1)`
    # over one row is another, and `min([])` is a ValueError — a section reporting `nan` as
    # a measured constant is worse than one that says it had nothing to measure. Guarding
    # § 1's `sd_b` alone would only have MOVED the single-lane failure here, which is what
    # running the case showed: §§ 1 and 2 completed and § 3 died three lines in.
    a = [r["a"] for r in rows if r["delivery"] == "Newscaster"]
    if len(a) < 2:
        print(f"  Newscaster A       n={len(a)} — fewer than 2 rows, so there is no mean, "
              f"sd or range to report.\n    Not a finding: this artifact simply does not "
              f"carry the lane this section is about.")
    else:
        print(f"  Newscaster A       n={len(a)} mean {np.mean(a):+.4f}  "
              f"sd {np.std(a, ddof=1):.4f}  range [{min(a):+.3f}, {max(a):+.3f}]")
    nv = [r["lufs_native"] for r in rows if r["campaign"] == "newscaster-v1"]
    if len(nv) < 2:
        print(f"  newscaster-v1 native LUFS  n={len(nv)} — no `newscaster-v1` campaign to "
              f"read a loudness target off.")
    else:
        print(f"  newscaster-v1 native LUFS  n={len(nv)} mean {np.mean(nv):.4f}"
              f"  sd {np.std(nv, ddof=1):.4f} dB   <- the campaign was rendered to one "
              f"target")

    h("4. FOUR CENTRES, NOT THREE — THE MIN_CAMPAIGN_N FALLBACK")
    offs = {round(r["lufs_offset"], 9) for r in rows}
    small = {k: sum(v.values()) for k, v in by_camp.items() if sum(v.values()) < 10}
    print(f"  distinct lufs_offset values in the file: {len(offs)}")
    print(f"  campaigns below MIN_CAMPAIGN_N={manifest['min_campaign_n']}: {len(small)}, "
          f"holding {sum(small.values())} rows, all on the bank-wide offset "
          f"{manifest['bank_offset_db']:.4f} dB")
    own = {round(r["lufs_offset"], 9) for r in rows if r["campaign"] not in small}
    print(f"  => {len(own)} campaign offsets + 1 bank offset = {len(own) + 1}"
          f"  (file carries {len(offs)})")
    anc = manifest["anchor_lufs"]
    print(f"\n  the anchor supplies the SCALE, un-rescaled: libritts_anchor over "
          f"{manifest['anchor_n_clips']:,} v4 clips, LUFS mean {anc['mean']:.3f} / "
          f"sd {anc['sd']:.3f}")


def loudness_targets(rows):
    """The chain set, derived — issue #36.

    `label_expressive_registers.py:44` hedges: "This bank holds **at least THREE** loudness
    targets". § 3b closed that to "three known targets". The hedge is the correct reading and
    the data is blunter than either: **one** of the three is a normalisation target in any
    useful sense. The other two are descriptions of un-normalised material, and a sixth of
    the bank matches none of them.
    """
    h("5. THE LOUDNESS-TARGET SET, DERIVED FROM MEASURED LUFS (issue #36)")
    nat = np.array([r["lufs_native"] for r in rows])
    n = len(nat)
    m23 = np.abs(nat - T_23) <= TOL
    m20 = np.abs(nat - T_20) <= TOL
    band = (nat >= BAND_LO - BAND_PAD) & (nat <= BAND_HI + BAND_PAD)
    tight = (nat >= BAND_LO) & (nat <= BAND_HI)
    print(f"  n = {n};  native LUFS min {nat.min():.3f}  median {np.median(nat):.3f}"
          f"  max {nat.max():.3f}")
    print(f"\n  within {TOL} dB of {T_23:5.1f}      {m23.sum():4d} ({100 * m23.sum() / n:5.1f}%)"
          "   <- a real target: loudnorm to -23 LUFS")
    print(f"  within {TOL} dB of {T_20:5.1f}      {m20.sum():4d} ({100 * m20.sum() / n:5.1f}%)"
          "   <- NOT a target; two rows is a coincidence")
    print(f"  inside [{BAND_LO}, {BAND_HI}]     {tight.sum():4d} "
          f"({100 * tight.sum() / n:5.1f}%)   <- a range, not a value")
    print(f"  ...widened by +-{BAND_PAD} dB      {band.sum():4d} "
          f"({100 * band.sum() / n:5.1f}%)")
    outside = ~(m23 | m20 | band)
    print(f"\n  OUTSIDE all three, on the generous reading: {outside.sum()} "
          f"({100 * outside.sum() / n:.1f}%)")
    print(f"  outside on the literal reading (no widening): {(~(m23 | m20 | tight)).sum()} "
          f"({100 * (~(m23 | m20 | tight)).sum() / n:.1f}%)")

    print("\n  per campaign. THE TEST IS MODAL, NOT DISPERSION-BASED: a declared target means")
    print("  most of the campaign sits ON it. sd alone mislabels `delivery-v1-narration-r2`,")
    print("  which is loudnorm'd to -23 (75% of rows within 0.05 dB) but carries a long tail")
    print("  that pushes its sd to 1.9. The two populations separate cleanly — no campaign")
    print("  lands between 33% and 73%.")
    cl = collections.defaultdict(list)
    for r in rows:
        cl[r["campaign"]].append(r["lufs_native"])
    buckets, n_rows = collections.defaultdict(list), len(rows)
    print(f"\n    {'campaign':34s} {'n':>4s} {'median':>9s} {'sd':>8s} {'@med':>6s}  verdict")
    for k in sorted(cl, key=lambda k: -float(np.median(cl[k]))):
        v = np.array(cl[k])
        med = float(np.median(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        at = float(np.mean(np.abs(v - med) <= TOL))
        if len(v) == 1:
            verdict = "n = 1, undecidable"
        elif at >= 0.5:
            verdict = f"normalised to {med:.1f} LUFS"
        else:
            verdict = "UN-NORMALISED — LUFS is measured, not declared"
        print(f"    {k:34s} {len(v):4d} {med:9.3f} {sd:8.3f} {100 * at:5.1f}%  {verdict}")
        buckets[verdict.split(" —")[0].split(", ")[0]].append((k, len(v)))

    print("\n  THE DERIVED CHAIN SET:")
    for label in sorted(buckets):
        camps = buckets[label]
        print(f"    {label:28s} {len(camps):2d} campaign(s), "
              f"{sum(n for _, n in camps):4d} rows "
              f"({100 * sum(n for _, n in camps) / n_rows:.1f}%)")
    # READ OFF `buckets`, not restated. "ONE declared target ... and 2 rows are n = 1" was
    # printed unconditionally directly beneath the block that derives both counts, so a run
    # on a different artifact contradicted itself two lines apart.
    declared = sorted(k for k in buckets if k.startswith("normalised to "))
    n1 = sum(n for _, n in buckets.get("n = 1", []))
    targets = ", ".join(k.replace("normalised to ", "") for k in declared) or "none"
    print(f"    => {len(declared)} declared loudness target(s) in this bank ({targets}).")
    print(f"       Everything else is un-normalised material whose LUFS is a measurement,")
    print(f"       and {n1} row(s) are n = 1 (undecidable either way).")


# --- part 2: the probe side, from measures.csv ---------------------------------------


def probe(rows):
    lanes = sorted({r["lane"] for r in rows}, key=lambda s: (s == "unknown", s))
    levels = sorted({r["energy"] for r in rows})

    h("6. THE PROBE'S TWO NOISE ESTIMATES (see the module docstring)")
    c30 = collections.defaultdict(list)
    c15 = collections.defaultdict(list)
    for r in rows:
        c30[(r["text"], r["lane"], r["energy"])].append(r["lufs"])
        c15[(r["lane"], r["energy"])].append(r["lufs"])
    # EVERY (lane, A) CELL THE REST OF THIS FUNCTION READS, CHECKED HERE. `means[(lane, e)]`
    # and `means[("unknown", zero)]` used to raise a bare KeyError on an absent cell — a
    # traceback where a read-only diagnostic owes a named refusal, and the same failure mode
    # `qc_verdict.build_anchors` was hardened against on this branch.
    absent = [(lane, e) for lane in lanes for e in levels if (lane, e) not in c15]
    if absent:
        sys.exit("DERIVE-FAIL: the probe is missing " + str(len(absent)) + " design cell(s): "
                 + ", ".join(f"({l}, A={e:+g})" for l, e in absent[:6])
                 + (" ..." if len(absent) > 6 else "")
                 + "\n  Sections 7 and 8 read every (lane, A) cell; there is nothing to "
                   "derive from a hole.")

    sd30, df30 = pooled_sd(c30)
    sd15, df15 = pooled_sd(c15)
    mean_of_sds = float(np.mean([np.std(v, ddof=0) for v in c30.values()]))
    # DERIVED FROM THE CELL SIZES, NOT THE LITERAL 6 THE DESIGN INTENDED. One wav that
    # failed to render, one row dropped in post, and `sqrt(2/6)` leaves every se too small
    # and every CI below too tight — silently, because `pooled_sd` already tolerates
    # unequal n and nothing else looks wrong.
    def se_of(a, b):
        return sd15 * float(np.sqrt(1 / len(c15[a]) + 1 / len(c15[b])))

    sizes = sorted({len(v) for v in c15.values()})
    balanced = len(sizes) == 1
    se = sd15 * float(np.sqrt(2 / sizes[-1]))   # the nominal, for the header line only
    print(f"  renders {len(rows)};  {len(c30)} cells of {sorted({len(v) for v in c30.values()})}"
          f" (text, lane, A);  {len(c15)} cells of {sizes} (lane, A)")
    print(f"  pooled within-cell sd, (text, lane, A)   {sd30:.4f} dB   df={df30}")
    print(f"  pooled within-cell sd, (lane, A)         {sd15:.4f} dB   df={df15}")
    if balanced:
        print(f"  se of a difference of two {sizes[0]}-render means {se:.4f} dB")
    else:
        print(f"  ⚠ THE DESIGN IS NOT BALANCED — cell sizes {sizes}. Every se below is "
              f"computed\n    per comparison from its own two cells, so they differ; the "
              f"single figure this\n    line used to print would have been wrong for most "
              f"of them.")
    print(f"\n  ⚠ mean of the {len(c30)} per-cell POPULATION sds (ddof=0) = {mean_of_sds:.4f} dB"
          " — this is the\n    number § 3b once called a 'pooled noise floor'. It is neither"
          " pooled nor a floor.")

    means = {k: float(np.mean(v)) for k, v in c15.items()}
    print("\n  cell means (LUFS):")
    print(f"    {'lane':12s}" + "".join(f"{f'A={e:+.0f}':>12s}" for e in levels))
    for lane in lanes:
        print(f"    {lane:12s}" + "".join(f"{means[(lane, e)]:12.3f}" for e in levels))

    h("7. SLOPE — dB of output movement across the dial, WITHIN each lane")
    tcrit = float(stats.t.ppf(0.975, df15))
    print(f"  A = {levels[-1]:+g} minus A = {levels[0]:+g}, per lane;"
          f"  95% CI = d +- t({df15}) x se, se per lane from its own two cells")
    slopes, slope_se = {}, {}
    for lane in lanes:
        d = means[(lane, levels[-1])] - means[(lane, levels[0])]
        s = se_of((lane, levels[-1]), (lane, levels[0]))
        slopes[lane], slope_se[lane] = d, s
        print(f"    {lane:12s} {d:+7.3f}   95% CI [{d - tcrit * s:+.2f}, {d + tcrit * s:+.2f}]"
              + ("   <- training A is a constant here" if lane == "Newscaster" else ""))
    spread = max(slopes.values()) - min(slopes.values())
    # The most conservative slope se in the set, so an unbalanced design cannot manufacture
    # a large q out of the one lane that happens to have the smallest denominator.
    se_slope = max(slope_se.values())
    se_ss = se_slope * np.sqrt(2)   # a difference of two slopes = 4 cell means
    # A STUDENTISED RANGE, NOT A t. `spread` is max-minus-min over 5 lanes — a maximum over
    # 10 pairs — and it was tested against the null distribution of ONE pre-specified pair.
    # That reference is far too narrow, so the p came out biased LOW: it overstated evidence
    # against equal gain, in the exact direction the prose below is careful about.
    #
    # ⚠ The denominator is the se of ONE SLOPE, not of a difference of two. Tukey's q is
    # (max - min) / se(group mean), and the "groups" here are lane slopes; using `se_ss`
    # would divide by an extra sqrt(2) and understate q. Same trap in the other direction.
    # `studentized_range` is undefined for k < 2 and returns `nan`. A nan p printed beside
    # "EQUAL gain is NOT established" in a script whose output is pasted into § 3b is worth
    # refusing rather than emitting — there is no range across one lane.
    if len(slopes) < 2:
        sys.exit(f"DERIVE-FAIL: the probe carries {len(slopes)} lane(s) — a range across "
                 f"lanes needs at least 2.\n  § 7's spread and § 8's between-lane "
                 f"differences are both undefined here.")
    q = spread / se_slope
    p = float(stats.studentized_range.sf(q, k=len(slopes), df=df15))
    print(f"  range across lanes: {min(slopes.values()):+.2f} to {max(slopes.values()):+.2f}"
          f"  (spread {spread:.2f} dB)")
    print(f"  se of ONE slope = {se_slope:.4f} dB;  of a DIFFERENCE of two = {se_ss:.4f} dB")
    print(f"  studentised range q = spread / se(slope) = {q:.2f}"
          f"  (k = {len(slopes)}, df = {df15})  =>  p = {p:.3f}")
    print("  ⚠ p is a RANGE statistic over all lane pairs, not a chosen pair's t. Testing")
    print("    a max-minus-min as if it were one contrast is how equal gain gets rejected")
    print("    on a spread that five noisy slopes produce on their own.")
    nonzero = [l for l, d in slopes.items() if abs(d) > tcrit * slope_se[l]]
    print(f"  ⇒ {len(nonzero)} of {len(slopes)} lane(s) have a gain distinguishable from"
          f" zero at 95%. EQUAL gain\n    is NOT established — nor excluded — at this n."
          f" ⚠ Comparing the spread against a\n    per-render sd is the wrong denominator;"
          f" these are {sizes[0] if balanced else 'multi'}-render means.")
    print("    AND ALL OF THIS IS A SLOPE. It says nothing about where A = 0 sits.")

    h("8. INTERCEPT — where A = 0 actually lands, BETWEEN lanes (issue #33)")
    # SELECTED BY VALUE, NOT BY POSITION. `levels[len(levels) // 2]` is the positional
    # midpoint, which is A = 0 only on a symmetric odd design. On A ∈ {-1, 0, +0.5, +1} the
    # midpoint is +0.5, and the guard below then exited claiming there is no A = 0 while
    # printing a level list whose second entry is 0.0 — a refusal pointing away from its own
    # condition, which is worse than the `assert` it replaced. `sys.exit` rather than
    # `assert` for the original reason: asserts vanish under `python -O`.
    if 0.0 not in levels:
        sys.exit(f"DERIVE-FAIL: the probe's A levels are {levels} — no A = 0 to take an "
                 f"intercept at.\n  § 8 is defined at A = 0 and nowhere else.")
    zero = 0.0
    if ("unknown", zero) not in c15:
        sys.exit("DERIVE-FAIL: the probe carries no `unknown` lane, which is § 8's "
                 "reference.\n  Every difference in this section is measured against it; "
                 "there is no substitute.")
    base = means[("unknown", zero)]
    print(f"  reference: `unknown` at A = 0 = {base:.3f} LUFS")
    diffs, diff_se = {}, {}
    for lane in lanes:
        if lane == "unknown":
            continue
        d = means[(lane, zero)] - base
        s = se_of((lane, zero), ("unknown", zero))
        diffs[lane], diff_se[lane] = d, s
        print(f"    {lane:12s} {means[(lane, zero)]:8.3f}   {d:+7.3f} dB vs unknown"
              f"   = {abs(d) / s:.1f} se")
    # DERIVED, not asserted. This sentence was a hardcoded string — "every named lane is
    # quieter than `unknown` … all four the same direction" — in the script written
    # specifically to stop § 3b's conclusions being claims. On a probe where three of four
    # lanes were LOUDER it printed the same sentence, with a descending range, and that
    # output is copied into the note as-is.
    if not diffs:
        # Unreachable while § 7's k < 2 refusal stands, and stated rather than assumed: the
        # `else` branch below indexes `mags[0]`, so an empty `diffs` was a bare IndexError
        # in the section this file hardened specifically to replace bare tracebacks.
        sys.exit("DERIVE-FAIL: the probe carries no lane other than `unknown` — § 8's "
                 "between-lane differences are undefined.\n  § 7's k < 2 refusal should "
                 "have caught this already; if you are reading this, that guard was lost.")
    quieter = [l for l, d in diffs.items() if d < 0]
    louder = [l for l, d in diffs.items() if d > 0]
    # A LANE AT EXACTLY ZERO IS IN NEITHER LIST, so the disagreement branch printed counts
    # that did not sum to the lanes it was accounting for — "3 quieter, 0 louder" over four
    # named lanes, with the parenthetical suppressed because `louder` is empty, i.e. a line
    # asserting disagreement while naming nothing that disagrees. Counted and named as its
    # own class: a lane sitting ON the reference is a real reading, not a rounding artefact
    # to be dropped out of an exhaustive-looking tally.
    level = [l for l, d in diffs.items() if d == 0.0]
    agree = not level and (len(quieter) == len(diffs) or len(louder) == len(diffs))
    # ⚠ RANKED SEPARATELY, because they are two different quantities. These were one list of
    # (magnitude, se-ratio) pairs sorted on magnitude, so the se endpoints were whichever
    # ratios rode along on the smallest- and largest-|d| lanes — not the smallest and
    # largest se. With the old constant `se` the two orderings coincided and the line was
    # right by construction; per-comparison `se_of` broke that, and the strongest
    # displacement in the section could sit outside a span a reader quotes as its maximum.
    # Both are sorted ASCENDING and printed low-to-high, so neither can read backwards.
    mags = sorted(abs(d) for d in diffs.values())
    ses = sorted(abs(d) / diff_se[l] for l, d in diffs.items())
    span = (f"{mags[0]:.2f} to {mags[-1]:.2f} dB "
            f"({ses[0]:.1f} to {ses[-1]:.1f} se)")
    if agree:
        side = "quieter" if quieter else "louder"
        print(f"\n  every named lane ({len(diffs)}) is {side} than `unknown` at A = 0, "
              f"by {span}, all the same direction.")
    else:
        # Every named lane appears in exactly one of the three counts, and they sum to
        # `len(diffs)` by construction — the tally reads as exhaustive, so it has to be.
        tally = (f"{len(quieter)} quieter, {len(louder)} louder, {len(level)} level"
                 f" (of {len(diffs)})")
        named_by = []
        for label, group in (("louder", louder), ("level", level)):
            if group:
                named_by.append(f"{label}: {', '.join(sorted(map(str, group)))}")
        print(f"\n  ⚠ THE NAMED LANES DO NOT AGREE IN DIRECTION at A = 0: {tally}"
              + (f" — {'; '.join(named_by)}." if named_by else ".")
              + f"\n    Magnitudes {span}. There is no single intercept displacement"
                "\n    to report — do not quote one.")
    named = [means[(l, zero)] for l in lanes if l != "unknown"]
    print(f"  named lanes differ from EACH OTHER at A = 0 by {max(named) - min(named):.2f} dB.")
    print("\n  ⚠ THIS IS NOT A CLEAN INTERCEPT TEST. It is confounded: the named lanes may")
    print("    genuinely be quieter deliveries, which is what a working delivery block")
    print("    SHOULD produce. Separating the two needs rendered A = 0 loudness compared")
    print("    against each lane's TRAINING-SET mean, which this probe does not carry.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--probe", default=PROBE)
    args = ap.parse_args()

    missing = [p for p in (args.labels, args.manifest, args.probe) if not os.path.exists(p)]
    if missing:
        print("missing input(s), nothing derived:", *missing, sep="\n  ", file=sys.stderr)
        return 2

    print(f"labels   {args.labels}")
    print(f"manifest {args.manifest}")
    print(f"probe    {args.probe}")

    with open(args.labels, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    corpus(rows, manifest)
    loudness_targets(rows)

    with open(args.probe, encoding="utf-8") as fh:
        pr = list(csv.DictReader(fh))
    for r in pr:
        r["energy"] = float(r["energy"])
        r["lufs"] = float(r["lufs"])
    probe(pr)

    print("\nread-only: nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
