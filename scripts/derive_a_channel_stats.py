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
    print(f"  between-lane structure surviving        {100 * sd_a / sd_b:.1f}%"
          f"   (removed {100 * (1 - sd_a / sd_b):.1f}%)")

    h("2. CAMPAIGN IS A NEAR-PERFECT PROXY FOR LANE — BUT ONLY ONE PAIR IS ONE-TO-ONE")
    by_camp = collections.defaultdict(collections.Counter)
    by_lane = collections.defaultdict(collections.Counter)
    for r in rows:
        by_camp[r["campaign"]][r["delivery"]] += 1
        by_lane[r["delivery"]][r["campaign"]] += 1
    n10 = {k: v for k, v in by_camp.items() if sum(v.values()) >= 10}
    ge90 = [k for k, v in n10.items() if max(v.values()) / sum(v.values()) >= 0.90]
    print(f"  campaigns with n >= 10: {len(n10)};  of those, >= 90% one lane: {len(ge90)}")

    print("\n  per lane: how concentrated is it in its top campaign?")
    for lane, c in sorted(by_lane.items(), key=lambda kv: str(kv[0])):
        tot = sum(c.values())
        camp, n = c.most_common(1)[0]
        camp_tot = sum(by_camp[camp].values())
        print(f"    {str(lane) or '(blank)':12s} n={tot:4d} from {len(c):2d} campaign(s); "
              f"top `{camp}` = {n}/{tot} ({100 * n / tot:.1f}% of the lane) and that "
              f"campaign is {n}/{camp_tot} ({100 * n / camp_tot:.1f}%) this lane")
    print("\n  ⚠ ONE lane/campaign pair is 100% in BOTH directions (Newscaster), not two.")
    sp = by_lane.get("Speech", collections.Counter())
    print("    Speech draws from " + ", ".join(f"`{k}` {v}" for k, v in sp.most_common())
          + f" = {sum(sp.values())}; its top campaign is "
          f"{100 * sp.most_common(1)[0][1] / sum(sp.values()):.1f}% of the lane.")

    h("3. NEWSCASTER'S A IS A CONSTANT — AND THE VARIANCE WAS NEVER THERE TO REMOVE")
    a = [r["a"] for r in rows if r["delivery"] == "Newscaster"]
    print(f"  Newscaster A       n={len(a)} mean {np.mean(a):+.4f}  sd {np.std(a, ddof=1):.4f}"
          f"  range [{min(a):+.3f}, {max(a):+.3f}]")
    nv = [r["lufs_native"] for r in rows if r["campaign"] == "newscaster-v1"]
    print(f"  newscaster-v1 native LUFS  n={len(nv)} mean {np.mean(nv):.4f}"
          f"  sd {np.std(nv, ddof=1):.4f} dB   <- the campaign was rendered to one target")

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
    print("    => ONE declared loudness target in this bank, not three. Everything else is")
    print("       un-normalised material whose LUFS is a measurement, and 2 rows are n = 1.")


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
    sd30, df30 = pooled_sd(c30)
    sd15, df15 = pooled_sd(c15)
    mean_of_sds = float(np.mean([np.std(v, ddof=0) for v in c30.values()]))
    se = sd15 * np.sqrt(2 / 6)
    print(f"  renders {len(rows)};  {len(c30)} cells of 3 (text, lane, A);"
          f"  {len(c15)} cells of 6 (lane, A)")
    print(f"  pooled within-cell sd, (text, lane, A)   {sd30:.4f} dB   df={df30}")
    print(f"  pooled within-cell sd, (lane, A)         {sd15:.4f} dB   df={df15}")
    print(f"  se of a difference of two 6-render means {se:.4f} dB")
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
    print(f"  A = +1 minus A = -1, per lane;  95% CI = d +- t({df15}) x se"
          f" = d +- {tcrit * se:.3f} dB")
    slopes = {}
    for lane in lanes:
        d = means[(lane, levels[-1])] - means[(lane, levels[0])]
        slopes[lane] = d
        print(f"    {lane:12s} {d:+7.3f}   95% CI [{d - tcrit * se:+.2f}, {d + tcrit * se:+.2f}]"
              + ("   <- training A is a constant here" if lane == "Newscaster" else ""))
    spread = max(slopes.values()) - min(slopes.values())
    se_ss = sd15 * np.sqrt(4 / 6)   # a slope is a difference of 2 means; two slopes = 4
    p = 2 * (1 - float(stats.t.cdf(spread / se_ss, df15)))
    print(f"  range across lanes: {min(slopes.values()):+.2f} to {max(slopes.values()):+.2f}"
          f"  (spread {spread:.2f} dB)")
    print(f"  se of a DIFFERENCE OF TWO SLOPES (4 cell means) = {se_ss:.4f} dB"
          f"  =>  spread = {spread / se_ss:.2f} se,  p = {p:.3f}")
    print("  ⇒ nonzero gain in every lane, including the constant-A one. EQUAL gain is NOT")
    print("    established — nor excluded — at this n. ⚠ Comparing the spread against a")
    print("    per-render sd is the wrong denominator; these are six-render means.")
    print("    AND ALL OF THIS IS A SLOPE. It says nothing about where A = 0 sits.")

    h("8. INTERCEPT — where A = 0 actually lands, BETWEEN lanes (issue #33)")
    zero = levels[len(levels) // 2]
    assert zero == 0.0, "expected an A = 0 level in the design"
    base = means[("unknown", zero)]
    print(f"  reference: `unknown` at A = 0 = {base:.3f} LUFS")
    diffs = {}
    for lane in lanes:
        if lane == "unknown":
            continue
        d = means[(lane, zero)] - base
        diffs[lane] = d
        print(f"    {lane:12s} {means[(lane, zero)]:8.3f}   {d:+7.3f} dB vs unknown"
              f"   = {abs(d) / se:.1f} se")
    lo, hi = min(diffs.values()), max(diffs.values())
    print(f"\n  every named lane is quieter than `unknown` at A = 0, by "
          f"{abs(hi):.2f} to {abs(lo):.2f} dB ({abs(hi) / se:.1f} to {abs(lo) / se:.1f} se),"
          " all four the same direction.")
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
