"""V/A/T labels for the v6 append set — global anchor, with a loudness offset on A.

THE LANE, decided 2026-08-10 (owner). The 832 appended rows are labelled on the
**global anchor** (`anchor_emilia_labels.libritts_anchor`), the same lane Emilia
uses, for the same reason: one speaker id per clip makes every appended row n = 1
under `derive_vat_corpus.per_spk_z`, which re-centres each clip on its own mean and
returns **exactly 0.000** on all three channels — a label indistinguishable from a
real at-speaker-mean one. `merge_emilia_corpus.py:20-29` exists because that already
bit this corpus once, on 756 one-clip speakers.

WITH ONE DEPARTURE, and this script exists mainly to carry it. A dry run of the
plain global anchor over these 846 clips returned:

    V  +0.004 / sd 0.584 /  21.4% clamped     (v5 emilia half: +0.307 / 0.609 / 30.0%)
    A  -0.970 / sd 0.135 /  94.4% clamped     (v5 emilia half: -0.003 / 0.592 / 11.5%)
    T  +0.262 / sd 0.576 /  22.8% clamped     (v5 emilia half: +0.747 / 0.362 / 54.0%)

V and T sit inside precedent. **A is pinned at -1 on 94.4% of the rows**, and the
cause is not performance:

    LibriTTS-R v4 (the anchor)   lufs mean -18.16  sd 1.86
    expressive-registers 846     lufs mean -23.34  sd 1.62   <- median exactly -23.00
    emilia probe                 lufs mean -17.73  sd 2.50

This bank is loudness-normalised to **-23 LUFS**; LibriTTS-R is not. That is a
5.18 dB offset = **2.8 anchor sd**, which `clamp2` maps to -1.39 and clips to -1.
Emilia's mean lands near LibriTTS's by accident of source, which is exactly why the
plain global anchor worked there and does not work here.

So the offset is not a tuning knob — without it, A does not merely go uninformative,
it teaches the model that every expressive-register clip is the quietest thing in the
corpus, encoding OUR ENCODER'S TARGET as a property of the performance.

WHAT IS CORRECTED, AND WHAT DELIBERATELY IS NOT. Only the **centre** moves: each
clip's LUFS is shifted so its source's mean lands on the anchor's, before `label()`
z-scores it. The bank's own **spread** is left alone, so A stays slightly less extreme
than the anchor rather than being stretched onto it. Rescaling the sd as well would
assert that this bank's loudness variation means the same thing as LibriTTS-R's, which
nothing has measured. Centre-only is the conservative half of that claim and is the
half the evidence supports: A means loud-vs-quiet WITHIN a source, and the offset is
what makes that reading true across sources.

⚠ **"SOURCE" IS THE CAMPAIGN, NOT THE BANK** — a single bank-wide offset was the first
fix and it is not enough. This bank holds at least THREE loudness targets:

    -23.0 LUFS   bulk1, newscaster-v1, stress-*, teacher-ab-v1, artifact-probe-*, ...
    -20.4 LUFS   quote-pilot-*
    -26.3..-27.1 book-librivox-*

One bank-wide offset centres on the weighted average of the three and leaves each one
displaced by up to 6.7 dB — so the 103 librivox rows come out at A = -0.835, labelled
uniformly quiet for a reason that lives in the encoder rather than the voice. That is
the same defect the offset was introduced to remove, one level down. Centring per
campaign removes it at every level it occurs.

**A CONSTANT A IS THE CORRECT ANSWER for most of these rows, and that is not a
failure.** 638 of 846 sit within 0.05 dB of exactly -23.00: loudness normalisation
already destroyed their performance loudness, and no offset can recover it. But the
model is trained to reproduce the NORMALISED audio, so "unremarkable loudness" is a
true statement about the target. Per-campaign centring says exactly that (A ~ 0) while
preserving the real variation wherever loudnorm did not flatten it — delivery-v1-
narration-r2 (sd 1.83), book-librivox-v2 (1.52), quote-pilot-v3 (1.13).

⚠ Pre-loudnorm loudness IS recoverable for **104 of the 846** (`v1/audio/loudnorm.jsonl`
records `lufs_in` per clip, and `_pre_loudnorm_v1/` holds the originals) — but only for
that slice. Using it would put 104 rows on a fourth scale, so it is left unused and
recorded here as the one lane that could carry real A signal if the sidecar were ever
backfilled across the bank.

Campaigns below `MIN_CAMPAIGN_N` fall back to the bank-wide offset: a campaign of one
centred on itself is A = 0.000 exactly, which is the n = 1 trap this whole lane exists
to avoid. 25 rows across 8 campaigns take that fallback.

V and T are NOT offset. Both are composites of measures that are not chain-scaled the
way integrated loudness is, and both already landed inside precedent — an offset there
would be a correction with no defect to correct.

`anchor_emilia_labels.py` is deliberately NOT modified. It is the shipped v5 labelling
path, and editing it to take an offset would put v5's reproducibility at risk to save
one line here. The shift is applied to the row before `label()` sees it, and every
row records `lufs_native`, `lufs_offset` and `lufs_adjusted` so the correction is
auditable rather than baked in.

Run:  .venv/bin/python scripts/label_expressive_registers.py \
          --measures /data/model-training/sonora/expressive_registers_measures/probe_measures.jsonl \
          --eiv /data/model-training/sonora/eiv_scores/expressive_registers_v6.jsonl \
          --out /data/model-training/sonora/expressive_registers_measures/labels_v6.jsonl
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anchor_emilia_labels as anchor_mod  # noqa: E402

V4_CORPUS = "/data/model-training/sonora/data/libritts_r_vat_v4"

# Below this, a campaign's own mean is too thin to centre on and the bank-wide offset is
# used instead. The floor exists for the n = 1 case: centring a lone clip on itself gives
# A = 0.000 exactly — a fabricated "average" wearing a measurement's clothes, which is the
# failure `merge_emilia_corpus.py:20-29` already documents for 756 one-clip speakers.
MIN_CAMPAIGN_N = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measures", required=True)
    ap.add_argument("--eiv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=V4_CORPUS,
                    help="corpus the global anchor is derived from (v5 used v4)")
    ap.add_argument("--offset-scope", choices=("campaign", "bank", "none"),
                    default="campaign",
                    help="granularity of the A-channel loudness centring. 'campaign' "
                         "(default) centres each campaign of >= MIN_CAMPAIGN_N clips on "
                         "the anchor; 'bank' uses one offset for all 846 and leaves the "
                         "three loudness targets displaced; 'none' reproduces the dry run "
                         "that motivated the correction and must not build a corpus")
    ap.add_argument("--ratings",
                    default="/data/model-training/datasets/sonora-expressive-registers/"
                            "ratings.csv",
                    help="re-read the MUTABLE metadata (delivery, register, campaign) from "
                         "here instead of trusting the copy frozen into --measures. "
                         "Acoustic measures are facts about a file and never change; "
                         "delivery is an ear verdict and does. Documentary was retired into "
                         "Neutral on 2026-08-10 and the stale copy silently survived a "
                         "relabel — pass --no-refresh-meta to reproduce that older output.")
    ap.add_argument("--no-refresh-meta", action="store_true",
                    help="trust --measures' frozen metadata (reproduces pre-2026-08-10 runs)")
    # SETTLED 2026-08-10 (owner): the 14 over-length clips are DROPPED, and the append set
    # is 832. On by default because the offsets below are a property of THIS SET's loudness
    # centres — labelling 846 changes `bank_offset`, changes every campaign mean that held
    # one of the 14, and so changes the A value of clips that were never over-length. A
    # default that produced 846 rows meant the documented Run: command no longer
    # reproduced the artifact on disk, and nothing downstream announced the divergence:
    # merge_expressive_registers.py catches the 14 as a `duration` drop labelled "a staging
    # bug, not a corpus decision", and 14/846 = 1.7% sails under the 10% drop ceiling.
    # `--keep-over-max` reproduces the older artifact, the same shape as --no-refresh-meta.
    ap.add_argument("--drop-over-max", dest="drop_over_max", action="store_true",
                    default=True,
                    help="exclude clips flagged over_max_seconds (DEFAULT since "
                         "2026-08-10, when the drop-or-raise call was made; the flag "
                         "is carried through either way)")
    ap.add_argument("--keep-over-max", dest="drop_over_max", action="store_false",
                    help="label the over-length clips too, reproducing the 846-row "
                         "pre-2026-08-10 artifact. Note this also shifts the A channel of "
                         "in-range clips, because the offsets are recomputed per set.")
    args = ap.parse_args()

    anchor_mod.CORPUS = args.base
    print(f"deriving the global anchor from {args.base} ...")
    anchor = anchor_mod.libritts_anchor()
    print(f"  {anchor['n_clips']} clips, {anchor['n_heads']} with EIV heads, "
          f"{len(anchor['weights'])} weighted heads")

    measures = {r["wav"]: r for r in anchor_mod._jsonl(args.measures)}
    heads = {r["wav"]: r for r in anchor_mod._jsonl(args.eiv)}

    # Refresh the mutable half against the SSOT. Read-only — the Auditions app owns this
    # file — and the mtime is recorded in the manifest so a concurrent edit is visible.
    ratings_mtime = None
    if not args.no_refresh_meta:
        import csv
        ratings_mtime = os.stat(args.ratings).st_mtime_ns
        with open(args.ratings, encoding="utf-8") as fh:
            live = {r["id"]: r for r in csv.DictReader(fh)}
        # ALL THREE FIELDS ARE COUNTED, not just delivery. `campaign` is the grouping key
        # for the A-channel offsets below — a rename or correction in ratings.csv moves
        # clips between offset groups and therefore changes their A value. Reporting only
        # `delivery` meant a run in which campaigns moved and deliveries did not printed
        # "no delivery changed since the measure pass", which an operator reads as "the
        # refresh was a no-op". It was not.
        changed, unmatched = collections.Counter(), 0
        for row in measures.values():
            cur = live.get(row.get("id"))
            if not cur:
                # Counted rather than skipped in silence. measure_expressive_registers.py
                # hard-aborts on an id missing from ratings.csv; the same condition here
                # was a no-op, so a reindexed id scheme would leave EVERY row on frozen
                # metadata and still print the reassuring line. Zero is the interesting
                # number, so it has to be printed to be trusted.
                unmatched += 1
                continue
            for field in ("delivery", "register", "campaign"):
                if field not in cur:
                    sys.exit(f"FATAL: {args.ratings} has no {field!r} column — the refresh "
                             f"cannot tell 'unchanged' from 'not present'.")
                if row.get(field) != cur[field]:
                    changed[f"{field}: {row.get(field) or '(blank)'} -> "
                            f"{cur[field] or '(blank)'}"] += 1
                    row[field] = cur[field]
        print(f"refreshed metadata from {args.ratings} (mtime_ns {ratings_mtime})")
        if unmatched:
            print(f"  ⚠ {unmatched} measured clip(s) have no row in ratings.csv — frozen "
                  f"metadata kept for them")
        for move, n in changed.most_common():
            print(f"  {move}: {n} clip(s)")
        if not changed:
            print("  no delivery/register/campaign changed since the measure pass")
    shared = sorted(set(measures) & set(heads))
    only_m, only_e = set(measures) - set(heads), set(heads) - set(measures)
    print(f"measures {len(measures)} · eiv {len(heads)} · joined {len(shared)}")
    if only_m or only_e:
        # A clip with one half of its label sources is not labellable, and quietly
        # dropping it moves the append count with no record of which rows left.
        sys.exit(f"FATAL: {len(only_m)} clip(s) have measures but no EIV heads, "
                 f"{len(only_e)} the reverse. Both halves must cover the same set.\n"
                 f"  e.g. measures-only: {sorted(only_m)[:2]}\n"
                 f"       eiv-only:      {sorted(only_e)[:2]}")

    if args.drop_over_max:
        kept = [w for w in shared if not measures[w].get("over_max_seconds")]
        print(f"dropping {len(shared) - len(kept)} clip(s) over MAX_SECONDS")
        shared = kept

    # Offsets are a property of THIS SET's loudness centres, so they are recomputed from
    # whatever is being labelled and written into the output. Changing the membership
    # (e.g. --drop-over-max) changes them, which is why they are recorded, not hard-coded.
    anchor_lufs_mean, anchor_lufs_sd = anchor["lufs"]
    src_lufs = np.array([measures[w]["lufs"] for w in shared], float)
    bank_offset = anchor_lufs_mean - float(src_lufs.mean())

    by_campaign = collections.defaultdict(list)
    for w in shared:
        by_campaign[measures[w].get("campaign") or "(none)"].append(measures[w]["lufs"])

    offsets, fell_back = {}, []
    for name, vals in by_campaign.items():
        if args.offset_scope == "none":
            offsets[name] = 0.0
        elif args.offset_scope == "bank" or len(vals) < MIN_CAMPAIGN_N:
            offsets[name] = bank_offset
            if args.offset_scope == "campaign":
                fell_back.append((name, len(vals)))
        else:
            offsets[name] = anchor_lufs_mean - float(np.mean(vals))

    print(f"\nA-channel loudness centring — scope: {args.offset_scope}")
    print(f"  anchor lufs   mean {anchor_lufs_mean:+7.2f}  sd {anchor_lufs_sd:5.2f}")
    print(f"  bank   lufs   mean {src_lufs.mean():+7.2f}  sd {src_lufs.std():5.2f}  "
          f"(n={len(src_lufs)}, bank-wide offset {bank_offset:+.3f} dB)")
    if args.offset_scope == "campaign":
        # The DOMINANT LANE is printed beside each campaign because campaign is close to a
        # proxy for delivery lane in this bank (`newscaster-v1` -> Newscaster,
        # `librivox-speech-v1` -> Speech, `delivery-v1-narration*` -> the narration lanes).
        # Wherever a campaign is lane-aligned, centring on its mean subtracts exactly the
        # component of loudness that differs BETWEEN lanes — so this column is what says
        # whether that is happening, rather than leaving it assumed either way.
        lane_of = collections.defaultdict(collections.Counter)
        for w in shared:
            lane_of[measures[w].get("campaign") or "(none)"][
                measures[w].get("delivery") or "(blank)"] += 1
        print(f"  {len(by_campaign)} campaigns; per-campaign offsets "
              f"(and the lane each is a proxy for):")
        for name, vals in sorted(by_campaign.items(), key=lambda kv: -len(kv[1])):
            tag = "  [fallback: bank-wide]" if len(vals) < MIN_CAMPAIGN_N else ""
            lane, lane_n = lane_of[name].most_common(1)[0]
            print(f"    n={len(vals):4d}  lufs {np.mean(vals):+7.2f}  "
                  f"offset {offsets[name]:+7.3f}  {name}{tag}\n"
                  f"           dominant lane {lane} ({100 * lane_n / len(vals):.0f}%)")
        if fell_back:
            print(f"  {sum(n for _, n in fell_back)} row(s) in {len(fell_back)} campaign(s) "
                  f"below n={MIN_CAMPAIGN_N} took the bank-wide fallback")
    if args.offset_scope == "none":
        print("  ⚠ NO CORRECTION APPLIED — dry-run mode; do not build a corpus from this.")

    labelled, errors, uncorrected_a = [], [], {}
    for w in shared:
        m, h = measures[w], heads[w]
        offset = offsets[m.get("campaign") or "(none)"]
        row = {**m, "lufs": m["lufs"] + offset,
               **{f"head:{k}": v for k, v in h.items() if k != "wav"}}
        try:
            v, a, t = anchor_mod.label(row, anchor)
            # The same clip's A with NO offset, kept only for the between-lane report
            # below. Pure arithmetic on the measures row — no audio is touched — and it is
            # deliberately not written to the output, so the artifact's schema is
            # unchanged and there is exactly one A per row on disk.
            a_raw = anchor_mod.label({**row, "lufs": m["lufs"]}, anchor)[1]
        except anchor_mod.LabelError as e:
            errors.append((m.get("id", w), str(e)))
            continue
        uncorrected_a[w] = a_raw
        labelled.append({
            "wav": w, "id": m.get("id"), "v": v, "a": a, "t": t,
            "lufs_native": m["lufs"], "lufs_offset": offset,
            "campaign_offset_scope": args.offset_scope,
            "lufs_adjusted": m["lufs"] + offset,
            "over_max_seconds": m.get("over_max_seconds", False),
            "resampled": m.get("resampled", False),
            "engine": m.get("engine"), "delivery": m.get("delivery"),
            "register": m.get("register"), "campaign": m.get("campaign"),
        })

    if errors:
        # Never write a partial label set: a missing row here becomes a clip that
        # derive_vat_corpus aborts on later, far from the cause.
        print(f"\n{len(errors)} clip(s) could not be labelled:")
        for cid, why in errors[:10]:
            print(f"  {cid}: {why}")
        sys.exit("FATAL: refusing to write a partial label set.")

    arr = np.array([[r["v"], r["a"], r["t"]] for r in labelled])
    print(f"\nlabelled {len(labelled)}/{len(shared)}")
    print("            mean      sd      min      max    clamped")
    for i, ch in enumerate("VAT"):
        c = arr[:, i]
        print(f"  {ch}      {c.mean():+7.4f} {c.std():7.4f} {c.min():+7.4f} {c.max():+7.4f}"
              f"   {(np.abs(c) > 0.99).mean() * 100:5.1f}%")

    bad = int((~np.isfinite(arr)).sum()) + int((np.abs(arr) > 1.0 + 1e-9).sum())
    if bad:
        sys.exit(f"FATAL: {bad} non-finite or out-of-range value(s) in the label set.")
    print("  0 non-finite / out-of-range")

    # ---- WHAT THE CENTRING REMOVED, measured rather than assumed ----------------------
    # Campaign is close to a proxy for delivery lane (see the dominant-lane column above),
    # so subtracting each campaign's mean LUFS also subtracts whatever loudness genuinely
    # differs BETWEEN lanes wherever a campaign is lane-aligned. That is not fatal — the
    # delivery one-hot can carry it — but it means the one-hot becomes the SOLE carrier of
    # between-lane energy, and rung 2 then cannot separate "the delivery channel taught the
    # model something" from "the A correction moved lane-level energy into it". quality-
    # gap-plan.md bills this rung as one lever; unmeasured, this quietly makes it two.
    #
    # So: the spread of the per-lane means, before and after. If it barely moves, the
    # campaigns are not lane-aligned in a way that matters and the concern evaporates —
    # measured, not assumed. If it collapses, the size of what was removed is on record
    # before the run rather than diagnosed after it.
    #
    # Measured in dB on LUFS and NOT on A, deliberately. A before the correction is pinned
    # at -1 on 94.4% of these rows, and a clamp destroys between-lane structure all by
    # itself — so an A-before/A-after comparison would confound the clamp with the offset
    # and understate what the centring removed. LUFS is the unclamped quantity the offset
    # actually acts on. Both A columns are printed beside it so the clamp's own share of
    # the flattening stays visible rather than being attributed to the centring.
    by_lane = collections.defaultdict(list)
    for r in labelled:
        by_lane[r.get("delivery") or "(blank)"].append(
            (r["lufs_native"], r["lufs_adjusted"], r["a"], uncorrected_a[r["wav"]]))
    if len(by_lane) > 1:
        def col(vals, i):
            return float(np.mean([v[i] for v in vals]))

        before = np.array([col(v, 0) for v in by_lane.values()])
        after = np.array([col(v, 1) for v in by_lane.values()])
        print(f"\nbetween-lane structure ({len(by_lane)} lanes) — the centring's "
              f"collateral, measured:")
        print("    lane                 n   lufs raw   lufs centred    A raw    A final")
        for lane, v in sorted(by_lane.items(), key=lambda kv: -len(kv[1])):
            print(f"    {lane:<18} {len(v):4d}   {col(v, 0):+8.2f}   {col(v, 1):+11.2f}"
                  f"  {col(v, 3):+7.3f}   {col(v, 2):+7.3f}")
        print(f"  sd of the lane means (LUFS): {before.std():.3f} dB raw -> "
              f"{after.std():.3f} dB centred")
        if before.std() > 1e-6:
            kept = 100 * after.std() / before.std()
            print(f"  {kept:.0f}% of the between-lane loudness spread survives the "
                  f"centring. Whatever does not must now be carried by the delivery "
                  f"one-hot, which makes it the sole carrier of lane-level energy.")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in labelled:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "lane": "global anchor (anchor_emilia_labels.libritts_anchor), A centred per "
                f"{args.offset_scope}",
        "anchor_base": args.base,
        "anchor_n_clips": anchor["n_clips"],
        "offset_scope": args.offset_scope,
        "min_campaign_n": MIN_CAMPAIGN_N,
        "bank_offset_db": bank_offset,
        "campaign_offsets_db": {k: offsets[k] for k in sorted(offsets)},
        "campaign_n": {k: len(v) for k, v in sorted(by_campaign.items())},
        "fell_back_to_bank_offset": sorted(fell_back),
        "anchor_lufs": {"mean": anchor_lufs_mean, "sd": anchor_lufs_sd},
        "source_lufs": {"mean": float(src_lufs.mean()), "sd": float(src_lufs.std()),
                        "n": len(src_lufs)},
        "ratings_csv_mtime_ns": ratings_mtime,
        "metadata_refreshed": not args.no_refresh_meta,
        "dropped_over_max_seconds": bool(args.drop_over_max),
        "n_labelled": len(labelled),
        "distribution": {ch: {"mean": float(arr[:, i].mean()), "sd": float(arr[:, i].std()),
                              "clamped_pct": float((np.abs(arr[:, i]) > 0.99).mean() * 100)}
                         for i, ch in enumerate("vat")},
    }
    mpath = os.path.join(os.path.dirname(args.out), "labels_v6_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n-> {args.out}\n-> {mpath}")


if __name__ == "__main__":
    main()
