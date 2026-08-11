"""Synthesis QC gate, stage 2 of 2 (label verdicts).

Merges qc_measures.jsonl (hard gates, phonation, LUFS) with EIV scores
(scripts/eiv_score.py output over qc_filelist.txt) and confirms each clip's
MEASURED affect direction against its INTENDED labels, on the same
LibriTTS-anchored z-scales the Emilia mining used. We never trust instruct
intent — a clip keeps its labels only if the instruments agree.

Axis check: for any axis with |intended| >= 0.3, the measured z must point
the same way with |z| >= --z-confirm. Near-neutral intents are unconstrained
(sanity-bounded). keep = hard_pass AND all axis checks.

ADVISORY — IT NEVER DROPS A CLIP (wired into synth_bank.sh 2026-08-11)
---------------------------------------------------------------------
`keep` here is a LABEL-CONFIRMATION verdict, not a quality verdict: it asks whether the
render points where the DIRECTOR said it should, so a beautiful clip whose Gemma-assigned
labels were wrong is a legitimate not-keep (gate_calibration.py says the same at length).
Wiring that fail-closed the way qc_gate is wired would therefore throw away good audio
because a label was wrong, which is why the caller runs this on the qc_engine_defects
pattern instead: always runs, writes verdicts, appends axis failures to qc_flags.txt so
the ear sees them, never removes anything from the queue.

The one thing that IS load-bearing is the ordering: the verdicts have to be frozen before
the owner rates the batch, because that comparison is the whole of what gate_calibration
can measure. Afterwards the round is spent — a backfill measures drift, it cannot recover
the round.

WHAT `A` MEANS HERE — and what it does not
------------------------------------------
Arousal is taken from the EIV Arousal head. The LibriTTS corpus lane derives A as the
per-speaker z-score of integrated loudness instead (derive_vat_corpus.py; the divergence
is deliberate and normalize_loudness.py's docstring records why). A clip can satisfy one
definition and fail the other, so each verdict row carries `measured_from` naming the
definition it was actually judged under rather than leaving that to be inferred.

CONTRACT v2 (delivery, 2026-08-07): delivery is the 4th..8th conditioning channel and is
NOT checked here — a lane is categorical, and there is no sign for a z-score to agree
with. V/A/T are the whole of what this instrument can confirm; `axes_checked` says so on
every row rather than letting "the verdict passed" read as "the delivery was verified".

Outputs: qc_verdicts.jsonl, keeps.jsonl (clips + confirmed labels), and a
console summary table for the owner audit.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LIB_MEASURES = os.path.join(REPO, "data/libritts_r_vat_v2/measures.jsonl")
LIB_EIV = "/data/model-training/sonora/eiv_scores/corpus_v1.jsonl"
LIB_FAM = "/data/model-training/sonora/eiv_scores/corpus_families.jsonl"
COMBO = "/data/model-training/sonora/eiv_scores/valence_combo_v1.json"

AXES = ("V", "A", "T")

# Which measurement each axis verdict is made against. Written onto every row because the
# lanes genuinely disagree about A and a row travels away from this file: gate_calibration
# reads rows, derive_markup_measures joins them, and "A failed" is unreadable without
# knowing which A.
MEASURED_FROM = {
    "V": "eiv:valence_combo_v1 z (LibriTTS anchor)",
    "A": "eiv:Arousal z (LibriTTS anchor) — NOT the corpus lane's per-speaker LUFS z",
    "T": "phonation (alpha_db+cpp-h1h2) - eiv:Soft_vs._Harsh, z (LibriTTS anchor)",
    "delivery": "not checked — categorical lane, no sign to confirm (contract v2)",
}

# Heads read on top of the valence combo's own list.
EIV_EXTRA_HEADS = ("Arousal", "Soft_vs._Harsh")


def jload(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def required_heads():
    """Every EIV head the measures below read.

    eiv_score.py DEFAULTS to four heads (Valence,Arousal,Distress,Soft_vs._Harsh) while V
    here is a weighted combination of twelve, and `e.get(h, 0.0)` used to absorb the gap
    without a word. That is not a small error: z(h, 0.0) is a fixed per-head offset, so
    the missing heads contribute a constant and V loses almost all of its variance.

    MEASURED on revisit-v1 — the last campaign this stage ever ran on — whose
    eiv_scores.jsonl holds exactly those four heads: 10 of the 12 combo heads were
    substituted with 0.0, and 33 of its 57 scored rows came out |V| < 0.25 (58%) against
    45 of 159 (28%) on synthetic_v1/bulk1, which was scored with the full set. The V axis
    was failing for want of a head, and it read as engines ignoring direction.
    """
    combo = json.load(open(COMBO, encoding="utf-8"))
    return list(dict.fromkeys(list(combo["heads"]) + list(EIV_EXTRA_HEADS)))


def intended_labels(row):
    """The row's intended V/A/T as floats — `{}` when it carries none.

    This was `r["intended"][axis]`, unguarded. Real-audio banks have no `intended` key at
    all (librivox-v1: 664 rows, librivox-v2: 1,366), so the first row took the whole
    campaign down with a KeyError. A bank with nothing to confirm is not a failed
    direction check; there is no direction, and it has to be a clean no-op.
    """
    lab = row.get("intended") or {}
    out = {}
    for ax in AXES:
        v = lab.get(ax)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[ax] = float(v)
    return out


def axis_verdicts(intended, measured, z_confirm, neutral_band):
    """-> (checks, notes) for one clip.

    An axis with no intended label is ABSENT from `checks` rather than False. `keep` is
    `all(checks.values())`, and an unlabelled axis has not failed anything — `all({})` is
    True, so an undirected clip's verdict is exactly its hard_pass, which is the honest
    answer when there is no claim to confirm.
    """
    checks, notes = {}, []
    for axis in AXES:
        if axis not in intended:
            continue
        want = intended[axis]
        if abs(want) <= neutral_band:  # inclusive: 0.3 intents are weak, not directional
            checks[axis] = abs(measured.get(axis, 0.0)) < 3.0 if measured else False
            continue
        got = measured.get(axis)
        ok = got is not None and np.sign(got) == np.sign(want) and abs(got) >= z_confirm
        checks[axis] = bool(ok)
        notes.append(f"{axis}:{'ok' if ok else 'FAIL'}({want:+.1f}->"
                     f"{'--' if got is None else format(got, '+.2f')})")
    return checks, notes


def build_anchors():
    lib_meas = jload(LIB_MEASURES)
    lib_eiv = {d["wav"]: d for d in jload(LIB_EIV)}
    lib_fam = {d["wav"]: d for d in jload(LIB_FAM)}
    combo = json.load(open(COMBO, encoding="utf-8"))
    heads, w = combo["heads"], np.array(combo["weights"])

    def gs(vals):
        v = np.asarray(vals, float)
        return float(v.mean()), float(v.std() + 1e-9)

    anchor = {k: gs([m[k] for m in lib_meas]) for k in ("alpha_db", "cpp", "h1h2")}
    joined = [{**lib_eiv[m["wav"]], **lib_fam.get(m["wav"], {})}
              for m in lib_meas if m["wav"] in lib_eiv]
    anchor["soft"] = gs([d["Soft_vs._Harsh"] for d in joined])
    anchor["arousal"] = gs([d["Arousal"] for d in joined])
    for h in heads:
        anchor[h] = gs([d.get(h, 0.0) for d in joined])
    return anchor, heads, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir")
    ap.add_argument("--eiv", help="eiv_score.py output jsonl")
    ap.add_argument("--print-heads", action="store_true",
                    help="print the EIV heads this stage needs (comma-joined) and exit. "
                         "synth_bank.sh passes the result to eiv_score.sh, so the head "
                         "set has ONE definition instead of a copy in the shell that can "
                         "drift from the combo file")
    ap.add_argument("--count-directed", action="store_true",
                    help="print how many rows of the campaign's qc_measures.jsonl carry "
                         "intended V/A/T labels, and exit. Callers use it to skip this "
                         "whole stage (including its GPU labelling pass) on real-audio "
                         "banks, which carry none")
    ap.add_argument("--append-flags", action="store_true",
                    help="append the ids of hard-passing clips that failed an axis to "
                         "<campaign>/qc_flags.txt (deduped) so pick_audit_subset --flags "
                         "routes them to the ear. ADVISORY: nothing is dropped")
    ap.add_argument("--z-confirm", type=float, default=0.25)
    ap.add_argument("--per-engine", action="store_true",
                    help="re-center measured z within each engine before direction "
                         "checks (pilot finding: engines are channels with large "
                         "offsets — Qwen A always positive, MOSS T always high). "
                         "Assumes the campaign's registers are roughly balanced per "
                         "engine so the engine mean approximates neutral. Requires "
                         "enough clips per engine for stable stats (>=15).")
    ap.add_argument("--neutral-band", type=float, default=0.3,
                    help="|intended| below this = no direction requirement")
    args = ap.parse_args()

    if args.print_heads:
        print(",".join(required_heads()))
        return
    if not args.campaign_dir:
        ap.error("--campaign-dir is required")
    measures_path = os.path.join(args.campaign_dir, "qc_measures.jsonl")

    if args.count_directed:
        # Deliberately quiet and cheap: it is a pre-flight, and the caller decides. A
        # missing measures file counts as zero rather than as an error, because "there is
        # nothing here to verify" and "the gate has not run" both mean: do not spend a
        # GPU labelling pass on this campaign.
        rows = jload(measures_path) if os.path.isfile(measures_path) else []
        print(sum(1 for r in rows if intended_labels(r)))
        return
    if not args.eiv:
        ap.error("--eiv is required (eiv_score.py output over qc_filelist.txt)")

    anchor, HEADS, W = build_anchors()

    def z(key, x):
        m, s = anchor[key]
        return (x - m) / s

    eiv_rows = jload(args.eiv)
    # The same list --print-heads gives the caller, so what is DEMANDED here and what was
    # REQUESTED of the labeler cannot drift.
    need = required_heads()
    absent = sorted({h for d in eiv_rows for h in need if h not in d})
    if absent:
        # Loud, because the silent version of this produced a verdict file that LOOKED
        # complete — see required_heads() for what it cost on revisit-v1. Re-scoring has
        # to start from a fresh file: eiv_score.py resumes by wav path, so appending to
        # the existing one would skip every clip it already holds and change nothing.
        sys.exit(
            f"QC-VERDICT-FAIL: {args.eiv} is missing EIV head(s): {', '.join(absent)}.\n"
            f"  V is a weighted combination of {len(HEADS)} heads; a missing head is "
            f"substituted with 0.0, which is a fixed offset, not a neutral value.\n"
            f"  Re-score with the full set (move the partial file aside first — "
            f"eiv_score.py resumes by wav path):\n"
            f"    mv {args.eiv} {args.eiv}.partial-heads\n"
            f"    scripts/eiv_score.sh {args.eiv} {args.campaign_dir}/qc_filelist.txt "
            f'-- --heads "{",".join(need)}"')

    # Keyed by resolved PATH, not basename: qc_gate keys its jobs by (engine dir, id), so
    # one campaign can hold the same clip id — and therefore the same wav basename — under
    # two engine dirs, and a basename index silently hands both of them one engine's
    # scores. The basename fallback is kept for paths that moved after scoring (a clip
    # swept to _dropped/), and only when it is unambiguous.
    eiv_by_path = {os.path.realpath(d["wav"]): d for d in eiv_rows}
    eiv_by_base = {}
    for d in eiv_rows:
        eiv_by_base.setdefault(os.path.basename(d["wav"]), []).append(d)
    rows = jload(measures_path)

    def eiv_for(r):
        hit = eiv_by_path.get(os.path.realpath(r["wav_abs"]))
        if hit is not None:
            return hit
        same = eiv_by_base.get(os.path.basename(r["wav_abs"]), [])
        return same[0] if len(same) == 1 else None

    # A clip re-rendered in place keeps its path, so eiv_score.py's resume-by-path skips
    # it and the verdict is made against the PREVIOUS take's scores. That is the same
    # shape as the loudnorm reroll defect (a sidecar keyed on a path that was rewritten
    # underneath it), and it is silent, so it is at least counted here.
    eiv_mtime = os.path.getmtime(args.eiv)
    stale = [r["id"] for r in rows
             if eiv_for(r) is not None and os.path.isfile(r["wav_abs"])
             and os.path.getmtime(r["wav_abs"]) > eiv_mtime]
    if stale:
        print(f"  !! {len(stale)} clip(s) are NEWER than {os.path.basename(args.eiv)} — "
              f"their scores are from an earlier take: {', '.join(stale[:5])}"
              + (" ..." if len(stale) > 5 else ""))
        print("  !! re-score them: drop those rows (or the file) and re-run eiv_score.sh")

    def raw_measures(r):
        e = eiv_for(r)
        if not (r.get("phonation") and e):
            return {}
        p = r["phonation"]
        return {"T": (z("alpha_db", p["alpha_db"]) + z("cpp", p["cpp"])
                      - z("h1h2", p["h1h2"]) - z("soft", e["Soft_vs._Harsh"])),
                "A": z("arousal", e["Arousal"]),
                "V": float(np.dot(W, [z(h, e.get(h, 0.0)) for h in HEADS]))}

    # Positional, not keyed by id, for the same reason the EIV index is keyed by path: two
    # engine dirs can hold the same clip id, and a dict would give the first row the
    # second row's measures.
    all_measured = [raw_measures(r) for r in rows]
    eng_stats = {}
    if args.per_engine:
        by_eng = {}
        for r, m in zip(rows, all_measured):
            if m:
                by_eng.setdefault(r.get("engine"), []).append(m)
        # Zero point = the engine's NEUTRAL_NARRATION takes (bulk-1 finding: the
        # full-pool mean is biased by register composition — Qwen's pool skews
        # soft/sad, MOSS's dark — so correct strong deliveries failed direction
        # checks against an already-shifted mean). Same logic as per-speaker z:
        # each channel's neutral defines its zero. Spread still from full pool.
        neutral = {}
        for r, m in zip(rows, all_measured):
            if r.get("register") == "neutral_narration" and m:
                neutral.setdefault(r.get("engine"), []).append(m)
        for eng, ms in by_eng.items():
            if len(ms) >= 15:
                zeros = neutral.get(eng) or ms
                eng_stats[eng] = {ax: (float(np.mean([m[ax] for m in zeros])),
                                       float(np.std([m[ax] for m in ms]) + 1e-6))
                                  for ax in ("V", "A", "T")}
        print(f"per-engine recentering active for: {sorted(eng_stats)} "
              f"(neutral-anchored: {sorted(k for k in eng_stats if k in neutral)})")

    verdicts, keeps, undirected = [], [], 0
    # The console log is what gets read back when a verdict is disputed, so it says which
    # A this was — the corpus lane's A is a different measurement of the same letter.
    for ax in ("V", "A", "T", "delivery"):
        print(f"  {ax:8s} {MEASURED_FROM[ax]}")
    print(f"{'id':26s} {'axis verdicts':40s} keep")
    for r, raw in zip(rows, all_measured):
        measured = dict(raw)
        st = eng_stats.get(r.get("engine"))
        if measured and st:
            measured = {ax: (measured[ax] - st[ax][0]) / st[ax][1] for ax in measured}

        intended = intended_labels(r)
        if not intended:
            undirected += 1
        checks, notes = axis_verdicts(intended, measured, args.z_confirm,
                                      args.neutral_band)
        keep = bool(r["hard_pass"] and all(checks.values()))
        v = {**r, "measured_z": measured or None, "axis_checks": checks, "keep": keep,
             "axes_checked": sorted(checks), "measured_from": MEASURED_FROM}
        verdicts.append(v)
        if keep:
            keeps.append(v)
        print(f"{r['id']:26s} {' '.join(notes):40s} {keep}")

    for name, data in (("qc_verdicts.jsonl", verdicts), ("keeps.jsonl", keeps)):
        with open(os.path.join(args.campaign_dir, name), "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    print(f"{len(keeps)}/{len(verdicts)} keeps -> keeps.jsonl")
    if undirected:
        # Said out loud rather than left to be read off the keep count: for these rows
        # `keep` is just `hard_pass`, and a verdict file where that is true of every row
        # confirms nothing about direction, however healthy it looks.
        print(f"  {undirected}/{len(verdicts)} row(s) carry NO intended V/A/T — nothing "
              f"to confirm for them; their keep is their hard_pass, unchanged.")

    # ADVISORY OUTPUT. Only hard-passing clips: a clip the gate already rejected is out of
    # the queue on its own merits, and its axes were scored against no EIV row anyway, so
    # flagging it would be noise on top of a rejection. Nothing here changes a clip's
    # status — the flag buys it an ear, which is the whole point.
    flagged = [v["id"] for v in verdicts
               if v["hard_pass"] and not all(v["axis_checks"].values())]
    by_axis = {ax: sum(1 for v in verdicts
                       if v["hard_pass"] and v["axis_checks"].get(ax) is False)
               for ax in AXES}
    print(f"direction failures among hard-pass clips: {len(flagged)} clip(s) — "
          + " ".join(f"{ax}:{n}" for ax, n in by_axis.items()))
    if args.append_flags and flagged:
        # Append-dedup, never clobber: register_audition.py and qc_engine_defects.py write
        # this same file, and whichever ran second used to erase the others' ids.
        fp = os.path.join(args.campaign_dir, "qc_flags.txt")
        have = set()
        if os.path.isfile(fp):
            with open(fp, encoding="utf-8") as f:
                have = {ln.strip() for ln in f if ln.strip()}
        new = [i for i in flagged if i not in have]
        with open(fp, "a", encoding="utf-8") as f:
            for i in new:
                f.write(i + "\n")
        print(f"appended {len(new)} new ids to {fp} "
              f"({len(flagged) - len(new)} already present)")
    print("QC-VERDICT-DONE")


if __name__ == "__main__":
    main()
