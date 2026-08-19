"""Synthesis QC gate, stage 2 of 2 (label verdicts).

Merges qc_measures.jsonl (hard gates, phonation, LUFS) with EIV scores
(scripts/stages/eiv_score.py output over qc_filelist.txt) and confirms each clip's
MEASURED affect direction against its INTENDED labels, on the same
LibriTTS-anchored z-scales the Emilia mining used. We never trust instruct
intent — a clip keeps its labels only if the instruments agree.

Axis check: for any axis with |intended| >= 0.3, the measured z must point
the same way with |z| >= --z-confirm. Near-neutral intents are unconstrained
(sanity-bounded). keep = hard_pass AND all axis checks.

FOUR OUTCOMES, NOT TWO: an axis check is True, False, None = UNMEASURED (no EIV row for
the clip, or no phonation from stage 1), or `UNREADABLE` (the intended label is present and
could not be parsed). Neither None nor `UNREADABLE` ever keeps — nothing was confirmed —
and neither counts as a direction failure or is flagged for the ear; both say so on the row
(`axes_unmeasured` / `axes_unreadable`) and in the summary. Collapsing None into False made
a scoring pass that produced nothing report a 100% direction-failure rate, which is the
same misreading the head guard below exists to end. Collapsing `UNREADABLE` into ABSENT was
the mirror image and cost the other way: `all(...)` over no checks is True, so a clip whose
labels were gibberish landed in keeps.jsonl as a confirmed keep.

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
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# ⚠ The sibling-bucket path, the same shape every other stage script uses. `schemas` owns the
# single definition of `coerce_axis`; keeping a second copy here is what issue #58 would look
# like on its second outing.
for _p in (os.path.abspath(REPO), os.path.abspath(os.path.join(REPO, "scripts", "lib"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import schemas  # noqa: E402
LIB_MEASURES = os.path.join(REPO, "data/libritts_r_vat_v2/measures.jsonl")
LIB_EIV = "/data/model-training/sonora/eiv_scores/corpus_v1.jsonl"
LIB_FAM = "/data/model-training/sonora/eiv_scores/corpus_families.jsonl"
COMBO = "/data/model-training/sonora/eiv_scores/valence_combo_v1.json"

AXES = ("V", "A", "T")

# A FOURTH outcome, and the reason it is not None. An axis whose intended label is PRESENT
# and unreadable ("A": "angry") confirmed nothing, exactly like an unmeasured one, so it
# must not keep. But it is a different fact about the bank — None means *we* could not
# measure the render, `UNREADABLE` means the DIRECTOR's label could not be parsed — and the
# repair is different too (fix the labels, versus score the clips). Telling them apart is
# the same discipline as None-is-not-False; collapsing them would report a labelling bug as
# a scoring gap. It is a string, not a bool or None, so every `is True` / `is False` /
# `is None` test downstream (audit_sampler, the by-axis tally, qc_flags) reads it correctly
# without being taught about it: it keeps nothing, fails nothing, and flags nothing.
UNREADABLE = "unreadable"

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


def coerce_axis(v):
    """One `intended` value as a float — None when it carries no usable number.

    ⚠ DELEGATES TO `schemas.coerce_axis`, WHICH IS THE SINGLE DEFINITION. Two copies of
    "what counts as a number" would drift, and the drift would be invisible: both sides
    return floats for every easy case, so only an odd input would reveal the disagreement,
    and by then it is in a manifest.

    The rule is unchanged. A JSON string that spells a number IS a label ("0.7" is 0.7); a
    string that spells anything else is not, and the difference has to stay visible — the old
    `isinstance(v, (int, float))` dropped "0.7" as silently as "very sad", so a whole directed
    bank counted as ZERO and was announced as real audio (issue #58).

    ⚠ The root of #58 is now fixed at the WRITER too (`book_ingest` validates through
    `schemas.intended_vat`), but this stays: `qc_measures.jsonl` files written before that
    change still exist, and six other readers still rely on it.
    """
    return schemas.coerce_axis(v)


def intended_labels(row, unusable=None):
    """The row's intended V/A/T as floats — `{}` when it carries none.

    This was `r["intended"][axis]`, unguarded. Real-audio banks have no `intended` key at
    all (librivox-v1: 664 rows, librivox-v2: 1,366), so the first row took the whole
    campaign down with a KeyError. A bank with nothing to confirm is not a failed
    direction check; there is no direction, and it has to be a clean no-op.

    `unusable`, when given, is a list that collects (id, axis, value) for every axis that
    is PRESENT and unreadable. "No labels" and "labels I could not read" are different
    facts about a bank and the caller has to be able to tell them apart — an absent axis
    (missing key, or an explicit null, which is how a writer says "no label here") is not
    collected, because there is nothing wrong with it.
    """
    lab = row.get("intended")
    if lab and not isinstance(lab, dict):
        if unusable is not None:
            unusable.append((row.get("id"), "intended", lab))
        lab = {}
    lab = lab or {}
    out = {}
    for ax in AXES:
        if lab.get(ax) is None:
            continue
        v = coerce_axis(lab[ax])
        if v is None:
            if unusable is not None:
                unusable.append((row.get("id"), ax, lab[ax]))
            continue
        out[ax] = v
    return out


def describe_unusable(unusable, limit=5):
    """One line naming the unreadable labels — they are never summarised as a count."""
    shown = ", ".join(f"{i}:{ax}={v!r}" for i, ax, v in unusable[:limit])
    return shown + (f" ... (+{len(unusable) - limit} more)" if len(unusable) > limit else "")


def unreadable_axes(entries):
    """-> the set of AXES named by one row's `unusable` entries.

    `intended_labels` records a whole non-dict `intended` under the pseudo-axis
    `"intended"`. That case names no axis because none could be read, so it stands for ALL
    of them: the row asserted a direction and not one letter of it survived parsing.
    """
    axes = {ax for _id, ax, _v in entries if ax in AXES}
    return set(AXES) if any(ax == "intended" for _id, ax, _v in entries) else axes


def axis_verdicts(intended, measured, z_confirm, neutral_band, unreadable=()):
    """-> (checks, notes) for one clip. True, False, None = UNMEASURED, `UNREADABLE`.

    An axis with no intended label is ABSENT from `checks` rather than False. `keep` is
    `all(...)` over the checks, and an unlabelled axis has not failed anything — an
    undirected clip's verdict is exactly its hard_pass, which is the honest answer when
    there is no claim to confirm.

    NONE IS NOT FALSE (issue #55). An axis with no EIV row was scored `False`, so a clip
    that was never measured was indistinguishable — in the verdict rows, in the by-axis
    tally and in qc_flags.txt — from one that was measured and pointed the wrong way. An
    empty eiv_scores.jsonl produced a 100% direction-failure rate, which is the exact
    misreading this stage exists to end (see required_heads() for what the first version
    of it cost). Unmeasured still cannot KEEP — nothing was confirmed — but it is not a
    failure of the engine, and it is not evidence of anything.

    AN UNREADABLE LABEL IS NOT AN ABSENT ONE. `intended_labels` drops an axis it cannot
    parse, so a row whose every label was unreadable used to arrive here with `intended ==
    {}` and leave with `checks == {}` — and `all(...)` over an empty dict is True, so
    `keep` came out as the bare `hard_pass` and the clip was written to keeps.jsonl, which
    the module docstring defines as "clips + CONFIRMED labels". Nothing was confirmed. The
    axes named in `unreadable` are therefore scored `UNREADABLE` rather than omitted, which
    also closes the partial case — 99 good rows and one malformed one, where the whole-row
    guard in `--count-directed` never fires because not every label is bad.

    Every axis that is not `ok` gets a note. The near-neutral lane used to return silently
    in every case, so a clip rejected on its sanity bound (or never measured at all)
    printed an empty console line beside a `False` verdict.
    """
    checks, notes = {}, []
    for axis in AXES:
        if axis in unreadable:
            # Before the `intended` test: an unreadable axis is by construction absent from
            # `intended`, so ordering these the other way round would drop it again.
            checks[axis] = UNREADABLE
            notes.append(f"{axis}:UNREADABLE")
            continue
        if axis not in intended:
            continue
        want = intended[axis]
        got = measured.get(axis) if measured else None
        if got is None:
            ok = None
        elif abs(want) <= neutral_band:  # inclusive: 0.3 intents are weak, not directional
            ok = bool(abs(got) < 3.0)  # sanity bound only: no direction to confirm
        else:
            ok = bool(np.sign(got) == np.sign(want) and abs(got) >= z_confirm)
        checks[axis] = ok
        if ok is not True:
            label = "unmeasured" if ok is None else "FAIL"
            notes.append(f"{axis}:{label}({want:+.1f}->"
                         f"{'--' if got is None else format(got, '+.2f')})")
        elif abs(want) > neutral_band:
            notes.append(f"{axis}:ok({want:+.1f}->{got:+.2f})")
    return checks, notes


def build_anchors():
    """(anchor, heads, weights) — the LibriTTS z-scales every measure below is read on.

    THE ANCHOR IS HELD TO THE SAME STANDARD AS THE CAMPAIGN FILE (issue #54). main()
    refuses a campaign eiv file that is missing a combo head; this side used to fill an
    absent head with `d.get(h, 0.0)`, which is worse, because `gs` returns
    `(mean, std + 1e-9)` — every value 0.0 collapses the std to the epsilon and `z(h, x)`
    becomes `x * 1e9`. A missing head on the campaign side is a fixed offset; a missing
    head HERE is a factor of a billion inside the V dot product.

    MEASURED against the shipped anchor (2026-08-11): of the 14 heads this stage needs,
    12 are present on all 30,351 joined rows. The two that are absent —
    Thankfulness_Gratitude and Disappointment — carry combo weight of exactly 0.0, so the
    dot product annihilates their blown-up z and V is unaffected: bulk1 measures
    min -2.755 / max 3.272 either way. V IS NOT CURRENTLY BROKEN. What is broken is that
    nothing stands between that state and a live one: a weight refit that gives either
    head a non-zero weight turns 1e9 z-scores into the V axis with no warning. So a
    missing head is now fatal when its weight can carry it into V, loud when it cannot,
    and never silently filled.
    """
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
    if not joined:
        sys.exit(f"QC-VERDICT-FAIL: the LibriTTS anchor is empty — no wav in\n"
                 f"  {LIB_MEASURES}\n  is present in {LIB_EIV}. Every z below would be "
                 f"measured against nothing.")

    thin, flat = [], []

    def head_stats(head, weight):
        """(mean, std) for one EIV head over the joined anchor rows.

        `weight` is what the head can contribute to V; the two failure modes below are
        fatal only when it is non-zero, because a zero-weight head cannot move a verdict
        however wrong its scale is. Both are reported either way — the weights are refit
        from data, and this is the file that would go quiet.
        """
        vals = [d[head] for d in joined if head in d]
        if len(vals) < len(joined):
            thin.append((head, len(vals), weight))
            if weight:
                sys.exit(
                    f"QC-VERDICT-FAIL: the LibriTTS anchor carries {head} on "
                    f"{len(vals)}/{len(joined)} rows, and it has combo weight {weight:+.6f}"
                    f" — it moves V.\n"
                    f"  Filling the gap with 0.0 does not give a neutral value: it drags "
                    f"the anchor mean and collapses the std toward the 1e-9 epsilon, so "
                    f"z becomes a multiplier, not a measurement.\n"
                    f"  Score {head} over the corpus and merge it in (the anchor is split "
                    f"across two files with different head sets):\n"
                    f"    {LIB_EIV}\n    {LIB_FAM}")
        if not vals:
            # Unreachable with a non-zero weight (refused above). (0.0, 1.0) keeps z finite
            # so the dot product stays a number rather than an inf/nan: 0.0 * huge is 0.0
            # in IEEE arithmetic, but 0.0 * inf is nan and would take the whole row out.
            return 0.0, 1.0
        v = np.asarray(vals, float)
        if float(v.std()) < 1e-9:
            # A channel with no variance carries no information and must not be divided
            # by — the epsilon in gs() exists to avoid ZeroDivisionError, not to license
            # dividing by 1e-9.
            flat.append((head, weight))
            if weight:
                sys.exit(f"QC-VERDICT-FAIL: the LibriTTS anchor's {head} is constant "
                         f"({float(v.mean()):+.6f} on all {len(vals)} rows) and it has "
                         f"combo weight {weight:+.6f}. Dividing by the 1e-9 floor turns "
                         f"it into a multiplier of ~1e9 inside V.")
            return float(v.mean()), 1.0
        return gs(vals)

    # The extras are read straight into A and T with full weight, so anything missing here
    # is fatal. This was a bare `d["Arousal"]`, i.e. a KeyError and a traceback.
    anchor["soft"] = head_stats("Soft_vs._Harsh", 1.0)
    anchor["arousal"] = head_stats("Arousal", 1.0)
    for h, wt in zip(heads, w.tolist()):
        anchor[h] = head_stats(h, wt)

    if thin or flat:
        # Loud, and it names the weight: this is the whole distance between "cannot affect
        # V" and "silently dominates V".
        for head, n, wt in thin:
            print(f"  !! anchor head {head} is on {n}/{len(joined)} LibriTTS rows "
                  f"(combo weight {wt:+.6f} — it cannot move V, so it is not fatal; a "
                  f"weight refit would make it fatal)")
        for head, wt in flat:
            print(f"  !! anchor head {head} is constant across the LibriTTS corpus "
                  f"(combo weight {wt:+.6f}); its z is not a measurement")
        print(f"  !! V is a weighted combination of {len(heads)} heads and "
              f"{len(thin) + len(flat)} of them are not anchored on data. Re-score the "
              f"corpus for those heads before any refit of {os.path.basename(COMBO)}.")
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
                         "banks, which carry none. Exits non-zero rather than printing 0 "
                         "when the labels are present and unreadable: 0 is what makes the "
                         "caller announce a real-audio bank")
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
        unusable = []
        directed = sum(1 for r in rows if intended_labels(r, unusable))
        if unusable:
            # A `0` on stdout is not a neutral answer: synth_bank.sh prints "(real-audio
            # bank)" on it and skips the stage. Never assert that about a campaign whose
            # labels we could not read — exit instead, which the caller already handles as
            # "the probe did not answer" and announces without claiming anything.
            print(f"  !! {len(unusable)} intended label(s) in {measures_path} are present "
                  f"and not numeric: {describe_unusable(unusable)}", file=sys.stderr)
            if not directed:
                sys.exit(
                    f"QC-VERDICT-FAIL: every intended label in {measures_path} is "
                    f"unreadable, so the directed-clip count would be 0 — which reads as "
                    f"'real-audio bank' and skips the direction check on a directed "
                    f"campaign.\n  Fix the labels (V/A/T must be numbers) and re-run.")
        print(directed)
        return
    if not args.eiv:
        ap.error("--eiv is required (eiv_score.py output over qc_filelist.txt)")

    anchor, HEADS, W = build_anchors()

    def z(key, x):
        m, s = anchor[key]
        return (x - m) / s

    if not os.path.isfile(args.eiv):
        # Same reachability as the empty file below: eiv_score.sh exits 0 whenever the
        # container came up, and a container that scored nothing writes nothing. A
        # traceback here says the same thing, but it says it as if the pipeline were
        # broken rather than the scoring pass empty-handed.
        sys.exit(f"QC-VERDICT-FAIL: {args.eiv} does not exist — eiv_score wrote no scores "
                 f"at all. Nothing was measured; no verdicts can be frozen for this bank.")
    eiv_rows = jload(args.eiv)
    # Keyed by resolved PATH, not basename: qc_gate keys its jobs by (engine dir, id), so
    # one campaign can hold the same clip id — and therefore the same wav basename — under
    # two engine dirs, and a basename index silently hands both of them one engine's
    # scores. Last row wins per path: eiv_score.py appends, so a re-scored clip is the
    # LATER row and the earlier one is a superseded take, not a second opinion.
    eiv_by_path = {os.path.realpath(d["wav"]): d for d in eiv_rows}
    eiv_rows = list(eiv_by_path.values())

    filelist = os.path.join(args.campaign_dir, "qc_filelist.txt")
    wanted = 0
    if os.path.isfile(filelist):
        with open(filelist, encoding="utf-8") as f:
            wanted = sum(1 for ln in f if ln.strip())
    if not eiv_rows:
        # The head guard below iterates ROWS, so with zero rows it finds nothing absent and
        # waves the file through — and then every axis of every clip reads as a direction
        # failure (issue #55). eiv_score.sh exits 0 whenever the container came up, which
        # it does for an empty filelist, unreadable wavs, an OOM-skip or a truncated
        # resume, so this is reachable without anything else looking wrong. Refusing writes
        # no verdict file at all, which is the honest outcome: nothing was measured, and
        # the caller announces it. A file of all-FAIL rows would be spent on the round.
        sys.exit(
            f"QC-VERDICT-FAIL: {args.eiv} holds 0 scored rows"
            + (f" for the {wanted} clip(s) in {filelist}" if wanted else "")
            + ".\n  Nothing was measured, so every axis would read as a direction failure "
              "on clips that were never scored.\n"
              f"  Re-score: scripts/stages/eiv_score.sh {args.eiv} {filelist} "
              f'-- --heads "{",".join(required_heads())}"')

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
            f"    scripts/stages/eiv_score.sh {args.eiv} {args.campaign_dir}/qc_filelist.txt "
            f'-- --heads "{",".join(need)}"')

    # The basename fallback is kept for paths that moved after scoring (a clip swept to
    # _dropped/), and only when it is unambiguous — two engine dirs can hold the same
    # basename, and then it must resolve to neither.
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
    #
    # PER CLIP, NOT PER FILE (issue #56). This compared each wav against the mtime of the
    # whole scores FILE, and eiv_score.sh appends to that file immediately before this
    # runs — so on any bank with even one newly-scored clip the file was newer than every
    # wav and the guard went silent. Reproduced: with one row appended it does not fire;
    # with none it does. The file's mtime is the time of the LAST APPEND, which is a fact
    # about the batch; `wav_mtime`, written per row by eiv_score.py, is a fact about the
    # clip, and one clip's freshness can no longer vouch for another's.
    eiv_mtime = os.path.getmtime(args.eiv)
    stale, unstamped = [], []
    for r in rows:
        e = eiv_for(r)
        if e is None or not os.path.isfile(r["wav_abs"]):
            continue
        scored_at = e.get("wav_mtime")
        if scored_at is None:
            # Rows written before eiv_score.py stamped them. Fall back to the file clock so
            # the check still runs, and say so — under that clock a stale clip CAN be
            # masked by any later append, so a silent pass here is weaker evidence.
            unstamped.append(r["id"])
            scored_at = eiv_mtime
        if os.path.getmtime(r["wav_abs"]) > scored_at + 1e-6:
            stale.append(r["id"])
    if stale:
        print(f"  !! {len(stale)} clip(s) were RE-RENDERED after they were scored — their "
              f"scores are from an earlier take: {', '.join(stale[:5])}"
              + (" ..." if len(stale) > 5 else ""))
        print("  !! re-score them: drop those rows (or the file) and re-run eiv_score.sh")
    if unstamped:
        print(f"  !! {len(unstamped)} scored row(s) carry no per-clip wav_mtime (written "
              f"before that stamp existed) — they were checked against the mtime of "
              f"{os.path.basename(args.eiv)} itself, which a later append can mask.")

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
    unusable, no_eiv, no_phonation = [], [], []
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
        # Two different absences, told apart: no score row at all, versus a scored clip the
        # gate could not measure phonation on. Both leave every axis unmeasured.
        #
        # ⚠ RECORDED ON THE ROW, not looked up by id afterwards. The summary needs to know
        # which absence this row has, and keying that on `r["id"]` reintroduces exactly what
        # L505-507 refuses: two engine dirs can hold the same clip id (`qc_gate.py` keys its
        # jobs on `(eng_dir, id)` for that reason), so an id set would hand one row the
        # other's cause and print the wrong repair for it. Positional by construction here,
        # because it is decided in the same pass that builds the row.
        cause = None
        if not raw:
            cause = "no_eiv" if eiv_for(r) is None else "no_phonation"
            (no_eiv if cause == "no_eiv" else no_phonation).append(r["id"])

        # Sliced per row, not read off the end: `unusable` accumulates across the whole
        # campaign, and this row's verdict may only be shaped by this row's bad labels.
        seen_bad = len(unusable)
        intended = intended_labels(r, unusable)
        bad_axes = unreadable_axes(unusable[seen_bad:])
        if not intended and not bad_axes:
            # "No labels" and "labels I could not read" are different facts, and the
            # sentence printed about `undirected` rows below ("their keep is their
            # hard_pass, unchanged") is false of the second. It used to count both.
            undirected += 1
        checks, notes = axis_verdicts(intended, measured, args.z_confirm,
                                      args.neutral_band, bad_axes)
        # `is True`, not truthiness: an unmeasured axis is None and an unreadable label is
        # `UNREADABLE`, and neither may KEEP (nothing was confirmed) any more than either
        # may FAIL (nothing was measured, nothing was even asked).
        keep = bool(r["hard_pass"] and all(c is True for c in checks.values()))
        v = {**r, "measured_z": measured or None, "axis_checks": checks, "keep": keep,
             "axes_checked": sorted(a for a, c in checks.items() if c is True or c is False),
             "axes_unmeasured": sorted(a for a, c in checks.items() if c is None),
             # `==`, not `is`: these rows are written to JSONL and read back elsewhere, and
             # a round-tripped string is a different object. `True == "unreadable"` is
             # False, so equality is still exact here.
             "axes_unreadable": sorted(a for a, c in checks.items() if c == UNREADABLE),
             # Which ABSENCE this row has, when it has one: `no_eiv` (never scored) or
             # `no_phonation` (scored, stage 1 measured nothing). None when measured. The
             # repairs differ, so the summary reads this rather than re-deriving it.
             "unmeasured_cause": cause,
             "measured_from": MEASURED_FROM}
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
    if unusable:
        # Not folded into `undirected`: a row whose labels could not be read is not a row
        # without labels, and reading the second off the first is how a directed bank
        # comes to look like real audio (issue #58).
        # "ON THAT ALONE" IS A CLAIM AND HAS TO BE TESTED, AND IT TAKES THREE BUCKETS.
        # It began as every row with an unreadable axis, which folded in clips rejected by
        # the hard gate or by a real `False`. Splitting that in two was still wrong in both
        # directions, because it asked only whether the READABLE axes confirmed:
        #
        #   * a row whose only intended axis is the unreadable one has no readable axes, so
        #     `all(...)` ran over an empty set and returned True — `hard_pass` alone put a
        #     clip with NO EIV ROW into "would otherwise keep". Fixing that label cannot
        #     make it keep; there is nothing to check the fixed label against.
        #   * `not only_the_label` then swept up every UNMEASURED row and reported it as
        #     "hard gate, or a measured direction disagreement". It is neither, and NONE IS
        #     NOT FALSE (issue #55) is this module's own doctrine — stating it backwards
        #     here is the same defect at a new address.
        #
        # So the question "will fixing the label bring this clip back" has three answers,
        # and each names the repair that would actually work.
        def verdict_bucket(v):
            if not v["hard_pass"]:
                return "rejected"          # the hard gate; a label cannot reach it
            readable = [c for a, c in v["axis_checks"].items()
                        if a not in v["axes_unreadable"]]
            if any(c is False for c in readable):
                return "rejected"          # measured, and pointed the wrong way
            if any(c is None for c in readable) or v["measured_z"] is None:
                # Nothing was measured — for a readable axis, or (the empty-readable case)
                # for the unreadable one itself. ⚠ TWO CAUSES, TWO REPAIRS, and this module
                # is emphatic they are different (see the `no_eiv` / `no_phonation` split
                # below): no EIV row at all means SCORE the clip; a scored clip stage 1
                # could not measure phonation on means RE-MEASURE stage 1. Telling the
                # second "score them first" sends an operator to re-run eiv_score.py, get
                # back the row it already had, and conclude the bucket is lying — while the
                # run says "were scored" about the same clip three lines later.
                #
                # Read off the row, NOT an id lookup — see `unmeasured_cause` at the build
                # site. A readable axis can also be None while the row itself measured
                # fine (no EIV row for that axis is impossible, but a future cause is not),
                # so `no_eiv` is the default rather than an assertion.
                return "no_phonation" if v["unmeasured_cause"] == "no_phonation" else "unscored"
            return "label"                 # would keep but for the unreadable label

        buckets = collections.defaultdict(list)
        for v in verdicts:
            if v["axes_unreadable"]:
                buckets[verdict_bucket(v)].append(v["id"])
        print(f"  !! {len(unusable)} intended label(s) are present and NOT NUMERIC, so "
              f"those axes CANNOT KEEP: {describe_unusable(unusable)}")
        if buckets["label"]:
            print(f"  !! {len(buckets['label'])} clip(s) are held out of keeps.jsonl on "
                  f"that alone — they stated a direction nobody could read and would "
                  f"otherwise keep. Fix the labels and re-run; this is a LABELLING repair.")
        if buckets["unscored"]:
            print(f"  !! {len(buckets['unscored'])} further clip(s) carry an unreadable "
                  f"label AND have NO EIV ROW. Fixing the label alone will not make them "
                  f"keep — there is nothing to check it against. Run the SCORING pass over "
                  f"them first; this is a scoring repair before it is a labelling one.")
        if buckets["no_phonation"]:
            print(f"  !! {len(buckets['no_phonation'])} further clip(s) carry an unreadable "
                  f"label AND were SCORED but carry no phonation measures from stage 1. "
                  f"Do NOT re-run the scoring pass — it will return the rows it already "
                  f"has. Re-measure stage 1 for them, then fix the labels.")
        if buckets["rejected"]:
            print(f"  !! {len(buckets['rejected'])} further clip(s) carry an unreadable "
                  f"label AND are rejected independently (hard gate, or a MEASURED "
                  f"direction disagreement). Fixing their labels will NOT make them keep.")

    # UNMEASURED IS NOT FAILED (issue #55). These clips have no direction verdict at all;
    # counting them among the failures is what turned a container that scored nothing into
    # a report of a 100% direction-failure rate.
    unmeasured = [v["id"] for v in verdicts if v["hard_pass"] and v["axes_unmeasured"]]
    if no_eiv or no_phonation or unmeasured:
        if no_eiv:
            print(f"  !! {len(no_eiv)}/{len(verdicts)} clip(s) had NO EIV row in "
                  f"{os.path.basename(args.eiv)}: {', '.join(no_eiv[:5])}"
                  + (" ..." if len(no_eiv) > 5 else ""))
        if no_phonation:
            print(f"  !! {len(no_phonation)}/{len(verdicts)} clip(s) were scored but carry "
                  f"no phonation measures from stage 1: {', '.join(no_phonation[:5])}"
                  + (" ..." if len(no_phonation) > 5 else ""))
        print(f"  !! {len(unmeasured)} hard-pass clip(s) have UNMEASURED axes — they are "
              f"not keeps (nothing was confirmed) and not failures (nothing was "
              f"measured), and they are not flagged for the ear on that basis.")
    if wanted and len(eiv_rows) < wanted:
        print(f"  !! {len(eiv_rows)} scored row(s) for the {wanted} clip(s) in "
              f"{os.path.basename(filelist)} — the direction check covers "
              f"{len(eiv_rows) / wanted:.0%} of this bank. Re-run eiv_score.sh to "
              f"complete it before reading these verdicts as the bank's.")

    # ADVISORY OUTPUT. Only hard-passing clips: a clip the gate already rejected is out of
    # the queue on its own merits, and its axes were scored against no EIV row anyway, so
    # flagging it would be noise on top of a rejection. Nothing here changes a clip's
    # status — the flag buys it an ear, which is the whole point.
    #
    # `is False`, not falsy: an unmeasured axis is None, and flagging it would send the
    # ear a clip on the strength of a measurement nobody made.
    flagged = [v["id"] for v in verdicts
               if v["hard_pass"] and any(c is False for c in v["axis_checks"].values())]
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
