#!/usr/bin/env python3
"""quote-pilot-v3d bank: Orpheus-3B (ft + pretrained arms) and F5-TTS (NC
benchmark arm) over the SAME 10 lines + G26 directions as v3/v3b/v3c.

Neither new engine consumes free-text directions, so the G26 `design` is
compiled into per-engine controls (owner ruling 2026-07-23):
  * orpheus_ft  — preset-voice pick by design gender (tara/leo) + the quote
                  text; instruct informs optional inline emotion tags.
  * orpheus_pt / f5 — REFERENCE-CLIP casting: the design selects an audited
                  certified keep (v1 dataset) by gender + intended-VAT
                  proximity + duration window; the reference's prosody
                  carries the delivery. Engine preference: casting-faithful
                  moss85/longcat over qwen (young-skew), dia last.

Output: <campaign_dir>/v3d_bank.json

B-L7, 2026-08-07. This scored references itself — gender match, VAT proximity, a duration
window, a variety bias — which is `ref_select.select_reference` rewritten. It cannot
simply CALL it, because it casts from this campaign's own keeps file rather than from the
certified pool `_load_pool()` reads, and that is a legitimate difference. What was not
legitimate is that the fork carried NONE of the guards the real one applies:

  * `REF_BLACKLIST` — five clips measured to render badly, by id, across chatterbox AND
    zonos. Nothing here excluded them, so this bank could and did cast them.
  * `MAX_REF_EXCURSION` — the pitch-excursion ceiling from the 2026-07-29 blind audition.
  * and `design_gender` / the VAT accessor were duplicated BYTE FOR BYTE from ref_select,
    which is the B-L5 shape again: a copied rule is a rule that can be fixed in one place
    and not the other.

The helpers are imported now and the guards applied. The scoring stays local, because the
pool genuinely differs; the guards do not, because they encode heard failures that are
true of a reference clip regardless of which file lists it.
"""
import json, os, sys
from pathlib import Path

# Sibling modules used to be reached with `sys.path.insert(0, dirname(__file__))`, which
# worked only while every script lived in one directory. After #26 step 3 they are split
# across scripts/{stages,lib,tools,gates}, so the anchor is the REPO ROOT and the search
# path is explicit. Uniform on purpose: every file under scripts/<bucket>/ is exactly two
# levels down, so this expression is the same everywhere and `tests/test_asset_paths.py`
# can check it.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_SONORA_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_SONORA_REPO, *(_os.path.join(_SONORA_REPO, "scripts", _b) for _b in ("lib",))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import schemas  # noqa: E402  -- the single definition of what counts as an axis number
from ref_select import (  # noqa: E402
    MAX_REF_EXCURSION,
    REF_BLACKLIST,
    _vat,
    _vat as keep_vat,
    _vat_distance,
    design_gender,
)

DS = Path("/data/model-training/datasets/sonora-expressive-registers")
SRC = DS / "quote-pilot-v3b" / "moss85_manifest.jsonl"
KEEPS = DS / "v1" / "metadata.jsonl"
OUT_DIR = DS / "quote-pilot-v3d"
ENGINE_PREF = {"moss85": 0.0, "longcat": 0.05, "qwen": 0.15, "dia": 0.3}
FT_VOICE = {"F": "tara", "M": "leo"}

AXES = ("V", "A", "T")


def rankable_vat(d):
    """True when AT LEAST ONE axis carries a usable number.

    ⚠ IT USED TO REQUIRE ALL THREE, AND THAT STOPPED BEING RIGHT ON 2026-08-18 (issue #112).
    It was written when the arithmetic below subtracted axes directly, so one `null` was a
    `TypeError` and the row genuinely could not be scored. `0e3b831` replaced that with
    `ref_select._vat_distance`, which ranks on the axes both sides label — so a row with V
    labelled and A/T null is now scored on V, perfectly well, and this gate was still
    throwing it away. The message even sent the operator off to re-direct those lines: a
    31B inference per line, to fix something that was no longer broken.

    ⚠ ONE axis is the floor rather than zero because zero is a different thing. A row that
    labels nothing scores `_vat_distance` = 0.0 against EVERY candidate — they all tie, the
    VAT term leaves the ranking, and `ENGINE_PREF` alone picks the reference. That is the
    silent degrade this file's guard exists to catch, so a row with no axes at all is still
    dropped, and now for the reason that is actually true.

    ⚠ DELEGATES TO `schemas.coerce_axis`, WHICH IS THE SINGLE DEFINITION (issue #113). The
    `isinstance(d.get(k), (int, float))` this replaces was the FOURTH copy of "what counts
    as a number" and disagreed with the definition on the one input #58 was filed about:
    `coerce_axis("0.7")` is `0.7`, the old test here said `False`. `0e3b831` came into this
    file to remove exactly this class of drift, took out the duplicated DISTANCE, and walked
    past the duplicated NUMBER TEST eight lines above the comment it wrote saying so.
    `coerce_axis` refuses bools, which the old test also had to say out loud.
    """
    return any(schemas.coerce_axis(d.get(k)) is not None for k in AXES)


# ⚠ `vat_dist` LIVED HERE AND IS GONE (2026-08-18, issue #107). It was
# `math.sqrt(sum((a[k] - b[k]) ** 2 for k in AXES))` — a second definition of the distance
# `select_reference` already owned, which is the drift this branch removed from `coerce_axis`
# in three other places. `ref_select._vat_distance` is the one definition, and it skips an
# axis neither side labels instead of subtracting `None`.

rows = [json.loads(l) for l in SRC.open()]
keeps = [json.loads(l) for l in KEEPS.open()]

bank = []
used = set()
# ⚠ SKIPPED AND COUNTED, NEVER SILENTLY DROPPED. A line that labels NO axis cannot be
# distance-ranked against anything — every candidate would tie and ENGINE_PREF would pick
# the reference by itself — so it has no place in this bank. But a bank that is quietly
# short is the failure this repo keeps paying for, so the count is printed.
unscoreable = [r for r in rows if not rankable_vat(r["intended"])]
rows = [r for r in rows if rankable_vat(r["intended"])]
if unscoreable:
    print(f"  ⚠ {len(unscoreable)} line(s) skipped: intended V/A/T labels no axis at all, so "
          f"every reference would score identically. A PARTIALLY labelled line is fine and "
          f"is kept — it ranks on the axes it has (issue #112). Re-direct these, or accept "
          f"a shorter bank.")
for r in rows:
    g = design_gender(r["direction"]["design"])
    cands = []
    for k in keeps:
        if k.get("gender", "")[:1].upper() != g:
            continue
        if not (4.0 <= float(k.get("duration", 0)) <= 10.0):   # owner floor 2026-07-25
            continue
        # B-L7: the two guards the fork dropped. Both are properties of the reference
        # CLIP, not of the pool file that happens to list it, so they apply here exactly
        # as they do in select_reference.
        if k.get("id") in REF_BLACKLIST:
            continue
        exc = k.get("ref_excursion_hz")
        if exc is not None and float(exc) >= MAX_REF_EXCURSION:
            continue
        # ⚠⚠ `k["intended_vat"]`, NOT `k` — RESTORED 2026-08-18 (issue #107), AND THE WAY
        # IT WENT MISSING IS THE POINT. This file's own `keep_vat` took a keeps RECORD and
        # did `iv = k["intended_vat"]` itself. `98734f9` (2026-08-07, B-L7) replaced it with
        # `_vat as keep_vat` to stop the fork keeping private copies — right intent — but
        # `_vat` takes the AXIS DICT, not the record, and the two call sites were not
        # changed. **A shared helper is only shared if its argument is the same thing.**
        #
        # From that day this scored every candidate against one constant: `_vat` looked for
        # `V`/`valence` at the top level of a record that keeps them under `intended_vat`,
        # found neither, and returned `{0.0, 0.0, 0.0}` for all 193 keeps. A constant
        # cancels out of a comparison, so ENGINE_PREF and the `used` penalty decided the
        # order by themselves — and nothing said so, because a bank still came out.
        # `9bb3607` then changed that default to `None`, which is why it now raises
        # `TypeError` instead of ranking badly in silence.
        #
        # Measured on the real data with the subscript restored: **9 of the 10 scoreable
        # rows choose a different reference.** `select_reference` has always written
        # `_vat(k["intended_vat"])`; B-L7 records two other guards this same fork dropped.
        # ⚠ BOTH SIDES NORMALISED, and only one of them was (issue #116). This passed
        # `r["intended"]` RAW while normalising the candidate, so the target kept whatever
        # the manifest happened to hold. `select_reference:842` writes `_vat(intended)` for
        # the same reason; this fork normalised the half it had just been fixing.
        #
        # ⚠ IT BECAME REACHABLE BECAUSE OF #113's OWN FIX. The old all-or-nothing
        # `isinstance` gate rejected any row with a non-float axis, which — entirely by
        # accident — was what protected this subtraction. Routing the gate through
        # `coerce_axis` admitted the numeric string `"0.7"` that #58 ruled IS a label, and
        # `"0.7" - 0.5` is a `TypeError` at module scope: the whole bank build dies on the
        # first such row. `True - 0.5` is worse, scoring 0.5 in silence.
        score = _vat_distance(_vat(r["intended"]), keep_vat(k["intended_vat"]))
        score += ENGINE_PREF.get(k.get("engine"), 0.2)
        if k["file"] in used:
            score += 0.5          # prefer distinct refs across the 10 lines
        cands.append((score, k))
    cands.sort(key=lambda x: x[0])
    # `cands[0]` with no emptiness check raised IndexError and took the whole bank with
    # it. With the guards above narrowing the pool it is a reachable state, not a
    # theoretical one — and one unfillable line must not cost the other nine.
    if not cands:
        print(f"!! {r['id']}: no eligible reference (gender {g}, 4-10 s, not blacklisted, "
              f"excursion < {MAX_REF_EXCURSION:.0f} Hz) — skipping this line")
        continue
    ref = cands[0][1]
    used.add(ref["file"])
    bank.append({
        "id": r["id"].replace("qp3b", "qp3d"),
        "text": r["text"],
        "register": r["register"],
        "intended": r["intended"],
        "direction": r["direction"],          # G26 originals, for the audition card
        "ft_voice": FT_VOICE[g],
        "ref_wav": str(DS / "v1" / ref["file"]),
        "ref_text": ref["text"],
        "ref_meta": {"id": ref["id"], "register": ref["register"],
                     "engine": ref["engine"], "gender": ref["gender"],
                     # ⚠ `ref["intended_vat"]`, NOT `ref` — same defect as the scorer
                     # above (issue #107), and this half would WRITE IT TO THE ARTIFACT:
                     # `ref_meta.intended_vat` recorded as `{V: 0, A: 0, T: 0}` for every
                     # clip is a forged neutral in a manifest, the failure `schemas.py`
                     # opens by naming and that #92 spent two passes taking out of the
                     # writers.
                     #
                     # ⚠ NO SHIPPED BANK CARRIES IT — checked, not assumed. The only
                     # `v3d_bank.json` on disk is dated 2026-07-23, two weeks BEFORE the
                     # 98734f9 refactor that broke this, and its `ref_meta.intended_vat`
                     # values are real and varied. The window between the break and this
                     # fix contains no run. Nothing needs regenerating.
                     "intended_vat": keep_vat(ref["intended_vat"]),
                     "score": round(cands[0][0], 3)},
    })

OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / "v3d_bank.json"
out.write_text(json.dumps(bank, indent=2))
print(f"wrote {out} ({len(bank)} lines)")
for b in bank:
    m = b["ref_meta"]
    print(f'{b["id"][:14]:14} {b["register"][:20]:20} ft={b["ft_voice"]:4} '
          f'ref={m["id"][:28]:28} ({m["engine"]}, {m["gender"]}, d={m["score"]})')
