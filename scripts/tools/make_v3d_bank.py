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
import json, math, os, sys
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
from ref_select import (  # noqa: E402
    MAX_REF_EXCURSION,
    REF_BLACKLIST,
    _vat as keep_vat,
    design_gender,
)

DS = Path("/data/model-training/datasets/sonora-expressive-registers")
SRC = DS / "quote-pilot-v3b" / "moss85_manifest.jsonl"
KEEPS = DS / "v1" / "metadata.jsonl"
OUT_DIR = DS / "quote-pilot-v3d"
ENGINE_PREF = {"moss85": 0.0, "longcat": 0.05, "qwen": 0.15, "dia": 0.3}
FT_VOICE = {"F": "tara", "M": "leo"}

AXES = ("V", "A", "T")


def complete_vat(d):
    """True when all three axes carry a real number.

    ⚠ `intended.V/A/T` MAY BE `null` SINCE 2026-08-17 (issue #92). `book_ingest` used to write
    `tag.get("valence", 0.0)`, so every row arrived with three floats and this file was written
    against that. An absent axis is now recorded honestly as `null` instead of being invented
    as a neutral 0.0 — which is the right call at the writer and makes the arithmetic below a
    `TypeError` here. A bool is refused too: `True - 0.4` is perfectly legal arithmetic and
    silently wrong.
    """
    return all(isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)
               for k in AXES)


def vat_dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in AXES))

rows = [json.loads(l) for l in SRC.open()]
keeps = [json.loads(l) for l in KEEPS.open()]

bank = []
used = set()
# ⚠ SKIPPED AND COUNTED, NEVER SILENTLY DROPPED. A line whose intended vector is incomplete
# cannot be distance-ranked against anything, so it has no place in this bank — but a bank
# that is quietly short is the failure this repo keeps paying for, so the count is printed.
unscoreable = [r for r in rows if not complete_vat(r["intended"])]
rows = [r for r in rows if complete_vat(r["intended"])]
if unscoreable:
    print(f"  ⚠ {len(unscoreable)} line(s) skipped: intended V/A/T is incomplete, so they "
          f"cannot be ranked by distance. Re-direct them or accept a shorter bank.")
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
        score = vat_dist(r["intended"], keep_vat(k))
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
                     "intended_vat": keep_vat(ref), "score": round(cands[0][0], 3)},
    })

OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / "v3d_bank.json"
out.write_text(json.dumps(bank, indent=2))
print(f"wrote {out} ({len(bank)} lines)")
for b in bank:
    m = b["ref_meta"]
    print(f'{b["id"][:14]:14} {b["register"][:20]:20} ft={b["ft_voice"]:4} '
          f'ref={m["id"][:28]:28} ({m["engine"]}, {m["gender"]}, d={m["score"]})')
