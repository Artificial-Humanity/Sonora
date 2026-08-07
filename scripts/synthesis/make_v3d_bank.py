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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

def vat_dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in ("V", "A", "T")))

rows = [json.loads(l) for l in SRC.open()]
keeps = [json.loads(l) for l in KEEPS.open()]

bank = []
used = set()
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
