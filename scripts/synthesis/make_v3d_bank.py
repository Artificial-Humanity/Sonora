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
"""
import json, math, re
from pathlib import Path

DS = Path("/data/model-training/datasets/sonora-expressive-registers")
SRC = DS / "quote-pilot-v3b" / "moss85_manifest.jsonl"
KEEPS = DS / "v1" / "metadata.jsonl"
OUT_DIR = DS / "quote-pilot-v3d"
ENGINE_PREF = {"moss85": 0.0, "longcat": 0.05, "qwen": 0.15, "dia": 0.3}
FT_VOICE = {"F": "tara", "M": "leo"}

def design_gender(design: str) -> str:
    d = design.lower()
    return "F" if re.search(r"\b(female|woman|maternal|girl)\b", d) else "M"

def vat_dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in ("V", "A", "T")))

def keep_vat(k):
    iv = k["intended_vat"]
    return {"V": iv.get("V", iv.get("valence", 0)),
            "A": iv.get("A", iv.get("arousal", iv.get("energy", 0))),
            "T": iv.get("T", iv.get("tension", 0))}

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
        if not (3.0 <= float(k.get("duration", 0)) <= 10.0):
            continue
        score = vat_dist(r["intended"], keep_vat(k))
        score += ENGINE_PREF.get(k.get("engine"), 0.2)
        if k["file"] in used:
            score += 0.5          # prefer distinct refs across the 10 lines
        cands.append((score, k))
    cands.sort(key=lambda x: x[0])
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
