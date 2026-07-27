"""Reference-clip casting for cloning engines (VibeVoice) — v3d-proven logic.

Given a direction design text + intended VAT, selects an audited keep from the
certified dataset (gender parse + intended-VAT proximity + duration window +
casting-faithful engine preference). This is the production home of the logic
piloted in make_v3d_bank.py.

Known limitation (v3d/v3e finding, 2026-07-23): the current reference pool is
own-synthesis keeps whose heritage skews young — age fidelity is bounded by the
pool, not the cloning engine (measured: render-vs-reference dF0 ~ +1-3%).
Real-speech pools + measured age norms (casting-attribute-norms brief) are the
upgrade path; swap POOL_PATH when they land.
"""
import json
import math
import re
from pathlib import Path

POOL_PATH = Path("/data/model-training/datasets/sonora-expressive-registers/v1/metadata.jsonl")
POOL_ROOT = POOL_PATH.parent
ACOUSTICS_PATH = POOL_ROOT / "pool_acoustics.json"
ENGINE_PREF = {"moss85": 0.0, "longcat": 0.05, "qwen": 0.15, "dia": 0.3}

# Age is carried by the reference's acoustics (v3d/v3e finding: renders copy the
# reference F0 register within ~2%), so the design's age band maps to an F0
# percentile target WITHIN gender. Crude but directionally correct until the
# measured casting norms land (casting-attribute-norms brief).
AGE_BANDS = [
    (r"\b(child|little (girl|boy)|kid)\b",          0.95),
    (r"\b(teen|adolescent|girlish|boyish)\b",       0.80),
    # 0.70 = the "adult" band. `young` is the classic trigger, but designs routinely say
    # "adult male" or give a decade ("mid-30s", "in her twenties") and those matched
    # NOTHING before 2026-07-27 — 23 of 39 historical rows recorded no age band despite
    # naming one in plain English.
    (r"\b(young|adult|twenties|thirties|20s|30s)\b",  0.70),
    (r"\b(middle.?aged|matronly|mature|forties|fifties|40s|50s)\b", 0.30),
    (r"\b(elderly|old (woman|man|lady)|aged|weathered|grandmother|grandfather)\b", 0.10),
]
AGE_WEIGHT = 0.8

_pool = None
_acoustics = None
_f0_pct = None


def _load_acoustics():
    global _acoustics, _f0_pct
    if _f0_pct is None:
        _acoustics = json.loads(ACOUSTICS_PATH.read_text()) if ACOUSTICS_PATH.exists() else {}
        _f0_pct = {}
        by_gender = {}
        for k in _load_pool():
            a = _acoustics.get(k["file"])
            if a:
                by_gender.setdefault(k.get("gender", "?")[:1].upper(), []).append((a["f0_median"], k["file"]))
        for g, vals in by_gender.items():
            vals.sort()
            n = max(len(vals) - 1, 1)
            for i, (_, f) in enumerate(vals):
                _f0_pct[f] = i / n
    return _f0_pct


# Owner taxonomy (casting-attribute-norms-brief.md): child / teen / adult /
# middle-aged / elderly. The 0.70 band is named "adult" to match the brief — note the
# TRIGGER WORD in a design is still "young" (that is what the regex matches and what a
# director writes); only the recorded band name is "adult".
AGE_BAND_NAMES = {0.95: "child", 0.80: "teen", 0.70: "adult", 0.30: "middle-aged", 0.10: "elderly"}


def design_age_target(design: str):
    d = (design or "").lower()
    for pat, target in AGE_BANDS:
        if re.search(pat, d):
            return target
    return None  # unspecified: no age term applied


def design_age_band(design: str):
    """Canonical age label for training attribution (owner taxonomy).

    Returns "" when the design names no age at all. Previously this returned "adult"
    for BOTH an explicit young/adult design and an unmarked one — and audit_sampler
    skips "adult" as "no age claim to betray", so explicitly-aged designs were being
    silently exempted from the age-mismatch check along with the genuinely unmarked
    (found 2026-07-27). An unmarked design has no claim; an explicit one does.
    """
    t = design_age_target(design)
    return AGE_BAND_NAMES[t] if t is not None else ""


def _load_pool():
    global _pool
    if _pool is None:
        _pool = [json.loads(l) for l in POOL_PATH.open()]
    return _pool


def design_gender(design: str) -> str:
    d = (design or "").lower()
    return "F" if re.search(r"\b(female|woman|maternal|girl)\b", d) else "M"


def _vat(v):
    return {"V": v.get("V", v.get("valence", 0)),
            "A": v.get("A", v.get("arousal", v.get("energy", 0))),
            "T": v.get("T", v.get("tension", 0))}


def select_reference(design: str, intended: dict, used: set | None = None):
    """Returns (ref_wav_path, ref_text, ref_meta). `used` biases toward variety."""
    used = used or set()
    want, g = _vat(intended), design_gender(design)
    age_target = design_age_target(design)
    best, best_score = None, 1e9
    for k in _load_pool():
        if k.get("gender", "")[:1].upper() != g:
            continue
        if not (4.0 <= float(k.get("duration", 0)) <= 10.0):   # owner floor 2026-07-25
            continue
        kv = _vat(k["intended_vat"])
        score = math.sqrt(sum((want[a] - kv[a]) ** 2 for a in "VAT"))
        score += ENGINE_PREF.get(k.get("engine"), 0.2)
        if age_target is not None:
            pct = _load_acoustics().get(k["file"])
            if pct is not None:
                score += AGE_WEIGHT * abs(pct - age_target)
        if k["file"] in used:
            score += 0.5
        if score < best_score:
            best, best_score = k, score
    if best is None:
        raise LookupError(f"no reference for gender={g}")
    used.add(best["file"])
    meta = {"id": best["id"], "register": best["register"], "engine": best["engine"],
            "gender": best["gender"], "score": round(best_score, 3)}
    pct = _load_acoustics().get(best["file"])
    if pct is not None:
        meta["ref_f0_pct"] = round(pct, 2)   # age evidence: within-gender F0 percentile
    return (str(POOL_ROOT / best["file"]), best["text"], meta)
