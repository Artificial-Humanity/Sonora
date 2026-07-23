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
ENGINE_PREF = {"moss85": 0.0, "longcat": 0.05, "qwen": 0.15, "dia": 0.3}

_pool = None


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
    best, best_score = None, 1e9
    for k in _load_pool():
        if k.get("gender", "")[:1].upper() != g:
            continue
        if not (3.0 <= float(k.get("duration", 0)) <= 10.0):
            continue
        kv = _vat(k["intended_vat"])
        score = math.sqrt(sum((want[a] - kv[a]) ** 2 for a in "VAT"))
        score += ENGINE_PREF.get(k.get("engine"), 0.2)
        if k["file"] in used:
            score += 0.5
        if score < best_score:
            best, best_score = k, score
    if best is None:
        raise LookupError(f"no reference for gender={g}")
    used.add(best["file"])
    return (str(POOL_ROOT / best["file"]), best["text"],
            {"id": best["id"], "register": best["register"], "engine": best["engine"],
             "gender": best["gender"], "score": round(best_score, 3)})
