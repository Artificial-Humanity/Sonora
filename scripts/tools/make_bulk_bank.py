"""Expand bulk_spec.json (registers x line pools x job templates x seeds) into
a flat bank the synth_* renderers consume. Job ids are deterministic:
<register>_<lineNN>_<engine>_<voice>_s<seed>.

B-M7, 2026-08-07. This built each engine's `direction` payload itself, which AGENTS.md
forbids in as many words: `build_direction()` in book_ingest.py "is the single source of
truth for what each engine actually receives — never bypass it". The bypass had already
cost something. Its Dia lines were `f"[S1] {dia_tags}{text}"` — no TRAILING `[S1]`, which
is not decoration but the end-of-audio guard nari-labs' generation guidelines prescribe;
without it Dia improvises a tail, so the bank renders long and wrong while looking
correctly directed. The same class of defect as the 2026-07-25 relay audit.

The bypass also explained itself: `build_direction` had no slot for `quality` (which
`synth_moss85.py` genuinely reads) or for Dia inline tags (which this spec genuinely
uses), so a builder that wanted them had to fork. Both slots now exist there, and this
script routes through it.

Ids gained the engine. Two jobs differing only by engine produced the SAME id and would
have collided in the shared output directory; the current spec escapes that only because
each engine happens to use a distinct `voice` name, which is a convention, not a guard.

Usage: python make_bulk_bank.py --spec bulk_spec.json --out bulk_bank.json
"""
import argparse
import json
import os
import sys

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
from book_ingest import build_direction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = json.load(open(args.spec, encoding="utf-8"))
    lines_out = []
    for reg, rdef in spec["registers"].items():
        for li, line in enumerate(rdef["lines"]):
            text = line["text"] if isinstance(line, dict) else line
            extras = line if isinstance(line, dict) else {}
            for job in rdef["jobs"]:
                for seed in job["seeds"]:
                    # Per-line placeholder substitution stays here: it is a property of
                    # THIS spec format ({tone}, {pace} …), not of the engine contract.
                    instruct = job.get("instruct", "")
                    for k, v in extras.items():
                        if k != "text":
                            instruct = instruct.replace("{" + k + "}", v)
                    tag = {
                        "engine": job["engine"],
                        "voice_design": job.get("design", ""),
                        "instruct": instruct,
                        "quality": job.get("quality"),
                        "dia_tags": job.get("dia_tags"),
                    }
                    engine, direction = build_direction(
                        tag, text, dia_guidance=job.get("guidance", 3.0))
                    lines_out.append({
                        "id": f"{reg}_{li:02d}_{engine}_{job['voice']}_s{seed}",
                        "engine": engine, "register": reg,
                        "intended": rdef["intended"], "seed": seed,
                        "text": text, "direction": direction,
                    })
    bank = {"version": spec["version"], "campaign": spec["campaign"],
            "license_note": spec["license_note"], "lines": lines_out}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=1, ensure_ascii=False)
    by_eng = {}
    for l in lines_out:
        by_eng[l["engine"]] = by_eng.get(l["engine"], 0) + 1
    print(f"{len(lines_out)} jobs -> {args.out}  {by_eng}")


if __name__ == "__main__":
    main()
